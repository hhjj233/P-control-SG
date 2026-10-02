"""Versioned full-DDIM training for atom-aware targets; original P stays input."""
from collections import Counter
import hashlib
import time
import numpy as np
import torch

from pcontrol.research import train_wide_coupled_risk_generator as legacy
from pcontrol.generation.torch_atom_aware_target import midrank_target_from_pieces
from pcontrol.generation.terminal_atom_aware import terminal_atom_losses
from pcontrol.generation.encounter_support_loss import batch_encounter_losses
from pcontrol.generation.wide_risk_sampling import coupled_request_noise
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY


def compute_auxiliary(model,schedule,features,p,noise,history_presence,teacher_batch,physical_batch,cfg,physics,coverage):
    slots=coverage['slots_per_history'];presence=history_presence.repeat_interleave(slots)
    reference,decoders,geometries=teacher_batch;multi=legacy.multi
    eligibility=multi.target_eligibility(reference,p,interior_margin=coverage['interior_margin'],atom_tolerance=coverage['atom_tolerance'])
    target,details=midrank_target_from_pieces(features[PIECES_KEY],p.detach(),return_details=True)
    target=target.detach()
    sampled=legacy.old.training_ddim_suffix(model,schedule,features,p,noise,steps=cfg['terminal_DDIM_steps'],
        grad_last_steps=cfg['grad_last_steps'],cfg_scale=cfg['terminal_CFG_scale'])
    sample=sampled['sample'];mask=features['agent_mask']
    term=terminal_atom_losses(sample,p,reference,decoders,geometries,mask,presence,target_pet=target,
        scalar_error_infimum=eligibility['optimistic_endpoint_only_error_floor'],exact_target_eligible=eligibility['eligible'],
        beta=cfg['terminal_beta'],minimum_density=cfg['minimum_density'],pet_beta=cfg['PET_target_beta'],fine_temperature=cfg['Fine_band_temperature'])
    def mean(values,support):return multi.grouped_value_mean(values,support,history_presence,slots)+sample.sum()*0.
    numeric=mean(term['per_scene_value_loss'],term['support_mask'])
    pet=term['PET_target_value'];pet_numeric=mean(pet['per_scene_loss'],pet['support_mask'])
    band=term['Fine_band'];fine_numeric=mean(band['per_scene_loss'],band['support_mask'])
    dims,roads=physical_batch
    physical=multi.physics_pilot.batch_physics_loss(sample,decoders,dims,roads,mask,presence,
        tail_fraction=physics['tail_fraction'],road_scale_m=physics['road_scale_m'],speed_scale_mps=physics['speed_scale_mps'])
    encounter=batch_encounter_losses(sample,decoders,dims,mask,term['exact_current_PET'],target,presence,margin_m=cfg['encounter_margin_m'])
    encounter_numeric=mean(encounter['per_scene_loss'],encounter['support_mask'])
    loss=(cfg['numeric_weight']*numeric+cfg['PET_target_weight']*pet_numeric+cfg['Fine_band_weight']*fine_numeric
          +cfg['encounter_weight']*encounter_numeric+physics['road_weight']*physical['road_loss']+physics['speed_weight']*physical['speed_loss'])
    return loss,dict(numeric=numeric,PET_numeric=pet_numeric,Fine_numeric=fine_numeric,encounter_numeric=encounter_numeric,
        terminal=term,physical=physical,eligibility=eligibility,sampling=sampled,encounter=encounter,target_details=details)


def auxiliary_backward(model,schedule,features,p,noise,history_presence,teacher_batch,physical_batch,cfg,physics,coverage):
    mode=model.training
    try:
        model.eval()
        loss,parts=compute_auxiliary(model,schedule,features,p,noise,history_presence,teacher_batch,physical_batch,cfg,physics,coverage)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite atom-aware terminal loss')
        loss.backward();term=parts['terminal'];phy=parts['physical'];elig=parts['eligibility']
        return dict(total=len(p),histories=len(history_presence),present_histories=int(history_presence.sum()),
            target_eligible=int(elig['eligible'].sum()),positive_floor=int((elig['optimistic_endpoint_only_error_floor']>1e-8).sum()),
            floor_sum=float(elig['optimistic_endpoint_only_error_floor'].sum()),supported=term['supported_scenes'],
            geometry_supported=term['geometry_supported_scenes'],reasons=term['geometry_reasons'],
            target_changed=int(parts['target_details']['target_changed'].sum()),
            encounter_numeric=float(parts['encounter_numeric'].detach()),recruit_count=parts['encounter']['recruit_count'],escape_count=parts['encounter']['escape_count'],
            numeric=float(parts['numeric'].detach()),Fine_numeric=float(parts['Fine_numeric'].detach()),PET_numeric=float(parts['PET_numeric'].detach()),
            PET_supported=term['PET_target_value']['supported_scenes'],
            PET_error_sum=term['PET_target_value']['all_request_PET_target_MAE_seconds']*len(p),point_sum=term['all_request_point_MAE']*len(p),
            excess_sum=float(term['all_request_excess_error'].sum()),road_loss=float(phy['road_loss'].detach()),speed_loss=float(phy['speed_loss'].detach()),
            road_events=sum(x['road_has_training_violation'] for x in phy['diagnostics']),
            speed_events=sum(x['speed_has_training_violation'] for x in phy['diagnostics']),
            forward_NFE=parts['sampling']['network_evaluations'],backward_NFE=parts['sampling']['backward_recompute_network_evaluations'])
    finally:model.train(mode)


def train_epoch(model,ema,schedule,optimizer,pack,p_values,cache,teacher,streams,aux_rng,cfg,physics,coverage,recipe,device,
                *,shape_values,slow_cycle,progress=None,max_updates=None):
    if recipe!='atom_aware':raise ValueError('explicit atom-aware arm required')
    model.train();order=streams.order(len(p_values));hashes={k:hashlib.sha256() for k in ('base_noise','drop','presence','aux_noise_plan')}
    target_hash=hashlib.sha256();slots=coverage['slots_per_history'];started=time.perf_counter();reasons=Counter();old=legacy.old
    stat=dict(encounter_numeric_sum=0.,recruit_count=0,escape_count=0,base_loss_scenes=0,updates=0,base_v_sum=0.,projection_support=0,jacobian_support=0,p_dropped=0,
        auxiliary_histories=0,auxiliary_requests=0,present_histories=0,target_eligible=0,positive_target_floor=0,
        supported=0,geometry_supported=0,point_sum=0.,floor_sum=0.,numeric_sum=0.,road_events=0,speed_events=0,
        road_loss_sum=0.,speed_loss_sum=0.,forward_NFE=0,backward_NFE=0,gradient_norm_max=0.,Fine_numeric_sum=0.,PET_numeric_sum=0.,PET_supported=0,PET_error_sum=0.,
        target_changed=0,excess_sum=0.)
    for start in range(0,len(order),cfg['batch_size']):
        if max_updates is not None and stat['updates']>=max_updates:break
        rows=order[start:start+cfg['batch_size']]
        f,clean,p=legacy.context_batch(pack,p_values,rows,shape_values,'wide_coupled',device);p=p.detach().requires_grad_(True)
        times,noise,u,present=streams.draw(clean.shape,schedule.steps,cfg['p_dropout_probability'])
        for a in (np.asarray(clean.shape,dtype=np.int64),times.numpy(),noise.numpy()):hashes['base_noise'].update(a.tobytes())
        hashes['drop'].update(u.tobytes());hashes['presence'].update(present.tobytes())
        presence=torch.from_numpy(present).to(device);t=times.to(device);optimizer.zero_grad(set_to_none=True)
        base=old.cfg_training_loss(model,schedule,clean,f,p,presence,timesteps=t,noise=noise.to(device))
        extra=old.natural_tangent_losses(base['model_prediction'],base['prediction_target'],base['noisy_coefficients'],p,
            schedule.alpha_bar(t,base['model_prediction']),f['agent_mask'],condition_present=presence,
            projection_weight=cfg['loss_weights']['projection'],jacobian_weight=cfg['loss_weights']['jacobian'],
            min_jacobian_sigma=cfg['min_jacobian_sigma'],**old.tangent.cache_batch(cache,rows,clean.shape[1],device))
        if extra['diagnostics']['jacobian_evaluated'] and extra['diagnostics']['p_graph_connected'] is False:raise RuntimeError('disconnected P condition')
        loss=base['loss']+extra['loss']
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite natural base loss')
        loss.backward();stat['base_v_sum']+=float(base['loss'].detach())*len(rows)
        stat['projection_support']+=extra['support_counts']['projection'];stat['jacobian_support']+=extra['support_counts']['jacobian']
        del base,extra,loss
        selected=slow_cycle.select(rows);repeated=np.repeat(selected,slots)
        af,ac,_=legacy.context_batch(pack,p_values,repeated,shape_values,'wide_coupled',device)
        natural=torch.tensor(p_values[selected],dtype=torch.float32,device=device)
        target=legacy.multi.make_targets(natural,'p_grid',grid=coverage['grid'])
        z=coupled_request_noise(tuple(ac.shape),len(selected),slots,generator=aux_rng)
        for a in (repeated.astype(np.int64),np.asarray(z.shape,dtype=np.int64),z.numpy()):hashes['aux_noise_plan'].update(a.tobytes())
        target_hash.update(target.detach().cpu().numpy().tobytes())
        aux=auxiliary_backward(model,schedule,af,target,z.to(device),presence[:len(selected)],teacher.batch(repeated),
            teacher.physical_batch(repeated),cfg,physics,coverage)
        for dst,src in [('auxiliary_histories','histories'),('auxiliary_requests','total'),('present_histories','present_histories'),
                ('target_eligible','target_eligible'),('positive_target_floor','positive_floor'),('floor_sum','floor_sum'),
                ('supported','supported'),('geometry_supported','geometry_supported'),('point_sum','point_sum'),('numeric_sum','numeric'),
                ('encounter_numeric_sum','encounter_numeric'),('recruit_count','recruit_count'),('escape_count','escape_count'),
                ('Fine_numeric_sum','Fine_numeric'),('PET_numeric_sum','PET_numeric'),('PET_supported','PET_supported'),('PET_error_sum','PET_error_sum'),
                ('road_events','road_events'),('speed_events','speed_events'),('forward_NFE','forward_NFE'),('backward_NFE','backward_NFE'),
                ('road_loss_sum','road_loss'),('speed_loss_sum','speed_loss'),('target_changed','target_changed'),('excess_sum','excess_sum')]:stat[dst]+=aux[src]
        reasons.update(aux['reasons'])
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
        if not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite combined gradient')
        stat['gradient_norm_max']=max(stat['gradient_norm_max'],float(norm));optimizer.step();legacy.update_adapter_ema(ema,model,cfg['EMA_decay'])
        stat['updates']+=1;stat['base_loss_scenes']+=len(rows);stat['p_dropped']+=int((~present).sum())
        if progress and stat['updates']%10==0:progress(dict(recipe=recipe,updates=stat['updates'],seconds=time.perf_counter()-started))
    stat.update(randomness={k:h.hexdigest() for k,h in hashes.items()},targets_sha256=target_hash.hexdigest(),
        order_sha256=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest(),geometry_reasons=dict(reasons),
        all_base_loss_rows_retained=stat['base_loss_scenes']==len(p_values),
        all_request_point_MAE=stat['point_sum']/stat['auxiliary_requests'],wall_seconds=time.perf_counter()-started)
    return stat
