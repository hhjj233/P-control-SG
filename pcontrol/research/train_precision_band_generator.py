#!/usr/bin/env python3
"""Fine-band continuation from saved raw/EMA/optimizer/RNG state; no new data."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import time
import numpy as np
import torch
from pcontrol.research import finetune_joint_risk_generator_v2 as previous
from pcontrol.generation.terminal_precision import terminal_precision_losses
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY,quantile_from_pieces
from pcontrol.generation.random_stream_state import paired_stream_state,slow_cycle_state,restore_paired_streams,restore_slow_cycle

old=previous.old;multi=previous.multi;ROOT=previous.ROOT
RulerContextPercentileDenoiser=previous.RulerContextPercentileDenoiser
speed_strata=previous.speed_strata;SlowHistoryCycle=previous.SlowHistoryCycle
update_adapter_ema=previous.update_adapter_ema
PROTOCOL='natural_GP_precision_band_continuation_v1'
POLICY_SHA='8760813ec8e2d4be221071ae0501947cd642e3320ba2d55b63a359677f64a5f9'
CODE=previous.CODE+('pcontrol/generation/fine_band_loss.py',
    'pcontrol/generation/terminal_precision.py','pcontrol/research/train_precision_band_generator.py')


def read_policy(binding):
    if binding.get('sha256')!=POLICY_SHA:raise ValueError('frozen precision-band policy required')
    patch=old.bound_json(binding)
    if patch['protocol']!=PROTOCOL or patch['recipes']!=['precision_band']:raise ValueError('wrong precision trial')
    policy=previous.read_policy(patch['base_policy'])
    policy.update(protocol=PROTOCOL,recipes=patch['recipes'],precision_base_policy=patch['base_policy'],
        resume_checkpoint=patch['resume_training_state'],resume_completed_epochs=patch['resume_completed_epochs'],
        output_root=patch['output_root'])
    policy['training'].update(patch['training_overrides'])
    policy['evaluation']['CFG_scale']=patch['evaluation_CFG_scale']
    for k in ('resume_training_state','reference_setting'):old.verify_binding(patch[k])
    return policy


def load_inputs(policy,device):
    source=previous.load_inputs(policy,device)
    source['codes'].update({p:old.sha256(ROOT/p) for p in CODE})
    return source


def context_values(policy,binding,source):
    b=policy['precision_base_policy'];return previous.context_values(previous.read_policy(b),b,source)


def context_batch(pack,p_values,rows,shape_values,recipe,device):
    if recipe!='precision_band':raise ValueError('precision-band recipe required')
    return previous.context_batch(pack,p_values,rows,shape_values,'joint_generator',device)

def compute_auxiliary(model,schedule,features,requested_p,noise,history_presence,teacher_batch,physical_batch,cfg,physics,coverage):
    slots=coverage['slots_per_history'];presence=history_presence.repeat_interleave(slots)
    reference,decoders,geometries=teacher_batch
    eligibility=multi.target_eligibility(reference,requested_p,interior_margin=coverage['interior_margin'],atom_tolerance=coverage['atom_tolerance'])
    sampled=old.training_ddim_suffix(model,schedule,features,requested_p,noise,steps=cfg['terminal_DDIM_steps'],
        grad_last_steps=cfg['grad_last_steps'],cfg_scale=cfg['terminal_CFG_scale'])
    target_pet=quantile_from_pieces(features[PIECES_KEY],1.-requested_p.detach().double()).detach()
    term=terminal_precision_losses(sampled['sample'],requested_p,reference,decoders,geometries,features['agent_mask'],
        presence&eligibility['eligible'],target_pet=target_pet,beta=cfg['terminal_beta'],minimum_density=cfg['minimum_density'],pet_beta=cfg['PET_target_beta'],fine_temperature=cfg['Fine_band_temperature'])
    numeric=multi.grouped_value_mean(term['per_scene_value_loss'],term['support_mask'],history_presence,slots)+sampled['sample'].sum()*0.
    pet=term['PET_target_value']
    pet_numeric=multi.grouped_value_mean(pet['per_scene_loss'],pet['support_mask'],history_presence,slots)+sampled['sample'].sum()*0.
    band=term['Fine_band']
    fine_numeric=multi.grouped_value_mean(band['per_scene_loss'],band['support_mask'],history_presence,slots)+sampled['sample'].sum()*0.
    dims,roads=physical_batch
    physical=multi.physics_pilot.batch_physics_loss(sampled['sample'],decoders,dims,roads,features['agent_mask'],presence,
        tail_fraction=physics['tail_fraction'],road_scale_m=physics['road_scale_m'],speed_scale_mps=physics['speed_scale_mps'])
    loss=numeric+cfg['PET_target_weight']*pet_numeric+cfg['Fine_band_weight']*fine_numeric+physics['road_weight']*physical['road_loss']+physics['speed_weight']*physical['speed_loss']
    return loss,dict(numeric=numeric,PET_numeric=pet_numeric,Fine_numeric=fine_numeric,terminal=term,physical=physical,eligibility=eligibility,sampling=sampled)

def weighted_auxiliary_backward(model,schedule,features,p,noise,history_presence,teacher_batch,physical_batch,cfg,physics,coverage):
    mode=model.training
    try:
        model.eval()
        loss,parts=compute_auxiliary(model,schedule,features,p,noise,history_presence,teacher_batch,physical_batch,cfg,physics,coverage)
        loss=loss+(cfg['numeric_weight']-1.)*parts['numeric']
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite coverage loss')
        loss.backward();term=parts['terminal'];physical=parts['physical'];eligible=parts['eligibility']
        return dict(total=len(p),histories=len(history_presence),present_histories=int(history_presence.sum()),
            target_eligible=int(eligible['eligible'].sum()),positive_floor=int((eligible['optimistic_endpoint_only_error_floor']>1e-8).sum()),
            floor_sum=float(eligible['optimistic_endpoint_only_error_floor'].sum()),
            supported=term['supported_scenes'],geometry_supported=term['geometry_supported_scenes'],reasons=term['geometry_reasons'],
            numeric=float(parts['numeric'].detach()),Fine_numeric=float(parts['Fine_numeric'].detach()),PET_numeric=float(parts['PET_numeric'].detach()),
            PET_supported=term['PET_target_value']['supported_scenes'],
            PET_error_sum=term['PET_target_value']['all_request_PET_target_MAE_seconds']*len(p),point_sum=term['all_request_point_MAE']*len(p),
            road_loss=float(physical['road_loss'].detach()),speed_loss=float(physical['speed_loss'].detach()),
            road_events=sum(x['road_has_training_violation'] for x in physical['diagnostics']),
            speed_events=sum(x['speed_has_training_violation'] for x in physical['diagnostics']),
            forward_NFE=parts['sampling']['network_evaluations'],backward_NFE=parts['sampling']['backward_recompute_network_evaluations'])
    finally:model.train(mode)
def train_epoch(model,ema,schedule,optimizer,pack,p_values,cache,teacher,streams,aux_rng,cfg,physics,coverage,recipe,device,
                *,shape_values,slow_cycle,progress=None,max_updates=None):
    model.train();order=streams.order(len(p_values));hashes={k:hashlib.sha256() for k in ('base_noise','drop','presence','aux_noise_plan')}
    target_hash=hashlib.sha256();slots=coverage['slots_per_history'];started=time.perf_counter();reasons=Counter()
    stat=dict(base_loss_scenes=0,updates=0,base_v_sum=0.,projection_support=0,jacobian_support=0,p_dropped=0,
        auxiliary_histories=0,auxiliary_requests=0,present_histories=0,target_eligible=0,positive_target_floor=0,
        supported=0,geometry_supported=0,point_sum=0.,floor_sum=0.,numeric_sum=0.,road_events=0,speed_events=0,
        road_loss_sum=0.,speed_loss_sum=0.,forward_NFE=0,backward_NFE=0,gradient_norm_max=0.,Fine_numeric_sum=0.,PET_numeric_sum=0.,PET_supported=0,PET_error_sum=0.)
    for start in range(0,len(order),cfg['batch_size']):
        if max_updates is not None and stat['updates']>=max_updates:break
        rows=order[start:start+cfg['batch_size']]
        f,c,p=context_batch(pack,p_values,rows,shape_values,recipe,device);p=p.detach().requires_grad_(True)
        times,noise,u,present=streams.draw(c.shape,schedule.steps,cfg['p_dropout_probability'])
        for a in (np.asarray(c.shape,dtype=np.int64),times.numpy(),noise.numpy()):hashes['base_noise'].update(a.tobytes())
        hashes['drop'].update(u.tobytes());hashes['presence'].update(present.tobytes())
        presence=torch.from_numpy(present).to(device);t=times.to(device);optimizer.zero_grad(set_to_none=True)
        base=old.cfg_training_loss(model,schedule,c,f,p,presence,timesteps=t,noise=noise.to(device))
        extra=old.natural_tangent_losses(base['model_prediction'],base['prediction_target'],base['noisy_coefficients'],p,
            schedule.alpha_bar(t,base['model_prediction']),f['agent_mask'],condition_present=presence,
            projection_weight=cfg['loss_weights']['projection'],jacobian_weight=cfg['loss_weights']['jacobian'],
            min_jacobian_sigma=cfg['min_jacobian_sigma'],**old.tangent.cache_batch(cache,rows,c.shape[1],device))
        if extra['diagnostics']['jacobian_evaluated'] and extra['diagnostics']['p_graph_connected'] is False:
            raise RuntimeError('base response is disconnected')
        loss=base['loss']+extra['loss']
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite base loss')
        loss.backward();stat['base_v_sum']+=float(base['loss'].detach())*len(rows)
        stat['projection_support']+=extra['support_counts']['projection'];stat['jacobian_support']+=extra['support_counts']['jacobian']
        del base,extra,loss
        selected=slow_cycle.select(rows);repeated=np.repeat(selected,slots)
        af,ac,_=context_batch(pack,p_values,repeated,shape_values,recipe,device)
        natural=torch.tensor(p_values[selected],dtype=torch.float32,device=device)
        target=multi.make_targets(natural,'p_grid',grid=coverage['grid']);z=torch.randn(tuple(ac.shape),generator=aux_rng)
        for a in (repeated.astype(np.int64),np.asarray(z.shape,dtype=np.int64),z.numpy()):hashes['aux_noise_plan'].update(a.tobytes())
        target_hash.update(target.detach().cpu().numpy().tobytes())
        aux=weighted_auxiliary_backward(model,schedule,af,target,z.to(device),presence[:len(selected)],teacher.batch(repeated),
            teacher.physical_batch(repeated),cfg,physics,coverage)
        stat['auxiliary_histories']+=aux['histories'];stat['auxiliary_requests']+=aux['total'];stat['present_histories']+=aux['present_histories']
        stat['target_eligible']+=aux['target_eligible'];stat['positive_target_floor']+=aux['positive_floor'];stat['floor_sum']+=aux['floor_sum']
        stat['supported']+=aux['supported'];stat['geometry_supported']+=aux['geometry_supported'];reasons.update(aux['reasons'])
        stat['point_sum']+=aux['point_sum'];stat['numeric_sum']+=aux['numeric']
        stat['Fine_numeric_sum']+=aux['Fine_numeric'];stat['PET_numeric_sum']+=aux['PET_numeric'];stat['PET_supported']+=aux['PET_supported'];stat['PET_error_sum']+=aux['PET_error_sum']
        for key in ('road_events','speed_events','forward_NFE','backward_NFE'):stat[key]+=aux[key]
        stat['road_loss_sum']+=aux['road_loss'];stat['speed_loss_sum']+=aux['speed_loss']
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
        if not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite combined gradient')
        stat['gradient_norm_max']=max(stat['gradient_norm_max'],float(norm));optimizer.step();update_adapter_ema(ema,model,cfg['EMA_decay'])
        stat['updates']+=1;stat['base_loss_scenes']+=len(rows);stat['p_dropped']+=int((~present).sum())
        if progress and stat['updates']%10==0:progress(dict(recipe=recipe,updates=stat['updates'],seconds=time.perf_counter()-started))
    stat.update(randomness={k:h.hexdigest() for k,h in hashes.items()},targets_sha256=target_hash.hexdigest(),
        order_sha256=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest(),geometry_reasons=dict(reasons),
        all_base_loss_rows_retained=stat['base_loss_scenes']==len(p_values),
        all_request_point_MAE=stat['point_sum']/stat['auxiliary_requests'],wall_seconds=time.perf_counter()-started)
    return stat
def train(binding,recipe,smoke=False):
    policy=read_policy(binding)
    if recipe not in policy['recipes']:raise ValueError('wrong recipe')
    device=old.configure(policy);cfg=policy['training'];torch.manual_seed(cfg['seed'])
    source=load_inputs(policy,device);shape_values,context_binding=context_values(policy,binding,source)
    kw={k:source['checkpoint']['architecture'][k] for k in old.ARCH_KEYS}
    model=RulerContextPercentileDenoiser(**kw,context_hidden_dim=policy['context']['hidden_dim'],dynamic_bottleneck=policy['dynamic_bottleneck'])
    resumed=torch.load(old.verify_binding(policy['resume_checkpoint']),map_location='cpu')
    if (resumed['protocol']!=previous.PROTOCOL or resumed['epoch']!=policy['resume_completed_epochs']
            or resumed['smoke'] or resumed['data']!=source['base']['data']
            or resumed['labels_manifest']!=source['base']['labels_manifest']):
        raise ValueError('wrong continuation checkpoint')
    old.check_codes(resumed['code_sha256'])
    model.load_state_dict(resumed['model_state'],strict=True);model.to(device)
    groups=model.finetuning_groups(cfg['core_learning_rate'],cfg['adapter_learning_rate'])
    trainable=sum(v.numel() for v in model.parameters() if v.requires_grad)
    initial_buffers={k:v.detach().cpu().clone() for k,v in model.named_buffers()}
    if not all(torch.equal(v.detach().cpu(),resumed['model_state'][k]) for k,v in model.state_dict().items()):
        raise ValueError('full-generator warm start changed initial tensors')
    ema=copy.deepcopy(model).requires_grad_(False);ema.load_state_dict(resumed['EMA_state'],strict=True)
    schedule=old.CosineDiffusionSchedule(100).to(device)
    root=old.resolve(policy['output_root'])/(recipe+'_smoke' if smoke else recipe);root.mkdir(parents=True,exist_ok=False)
    freeze=old.write_json(root/'freeze_before_training.json',dict(protocol=PROTOCOL,policy=binding,
        resolved_policy=policy,recipe=recipe,smoke=smoke,initial_state_sha256=old.state_hash(model),
        context=context_binding,code_sha256=source['codes'],trainable_parameters=trainable,resume_checkpoint=policy['resume_checkpoint'],
        OOF_p_replay_error=source['teacher'].replay_max_error,STOP_CAL_AUDIT_observations_accessed=False))
    optimizer=torch.optim.AdamW(groups,weight_decay=cfg['weight_decay']);optimizer.load_state_dict(resumed['optimizer_state'])
    streams=old.PairedStreams(cfg['seed'],cfg['p_dropout_seed']);rng=torch.Generator().manual_seed(cfg['terminal_noise_seed'])
    speed,strata=speed_strata(source['teacher'].physical['history'],source['pack']['agent_mask'],source['pack']['role'])
    slow_cycle=SlowHistoryCycle(source['pack']['recording_id'],strata,seed=policy['history_sampling']['seed'])
    restore_paired_streams(streams,resumed['paired_streams_state']);rng.set_state(resumed['auxiliary_generator_state'].cpu())
    restore_slow_cycle(slow_cycle,resumed['slow_history_cycle_state'])
    torch.set_rng_state(resumed['torch_CPU_RNG'].cpu())
    if device.type=='cuda' and resumed['torch_device_RNG'] is not None:torch.cuda.set_rng_state(resumed['torch_device_RNG'].cpu(),device)
    if [g['lr'] for g in optimizer.param_groups]!=[cfg['core_learning_rate'],cfg['adapter_learning_rate']]:
        raise ValueError('restored optimizer learning rates changed')
    checkpoints={};restart_states={};started=time.perf_counter()
    def save(epoch):
        for candidate in (model,ema):
            if any(not torch.equal(v.detach().cpu(),initial_buffers[k]) for k,v in candidate.named_buffers()):
                raise ValueError('fixed generator buffers changed')
        path=root/('ema_epoch_%03d.pt'%epoch)
        data=source['data'];base=source['base']
        checkpoint=dict(protocol=PROTOCOL,policy=binding,recipe=recipe,epoch=epoch,smoke=smoke,EMA=True,
            state_dict={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},architecture=ema.architecture_config(),
            schedule=schedule.as_dict(),prediction_type='v',data=base['data'],labels_manifest=base['labels_manifest'],
            parent_checkpoint=policy['parent_checkpoint'],resume_checkpoint=policy['resume_checkpoint'],risk_cache=policy['risk_cache'],basis=data['basis'],
            coefficient_normalizer=data['coefficient_normalizer'],history_normalizer=data['history_normalizer'],
            context=context_binding,code_sha256=source['codes'])
        with path.open('xb') as handle:torch.save(checkpoint,handle)
        checkpoints[str(epoch)]=dict(path=str(path),sha256=old.sha256(path))
    save(0)
    with (root/'epochs.jsonl').open('x') as log:
        for epoch in range(1,(1 if smoke else cfg['epochs'])+1):
            record=train_epoch(model,ema,schedule,optimizer,source['pack'],source['p'],source['cache'],source['teacher'],
                streams,rng,cfg,policy['physics'],policy['coverage'],recipe,device,shape_values=shape_values,slow_cycle=slow_cycle,
                max_updates=1 if smoke else None,progress=lambda r:print(json.dumps(dict(epoch=epoch,**r)),flush=True))
            record.update(epoch=epoch,global_epoch=policy['resume_completed_epochs']+epoch,recipe=recipe)
            log.write(json.dumps(record,sort_keys=True)+'\n');log.flush()
            if epoch in cfg['snapshot_epochs'] or smoke:save(epoch)
            state_path=root/('training_state_epoch_%03d.pt'%epoch)
            state=dict(protocol=PROTOCOL,policy=binding,epoch=epoch,global_epoch=policy['resume_completed_epochs']+epoch,smoke=smoke,architecture=model.architecture_config(),
                model_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                EMA_state={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},
                optimizer_state=optimizer.state_dict(),paired_streams_state=paired_stream_state(streams),
                auxiliary_generator_state=rng.get_state(),slow_history_cycle_state=slow_cycle_state(slow_cycle),
                torch_CPU_RNG=torch.get_rng_state(),
                torch_device_RNG=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,
                data=source['base']['data'],labels_manifest=source['base']['labels_manifest'],context=context_binding,
                code_sha256=source['codes'],exact_restart_not_yet_tested=True)
            with state_path.open('xb') as handle:torch.save(state,handle)
            restart_states[str(epoch)]=dict(path=str(state_path),sha256=old.sha256(state_path))
            print(json.dumps(record,sort_keys=True),flush=True)
    old.check_codes(source['codes'])
    result=dict(protocol=PROTOCOL,status='smoke_complete' if smoke else 'complete',policy=binding,
        recipe=recipe,smoke=smoke,epochs_completed=1 if smoke else cfg['epochs'],checkpoints=checkpoints,freeze=freeze,
        code_sha256=source['codes'],architecture=ema.architecture_config(),trainable_parameters=trainable,
        data=source['base']['data'],labels_manifest=source['base']['labels_manifest'],parent_checkpoint=policy['parent_checkpoint'],resume_checkpoint=policy['resume_checkpoint'],
        epochs=dict(path=str(root/'epochs.jsonl'),sha256=old.sha256(root/'epochs.jsonl')),
        context=context_binding,wall_seconds=time.perf_counter()-started,parent_parameters_unchanged=False,generator_core_updates_allowed=True,restart_states=restart_states,
        optimizer_and_streams_restored=True,production_default_changed=False,STOP_CAL_AUDIT_observations_accessed=False)
    print(json.dumps(dict(training_complete=old.write_json(root/'result.json',result))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',default='configs/natural_percentile/precision_band_continuation_v1.json')
    parser.add_argument('--policy-sha256',default=POLICY_SHA);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();train(dict(path=args.policy,sha256=args.policy_sha256),'precision_band',args.smoke)

