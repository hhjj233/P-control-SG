#!/usr/bin/env python3
"""Matched natural-data target-coverage pilot, with no inference intervention."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from pcontrol.research import pilot_natural_terminal_physics as physics_pilot
from pcontrol.generation.percentile_target_coverage import target_eligibility,grouped_value_mean,make_targets

old=physics_pilot.old
ROOT=physics_pilot.ROOT
PROTOCOL='natural_multi_P_coverage_paired_training_v1'
POLICY_SHA='56404622fadd8dc4477356e13130d2f94e442d816839d1956894599becb0c081'
CODE=physics_pilot.CODE+('pcontrol/research/pilot_natural_multi_p.py','pcontrol/generation/percentile_target_coverage.py')


def read_policy(binding):
    if binding.get('sha256')!=POLICY_SHA:raise ValueError('only the frozen multi-p pilot is allowed')
    patch=old.bound_json(binding)
    if patch['protocol']!=PROTOCOL or patch['recipes']!=['natural_multi_noise','p_grid']:
        raise ValueError('wrong target coverage experiment')
    policy=physics_pilot.read_policy(patch['base_policy'])
    policy.update(protocol=PROTOCOL,recipes=patch['recipes'],runtime=patch['runtime'],coverage=patch['coverage'],
        output_root=patch['output_root'],coverage_base_policy=patch['base_policy'],
        previous_STOP=patch['previous_STOP'],historical_comparator=patch['historical_comparator'])
    policy['training']['terminal_weight']={r:1. for r in policy['recipes']}
    policy['training']['terminal_targets']='coverage_recipe; observed base targets remain own natural OOF p'
    budget=patch['training_budget'];cfg=policy['training']
    if (cfg['epochs']!=budget['epochs'] or cfg['batch_size']!=budget['base_batch']
            or cfg['terminal_per_batch']!=budget['histories_per_aux_batch']
            or patch['coverage']['slots_per_history']!=3 or budget['aux_requests_per_batch']!=12):
        raise ValueError('inherited budget mismatch')
    old.verify_binding(patch['previous_STOP'])
    return policy


def compute_auxiliary(model,schedule,features,requested_p,noise,history_presence,teacher_batch,physical_batch,cfg,physics,coverage):
    slots=coverage['slots_per_history'];presence=history_presence.repeat_interleave(slots)
    reference,decoders,geometries=teacher_batch
    eligibility=target_eligibility(reference,requested_p,interior_margin=coverage['interior_margin'],atom_tolerance=coverage['atom_tolerance'])
    sampled=old.training_ddim_suffix(model,schedule,features,requested_p,noise,steps=cfg['terminal_DDIM_steps'],
        grad_last_steps=cfg['grad_last_steps'],cfg_scale=cfg['terminal_CFG_scale'])
    term=old.terminal_batch_loss(sampled['sample'],requested_p,reference,decoders,geometries,features['agent_mask'],
        presence&eligibility['eligible'],beta=cfg['terminal_beta'],minimum_density=cfg['minimum_density'])
    numeric=grouped_value_mean(term['per_scene_value_loss'],term['support_mask'],history_presence,slots)+sampled['sample'].sum()*0.
    dims,roads=physical_batch
    physical=physics_pilot.batch_physics_loss(sampled['sample'],decoders,dims,roads,features['agent_mask'],presence,
        tail_fraction=physics['tail_fraction'],road_scale_m=physics['road_scale_m'],speed_scale_mps=physics['speed_scale_mps'])
    loss=numeric+physics['road_weight']*physical['road_loss']+physics['speed_weight']*physical['speed_loss']
    return loss,dict(numeric=numeric,terminal=term,physical=physical,eligibility=eligibility,sampling=sampled)


def add_auxiliary_backward(model,schedule,features,p,noise,history_presence,teacher_batch,physical_batch,cfg,physics,coverage):
    mode=model.training
    try:
        model.eval()
        loss,parts=compute_auxiliary(model,schedule,features,p,noise,history_presence,teacher_batch,physical_batch,cfg,physics,coverage)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite coverage loss')
        loss.backward();term=parts['terminal'];physical=parts['physical'];eligible=parts['eligibility']
        return dict(total=len(p),histories=len(history_presence),present_histories=int(history_presence.sum()),
            target_eligible=int(eligible['eligible'].sum()),positive_floor=int((eligible['optimistic_endpoint_only_error_floor']>1e-8).sum()),
            floor_sum=float(eligible['optimistic_endpoint_only_error_floor'].sum()),
            supported=term['supported_scenes'],geometry_supported=term['geometry_supported_scenes'],reasons=term['geometry_reasons'],
            numeric=float(parts['numeric'].detach()),point_sum=term['all_request_point_MAE']*len(p),
            road_loss=float(physical['road_loss'].detach()),speed_loss=float(physical['speed_loss'].detach()),
            road_events=sum(x['road_has_training_violation'] for x in physical['diagnostics']),
            speed_events=sum(x['speed_has_training_violation'] for x in physical['diagnostics']),
            forward_NFE=parts['sampling']['network_evaluations'],backward_NFE=parts['sampling']['backward_recompute_network_evaluations'])
    finally:model.train(mode)


def train_epoch(model,ema,schedule,optimizer,pack,p_values,cache,teacher,streams,aux_rng,cfg,physics,coverage,recipe,device,
                *,progress=None,max_updates=None):
    model.train();order=streams.order(len(p_values));hashes={k:hashlib.sha256() for k in ('base_noise','drop','presence','aux_noise_plan')}
    target_hash=hashlib.sha256();slots=coverage['slots_per_history'];started=time.perf_counter();reasons=Counter()
    stat=dict(base_loss_scenes=0,updates=0,base_v_sum=0.,projection_support=0,jacobian_support=0,p_dropped=0,
        auxiliary_histories=0,auxiliary_requests=0,present_histories=0,target_eligible=0,positive_target_floor=0,
        supported=0,geometry_supported=0,point_sum=0.,floor_sum=0.,numeric_sum=0.,road_events=0,speed_events=0,
        road_loss_sum=0.,speed_loss_sum=0.,forward_NFE=0,backward_NFE=0,gradient_norm_max=0.)
    for start in range(0,len(order),cfg['batch_size']):
        if max_updates is not None and stat['updates']>=max_updates:break
        rows=order[start:start+cfg['batch_size']]
        f,c,p=old.direct.tensor_batch(pack,p_values,rows,device);p=p.detach().requires_grad_(True)
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
        selected=rows[:min(cfg['terminal_per_batch'],len(rows))];repeated=np.repeat(selected,slots)
        af,ac,_=old.direct.tensor_batch(pack,p_values,repeated,device)
        natural=torch.tensor(p_values[selected],dtype=torch.float32,device=device)
        target=make_targets(natural,recipe,grid=coverage['grid']);z=torch.randn(tuple(ac.shape),generator=aux_rng)
        for a in (repeated.astype(np.int64),np.asarray(z.shape,dtype=np.int64),z.numpy()):hashes['aux_noise_plan'].update(a.tobytes())
        target_hash.update(target.detach().cpu().numpy().tobytes())
        aux=add_auxiliary_backward(model,schedule,af,target,z.to(device),presence[:len(selected)],teacher.batch(repeated),
            teacher.physical_batch(repeated),cfg,physics,coverage)
        stat['auxiliary_histories']+=aux['histories'];stat['auxiliary_requests']+=aux['total'];stat['present_histories']+=aux['present_histories']
        stat['target_eligible']+=aux['target_eligible'];stat['positive_target_floor']+=aux['positive_floor'];stat['floor_sum']+=aux['floor_sum']
        stat['supported']+=aux['supported'];stat['geometry_supported']+=aux['geometry_supported'];reasons.update(aux['reasons'])
        stat['point_sum']+=aux['point_sum'];stat['numeric_sum']+=aux['numeric']
        for key in ('road_events','speed_events','forward_NFE','backward_NFE'):stat[key]+=aux[key]
        stat['road_loss_sum']+=aux['road_loss'];stat['speed_loss_sum']+=aux['speed_loss']
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
        if not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite combined gradient')
        stat['gradient_norm_max']=max(stat['gradient_norm_max'],float(norm));optimizer.step();old.direct.prior.update_ema(ema,model,cfg['EMA_decay'])
        stat['updates']+=1;stat['base_loss_scenes']+=len(rows);stat['p_dropped']+=int((~present).sum())
        if progress and stat['updates']%10==0:progress(dict(recipe=recipe,updates=stat['updates'],seconds=time.perf_counter()-started))
    stat.update(randomness={k:h.hexdigest() for k,h in hashes.items()},targets_sha256=target_hash.hexdigest(),
        order_sha256=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest(),geometry_reasons=dict(reasons),
        all_base_loss_rows_retained=stat['base_loss_scenes']==len(p_values),
        all_request_point_MAE=stat['point_sum']/stat['auxiliary_requests'],wall_seconds=time.perf_counter()-started)
    return stat


def load_inputs(policy,device):
    source=physics_pilot.load_inputs(policy,device)
    source['codes'].update({path:old.sha256(ROOT/path) for path in CODE})
    old.check_codes(source['codes'])
    return source


def train(binding,recipe,*,smoke=False):
    policy=read_policy(binding)
    if binding['sha256']!=POLICY_SHA:raise ValueError('only frozen target coverage training')
    if recipe not in policy['recipes']:raise ValueError('unregistered recipe')
    device=old.configure(policy);cfg=policy['training'];torch.manual_seed(cfg['seed']);source=load_inputs(policy,device)
    model=old.warmed(source['checkpoint'],device).requires_grad_(True);initial=old.state_hash(model)
    ema=copy.deepcopy(model).requires_grad_(False);schedule=old.CosineDiffusionSchedule(100).to(device)
    if schedule.as_dict()!=source['checkpoint']['schedule']:raise ValueError('schedule changed')
    output=old.resolve(policy['output_root'])/(recipe+'_smoke' if smoke else recipe);output.mkdir(parents=True,exist_ok=False)
    freeze=old.write_json(output/'freeze_before_training.json',dict(protocol=PROTOCOL,policy=binding,resolved_policy=policy,
        recipe=recipe,smoke=smoke,initial_state_sha256=initial,join=source['join'],cache_join=source['cache_join'],
        code_sha256=source['codes'],teacher_original_p_replay_max_error=source['teacher'].replay_max_error,
        STOP_CAL_AUDIT_observations_accessed=False))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    streams=old.PairedStreams(cfg['seed'],cfg['p_dropout_seed']);rng=torch.Generator().manual_seed(cfg['terminal_noise_seed'])
    checkpoints={};started=time.perf_counter()
    def save(epoch):
        path=output/('ema_epoch_%03d.pt'%epoch);data=source['data'];base=source['base']
        value=dict(protocol=PROTOCOL,policy=binding,recipe=recipe,epoch=epoch,smoke=smoke,EMA=True,
            state_dict={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},architecture=ema.architecture_config(),
            schedule=schedule.as_dict(),prediction_type='v',data=base['data'],labels_manifest=base['labels_manifest'],
            parent_checkpoint=policy['parent_checkpoint'],risk_cache=policy['risk_cache'],basis=data['basis'],
            coefficient_normalizer=data['coefficient_normalizer'],history_normalizer=data['history_normalizer'],code_sha256=source['codes'])
        with path.open('xb') as handle:torch.save(value,handle)
        checkpoints[str(epoch)]=dict(path=str(path),sha256=old.sha256(path))
    save(0)
    with (output/'epochs.jsonl').open('x') as log:
        for epoch in range(1,(1 if smoke else cfg['epochs'])+1):
            record=train_epoch(model,ema,schedule,optimizer,source['pack'],source['p'],source['cache'],source['teacher'],
                streams,rng,cfg,policy['physics'],policy['coverage'],recipe,device,max_updates=1 if smoke else None,
                progress=lambda r:print(json.dumps(dict(epoch=epoch,**r)),flush=True))
            record.update(epoch=epoch,recipe=recipe);log.write(json.dumps(record,sort_keys=True)+'\n');log.flush()
            if epoch in cfg['snapshot_epochs'] or smoke:save(epoch)
            print(json.dumps(record,sort_keys=True),flush=True)
    old.check_codes(source['codes'])
    result=dict(protocol=PROTOCOL,status='smoke_complete' if smoke else 'complete',policy=binding,recipe=recipe,smoke=smoke,
        epochs_completed=1 if smoke else cfg['epochs'],checkpoints=checkpoints,freeze=freeze,code_sha256=source['codes'],
        architecture=ema.architecture_config(),data=source['base']['data'],labels_manifest=source['base']['labels_manifest'],
        epochs=dict(path=str(output/'epochs.jsonl'),sha256=old.sha256(output/'epochs.jsonl')),wall_seconds=time.perf_counter()-started,
        parent_checkpoint=policy['parent_checkpoint'],production_default_changed=False,STOP_CAL_AUDIT_observations_accessed=False)
    print(json.dumps(dict(training_complete=old.write_json(output/'result.json',result))),flush=True)


def coverage_probe(binding):
    policy=read_policy(binding);device=old.configure(policy);source=load_inputs(policy,device)
    teacher=source['teacher'];pack=source['pack'];count=pack['agent_mask'].sum(1)
    repeated=np.repeat(np.arange(len(count)),3)
    ref=old.FrozenTorchCDF(teacher.masses[repeated],count[repeated],row_nodes=teacher.nodes[repeated])
    natural=torch.tensor(source['p'],dtype=torch.float32)
    reports={}
    for recipe in policy['recipes']:
        target=make_targets(natural,recipe,grid=policy['coverage']['grid'])
        result=target_eligibility(ref,target,interior_margin=policy['coverage']['interior_margin'],
                                 atom_tolerance=policy['coverage']['atom_tolerance'])
        eligible=result['eligible'].reshape(-1,3);floor=result['optimistic_endpoint_only_error_floor'].reshape(-1,3)
        reports[recipe]=dict(histories=len(count),requests=len(target),eligible=int(eligible.sum()),
            eligible_fraction=float(eligible.double().mean()),no_eligible_history=int((~eligible.any(1)).sum()),
            all_three_eligible_histories=int(eligible.all(1).sum()),
            eligible_by_slot=eligible.sum(0).tolist(),floor_mean_by_slot=floor.mean(0).tolist(),
            positive_floor_requests=int((floor>1e-8).sum()),
            by_count={name:dict(histories=int(mask.sum()),eligible=int(eligible[mask].sum()),
                requests=int(mask.sum())*3) for name,mask in (
                    ('N3_5',count<=5),('N6_8',(count>=6)&(count<=8)),('N9_plus',count>=9))})
    output=old.resolve(policy['output_root']);output.mkdir(parents=True,exist_ok=True)
    result=dict(protocol=PROTOCOL,policy=binding,reports=reports,code_sha256=source['codes'],
        generator_forward_calls=0,optimizer_steps=0,FIT_only=True,CDF_changed=False,targets_replaced=False)
    print(json.dumps(dict(coverage_probe=old.write_json(output/'FIT_target_coverage.json',result),reports=reports)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',default='configs/natural_percentile/multi_p_pilot_v1.json')
    parser.add_argument('--policy-sha256',default=POLICY_SHA)
    parser.add_argument('--stage',choices=['probe','train'],required=True)
    parser.add_argument('--recipe',choices=['natural_multi_noise','p_grid'])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();binding=dict(path=args.policy,sha256=args.policy_sha256)
    if args.stage=='probe':coverage_probe(binding)
    else:train(binding,args.recipe,smoke=args.smoke)
