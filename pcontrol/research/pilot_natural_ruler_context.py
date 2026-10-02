#!/usr/bin/env python3
"""Paired FIT-only CDF-shape adapter training, no production replacement."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch

from pcontrol.research import pilot_natural_multi_p as multi
from pcontrol.research.evaluate_natural_terminal_physics import same_artifact
from pcontrol.generation.cdf_shape_context import CONTEXT_KEY, shape_from_masses, descriptor_contract
from pcontrol.generation.ruler_context_direct_p import RulerContextPercentileDenoiser

old=multi.old;ROOT=multi.ROOT
PROTOCOL='natural_ruler_context_adapter_paired_training_v1'
POLICY_SHA='c7a8df943946e27786a68df82d2334348f7d01281ea32ad02702c3b57e014d4c'
CODE=multi.CODE+('pcontrol/generation/cdf_shape_context.py',
    'pcontrol/generation/ruler_context_direct_p.py','pcontrol/research/pilot_natural_ruler_context.py')


def read_policy(binding):
    if binding.get('sha256') != POLICY_SHA:raise ValueError('frozen ruler policy required')
    patch=old.bound_json(binding)
    if patch['protocol']!=PROTOCOL or patch['recipes']!=['constant_shape','oof_shape']:
        raise ValueError('wrong ruler experiment')
    policy=multi.read_policy(patch['base_policy'])
    policy.update(protocol=PROTOCOL,recipes=patch['recipes'],runtime=patch['runtime'],
        context=patch['context'],output_root=patch['output_root'],ruler_base_policy=patch['base_policy'],
        parent_STOP_CPU=patch['parent_STOP_CPU'],evaluation_device=patch['evaluation_device'])
    policy['training'].update(patch['training_overrides'])
    policy['training']['terminal_targets']='three_grid_P; natural_base_labels_unchanged'
    policy['training']['terminal_weight']={r:1. for r in patch['recipes']}
    for k in ('parent_STOP_CPU','smoke','evidence'):old.verify_binding(patch[k])
    return policy


def context_batch(pack,p_values,rows,shape_values,recipe,device):
    f,c,p=old.direct.tensor_batch(pack,p_values,rows,device)
    if recipe=='oof_shape':values=shape_values[rows]
    elif recipe=='constant_shape':values=np.tile(np.linspace(0.,1.,65,dtype=np.float32),(len(rows),1))
    else:raise ValueError('unregistered context recipe')
    f[CONTEXT_KEY]=torch.tensor(values,dtype=torch.float32,device=device)
    return f,c,p


def update_adapter_ema(ema,model,decay):
    # Applying floating EMA arithmetic even to equal frozen weights may drift.
    # Touch ONLY new trainable parameters; parent buffers/weights stay exact.
    with torch.no_grad():
        target=dict(ema.named_parameters())
        for name,value in model.named_parameters():
            if name.startswith('ruler_encoder.'):target[name].mul_(decay).add_(value,alpha=1.-decay)


def parent_unchanged(model,parent_state):
    state=model.state_dict()
    return all(torch.equal(state[k].detach().cpu(),v.cpu()) for k,v in parent_state.items())


def load_inputs(policy,device):
    source=multi.load_inputs(policy,device)
    source['codes'].update({p:old.sha256(ROOT/p) for p in CODE})
    return source


def prepare(binding):
    policy=read_policy(binding);device=old.configure(policy);source=load_inputs(policy,device)
    root=old.resolve(policy['output_root']);root.mkdir(parents=True,exist_ok=False)
    teacher=source['teacher'];pack=source['pack']
    old.write_json(root/'freeze_before_context.json',dict(protocol=PROTOCOL,policy=binding,
        code_sha256=source['codes'],FIT_only=True,context=descriptor_contract(),
        risk_cache=policy['risk_cache'],rows=len(source['p'])))
    values=[]
    for i in range(len(source['p'])):
        values.append(shape_from_masses(teacher.masses[i],teacher.nodes[i],int(pack['agent_mask'][i].sum())))
        if (i+1)%2000==0:print(json.dumps(dict(context_rows=i+1)),flush=True)
    values=np.asarray(values,dtype=np.float32)
    path=root/'FIT_OOF_context.npz'
    with path.open('xb') as handle:
        np.savez_compressed(handle,shape=values,scene_id=pack['scene_id'],recording_id=pack['recording_id'],role=pack['role'])
    old.check_codes(source['codes'])
    manifest=dict(protocol=PROTOCOL,policy=binding,code_sha256=source['codes'],rows=len(values),role='FIT',
        context=dict(path=str(path),sha256=old.sha256(path)),risk_cache=policy['risk_cache'],
        labels_unchanged=True,CAL_AUDIT_observations_accessed=False)
    print(json.dumps(dict(prepared=old.write_json(root/'prepared_context.json',manifest))),flush=True)


def context_values(policy,binding,source):
    path=old.resolve(policy['output_root'])/'prepared_context.json'
    context_binding=dict(path=str(path),sha256=old.sha256(path));manifest=old.bound_json(context_binding)
    if manifest['protocol']!=PROTOCOL or not same_artifact(manifest['policy'],binding) or manifest['rows']!=9913:
        raise ValueError('wrong context manifest')
    old.check_codes(manifest['code_sha256'])
    with np.load(old.verify_binding(manifest['context']),allow_pickle=False) as a:
        for key in ('scene_id','recording_id','role'):
            if not np.array_equal(a[key],source['pack'][key]):raise ValueError('CDF context row identity mismatch')
        values=a['shape']
    if values.shape!=(9913,65) or not np.isfinite(values).all():raise ValueError('wrong descriptor cache')
    return values,context_binding

def train_epoch(model,ema,schedule,optimizer,pack,p_values,cache,teacher,streams,aux_rng,cfg,physics,coverage,recipe,device,
                *,shape_values,progress=None,max_updates=None):
    model.train();order=streams.order(len(p_values));hashes={k:hashlib.sha256() for k in ('base_noise','drop','presence','aux_noise_plan')}
    target_hash=hashlib.sha256();slots=coverage['slots_per_history'];started=time.perf_counter();reasons=Counter()
    stat=dict(base_loss_scenes=0,updates=0,base_v_sum=0.,projection_support=0,jacobian_support=0,p_dropped=0,
        auxiliary_histories=0,auxiliary_requests=0,present_histories=0,target_eligible=0,positive_target_floor=0,
        supported=0,geometry_supported=0,point_sum=0.,floor_sum=0.,numeric_sum=0.,road_events=0,speed_events=0,
        road_loss_sum=0.,speed_loss_sum=0.,forward_NFE=0,backward_NFE=0,gradient_norm_max=0.)
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
        selected=rows[:min(cfg['terminal_per_batch'],len(rows))];repeated=np.repeat(selected,slots)
        af,ac,_=context_batch(pack,p_values,repeated,shape_values,recipe,device)
        natural=torch.tensor(p_values[selected],dtype=torch.float32,device=device)
        target=multi.make_targets(natural,'p_grid',grid=coverage['grid']);z=torch.randn(tuple(ac.shape),generator=aux_rng)
        for a in (repeated.astype(np.int64),np.asarray(z.shape,dtype=np.int64),z.numpy()):hashes['aux_noise_plan'].update(a.tobytes())
        target_hash.update(target.detach().cpu().numpy().tobytes())
        aux=multi.add_auxiliary_backward(model,schedule,af,target,z.to(device),presence[:len(selected)],teacher.batch(repeated),
            teacher.physical_batch(repeated),cfg,physics,coverage)
        stat['auxiliary_histories']+=aux['histories'];stat['auxiliary_requests']+=aux['total'];stat['present_histories']+=aux['present_histories']
        stat['target_eligible']+=aux['target_eligible'];stat['positive_target_floor']+=aux['positive_floor'];stat['floor_sum']+=aux['floor_sum']
        stat['supported']+=aux['supported'];stat['geometry_supported']+=aux['geometry_supported'];reasons.update(aux['reasons'])
        stat['point_sum']+=aux['point_sum'];stat['numeric_sum']+=aux['numeric']
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
    model=RulerContextPercentileDenoiser(**kw,context_hidden_dim=policy['context']['hidden_dim'])
    model.load_from_parent_state_dict(source['checkpoint']['state_dict']);model.to(device)
    trainable=model.train_adapter_only();ema=copy.deepcopy(model).requires_grad_(False)
    schedule=old.CosineDiffusionSchedule(100).to(device)
    root=old.resolve(policy['output_root'])/(recipe+'_smoke' if smoke else recipe);root.mkdir(exist_ok=False)
    freeze=old.write_json(root/'freeze_before_training.json',dict(protocol=PROTOCOL,policy=binding,
        resolved_policy=policy,recipe=recipe,smoke=smoke,initial_state_sha256=old.state_hash(model),
        context=context_binding,code_sha256=source['codes'],trainable_parameters=trainable,
        OOF_p_replay_error=source['teacher'].replay_max_error,STOP_CAL_AUDIT_observations_accessed=False))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    streams=old.PairedStreams(cfg['seed'],cfg['p_dropout_seed']);rng=torch.Generator().manual_seed(cfg['terminal_noise_seed'])
    checkpoints={};started=time.perf_counter()
    def save(epoch):
        for candidate in (model,ema):
            if not parent_unchanged(candidate,source['checkpoint']['state_dict']):raise ValueError('frozen parent weights drifted')
        path=root/('ema_epoch_%03d.pt'%epoch)
        data=source['data'];base=source['base']
        checkpoint=dict(protocol=PROTOCOL,policy=binding,recipe=recipe,epoch=epoch,smoke=smoke,EMA=True,
            state_dict={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},architecture=ema.architecture_config(),
            schedule=schedule.as_dict(),prediction_type='v',data=base['data'],labels_manifest=base['labels_manifest'],
            parent_checkpoint=policy['parent_checkpoint'],risk_cache=policy['risk_cache'],basis=data['basis'],
            coefficient_normalizer=data['coefficient_normalizer'],history_normalizer=data['history_normalizer'],
            context=context_binding,code_sha256=source['codes'])
        with path.open('xb') as handle:torch.save(checkpoint,handle)
        checkpoints[str(epoch)]=dict(path=str(path),sha256=old.sha256(path))
    save(0)
    with (root/'epochs.jsonl').open('x') as log:
        for epoch in range(1,(1 if smoke else cfg['epochs'])+1):
            record=train_epoch(model,ema,schedule,optimizer,source['pack'],source['p'],source['cache'],source['teacher'],
                streams,rng,cfg,policy['physics'],policy['coverage'],recipe,device,shape_values=shape_values,
                max_updates=1 if smoke else None,progress=lambda r:print(json.dumps(dict(epoch=epoch,**r)),flush=True))
            record.update(epoch=epoch,recipe=recipe)
            log.write(json.dumps(record,sort_keys=True)+'\n');log.flush()
            if epoch in cfg['snapshot_epochs'] or smoke:save(epoch)
            print(json.dumps(record,sort_keys=True),flush=True)
    old.check_codes(source['codes'])
    result=dict(protocol=PROTOCOL,status='smoke_complete' if smoke else 'complete',policy=binding,
        recipe=recipe,smoke=smoke,epochs_completed=1 if smoke else cfg['epochs'],checkpoints=checkpoints,freeze=freeze,
        code_sha256=source['codes'],architecture=ema.architecture_config(),trainable_parameters=trainable,
        data=source['base']['data'],labels_manifest=source['base']['labels_manifest'],parent_checkpoint=policy['parent_checkpoint'],
        epochs=dict(path=str(root/'epochs.jsonl'),sha256=old.sha256(root/'epochs.jsonl')),
        context=context_binding,wall_seconds=time.perf_counter()-started,parent_parameters_unchanged=True,
        production_default_changed=False,STOP_CAL_AUDIT_observations_accessed=False)
    print(json.dumps(dict(training_complete=old.write_json(root/'result.json',result))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',default='configs/natural_percentile/ruler_context_pilot_v1.json')
    parser.add_argument('--policy-sha256',default=POLICY_SHA)
    parser.add_argument('--stage',choices=['prepare','train'],required=True)
    parser.add_argument('--recipe',choices=['constant_shape','oof_shape'])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();binding=dict(path=args.policy,sha256=args.policy_sha256)
    if args.stage=='prepare':prepare(binding)
    else:train(binding,args.recipe,args.smoke)

