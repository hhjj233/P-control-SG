#!/usr/bin/env python3
"""Full50-DDIM parameter-gradient continuation; original raw inference retained."""
import argparse
import copy
import json
import time
import torch
from pcontrol.research import pilot_dynamic_dual_value as previous

old=previous.old;multi=previous.multi;ROOT=previous.ROOT
RulerContextPercentileDenoiser=previous.RulerContextPercentileDenoiser
parent_unchanged=previous.parent_unchanged
speed_strata=previous.speed_strata;SlowHistoryCycle=previous.SlowHistoryCycle
PROTOCOL='natural_dynamic_full_DDIM_gradient_training_v1'
POLICY_SHA='b380e74aeff6eabc6358c92caad0f3d1cc85b7bbfb68e90d4602b5871734cafa'
CODE=previous.CODE+('pcontrol/research/pilot_dynamic_full_chain.py',)


def read_policy(binding):
    if binding.get('sha256')!=POLICY_SHA:raise ValueError('frozen full-chain policy required')
    patch=old.bound_json(binding)
    if patch['protocol']!=PROTOCOL or patch['recipes']!=['full_chain']:raise ValueError('wrong full-chain continuation')
    policy=previous.read_policy(patch['base_policy'])
    policy.update(protocol=PROTOCOL,recipes=patch['recipes'],resume_checkpoint=patch['resume_checkpoint'],
        full_chain_base_policy=patch['base_policy'],output_root=patch['output_root'],previous_STOP=patch['previous_STOP'])
    policy['training'].update(patch['training_overrides']);policy['history_sampling']['seed']=patch['history_sampling_seed']
    probe=old.bound_json(patch['FIT_gradient_probe'])
    if probe['status']!='complete' or not probe['forward_samples_exactly_equal'] or not probe['restored_model_state']:
        raise ValueError('full-gradient FIT validation required')
    if policy['training']['grad_last_steps']!=50 or policy['training']['terminal_DDIM_steps']!=50:
        raise ValueError('full50 forward and parameter backward required')
    for key in ('resume_checkpoint','previous_STOP'):old.verify_binding(patch[key])
    return policy


def load_inputs(policy,device):
    source=previous.load_inputs(policy,device)
    source['codes'].update({p:old.sha256(ROOT/p) for p in CODE})
    return source


def context_values(policy,binding,source):
    original_binding=policy['full_chain_base_policy'];original=previous.read_policy(original_binding)
    return previous.context_values(original,original_binding,source)


def train_epoch(model,ema,schedule,optimizer,pack,p_values,cache,teacher,streams,aux_rng,cfg,physics,coverage,recipe,device,
                *,shape_values,slow_cycle,progress=None,max_updates=None):
    if recipe!='full_chain':raise ValueError('full chain recipe required')
    return previous.train_epoch(model,ema,schedule,optimizer,pack,p_values,cache,teacher,streams,aux_rng,
        cfg,physics,coverage,'dynamic_dual',device,shape_values=shape_values,slow_cycle=slow_cycle,
        progress=(lambda r:progress(dict(r,recipe=recipe))) if progress else None,max_updates=max_updates)

def train(binding,recipe,smoke=False):
    policy=read_policy(binding)
    if recipe not in policy['recipes']:raise ValueError('wrong recipe')
    device=old.configure(policy);cfg=policy['training'];torch.manual_seed(cfg['seed'])
    source=load_inputs(policy,device);shape_values,context_binding=context_values(policy,binding,source)
    kw={k:source['checkpoint']['architecture'][k] for k in old.ARCH_KEYS}
    model=RulerContextPercentileDenoiser(**kw,context_hidden_dim=policy['context']['hidden_dim'],dynamic_bottleneck=policy['dynamic_bottleneck'])
    resumed=torch.load(old.verify_binding(policy['resume_checkpoint']),map_location='cpu')
    if (resumed['protocol']!=previous.PROTOCOL or resumed['recipe']!='dynamic_dual' or resumed['epoch']!=3
            or resumed['smoke'] or not resumed['EMA'] or resumed['data']!=source['base']['data']
            or resumed['labels_manifest']!=source['base']['labels_manifest']):
        raise ValueError('wrong continuation checkpoint')
    old.check_codes(resumed['code_sha256'])
    model.load_state_dict(resumed['state_dict'],strict=True);model.to(device)
    trainable=model.train_adapter_only();ema=copy.deepcopy(model).requires_grad_(False)
    schedule=old.CosineDiffusionSchedule(100).to(device)
    root=old.resolve(policy['output_root'])/(recipe+'_smoke' if smoke else recipe);root.mkdir(parents=True,exist_ok=False)
    freeze=old.write_json(root/'freeze_before_training.json',dict(protocol=PROTOCOL,policy=binding,
        resolved_policy=policy,recipe=recipe,smoke=smoke,initial_state_sha256=old.state_hash(model),
        context=context_binding,code_sha256=source['codes'],trainable_parameters=trainable,resume_checkpoint=policy['resume_checkpoint'],
        OOF_p_replay_error=source['teacher'].replay_max_error,STOP_CAL_AUDIT_observations_accessed=False))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    streams=old.PairedStreams(cfg['seed'],cfg['p_dropout_seed']);rng=torch.Generator().manual_seed(cfg['terminal_noise_seed'])
    speed,strata=speed_strata(source['teacher'].physical['history'],source['pack']['agent_mask'],source['pack']['role'])
    slow_cycle=SlowHistoryCycle(source['pack']['recording_id'],strata,seed=policy['history_sampling']['seed'])
    checkpoints={};started=time.perf_counter()
    def save(epoch):
        for candidate in (model,ema):
            if not parent_unchanged(candidate,source['checkpoint']['state_dict']):raise ValueError('frozen parent weights drifted')
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
            record.update(epoch=epoch,recipe=recipe)
            log.write(json.dumps(record,sort_keys=True)+'\n');log.flush()
            if epoch in cfg['snapshot_epochs'] or smoke:save(epoch)
            print(json.dumps(record,sort_keys=True),flush=True)
    old.check_codes(source['codes'])
    result=dict(protocol=PROTOCOL,status='smoke_complete' if smoke else 'complete',policy=binding,
        recipe=recipe,smoke=smoke,epochs_completed=1 if smoke else cfg['epochs'],checkpoints=checkpoints,freeze=freeze,
        code_sha256=source['codes'],architecture=ema.architecture_config(),trainable_parameters=trainable,
        data=source['base']['data'],labels_manifest=source['base']['labels_manifest'],parent_checkpoint=policy['parent_checkpoint'],resume_checkpoint=policy['resume_checkpoint'],
        epochs=dict(path=str(root/'epochs.jsonl'),sha256=old.sha256(root/'epochs.jsonl')),
        context=context_binding,wall_seconds=time.perf_counter()-started,parent_parameters_unchanged=True,
        production_default_changed=False,STOP_CAL_AUDIT_observations_accessed=False)
    print(json.dumps(dict(training_complete=old.write_json(root/'result.json',result))),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',default='configs/natural_percentile/dynamic_full_chain_v1.json')
    parser.add_argument('--policy-sha256',default=POLICY_SHA);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();train(dict(path=args.policy,sha256=args.policy_sha256),'full_chain',args.smoke)

