#!/usr/bin/env python3
"""Common natural v-MSE adaptation before the paired target-policy study."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.publication_pipeline import generator_data as gd
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.direct_p_cfg import cfg_training_loss
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import train_time_attention_generator as prior
from pcontrol.research.refine_natural_direct_p import PairedStreams,state_hash
from pcontrol.research.finetune_joint_risk_generator_v2 import update_adapter_ema

PROTOCOL='guarded_reference_common_generator_adaptation_v1'
CODE=tuple(dict.fromkeys((*prior.CODE,'pcontrol/research/train_guarded_generator_adaptation.py',
    'pcontrol/publication_pipeline/generator_data.py',
    'pcontrol/generation/torch_atom_aware_target.py')))


def prepare(pb):
    source=gd.load_inputs(pb);p=source['policy']
    probe_b=c.bind('outputs/natural_percentile/transformer_publication_v1_20260916/guarded_atom_target_math_v1/result.json')
    probe=c.json_file(probe_b)
    if probe['status']!='pass' or probe['queries']!=39652:raise ValueError('complete scalar CPU/CUDA verification required')
    if not gd.same_binding(c.json_file(probe['freeze'])['labels'],p['labels_manifest']):raise ValueError('probe reference mismatch')
    root=io.resolve(p['output_root'])/'data';root.mkdir(parents=True,exist_ok=False)
    codes=c.source_bindings(CODE)
    freeze=io.write_json(root/'freeze.json',dict(policy=pb,labels_manifest=p['labels_manifest'],code_sha256=codes))
    cache,summary=gd.regenerate_tangent(source)
    cb=io.save_pack(root/'FIT_tangents.npz',cache)
    c.verify_sources(codes)
    result=dict(protocol=PROTOCOL,status='prepared',policy=pb,reference_manifest=p['reference_manifest'],
        labels_manifest=p['labels_manifest'],generator_data=p['generator_data'],tangent_cache=cb,tangent_summary=summary,
        label_joins=source['joins'],atom_target_probe=probe_b,freeze=freeze,code_sha256=codes,
        natural_coefficients_unchanged=True,CAL_AUDIT_generator_examples=False,old_density_reused=False)
    io.write_json(root/'manifest.json',result);print(json.dumps(dict(stage='prepared',summary=summary,joins=source['joins'])),flush=True)


def load_source(pb):
    p=c.json_file(pb);mb=c.bind(io.resolve(p['output_root'])/'data/manifest.json');m=c.json_file(mb)
    if not gd.same_binding(m['policy'],pb) or m['status']!='prepared':raise ValueError('complete own preparation required')
    c.verify_sources(m['code_sha256']);s=gd.load_inputs(pb)
    if not gd.same_binding(s['labels_binding'],m['labels_manifest']):raise ValueError('label drift')
    s.update(preparation=m,preparation_binding=mb,cache=c.arrays(m['tangent_cache']))
    return s


def checkpoint(path,model,source,pb,epoch,extra=None):
    m=source['preparation'];d=source['data']
    payload=dict(protocol=PROTOCOL,policy=pb,stage='common_adaptation',epoch=epoch,state_dict=prior.cpu_state(model),
        architecture=model.architecture_config(),generator_data=m['generator_data'],labels_manifest=m['labels_manifest'],
        reference_manifest=m['reference_manifest'],generator_preparation=source['preparation_binding'],basis=d['basis'],
        coefficient_normalizer=d['coefficient_normalizer'],history_normalizer=d['history_normalizer'],
        code_sha256=m['code_sha256'],**(extra or {}))
    with path.open('xb') as f:torch.save(payload,f)
    return c.bind(path)


def adapt(pb):
    s=load_source(pb);p=s['policy'];cfg=p['generator'];device=prior.runtime(p)
    parent=torch.load(io.verify_binding(p['generator_warm_start']),map_location='cpu',weights_only=False)
    c.verify_sources(parent['code_sha256'])
    if (parent['protocol']!=prior.PROTOCOL or parent['stage']!='terminal' or parent['epoch']!=3 or parent['smoke']
            or not gd.same_binding(parent['generator_data'],p['generator_data'])):raise ValueError('wrong complete warm-start')
    model=prior.make_model(parent['architecture'],parent['state_dict'],device);initial=state_hash(model)
    groups=model.finetuning_groups(cfg['core_learning_rate'],cfg['adapter_learning_rate'])
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4);ema=copy.deepcopy(model).requires_grad_(False)
    schedule=CosineDiffusionSchedule(100).to(device)
    root=io.resolve(p['output_root'])/'adaptation';root.mkdir(exist_ok=False)
    freeze=io.write_json(root/'freeze.json',dict(policy=pb,preparation=s['preparation_binding'],initial_state_sha256=initial,
        warm_start=p['generator_warm_start'],old_optimizer_reused=False,old_reference_labels_reused=False,
        code_sha256=s['preparation']['code_sha256'],fixed_STOP_noise_seed=20261015,
        selection='fixed_noise_STOP_EMA_v_MSE_not_generated_Fine',initial_state_reference=parent['reference_manifest']))
    best=prior.validation(ema,schedule,s,device);initial_score=copy.deepcopy(best)
    best_state=prior.cpu_state(ema);best_epoch=0;stale=0;significant=best['mean']
    streams=PairedStreams(cfg['seed'],cfg['seed']+1);started=time.monotonic();updates=0
    with (root/'epochs.jsonl').open('x') as log:
        for epoch in range(1,cfg['adaptation_max_epochs']+1):
            model.train();order=streams.order(9913);total=0.;dropped=0
            for start in range(0,9913,cfg['batch_size']):
                rows=order[start:start+cfg['batch_size']];f,clean,labels=prior.model_batch(s,'FIT',rows,device)
                times,noise,_,present=streams.draw(clean.shape,100,cfg['p_dropout_probability'])
                optimizer.zero_grad(set_to_none=True)
                result=cfg_training_loss(model,schedule,clean,f,labels,torch.tensor(present,device=device),timesteps=times.to(device),noise=noise.to(device))
                loss=result['loss']
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite natural adaptation')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite adaptation gradient')
                optimizer.step();update_adapter_ema(ema,model,cfg['EMA_decay'])
                total+=float(loss.detach())*len(rows);dropped+=int((~present).sum());updates+=1
            score=prior.validation(ema,schedule,s,device)
            if score['mean']<best['mean']:best,best_epoch,best_state=score,epoch,prior.cpu_state(ema)
            if score['mean']<significant-1e-5:significant,stale=score['mean'],0
            else:stale+=1
            row=dict(epoch=epoch,FIT_rows=9913,mean_train_v_MSE=total/9913,STOP=score,best_epoch=best_epoch,
                best_STOP=best,stale=stale,optimizer_updates=updates,p_dropped=dropped,
                order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),seconds=time.monotonic()-started)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
            if stale>=cfg['adaptation_patience']:break
    ema.load_state_dict(best_state,strict=True)
    chosen=checkpoint(root/'best_ema.pt',ema,s,pb,best_epoch,dict(EMA=True,warm_start=p['generator_warm_start']))
    c.verify_sources(s['preparation']['code_sha256'])
    result=dict(protocol=PROTOCOL,status='complete',policy=pb,preparation=s['preparation_binding'],checkpoint=chosen,
        freeze=freeze,initial_validation=initial_score,best_validation=best,best_epoch=best_epoch,epochs_completed=epoch,
        optimizer_updates=updates,all_FIT_rows_each_epoch=True,labels_manifest=p['labels_manifest'],reference_manifest=p['reference_manifest'],
        code_sha256=s['preparation']['code_sha256'],warm_start=p['generator_warm_start'],
        initial_state_sha256=initial,selected_state_sha256=state_hash(ema),old_optimizer_reused=False,
        paired_terminal_training_completed=False,protected_data_access=False,wall_seconds=time.monotonic()-started)
    io.write_json(root/'result.json',result);print(json.dumps(dict(stage='common_adaptation_complete',best_epoch=best_epoch,STOP=best)),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','adapt'])
    parser.add_argument('--policy',default='configs/natural_percentile/guarded_generator_adaptation_v1.json')
    parser.add_argument('--policy-sha256',required=True)
    a=parser.parse_args();pb=dict(path=str(io.resolve(a.policy)),sha256=a.policy_sha256)
    {'prepare':prepare,'adapt':adapt}[a.command](pb)
