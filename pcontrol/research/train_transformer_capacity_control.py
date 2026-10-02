#!/usr/bin/env python3
"""Fresh capacity control with the already declared FIT/STOP training rules."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.reference.capacity_matched_mlp import make_capacity_control
from pcontrol.reference.group_guarded_selection import grouped_scores,eligible
from pcontrol.reference.scores import crps_from_params
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import pilot_time_attention_cdf as pilot
from pcontrol.research.run_transformer_reference_ablation import common_hash

PROTOCOL='transformer_near_capacity_matched_MLP_control_v1'
CODE=('pcontrol/research/train_transformer_capacity_control.py','pcontrol/reference/capacity_matched_mlp.py',
      'pcontrol/reference/group_guarded_selection.py','pcontrol/research/run_transformer_reference_ablation.py',*pilot.CODE)


def inputs(pb):
    p=c.json_file(pb)
    if (p['protocol']!=PROTOCOL or p['arm']!='M2_WideMLP' or p['CAL_AUDIT_access_in_training'] or p['protected_data_access']
            or p['multi_training_seed_campaign'] or not p['all_results_reported']):raise ValueError('wrong capacity-control scope')
    parent=pilot.validate_policy(c.json_file(p['parent_policy']));guard=c.json_file(p['group_guard_policy'])
    prepared,_=io.read_prepared(parent['reference_prepared']);reference=c.json_file(p['matched_TimeAttn_result'])
    if guard['training']['epochs']!=30 or guard['training']['seed']!=parent['training']['seed']:raise ValueError('guard recipe mismatch')
    c.verify_sources(reference['code_sha256'])
    return p,parent,guard,prepared,reference,io.resolve(p['output_root'])


def prepare(pb):
    p,parent,guard,data,reference,root=inputs(pb)
    model=make_capacity_control(parent['training']['seed'],p['encoder_hidden_width']);count=sum(v.numel() for v in model.parameters())
    if count!=p['parameter_count'] or abs(count/p['TimeAttn_parameter_count']-1)>p['maximum_relative_parameter_difference']:
        raise ValueError('capacity budget mismatch')
    initial=common_hash(model)
    if initial!=reference['initial_nonencoder_sha256']:raise ValueError('shared weights differ')
    neighbors={str(width):sum(v.numel() for v in make_capacity_control(parent['training']['seed'],width).parameters())
               for width in (p['encoder_hidden_width']-1,p['encoder_hidden_width'],p['encoder_hidden_width']+1)}
    if min(neighbors,key=lambda x:abs(neighbors[x]-p['TimeAttn_parameter_count']))!=str(p['encoder_hidden_width']):
        raise ValueError('width is not nearest shared integer')
    root.mkdir(exist_ok=False)
    result=dict(protocol=PROTOCOL,status='prepared',policy=pb,parent_policy=p['parent_policy'],group_guard_policy=p['group_guard_policy'],
        source_prepared=parent['reference_prepared'],data=data['data'],normalizer=data['normalizer'],packs=data['packs'],
        architecture=model.architecture_config(),initial_nonencoder_sha256=initial,width_parameter_counts=neighbors,
        signed_parameter_difference=count-p['TimeAttn_parameter_count'],relative_parameter_difference=count/p['TimeAttn_parameter_count']-1,
        code_sha256=c.source_bindings(CODE),CAL_AUDIT_decoded=False,protected_data_access=False,main_reference_changed=False)
    io.write_json(root/'prepared.json',result);print(json.dumps(dict(stage='capacity_control_prepared',parameters=count,neighbor_widths=neighbors)),flush=True)


def train(pb):
    p,parent,guard,data,reference,root=inputs(pb);fb=c.bind(root/'prepared.json');f=c.json_file(fb)
    if f['policy']!=pb:raise ValueError('policy drift')
    c.verify_sources(f['code_sha256']);device=pilot.seed_runtime(parent);cfg=guard['training'];arm=p['arm']
    out=root/'training';out.mkdir(exist_ok=False)
    fp=pilot.load_allowed_pack(data['packs']['FIT'],'FIT',data);sp=pilot.load_allowed_pack(data['packs']['STOP'],'STOP',data)
    fit,stop=pilot.resident(fp,device),pilot.resident(sp,device);n=fp['agent_mask'].sum(1);sn=sp['agent_mask'].sum(1)
    model=make_capacity_control(parent['training']['seed'],p['encoder_hidden_width'])
    if common_hash(model)!=f['initial_nonencoder_sha256']:raise ValueError('initialization drift')
    model.to(device);header=dict(protocol=PROTOCOL,arm=arm,policy=pb,prepared=fb,data=data['data'],normalizer=data['normalizer'],
        architecture=model.architecture_config(),code_sha256=f['code_sha256'],seed=cfg['seed'])
    started=time.monotonic();torch.cuda.reset_peak_memory_stats(device)
    first=pilot.train_stage(model,fit,stop,n,sn,parent,arm,'stage1',out/'stage1',header)
    base_arrays=pilot.prediction_arrays(model,stop,sp,parent);base=grouped_scores(base_arrays['crps_seconds'],sn,sp['recording_id'])
    first_prediction=io.save_pack(out/'stage1_STOP_predictions.npz',base_arrays)
    refinements=out/'fixed30_refinement';refinements.mkdir()
    selections={rule:dict(epoch=0,score=base['selection'],STOP=base,checkpoint=first['checkpoint'],predictions=first_prediction)
                for rule in ('unguarded','group_guarded')}
    snapshots={'0':dict(checkpoint=first['checkpoint'],predictions=first_prediction,STOP=base)}
    io.write_json(refinements/'freeze.json',dict(policy=pb,prepared=fb,initial=first['checkpoint'],initial_STOP=base,
        candidate_epochs=list(range(31)),same_complete_candidates_for_both_rules=True,STOP_only_selection=True))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    raw=np.where(n>=9,cfg['highN_weight'],1.);weights=torch.tensor(raw/raw.mean(),dtype=torch.float64,device=device)
    rng=np.random.default_rng(cfg['seed']+1000);updates=0
    with (refinements/'epochs.jsonl').open('x') as logfile:
        for epoch in range(1,31):
            model.train();order=rng.permutation(len(n));total=0.
            for start in range(0,len(order),cfg['batch_size']):
                rows=order[start:start+cfg['batch_size']];features,y=pilot.batch(fit,rows)
                optimizer.zero_grad(set_to_none=True)
                loss=(crps_from_params(model(features),y,normalized=True)*weights[torch.tensor(rows,device=device)]).mean()
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite capacity-control loss')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
                if not torch.isfinite(norm):raise FloatingPointError('nonfinite capacity-control gradient')
                optimizer.step();updates+=1;total+=float(loss.detach())*len(rows)*4
            arrays=pilot.prediction_arrays(model,stop,sp,parent);scores=grouped_scores(arrays['crps_seconds'],sn,sp['recording_id'])
            cp=pilot.save_checkpoint(refinements/f'epoch_{epoch:03d}.pt',model,dict(header,stage='fixed30_refinement',epoch=epoch))
            predictions=io.save_pack(refinements/f'epoch_{epoch:03d}_STOP_predictions.npz',arrays)
            snapshots[str(epoch)]=dict(checkpoint=cp,predictions=predictions,STOP=scores);checks={}
            for rule in selections:
                ok,why=eligible(scores,base,guard['selection'],rule);checks[rule]=dict(eligible=ok,checks=why)
                if ok and scores['selection']<selections[rule]['score']:
                    selections[rule]=dict(epoch=epoch,score=scores['selection'],STOP=scores,checkpoint=cp,predictions=predictions)
            row=dict(epoch=epoch,FIT_rows=len(n),optimizer_steps=updates,STOP=scores,guards=checks,
                train_weighted_CRPS=total/len(n),selected_epochs={key:value['epoch'] for key,value in selections.items()},
                order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),seconds=time.monotonic()-started)
            logfile.write(json.dumps(row)+'\n');logfile.flush();print(json.dumps(dict(stage='capacity_refinement',epoch=epoch,CRPS=scores['overall'],selected=row['selected_epochs'])),flush=True)
    if updates!=2340:raise ValueError('fixed highN budget incomplete')
    c.verify_sources(f['code_sha256'])
    result=dict(protocol=PROTOCOL,status='complete',policy=pb,prepared=fb,arm=arm,architecture=model.architecture_config(),
        stage1=first,stage1_predictions=first_prediction,stage1_STOP=base,selections=selections,snapshots=snapshots,
        refinement_trace=c.bind(refinements/'epochs.jsonl'),refinement_epochs=30,refinement_updates=updates,
        data=data['data'],normalizer=data['normalizer'],code_sha256=f['code_sha256'],initial_nonencoder_sha256=f['initial_nonencoder_sha256'],
        wall_seconds=time.monotonic()-started,peak_allocated_GPU_MiB=torch.cuda.max_memory_allocated(device)/2**20,
        shared_GPU_not_latency_benchmark=True,CAL_AUDIT_decoded=False,protected_data_access=False,main_reference_changed=False)
    io.write_json(root/'result.json',result);print(json.dumps(dict(stage='capacity_control_complete',base_CRPS=base['overall'],selected={k:v['epoch'] for k,v in selections.items()})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','train'])
    parser.add_argument('--policy',default='configs/natural_percentile/transformer_capacity_control_v1.json');parser.add_argument('--policy-sha256',required=True)
    a=parser.parse_args();pb=dict(path=str(io.resolve(a.policy)),sha256=a.policy_sha256);{'prepare':prepare,'train':train}[a.command](pb)
