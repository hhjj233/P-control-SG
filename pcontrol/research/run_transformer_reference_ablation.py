#!/usr/bin/env python3
"""Publication-stage mechanism controls; immutable FIT/STOP data and parent recipe."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import pilot_time_attention_cdf as pilot
from pcontrol.time_attention_pipeline import common as c
from pcontrol.reference.history_encoder_ablation import NEW_ARMS,make_ablation

PROTOCOL='transformer_publication_reference_ablation_v1'
CODE=('pcontrol/research/run_transformer_reference_ablation.py','pcontrol/reference/history_encoder_ablation.py',*pilot.CODE)


def inputs(pb):
    p=c.json_file(pb)
    if (p['protocol']!=PROTOCOL or p['new_arms']!=list(NEW_ARMS) or p['selection_roles']!=['FIT','STOP']
            or p['protected_data_access'] or p['CAL_AUDIT_access'] or p['multitraining_seed_campaign']
            or p['equal_parameter_or_FLOP_claim'] or not p['all_arms_reported']):raise ValueError('bounded matched protocol required')
    parent=pilot.validate_policy(c.json_file(p['parent_policy']))
    prepared,_=io.read_prepared(parent['reference_prepared'])
    reused={arm:c.json_file(b) for arm,b in p['reuse_completed'].items()}
    if set(reused)!={'M2_MLP','M2_TimeAttn'}:raise ValueError('both completed matched controls required')
    for arm,r in reused.items():
        if r['status']!='complete' or r['arm']!=arm or r['policy']['sha256']!=p['parent_policy']['sha256']:
            raise ValueError('wrong reusable completed arm')
        c.verify_sources(r['code_sha256']);io.verify_binding(r['checkpoint'])
    return p,parent,prepared,reused,io.resolve(p['output_root'])


def common_hash(model):
    return pilot.tensor_hash({k:v for k,v in model.state_dict().items() if not k.startswith(pilot.ENCODERS)})


def prepare(pb):
    p,parent,prepared,reused,root=inputs(pb)
    models={arm:make_ablation(arm,parent['training']['seed']) for arm in NEW_ARMS}
    expected=reused['M2_TimeAttn']['initial_nonencoder_sha256']
    if any(common_hash(m)!=expected for m in models.values()):raise ValueError('shared initial weights differ')
    root.mkdir(parents=True,exist_ok=False)
    report=dict(protocol=PROTOCOL,status='prepared',policy=pb,parent_policy=p['parent_policy'],
        reference_prepared=parent['reference_prepared'],data=prepared['data'],normalizer=prepared['normalizer'],
        packs=prepared['packs'],counts=prepared['counts'],reuse_completed=p['reuse_completed'],
        initial_nonencoder_sha256=expected,architectures={a:m.architecture_config() for a,m in models.items()},
        code_sha256=c.source_bindings(CODE),CAL_AUDIT_decoded=False,protected_data_access=False,
        all9913_FIT_once_per_epoch=True,training_recipe_unchanged=parent['training'],
        all_arms_to_be_reported=True,main_reference_not_automatically_replaced=True)
    io.write_json(root/'prepared.json',report)
    print(json.dumps(dict(stage='prepared',root=str(root),parameters={a:m.architecture_config()['parameter_count'] for a,m in models.items()})),flush=True)


def frozen(pb):
    p,parent,prepared,reused,root=inputs(pb)
    fb=c.bind(root/'prepared.json');f=c.json_file(fb)
    if f['policy']!=pb or f['packs']!=prepared['packs']:raise ValueError('preparation drift')
    c.verify_sources(f['code_sha256'])
    return p,parent,prepared,reused,root,fb,f


def train(pb,arm,smoke=False):
    p,parent,prepared,reused,root,fb,f=frozen(pb)
    if arm not in NEW_ARMS:raise ValueError('undeclared new arm')
    device=pilot.seed_runtime(parent)
    out=root/('smoke' if smoke else 'models')/arm;out.mkdir(parents=True,exist_ok=False)
    fit_pack=pilot.load_allowed_pack(prepared['packs']['FIT'],'FIT',prepared)
    stop_pack=pilot.load_allowed_pack(prepared['packs']['STOP'],'STOP',prepared)
    fit,stop=pilot.resident(fit_pack,device),pilot.resident(stop_pack,device)
    fit_n=fit_pack['agent_mask'].sum(1);stop_n=stop_pack['agent_mask'].sum(1)
    model=make_ablation(arm,parent['training']['seed'])
    if common_hash(model)!=f['initial_nonencoder_sha256']:raise ValueError('initialization mismatch')
    model=model.to(device);actual=json.loads(json.dumps(parent))
    if smoke:
        for stage in ('stage1','stage2'):actual['training'][stage].update(epochs=1,patience=1)
    header=dict(protocol=PROTOCOL,arm=arm,policy=pb,parent_policy=p['parent_policy'],prepared=fb,
        data=prepared['data'],normalizer=prepared['normalizer'],code_sha256=f['code_sha256'],
        seed=parent['training']['seed'],architecture=model.architecture_config(),smoke=smoke)
    torch.cuda.reset_peak_memory_stats(device);started=time.monotonic();stages={};metrics={};predictions={}
    for stage in ('stage1','stage2'):
        stages[stage]=pilot.train_stage(model,fit,stop,fit_n,stop_n,actual,arm,stage,out/stage,header)
        arrays=pilot.prediction_arrays(model,stop,stop_pack,actual)
        predictions[stage]=io.save_pack(out/(stage+'_STOP_predictions.npz'),arrays)
        metrics[stage]=pilot.diagnostics(arrays,actual)
    c.verify_sources(f['code_sha256'])
    result=dict(protocol=PROTOCOL,status='smoke_complete' if smoke else 'complete',arm=arm,smoke=smoke,
        policy=pb,prepared=fb,parent_policy=p['parent_policy'],architecture=model.architecture_config(),
        stages=stages,predictions=predictions,STOP_metrics=metrics,checkpoint=stages['stage2']['checkpoint'],
        initial_nonencoder_sha256=f['initial_nonencoder_sha256'],code_sha256=f['code_sha256'],
        wall_seconds=time.monotonic()-started,peak_allocated_gpu_MiB=torch.cuda.max_memory_allocated(device)/2**20,
        data=prepared['data'],normalizer=prepared['normalizer'],CAL_AUDIT_decoded=False,protected_data_access=False,
        generation_called=False,equal_parameter_or_FLOP_claim=False)
    rb=io.write_json(out/'result.json',result)
    print(json.dumps(dict(stage='arm_complete',arm=arm,smoke=smoke,result=rb,
        CRPS=metrics['stage2']['CRPS_seconds'],highN=metrics['stage2']['highN_CRPS_seconds'])),flush=True)


def summarize(pb):
    p,parent,prepared,reused,root,fb,f=frozen(pb)
    bindings=dict(p['reuse_completed'])
    for arm in NEW_ARMS:bindings[arm]=c.bind(root/'models'/arm/'result.json')
    reports={a:c.json_file(b) for a,b in bindings.items()}
    for arm in NEW_ARMS:
        r=reports[arm]
        if r['status']!='complete' or r['smoke'] or r['policy']!=pb or r['prepared']!=fb:raise ValueError('actual complete new training required')
        c.verify_sources(r['code_sha256'])
    reference=reports['M2_TimeAttn'];order_checks={}
    ids=c.arrays(reference['predictions']['stage2'])['scene_id']
    for arm,r in reports.items():
        if not np.array_equal(c.arrays(r['predictions']['stage2'])['scene_id'],ids):raise ValueError('STOP identities differ')
        for stage in ('stage1','stage2'):
            first=[json.loads(v) for v in io.verify_binding(reference['stages'][stage]['epochs']).read_text().splitlines()]
            second=[json.loads(v) for v in io.verify_binding(r['stages'][stage]['epochs']).read_text().splitlines()]
            n=min(len(first),len(second))
            if any(first[i]['order_sha256']!=second[i]['order_sha256'] or second[i]['FIT_observations']!=9913 for i in range(n)):
                raise ValueError('matched FIT orders or exposure changed')
            order_checks[arm+'/'+stage]=n
    result=dict(protocol=PROTOCOL,status='complete',policy=pb,prepared=fb,arms=bindings,
        STOP={a:r['STOP_metrics']['stage2'] for a,r in reports.items()},architectures={a:r['architecture'] for a,r in reports.items()},
        matching_order_epochs=order_checks,CAL_AUDIT_decoded=False,protected_data_access=False,
        code_sha256=f['code_sha256'],main_Transformer_not_replaced=True,
        scope='single_training_seed_development_mechanism_controls; later_CAL_and_final_validation_separate')
    io.write_json(root/'result.json',result)
    print(json.dumps(dict(stage='reference_ablation_complete',CRPS={a:r['STOP_metrics']['stage2']['CRPS_seconds'] for a,r in reports.items()})),flush=True)


def suite(pb):
    p,_,_,_,root,_,_=frozen(pb);logs=root/'runlogs';logs.mkdir(exist_ok=True)
    for arm in NEW_ARMS:
        out=root/'models'/arm
        if out.exists():
            rb=c.bind(out/'result.json');r=c.json_file(rb)
            if r['status']!='complete' or r['policy']!=pb or r['smoke']:raise ValueError('inspect existing incomplete process; never auto-restart')
            c.verify_sources(r['code_sha256']);continue
        argv=[sys.executable,str(Path(__file__).resolve()),'train','--policy',pb['path'],'--policy-sha256',pb['sha256'],'--arm',arm]
        print(json.dumps(dict(starting=arm)),flush=True)
        with (logs/(arm+'.log')).open('x') as log:
            subprocess.run(argv,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdout=log,stderr=subprocess.STDOUT,check=True)
        r=c.json_file(c.bind(out/'result.json'))
        print(json.dumps(dict(completed=arm,CRPS=r['STOP_metrics']['stage2']['CRPS_seconds'],seconds=r['wall_seconds'])),flush=True)
    summarize(pb)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','train','summarize','suite'])
    parser.add_argument('--policy',required=True);parser.add_argument('--policy-sha256',required=True)
    parser.add_argument('--arm',choices=NEW_ARMS);parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args();pb=dict(path=str(io.resolve(a.policy)),sha256=a.policy_sha256)
    if a.command=='train':train(pb,a.arm,a.smoke)
    else:{'prepare':prepare,'summarize':summarize,'suite':suite}[a.command](pb)
