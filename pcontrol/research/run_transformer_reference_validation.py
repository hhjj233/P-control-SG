#!/usr/bin/env python3
"""Matched CAL-only calibration of six reference architectures, then dev AUDIT.

Every CAL selection freezes before any new AUDIT prediction. No neural weight
selection/refit uses CAL/AUDIT, and no protected raw recording is read.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.reference.history_encoder_ablation import make_ablation
from pcontrol.reference.publication_calibration import CalibrationScoreCache,fit_cached,eligibility,select_candidate,SELECTORS
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import pilot_time_attention_cdf as pilot
from pcontrol.research.validate_time_attention_reference import get_predictions
from pcontrol.research.audit_time_attention_full_pipeline import verify_nested

PROTOCOL='transformer_publication_reference_validation_v1'
ARMS=('M2_MLP','M2_TimeAttn','M2_TimeAttn_absolute_only','M2_TimeAttn_relative_only','M1_TimeAttn','M2_GRU')
STAGES=('stage1','stage2')
CODE=('pcontrol/research/run_transformer_reference_validation.py','pcontrol/reference/publication_calibration.py',
      'pcontrol/reference/history_encoder_ablation.py','pcontrol/time_attention_pipeline/common.py',
      'pcontrol/reference/scene_calibration.py','pcontrol/reference/scene_context_calibration.py',
      'pcontrol/reference/scene_context_plugin.py','pcontrol/research/validate_time_attention_reference.py',*pilot.CODE)


def read_policy(pb):
    p=c.json_file(pb)
    if (p['protocol']!=PROTOCOL or p['arms']!=list(ARMS) or p['stages']!=list(STAGES)
            or p['protected_val_test_or_recording18_access'] or p['raw_CSV_access'] or p['generator_training_or_sampling']
            or p['primary_reference']!=dict(arm='M2_TimeAttn',stage='stage2',selector='count_balanced')
            or set(p['calibration']['selectors'])!=set(SELECTORS)
            or not p['evaluation']['all12_CAL_selections_frozen_before_any_new_AUDIT_prediction']):
        raise ValueError('wrong validation scope')
    return p


def runtime(p):
    torch.set_num_threads(p['runtime']['threads']);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False


def prepare(pb):
    p=read_policy(pb);study=c.json_file(p['source_study']);audit=c.json_file(p['source_audit'])
    if study['status']!='complete' or audit['status']!='pass' or audit['source']['sha256']!=p['source_study']['sha256']:
        raise ValueError('completed verified six-arm study required')
    c.verify_sources(study['code_sha256']);seen=set();verify_nested(study,seen)
    model_sources={};data=normalizer=None
    for arm in ARMS:
        r=c.json_file(study['arms'][arm]);c.verify_sources(r['code_sha256'])
        for stage in STAGES:
            b=r['stages'][stage]['checkpoint'];cp=torch.load(io.verify_binding(b),map_location='cpu',weights_only=False)
            if cp['arm']!=arm or cp['stage']!=stage:raise ValueError('checkpoint identity mismatch')
            if data is None:data,normalizer=cp['data'],cp['normalizer']
            if data!=cp['data'] or normalizer!=cp['normalizer']:raise ValueError('unmatched source population')
            model_sources[arm+'/'+stage]=dict(checkpoint=b,training_result=study['arms'][arm],architecture=r['architecture'])
    training_parent=c.json_file(c.json_file(study['policy'])['parent_policy'])
    prepared,_=io.read_prepared(training_parent['reference_prepared'])
    if prepared['data']!=data or prepared['normalizer']!=normalizer:raise ValueError('source pack changed')
    root=io.resolve(p['output_root']);root.mkdir(parents=True,exist_ok=False)
    report=dict(protocol=PROTOCOL,status='prepared',policy=pb,source_study=p['source_study'],source_audit=p['source_audit'],
        source_prepared=training_parent['reference_prepared'],data=data,normalizer=normalizer,
        models=model_sources,seed=training_parent['training']['seed'],code_sha256=c.source_bindings(CODE),
        CAL_AUDIT_arrays_decoded=False,base_checkpoint_choices_frozen=True,protected_data_access=False,
        primary_reference=p['primary_reference'],main_architecture_or_stage_selected_from_CAL_AUDIT=False)
    io.write_json(root/'prepared.json',report)
    print(json.dumps(dict(stage='validation_prepared',checkpoints=len(model_sources),root=str(root))),flush=True)


def frozen(pb):
    p=read_policy(pb);root=io.resolve(p['output_root']);fb=c.bind(root/'prepared.json');f=c.json_file(fb)
    if f['policy']!=pb or f['status']!='prepared':raise ValueError('immutable preparation required')
    c.verify_sources(f['code_sha256']);prepared,_=io.read_prepared(f['source_prepared'])
    return p,root,fb,f,prepared


def predict(p,f,prepared,arm,stage,role):
    if arm not in ARMS or stage not in STAGES or role not in ('CAL','AUDIT'):raise ValueError('undeclared model/stage/role')
    b=f['models'][arm+'/'+stage]['checkpoint'];cp=torch.load(io.verify_binding(b),map_location='cpu',weights_only=False)
    model=make_ablation(arm,f['seed'])
    if model.architecture_config()!=f['models'][arm+'/'+stage]['architecture']:raise ValueError('model architecture changed')
    model.load_state_dict(cp['state_dict'],strict=True);model.eval().requires_grad_(False)
    raw,_=get_predictions(model,prepared,role,p)
    raw['role']=np.full(len(raw['target']),role)
    if any(v.grad is not None for v in model.parameters()):raise ValueError('prediction mutated base model')
    del model
    return raw


def calibrate(pb,arm,stage):
    p,root,fb,f,prepared=frozen(pb);runtime(p)
    out=root/'CAL'/arm/stage;out.mkdir(parents=True,exist_ok=False)
    key=arm+'/'+stage;io.write_json(out/'freeze_before_CAL.json',dict(preparation=fb,model=f['models'][key],
        role='CAL',base_weights_frozen=True,new_AUDIT_decoded=False,code_sha256=f['code_sha256']))
    started=time.monotonic();raw=predict(p,f,prepared,arm,stage,'CAL')
    raw_b=io.save_pack(out/'raw_predictions.npz',raw)
    cache=CalibrationScoreCache(raw,p['calibration']);identity=c.StableCountWarp.identity()
    raw_nodes=identity.row_nodes(raw['num_agents']);raw_a,raw_m=cache.score(raw_nodes)
    def candidate(name,family,ridge,a,m,folds):
        elig={};checks={}
        for selector in SELECTORS:elig[selector],checks[selector]=eligibility(m,raw_m,p['calibration'],selector)
        return dict(family=family,ridge=ridge,metrics=m,selection_scores=m['selection_scores'],eligible=elig,
            guards=checks,folds=folds,predictions=io.save_pack(out/(name+'_CAL_OOF.npz'),a))
    candidates={'identity':candidate('identity','identity',None,raw_a,raw_m,[])}
    records=np.unique(raw['recording_id'])
    if len(records)!=6:raise ValueError('unchanged six-recording CAL role required')
    for family in ('global','count'):
        for ridge in p['calibration']['ridge_grid']:
            name=f'{family}_ridge_{ridge:g}';nodes=np.empty_like(raw_nodes);folds=[]
            for rec in records:
                held=raw['recording_id']==rec
                warp,fit=fit_cached(cache.quadratics['CRPS_seconds'],raw['num_agents'],~held,family=family,ridge=ridge)
                nodes[held]=warp.row_nodes(raw['num_agents'][held])
                folds.append(dict(held_recording=str(rec),train_recordings=records[records!=rec].tolist(),
                    fitting_rows=int((~held).sum()),held_rows=int(held.sum()),fit=fit))
            arrays,metrics=cache.score(nodes)
            candidates[name]=candidate(name,family,ridge,arrays,metrics,folds)
    selections={};fitted={}
    for selector in SELECTORS:
        chosen=select_candidate(candidates,selector);item=candidates[chosen]
        if chosen not in fitted:
            if chosen=='identity':warp=identity;fit=dict(success=True,identity=True,rows=len(raw['target']))
            else:warp,fit=fit_cached(cache.quadratics['CRPS_seconds'],raw['num_agents'],np.ones(len(raw['target']),bool),family=item['family'],ridge=item['ridge'])
            model_b=io.write_json(out/(chosen+'_full_CAL_model.json'),warp.as_dict())
            fitted[chosen]=dict(calibration_model=model_b,fit=fit)
        selections[selector]=dict(chosen=chosen,family=item['family'],ridge=item['ridge'],
            CAL_OOF_selection_score=item['selection_scores'][selector],**fitted[chosen])
    c.verify_sources(f['code_sha256'])
    report=dict(protocol=PROTOCOL,status='calibration_frozen',policy=pb,prepared=fb,arm=arm,stage=stage,
        checkpoint=f['models'][key]['checkpoint'],normalizer=f['normalizer'],data=f['data'],
        raw_CAL_predictions=raw_b,candidates=candidates,selections=selections,CAL_rows=504,CAL_recordings=records.tolist(),
        base_weights_frozen=True,AUDIT_used_for_selection=False,new_AUDIT_decoded=False,
        stage_or_architecture_selection_performed=False,code_sha256=f['code_sha256'],wall_seconds=time.monotonic()-started)
    io.write_json(out/'selection.json',report)
    print(json.dumps(dict(stage='CAL_complete',model=key,selected={k:v['chosen'] for k,v in selections.items()},seconds=report['wall_seconds'])),flush=True)


def freeze_all(pb):
    p,root,fb,f,_=frozen(pb);selections={};identities=None
    for arm in ARMS:
        for stage in STAGES:
            key=arm+'/'+stage;b=c.bind(root/'CAL'/arm/stage/'selection.json');r=c.json_file(b)
            if r['policy']!=pb or r['prepared']!=fb or r['status']!='calibration_frozen' or r['checkpoint']!=f['models'][key]['checkpoint']:
                raise ValueError('incomplete or mismatched CAL choice')
            c.verify_sources(r['code_sha256']);raw=c.arrays(r['raw_CAL_predictions'])
            now={k:raw[k] for k in ('scene_id','recording_id','target','num_agents')}
            if identities is None:identities=now
            elif any(not np.array_equal(identities[k],now[k]) for k in identities):raise ValueError('CAL populations differ across arms')
            selections[key]=b
    report=dict(protocol=PROTOCOL,status='all_CAL_selections_frozen',policy=pb,prepared=fb,selections=selections,
        primary_reference=p['primary_reference'],checkpoints=12,CAL_rows=504,new_AUDIT_decoded=False,
        base_checkpoint_choice_unchanged=True,code_sha256=f['code_sha256'])
    io.write_json(root/'selections_before_AUDIT.json',report)
    print(json.dumps(dict(stage='all_CAL_frozen',checkpoints=12)),flush=True)


def validate(pb,arm,stage):
    p,root,fb,f,prepared=frozen(pb);runtime(p)
    barrier_b=c.bind(root/'selections_before_AUDIT.json');barrier=c.json_file(barrier_b)
    if barrier['status']!='all_CAL_selections_frozen' or barrier['policy']!=pb or len(barrier['selections'])!=12:
        raise ValueError('all12 choices must freeze before any AUDIT decode')
    seen=set();verify_nested(barrier,seen);selection_b=barrier['selections'][arm+'/'+stage];selection=c.json_file(selection_b)
    out=root/'development_AUDIT'/arm/stage;out.mkdir(parents=True,exist_ok=False)
    io.write_json(out/'freeze_before_AUDIT.json',dict(barrier=barrier_b,selection=selection_b,
        AUDIT_previously_used=True,this_round_AUDIT_used_for_selection=False,not_a_fresh_blind_test=True))
    raw=predict(p,f,prepared,arm,stage,'AUDIT');raw_b=io.save_pack(out/'raw_predictions.npz',raw)
    cache=CalibrationScoreCache(raw,p['calibration']);variants={};predictions={}
    for name in p['evaluation']['representations']:
        warp=c.StableCountWarp.identity() if name=='raw' else c.StableCountWarp.from_dict(c.json_file(selection['selections'][name]['calibration_model']))
        a,m=cache.score(warp.row_nodes(raw['num_agents']));variants[name]=m
        predictions[name]=io.save_pack(out/(name+'_predictions.npz'),a)
    c.verify_sources(f['code_sha256'])
    report=dict(protocol=PROTOCOL,status='complete',policy=pb,prepared=fb,barrier=barrier_b,CAL_selection=selection_b,
        arm=arm,stage=stage,checkpoint=selection['checkpoint'],raw_predictions=raw_b,models=variants,predictions=predictions,
        data=f['data'],normalizer=f['normalizer'],scope='reused_development487_not_blind',
        this_round_AUDIT_used_for_selection=False,protected_data_access=False,code_sha256=f['code_sha256'])
    io.write_json(out/'result.json',report)
    print(json.dumps(dict(stage='AUDIT_complete',model=arm+'/'+stage,CRPS={k:v['overall']['CRPS_seconds'] for k,v in variants.items()})),flush=True)


def paired(left,right,p):
    result={}
    for group,mask in [('overall',np.ones(len(left['target']),bool)),('N9_plus',left['num_agents']>=9)]:
        result[group]={}
        for metric in ('CRPS_seconds','twCRPS_1s','twCRPS_2s'):
            l={k:left[k][mask] for k in ('scene_id','recording_id','target')};l['crps_seconds']=left[metric][mask]
            r={k:right[k][mask] for k in ('scene_id','recording_id','target')};r['crps_seconds']=right[metric][mask]
            if len(np.unique(l['recording_id']))<2:
                result[group][metric]=dict(status='insufficient_recordings',scenes=int(mask.sum()));continue
            result[group][metric]=io.paired_recording_bootstrap(l,r,p)
    return result


def summarize(pb):
    p,root,fb,f,_=frozen(pb);barrier_b=c.bind(root/'selections_before_AUDIT.json');barrier=c.json_file(barrier_b)
    results={};sources={};pred={};identity=None
    for arm in ARMS:
        for stage in STAGES:
            key=arm+'/'+stage;b=c.bind(root/'development_AUDIT'/arm/stage/'result.json');r=c.json_file(b)
            if r['status']!='complete' or r['barrier']!=barrier_b or r['policy']!=pb:raise ValueError('complete matched validation required')
            c.verify_sources(r['code_sha256']);results[key]=r;sources[key]=b
            for mode,binding in r['predictions'].items():
                a=c.arrays(binding);pred[key+'/'+mode]=a
                if identity is None:identity={k:a[k] for k in ('scene_id','recording_id','target','num_agents')}
                elif any(not np.array_equal(identity[k],a[k]) for k in identity):raise ValueError('AUDIT rows differ')
    comparisons={}
    for mode in p['evaluation']['representations']:
        left='M2_TimeAttn/stage2/'+mode
        for arm in ARMS:
            if arm=='M2_TimeAttn':continue
            right=arm+'/stage2/'+mode;comparisons[left+'__minus__'+right]=paired(pred[left],pred[right],p)
    for arm in ARMS:
        left=arm+'/stage2/count_balanced';right=arm+'/stage2/legacy_global'
        comparisons[left+'__minus__'+right]=paired(pred[left],pred[right],p)
        left=arm+'/stage2/raw';right=arm+'/stage1/raw'
        comparisons[left+'__minus__'+right]=paired(pred[left],pred[right],p)
    selected=results['M2_TimeAttn/stage2'];sel=c.json_file(selected['CAL_selection'])
    reference=dict(protocol='transformer_publication_frozen_reference_v1',status='complete',architecture='M2_TimeAttn',stage='stage2',
        policy=pb,prepared=fb,checkpoint=selected['checkpoint'],data=f['data'],normalizer=f['normalizer'],
        calibration_model=sel['selections']['count_balanced']['calibration_model'],
        calibration_selection=selected['CAL_selection'],selector='count_balanced',
        base_weights_selected_on='STOP_only_in_P1',calibration_selected_on='CAL_record_LOO_only',
        selected_calibration=sel['selections']['count_balanced'],development_evaluation=sources['M2_TimeAttn/stage2'],
        calibration_quality_not_guaranteed=True,production_default_changed=False,code_sha256=f['code_sha256'])
    ref_b=io.write_json(root/'reference_manifest.json',reference)
    report=dict(protocol=PROTOCOL,status='complete',policy=pb,prepared=fb,barrier=barrier_b,results=sources,
        models={k:r['models'] for k,r in results.items()},comparisons=comparisons,reference_manifest=ref_b,
        checkpoints=12,calibration_selectors=list(SELECTORS),AUDIT_rows_per_checkpoint=487,
        evaluation_representations=36,AUDIT_scope='reused_development_not_blind',
        base_checkpoint_selection_not_changed=True,all_arms_reported=True,
        no_multiple_comparison_adjusted_significance_claim=True,protected_data_access=False,
        generator_changed=False,code_sha256=f['code_sha256'])
    io.write_json(root/'result.json',report)
    print(json.dumps(dict(stage='reference_validation_complete',main_calibration=reference['selected_calibration']['chosen'],
        stage2={arm:{mode:results[arm+'/stage2']['models'][mode]['overall']['CRPS_seconds'] for mode in p['evaluation']['representations']} for arm in ARMS})),flush=True)


def suite(pb):
    p,root,_,_,_=frozen(pb);logs=root/'runlogs';logs.mkdir(exist_ok=True)
    def execute(command,arm=None,stage=None):
        argv=[sys.executable,str(Path(__file__).resolve()),command,'--policy',pb['path'],'--policy-sha256',pb['sha256']]
        if arm:argv+=['--arm',arm,'--stage',stage]
        tag='_'.join(v for v in (command,arm,stage) if v)
        print(json.dumps(dict(starting=tag)),flush=True)
        with (logs/(tag+'.log')).open('x') as log:
            subprocess.run(argv,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(completed=tag)),flush=True)
    for arm in ARMS:
        for stage in STAGES:execute('calibrate',arm,stage)
    execute('freeze')
    for arm in ARMS:
        for stage in STAGES:execute('validate',arm,stage)
    execute('summarize')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','calibrate','freeze','validate','summarize','suite'])
    parser.add_argument('--policy',required=True);parser.add_argument('--policy-sha256',required=True)
    parser.add_argument('--arm',choices=ARMS);parser.add_argument('--stage',choices=STAGES)
    a=parser.parse_args();pb=dict(path=str(io.resolve(a.policy)),sha256=a.policy_sha256)
    if a.command in ('calibrate','validate'):{'calibrate':calibrate,'validate':validate}[a.command](pb,a.arm,a.stage)
    else:{'prepare':prepare,'freeze':freeze_all,'summarize':summarize,'suite':suite}[a.command](pb)
