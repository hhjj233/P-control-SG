#!/usr/bin/env python3
"""Frozen CAL selection and development comparison for the wide-MLP control."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.publication_pipeline.generator_data import same_binding
from pcontrol.reference.capacity_matched_mlp import make_capacity_control
from pcontrol.reference.publication_calibration import CalibrationScoreCache,fit_cached,eligibility,select_candidate,SELECTORS
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import run_transformer_reference_validation as previous
from pcontrol.research.pilot_time_attention_cdf import tensor_hash
from pcontrol.research.validate_time_attention_reference import get_predictions

PROTOCOL='transformer_capacity_control_CAL_development_validation_v1'
CODE=('pcontrol/research/validate_transformer_capacity_control.py','pcontrol/reference/capacity_matched_mlp.py',*previous.CODE)


def prepare(pb):
    p=c.json_file(pb)
    if p['protocol']!=PROTOCOL or p['protected_data_access'] or p['existing_primary_reference_or_generator_changed']:raise ValueError('wrong comparison scope')
    r=c.json_file(p['training_result']);tp=c.json_file(r['policy']);parent=c.json_file(tp['parent_policy'])
    ab=c.bind(io.verify_binding(p['training_result']).parent/'independent_audit.json');audit=c.json_file(ab)
    if r['status']!='complete' or audit['status']!='pass' or not same_binding(audit['source'],p['training_result']):raise ValueError('audited actual training required')
    c.verify_sources(r['code_sha256']);reference=c.json_file(p['baseline_reference'])
    if r['data']!=reference['data'] or r['normalizer']!=reference['normalizer']:raise ValueError('comparator population mismatch')
    entries={};models={}
    for key in p['variants']:
        chosen=dict(epoch=0,checkpoint=r['stage1']['checkpoint']) if key=='stage1' else r['selections'][key]
        ident='epoch'+str(chosen['epoch']);cp=torch.load(io.verify_binding(chosen['checkpoint']),map_location='cpu',weights_only=False)
        record=dict(checkpoint=chosen['checkpoint'],architecture=cp['architecture'],state_sha256=tensor_hash(cp['state_dict']))
        if ident in models and models[ident]!=record:raise ValueError('inconsistent shared stage/guard state')
        models[ident]=record;entries[key]=ident
    root=io.resolve(p['output_root']);root.mkdir(exist_ok=False)
    manifest=dict(protocol=PROTOCOL,status='prepared',policy=pb,training_result=p['training_result'],training_audit=ab,
        calibration_policy=p['calibration_policy'],source_prepared=parent['reference_prepared'],models=models,entries=entries,
        data=r['data'],normalizer=r['normalizer'],baseline_reference=p['baseline_reference'],
        baseline_evaluation=reference['validation_reuse']['source'],code_sha256=c.source_bindings(CODE),
        primary_comparison='group_guarded/count_balanced',base_checkpoint_choices_frozen=True,protected_data_access=False)
    io.write_json(root/'prepared.json',manifest);print(json.dumps(dict(stage='capacity_validation_prepared',unique_models=len(models),entries=entries)),flush=True)


def frozen(pb):
    p=c.json_file(pb);root=io.resolve(p['output_root']);fb=c.bind(root/'prepared.json');f=c.json_file(fb)
    if not same_binding(f['policy'],pb):raise ValueError('preparation drift')
    c.verify_sources(f['code_sha256']);ep=c.json_file(f['calibration_policy']);prepared,_=io.read_prepared(f['source_prepared'])
    return p,ep,root,fb,f,prepared


def predict(record,prepared,ep,role):
    if role not in ('CAL','AUDIT'):raise ValueError('undeclared development role')
    cp=torch.load(io.verify_binding(record['checkpoint']),map_location='cpu',weights_only=False)
    model=make_capacity_control()
    if model.architecture_config()!=record['architecture'] or tensor_hash(cp['state_dict'])!=record['state_sha256']:raise ValueError('model drift')
    model.load_state_dict(cp['state_dict'],strict=True);model.eval().requires_grad_(False)
    raw,_=get_predictions(model,prepared,role,ep);raw['role']=np.full(len(raw['target']),role);return raw


def calibrate(pb):
    p,ep,root,fb,f,prepared=frozen(pb);previous.runtime(ep);selections={}
    for ident,record in f['models'].items():
        out=root/'CAL'/ident;out.mkdir(parents=True,exist_ok=False)
        raw=predict(record,prepared,ep,'CAL');raw_b=io.save_pack(out/'base_CAL_predictions.npz',raw)
        cache=CalibrationScoreCache(raw,ep['calibration']);identity=c.StableCountWarp.identity();nodes0=identity.row_nodes(raw['num_agents'])
        a0,m0=cache.score(nodes0);candidates={}
        def store(name,family,ridge,arrays,metrics,folds):
            decisions={s:eligibility(metrics,m0,ep['calibration'],s) for s in SELECTORS}
            return dict(family=family,ridge=ridge,metrics=metrics,selection_scores=metrics['selection_scores'],
                eligible={s:v[0] for s,v in decisions.items()},guards={s:v[1] for s,v in decisions.items()},folds=folds,
                predictions=io.save_pack(out/(name+'_scores.npz'),arrays))
        candidates['identity']=store('identity','identity',None,a0,m0,[]);records=np.unique(raw['recording_id'])
        if len(raw['target'])!=504 or len(records)!=6:raise ValueError('CAL denominator mismatch')
        for family in ('global','count'):
            for ridge in ep['calibration']['ridge_grid']:
                name=f'{family}_ridge_{ridge:g}';nodes=np.empty_like(nodes0);folds=[]
                for rec in records:
                    held=raw['recording_id']==rec
                    warp,fit=fit_cached(cache.quadratics['CRPS_seconds'],raw['num_agents'],~held,family=family,ridge=ridge)
                    nodes[held]=warp.row_nodes(raw['num_agents'][held])
                    folds.append(dict(held_recording=str(rec),train_recordings=records[records!=rec].tolist(),fit=fit,calibration_model=warp.as_dict()))
                arrays,metrics=cache.score(nodes);candidates[name]=store(name,family,ridge,arrays,metrics,folds)
        choices={};fits={}
        for rule in SELECTORS:
            name=select_candidate(candidates,rule);item=candidates[name]
            if name not in fits:
                if name=='identity':warp=identity;fit=dict(success=True,identity=True)
                else:warp,fit=fit_cached(cache.quadratics['CRPS_seconds'],raw['num_agents'],np.ones(len(raw['target']),bool),family=item['family'],ridge=item['ridge'])
                fits[name]=dict(calibration_model=io.write_json(out/(name+'_full_CAL.json'),warp.as_dict()),fit=fit)
            choices[rule]=dict(chosen=name,family=item['family'],ridge=item['ridge'],**fits[name])
        selections[ident]=io.write_json(out/'selection.json',dict(status='calibration_frozen',prepared=fb,model=record,
            raw_CAL_predictions=raw_b,candidates=candidates,selections=choices,AUDIT_used_for_selection=False,code_sha256=f['code_sha256']))
        print(json.dumps(dict(stage='capacity_CAL_complete',model=ident,choices={k:v['chosen'] for k,v in choices.items()})),flush=True)
    c.verify_sources(f['code_sha256'])
    io.write_json(root/'selections_before_AUDIT.json',dict(status='all_new_CAL_choices_frozen',policy=pb,prepared=fb,
        selections=selections,base_weight_choices_unchanged=True,new_AUDIT_decoded=False))


def validate(pb):
    p,ep,root,fb,f,prepared=frozen(pb);previous.runtime(ep)
    barrier_b=c.bind(root/'selections_before_AUDIT.json');barrier=c.json_file(barrier_b)
    if barrier['status']!='all_new_CAL_choices_frozen' or not same_binding(barrier['prepared'],fb) or set(barrier['selections'])!=set(f['models']):
        raise ValueError('all CAL choices must be complete before AUDIT')
    evaluations={}
    for ident,record in f['models'].items():
        out=root/'development_AUDIT'/ident;out.mkdir(parents=True,exist_ok=False)
        raw=predict(record,prepared,ep,'AUDIT');raw_b=io.save_pack(out/'base_AUDIT_predictions.npz',raw)
        cache=CalibrationScoreCache(raw,ep['calibration']);selection=c.json_file(barrier['selections'][ident]);models={};predictions={}
        for mode in ('raw',*SELECTORS):
            warp=c.StableCountWarp.identity() if mode=='raw' else c.StableCountWarp.from_dict(c.json_file(selection['selections'][mode]['calibration_model']))
            a,m=cache.score(warp.row_nodes(raw['num_agents']));models[mode]=m;predictions[mode]=io.save_pack(out/(mode+'_scores.npz'),a)
        evaluations[ident]=io.write_json(out/'result.json',dict(status='complete',prepared=fb,barrier=barrier_b,model=record,
            raw_predictions=raw_b,models=models,predictions=predictions,CAL_selection=barrier['selections'][ident],
            code_sha256=f['code_sha256'],scope='reused_development487_not_blind',base_or_calibration_reselection=False))
        print(json.dumps(dict(stage='capacity_AUDIT_complete',model=ident,CRPS={k:v['overall']['CRPS_seconds'] for k,v in models.items()})),flush=True)
    baseline=c.json_file(f['baseline_evaluation']);entries={};comparisons={}
    for key,ident in f['entries'].items():
        r=c.json_file(evaluations[ident]);entries[key]=dict(model_id=ident,evaluation=evaluations[ident],models=r['models'],predictions=r['predictions'])
    for mode in ('raw',*SELECTORS):
        left=c.arrays(baseline['predictions'][mode]);right=c.arrays(entries['group_guarded']['predictions'][mode])
        for name in ('scene_id','recording_id','target','num_agents'):
            if not np.array_equal(left[name],right[name]):raise ValueError('baseline/comparator AUDIT identity mismatch')
        comparisons[mode]=previous.paired(left,right,ep)
    c.verify_sources(f['code_sha256'])
    result=dict(protocol=PROTOCOL,status='complete',policy=pb,prepared=fb,barrier=barrier_b,unique_evaluations=evaluations,
        entries=entries,baseline_evaluation=f['baseline_evaluation'],TimeAttn_minus_WideMLP=comparisons,
        all_variants_reported=True,main_reference_unchanged=True,protected_data_access=False,development_not_blind=True,
        code_sha256=f['code_sha256'])
    io.write_json(root/'result.json',result);print(json.dumps(dict(stage='capacity_validation_complete',guarded=entries['group_guarded']['models']['count_balanced']['overall'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','calibrate','validate'])
    parser.add_argument('--policy',default='configs/natural_percentile/transformer_capacity_validation_v1.json');parser.add_argument('--policy-sha256',required=True)
    a=parser.parse_args();pb=dict(path=str(io.resolve(a.policy)),sha256=a.policy_sha256)
    {'prepare':prepare,'calibrate':calibrate,'validate':validate}[a.command](pb)
