#!/usr/bin/env python3
"""Validate STOP-selected guarded/unguarded models with exact audited reuse.

Fresh CAL/AUDIT work is performed only for genuinely new tensor states. Existing
results remain explicitly reused development measurements, never fresh tests.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.publication_pipeline.reference import PROTOCOL as REF_PROTOCOL
from pcontrol.reference.history_encoder_ablation import make_ablation
from pcontrol.reference.publication_calibration import CalibrationScoreCache,fit_cached,eligibility,select_candidate,SELECTORS
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import run_transformer_reference_validation as previous
from pcontrol.research.pilot_time_attention_cdf import tensor_hash
from pcontrol.research.validate_time_attention_reference import get_predictions

PROTOCOL='guarded_reference_validation_v1'
CODE=('pcontrol/research/validate_guarded_reference.py','pcontrol/publication_pipeline/reference.py',*previous.CODE)


def sources(pb):
    p=c.json_file(pb)
    if p['protocol']!=PROTOCOL or p['protected_data_access'] or p['primary']!=dict(arm='M2_TimeAttn',rule='group_guarded',calibration_selector='count_balanced'):
        raise ValueError('wrong guarded validation scope')
    guard=c.json_file(p['guarded_study']);ga=c.json_file(p['guarded_audit'])
    old=c.json_file(p['previous_validation']);oa=c.json_file(p['previous_audit'])
    if ga['status']!='pass' or ga['source']['sha256']!=p['guarded_study']['sha256'] or oa['status']!='pass' or oa['result']['sha256']!=p['previous_validation']['sha256']:
        raise ValueError('audited source studies required')
    c.verify_sources(guard['code_sha256']);c.verify_sources(old['code_sha256'])
    return p,guard,old,c.json_file(old['prepared']),c.json_file(old['policy']),io.resolve(p['output_root'])


def load_cp(binding):return torch.load(io.verify_binding(binding),map_location='cpu',weights_only=False)


def prepare(pb):
    p,g,old,op,ep,root=sources(pb);old_states={}
    for key,v in op['models'].items():
        cp=load_cp(v['checkpoint']);old_states[key]=(tensor_hash(cp['state_dict']),cp['data'],cp['normalizer'],v['architecture'])
    entries={};unique={}
    for arm,s in g['selections'].items():
        for rule,item in s.items():
            cp=load_cp(item['checkpoint']);state=tensor_hash(cp['state_dict']);ident=arm+'_epoch'+str(item['epoch'])
            signature=(state,cp['data'],cp['normalizer'],cp['architecture'])
            matches=[key for key,v in old_states.items() if key.split('/')[0]==arm and v==signature]
            record=dict(arm=arm,refinement_epoch=item['epoch'],checkpoint=item['checkpoint'],state_tensor_sha256=state,
                architecture=cp['architecture'],data=cp['data'],normalizer=cp['normalizer'],reuse_key=matches[0] if matches else None)
            if len(matches)>1:raise ValueError('ambiguous reuse; inspect explicitly')
            if ident in unique and unique[ident]!=record:raise ValueError('same epoch identifier has different state')
            unique[ident]=record;entries[arm+'/'+rule]=dict(model_id=ident,selection=item)
    root.mkdir(parents=True,exist_ok=False)
    f=dict(protocol=PROTOCOL,status='prepared',policy=pb,source_study=p['guarded_study'],source_audit=p['guarded_audit'],
        previous_validation=p['previous_validation'],previous_audit=p['previous_audit'],
        previous_prepared=old['prepared'],calibration_policy=old['policy'],source_prepared=op['source_prepared'],
        data=op['data'],normalizer=op['normalizer'],seed=op['seed'],unique_models=unique,entries=entries,
        primary=p['primary'],code_sha256=c.source_bindings(CODE),
        exact_tensor_state_reuse=True,reused_results_not_independent_new_measurements=True,protected_data_access=False)
    fb=io.write_json(root/'prepared.json',f)
    # The primary is independently frozen before the comparison work. Reuse is
    # permitted here only because the exact model AND CAL selection were audited.
    main=entries['M2_TimeAttn/group_guarded'];record=unique[main['model_id']]
    if record['reuse_key'] is None:raise ValueError('early primary freeze requires audited exact reuse')
    old_eval_b=old['results'][record['reuse_key']];old_eval=c.json_file(old_eval_b)
    cal_b=old_eval['CAL_selection'];cal=c.json_file(cal_b);choice=cal['selections']['count_balanced']
    main_training=c.json_file(g['arms']['M2_TimeAttn'])
    base_training_cp=load_cp(main_training['initial_model']['checkpoint'])
    if base_training_cp['stage']!='stage1':raise ValueError('ordinary-stage epoch provenance required')
    ref=dict(protocol=REF_PROTOCOL,status='complete',architecture='M2_TimeAttn',policy=pb,prepared=fb,
        checkpoint=record['checkpoint'],tensor_state_sha256=record['state_tensor_sha256'],data=record['data'],normalizer=record['normalizer'],
        base_selection='STOP_group_guarded',base_selection_result=p['guarded_study'],refinement_epoch=record['refinement_epoch'],
        calibration_model=choice['calibration_model'],calibration_selection=cal_b,calibration_selected_on='CAL_record_LOO_only',
        calibration_selector='count_balanced',selected_family=choice['family'],selected_ridge=choice['ridge'],
        validation_reuse=dict(source=old_eval_b,source_audit=p['previous_audit'],source_key=record['reuse_key'],
            exact_parameters_data_normalizer_architecture=True,fresh_measurement=False,scope='reused_development487'),
        model_training_epochs=dict(base=base_training_cp['best_epoch'],highN=record['refinement_epoch']),
        code_sha256=f['code_sha256'],production_default_changed=False,reference_is_estimated=True)
    io.write_json(root/'reference_manifest.json',ref)
    print(json.dumps(dict(stage='guarded_reference_prepared',entries=len(entries),unique_models=len(unique),
        reused_unique=sum(v['reuse_key'] is not None for v in unique.values()),new_unique=sum(v['reuse_key'] is None for v in unique.values()),
        primary_reuses=record['reuse_key'],primary_calibration=choice['chosen'])),flush=True)


def frozen(pb):
    p,guard,old,op,ep,root=sources(pb);fb=c.bind(root/'prepared.json');f=c.json_file(fb)
    if f['policy']!=pb:raise ValueError('policy drift')
    c.verify_sources(f['code_sha256']);prepared,_=io.read_prepared(f['source_prepared'])
    return p,old,ep,root,fb,f,prepared


def prediction(ep,f,prepared,record,role):
    cp=load_cp(record['checkpoint']);model=make_ablation(record['arm'],f['seed'])
    if model.architecture_config()!=record['architecture'] or tensor_hash(cp['state_dict'])!=record['state_tensor_sha256']:
        raise ValueError('checkpoint drift')
    model.load_state_dict(cp['state_dict'],strict=True);model.eval().requires_grad_(False)
    raw,_=get_predictions(model,prepared,role,ep);raw['role']=np.full(len(raw['target']),role)
    return raw


def calibrate_new(pb):
    p,old,ep,root,fb,f,prepared=frozen(pb);previous.runtime(ep);selections={}
    for ident,record in f['unique_models'].items():
        if record['reuse_key'] is not None:
            ev=c.json_file(old['results'][record['reuse_key']]);selections[ident]=dict(reused=True,selection=ev['CAL_selection']);continue
        out=root/'CAL'/ident;out.mkdir(parents=True,exist_ok=False)
        raw=prediction(ep,f,prepared,record,'CAL');raw_b=io.save_pack(out/'base_predictions.npz',raw)
        cache=CalibrationScoreCache(raw,ep['calibration']);identity=c.StableCountWarp.identity()
        nodes0=identity.row_nodes(raw['num_agents']);a0,m0=cache.score(nodes0);candidates={}
        def store(name,family,ridge,arrays,metrics,folds):
            decisions={s:eligibility(metrics,m0,ep['calibration'],s) for s in SELECTORS}
            return dict(family=family,ridge=ridge,metrics=metrics,selection_scores=metrics['selection_scores'],
                eligible={s:v[0] for s,v in decisions.items()},guards={s:v[1] for s,v in decisions.items()},folds=folds,
                predictions=io.save_pack(out/(name+'_CAL_OOF.npz'),arrays))
        candidates['identity']=store('identity','identity',None,a0,m0,[]);records=np.unique(raw['recording_id'])
        for family in ('global','count'):
            for ridge in ep['calibration']['ridge_grid']:
                name=f'{family}_ridge_{ridge:g}';nodes=np.empty_like(nodes0);folds=[]
                for rec in records:
                    held=raw['recording_id']==rec
                    warp,fit=fit_cached(cache.quadratics['CRPS_seconds'],raw['num_agents'],~held,family=family,ridge=ridge)
                    nodes[held]=warp.row_nodes(raw['num_agents'][held]);folds.append(dict(held_recording=str(rec),
                        train_recordings=records[records!=rec].tolist(),fitting_rows=int((~held).sum()),held_rows=int(held.sum()),
                        fit=fit,calibration_model=warp.as_dict()))
                arrays,metrics=cache.score(nodes);candidates[name]=store(name,family,ridge,arrays,metrics,folds)
        chosen={};fits={}
        for rule in SELECTORS:
            name=select_candidate(candidates,rule);item=candidates[name]
            if name not in fits:
                if name=='identity':warp=identity;fit=dict(success=True,identity=True)
                else:warp,fit=fit_cached(cache.quadratics['CRPS_seconds'],raw['num_agents'],np.ones(len(raw['target']),bool),family=item['family'],ridge=item['ridge'])
                fits[name]=dict(calibration_model=io.write_json(out/(name+'_full_CAL.json'),warp.as_dict()),fit=fit)
            chosen[rule]=dict(chosen=name,family=item['family'],ridge=item['ridge'],**fits[name])
        report=dict(protocol=PROTOCOL,status='calibration_frozen',policy=pb,prepared=fb,model_id=ident,model=record,
            raw_CAL_predictions=raw_b,candidates=candidates,selections=chosen,code_sha256=f['code_sha256'],new_AUDIT_decoded=False)
        selections[ident]=dict(reused=False,selection=io.write_json(out/'selection.json',report))
        print(json.dumps(dict(stage='new_CAL_complete',model=ident,selected={k:v['chosen'] for k,v in chosen.items()})),flush=True)
    c.verify_sources(f['code_sha256'])
    io.write_json(root/'selections_before_new_AUDIT.json',dict(protocol=PROTOCOL,status='all_selections_frozen',policy=pb,
        prepared=fb,selections=selections,all_new_choices_before_new_AUDIT=True,reused_evaluations_already_known=True))


def validate_new(pb):
    p,old,ep,root,fb,f,prepared=frozen(pb);previous.runtime(ep)
    barrier_b=c.bind(root/'selections_before_new_AUDIT.json');barrier=c.json_file(barrier_b)
    if barrier['status']!='all_selections_frozen' or barrier['policy']!=pb or set(barrier['selections'])!=set(f['unique_models']):raise ValueError('all choices must freeze')
    results={}
    for ident,record in f['unique_models'].items():
        item=barrier['selections'][ident]
        if item['reused']:
            results[ident]=dict(reused=True,result=old['results'][record['reuse_key']]);continue
        out=root/'development_AUDIT'/ident;out.mkdir(parents=True,exist_ok=False)
        io.write_json(out/'freeze.json',dict(barrier=barrier_b,selection=item['selection'],model=record))
        raw=prediction(ep,f,prepared,record,'AUDIT');raw_b=io.save_pack(out/'base_predictions.npz',raw)
        cache=CalibrationScoreCache(raw,ep['calibration']);sel=c.json_file(item['selection']);models={};predictions={}
        for mode in ('raw',*SELECTORS):
            warp=c.StableCountWarp.identity() if mode=='raw' else c.StableCountWarp.from_dict(c.json_file(sel['selections'][mode]['calibration_model']))
            arrays,metrics=cache.score(warp.row_nodes(raw['num_agents']));models[mode]=metrics
            predictions[mode]=io.save_pack(out/(mode+'_scores.npz'),arrays)
        report=dict(protocol=PROTOCOL,status='complete',policy=pb,prepared=fb,barrier=barrier_b,model_id=ident,
            model=record,raw_predictions=raw_b,models=models,predictions=predictions,CAL_selection=item['selection'],
            code_sha256=f['code_sha256'],scope='reused_development487_not_blind',base_or_calibration_reselection=False)
        results[ident]=dict(reused=False,result=io.write_json(out/'result.json',report))
        print(json.dumps(dict(stage='new_AUDIT_complete',model=ident,CRPS={k:v['overall']['CRPS_seconds'] for k,v in models.items()})),flush=True)
    entries={};comparisons={};primary=None
    for key,entry in f['entries'].items():
        source=results[entry['model_id']];data=c.json_file(source['result'])
        entries[key]=dict(entry,evaluation=source,models=data['models'],predictions=data['predictions'])
        if key=='M2_TimeAttn/group_guarded':primary=entries[key]
    for key,value in entries.items():
        arm,rule=key.split('/')
        for mode in ('raw',*SELECTORS):
            if rule=='group_guarded' and arm!='M2_TimeAttn':
                comparisons['TimeAttn_minus_'+key+'/'+mode]=previous.paired(c.arrays(primary['predictions'][mode]),c.arrays(value['predictions'][mode]),ep)
            if rule=='group_guarded':
                other=entries[arm+'/unguarded']
                comparisons[key+'_minus_unguarded/'+mode]=previous.paired(c.arrays(value['predictions'][mode]),c.arrays(other['predictions'][mode]),ep)
    c.verify_sources(f['code_sha256'])
    report=dict(protocol=PROTOCOL,status='complete',policy=pb,prepared=fb,barrier=barrier_b,unique_evaluations=results,
        entries=entries,comparisons=comparisons,primary_reference=c.bind(root/'reference_manifest.json'),
        reused_unique=sum(v['reused'] for v in results.values()),new_unique=sum(not v['reused'] for v in results.values()),
        observations_per_evaluation=487,all_choices_on_STOP_and_CAL=True,no_new_blind_measurements=True,
        code_sha256=f['code_sha256'],protected_data_access=False)
    io.write_json(root/'result.json',report)
    print(json.dumps(dict(stage='guarded_validation_complete',reused=report['reused_unique'],new=report['new_unique'],
        guarded_CRPS={k:v['models']['count_balanced']['overall']['CRPS_seconds'] for k,v in entries.items() if k.endswith('/group_guarded')})),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','calibrate','validate'])
    parser.add_argument('--policy',required=True);parser.add_argument('--policy-sha256',required=True)
    a=parser.parse_args();pb=dict(path=str(io.resolve(a.policy)),sha256=a.policy_sha256)
    {'prepare':prepare,'calibrate':calibrate_new,'validate':validate_new}[a.command](pb)
