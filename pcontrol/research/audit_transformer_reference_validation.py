#!/usr/bin/env python3
"""Independent score/CV/selection audit and actual CAL/AUDIT model replay.

Recomputes calibration folds with the original noncached optimizer, keeping
held recordings out. This does not change a choice or fit a new deployed model.
"""
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
from pcontrol.reference.scene_calibration import fit_scene_calibration
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import run_transformer_reference_validation as study
from pcontrol.research.audit_scene_context_diagnostics import independent_scores
from pcontrol.research.audit_time_attention_full_pipeline import verify_nested

POLICY=dict(path=str(ROOT/'configs/natural_percentile/transformer_reference_validation_v1.json'),
            sha256='024114be2e51dec83568d3d4f4ab89b960ab276abe5300ccfa5c8d934b6bd42a')


def own_metrics(a,cfg):
    """Selection arithmetic from saved, independently numerically checked rows."""
    y=a['target'];n=a['num_agents'];grid=np.asarray(cfg['PIT_grid']);thresholds=np.asarray(cfg['thresholds_seconds'])
    def group(mask):
        if not mask.any():return dict(scenes=0)
        left=a['cdf_left'][mask];right=a['cdf_right'][mask];width=right-left
        fractions=(grid[None]-left[:,None])/np.where(width>0,width,1.)[:,None]
        contributions=np.where((width>0)[:,None],np.minimum(np.maximum(fractions,0.),1.),right[:,None]<=grid)
        pit=abs(contributions.mean(0)-grid)
        threshold=float(abs(a['threshold_cdf'][mask].mean(0)-(y[mask,None]<=thresholds).mean(0)).mean())
        return dict(scenes=int(mask.sum()),CRPS_seconds=float(a['CRPS_seconds'][mask].mean()),
            twCRPS_1s=float(a['twCRPS_1s'][mask].mean()),twCRPS_2s=float(a['twCRPS_2s'][mask].mean()),
            expected_PIT_grid_MAE=float(pit.mean()),expected_PIT_grid_KS=float(pit.max()),threshold_MAE=threshold,
            local_component_score=float(pit.mean()+2*threshold))
    return dict(overall=group(np.ones(len(y),bool)),groups={'N':{
        'N3_5':group(n<6),'N6_8':group((n>=6)&(n<9)),'N9_plus':group(n>=9)}})


def own_selection(candidates,raw,cfg,selector):
    ranking=[]
    for name,item in candidates.items():
        m=item['metrics'];a=m['overall'];b=raw['overall'];rule=cfg['selectors'][selector]
        ok=a['CRPS_seconds']<=rule['overall_CRPS_ratio']*b['CRPS_seconds']
        if selector=='legacy_global':
            if raw['groups']['N']['N9_plus']['scenes']:
                ok &= m['groups']['N']['N9_plus']['CRPS_seconds']<=rule['highN_CRPS_ratio']*raw['groups']['N']['N9_plus']['CRPS_seconds']
            score=a['expected_PIT_grid_KS']+2*a['threshold_MAE']
        else:
            local=[]
            for k,v in raw['groups']['N'].items():
                if v['scenes']>=cfg['minimum_group_rows']:
                    q=m['groups']['N'][k];ok &= q['CRPS_seconds']<=rule['supported_group_CRPS_ratio']*v['CRPS_seconds']
                    local.append(q['local_component_score'])
            for key in ('twCRPS_1s','twCRPS_2s'):ok &= a[key]<=rule['tail_CRPS_ratio']*b[key]
            score=.5*a['local_component_score']+.5*np.mean(local) if local else a['local_component_score']
        if bool(ok)!=item['eligible'][selector] or abs(score-item['selection_scores'][selector])>2e-12:
            raise ValueError('CAL eligibility/selection score does not replay')
        if ok:ranking.append((score,0 if name=='identity' else 1 if item['family']=='global' else 2,-(item['ridge'] or 0.),name))
    if not ranking:raise ValueError('identity fallback missing')
    return min(ranking)[-1]


def numeric_check(a,cfg):
    maximum=0.;u=np.array([0.,.05,.1,.25,.5,.75,.9,.95,1.]);thresholds=np.asarray(cfg['thresholds_seconds'])
    for i,(mass,y,node) in enumerate(zip(a['joint_masses'],a['target'],a['effective_nodes'])):
        got=independent_scores(mass,y,node,u,thresholds)
        for key,value in got.items():maximum=max(maximum,float(np.max(abs(value-a[key][i]))))
    if maximum>2e-10:raise ValueError('independent score/CDF disagreement')
    return maximum


def run():
    p,root,fb,f,prepared=study.frozen(POLICY);study.runtime(p)
    source=c.bind(root/'result.json');result=c.json_file(source);barrier=c.json_file(result['barrier'])
    execution_b=c.bind(root/'execution_completion.json');execution=c.json_file(execution_b)
    if result['status']!='complete' or execution['result']!=source or result['checkpoints']!=12:raise ValueError('complete matched study required')
    seen=set();verify_nested(result,seen);verify_nested(execution,seen);c.verify_sources(execution['code_sha256'])
    maximum=dict(score=0.,CAL_fold_nodes=0.,final_nodes=0.,base_probability_replay=0.,summary=0.)
    scored=fold_count=forward_rows=0
    for arm in study.ARMS:
        for stage in study.STAGES:
            key=arm+'/'+stage;selection_b=barrier['selections'][key];selection=c.json_file(selection_b)
            if selection['checkpoint']!=f['models'][key]['checkpoint'] or selection['AUDIT_used_for_selection']:
                raise ValueError('base checkpoint or selection role changed')
            verify_nested(selection,seen);c.verify_sources(selection['code_sha256'])
            raw=c.arrays(selection['raw_CAL_predictions']);records=set(raw['recording_id']);own_candidates={}
            if len(raw['target'])!=504 or len(records)!=6 or not (raw['role']=='CAL').all():raise ValueError('CAL population mismatch')
            for name,item in selection['candidates'].items():
                a=c.arrays(item['predictions'])
                for field in ('scene_id','recording_id','target','num_agents'):
                    if not np.array_equal(raw[field],a[field]):raise ValueError('candidate changed an observation')
                maximum['score']=max(maximum['score'],numeric_check(a,p['calibration']));scored+=len(a['target'])
                own=own_metrics(a,p['calibration'])
                for axis,m in [('overall',own['overall'])]+[(g,v) for g,v in own['groups']['N'].items()]:
                    stored=item['metrics']['overall'] if axis=='overall' else item['metrics']['groups']['N'][axis]
                    for metric,value in m.items():maximum['summary']=max(maximum['summary'],abs(value-stored[metric]))
                own_candidates[name]=dict(item,metrics=own)
                if name=='identity':continue
                if len(item['folds'])!=6:raise ValueError('missing CAL exclusion folds')
                for fold in item['folds']:
                    rec=fold['held_recording'];held=raw['recording_id']==rec;train=set(fold['train_recordings'])
                    if rec in train or train|{rec}!=records or fold['fitting_rows']!=int((~held).sum()) or fold['held_rows']!=int(held.sum()):
                        raise ValueError('held calibration recording entered its fit')
                    fit=fit_scene_calibration(raw['joint_masses'][~held],raw['num_agents'][~held],raw['target'][~held],family=item['family'],ridge=item['ridge'])
                    if not fit.report['success']:raise ValueError('independent fold refit failed')
                    warp=c.StableCountWarp(fit.warp.family,fit.warp.node_values)
                    maximum['CAL_fold_nodes']=max(maximum['CAL_fold_nodes'],float(abs(warp.row_nodes(raw['num_agents'][held])-a['effective_nodes'][held]).max()))
                    fold_count+=1
            raw_m=own_candidates['identity']['metrics']
            for selector,sel in selection['selections'].items():
                chosen=own_selection(own_candidates,raw_m,p['calibration'],selector)
                if chosen!=sel['chosen']:raise ValueError('CAL-selected candidate changed')
                deployed=c.StableCountWarp.from_dict(c.json_file(sel['calibration_model']))
                if chosen=='identity':reference=c.StableCountWarp.identity()
                else:
                    fit=fit_scene_calibration(raw['joint_masses'],raw['num_agents'],raw['target'],family=sel['family'],ridge=sel['ridge'])
                    if not fit.report['success']:raise ValueError('independent final refit failed')
                    reference=c.StableCountWarp(fit.warp.family,fit.warp.node_values)
                maximum['final_nodes']=max(maximum['final_nodes'],float(abs(reference.node_values-deployed.node_values).max()))
            audited=c.json_file(result['results'][key]);verify_nested(audited,seen)
            if audited['barrier']!=result['barrier'] or audited['CAL_selection']!=selection_b:raise ValueError('AUDIT bypassed the global freeze')
            audit_raw=c.arrays(audited['raw_predictions'])
            if set(audit_raw['recording_id'])&records:raise ValueError('CAL/AUDIT recording overlap')
            for b in audited['predictions'].values():
                a=c.arrays(b)
                if len(a['target'])!=487 or not (a['role']=='AUDIT').all():raise ValueError('AUDIT denominator changed')
                maximum['score']=max(maximum['score'],numeric_check(a,p['calibration']));scored+=len(a['target'])
            for role,stored in [('CAL',raw),('AUDIT',audit_raw)]:
                replay=study.predict(p,f,prepared,arm,stage,role)
                maximum['base_probability_replay']=max(maximum['base_probability_replay'],float(abs(replay['joint_masses']-stored['joint_masses']).max()))
                forward_rows+=len(replay['target'])
            print(json.dumps(dict(audited=key,scored_rows=scored,refitted_exclusion_folds=fold_count)),flush=True)
    if (scored!=84060 or fold_count!=720 or forward_rows!=11892 or maximum['CAL_fold_nodes']>2e-7
            or maximum['final_nodes']>2e-7 or maximum['base_probability_replay']>2e-5 or maximum['summary']>2e-12):
        raise ValueError('full audit coverage or numerical tolerance failure: '+str(maximum))
    report=dict(status='pass',result=source,execution=execution_b,scored_rows=scored,refitted_exclusion_folds=fold_count,
        neural_prediction_rows_replayed=forward_rows,checkpoints=12,representations=36,
        maximum_errors=maximum,artifact_bindings_verified=len(seen),CAL_choices_independently_replayed=True,
        raw_fitting_implementation='original_non_cached_convex_optimizer_on_excluded_recording_subset',
        scoring_implementation='independent_piecewise_linear_polynomial_integrals',
        no_selections_changed=True,no_new_observations_or_protected_data=True,
        not_an_independent_population_test=True,code_sha256=io.sha256(__file__))
    io.write_json(root/'independent_audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':run()
