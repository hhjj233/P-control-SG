#!/usr/bin/env python3
"""Replay saved trajectories, raw ranks and complete paired denominators.

PET uses the same declared all-actor oracle (not an independent definition).
CDF interpolation, target infimum, quality arithmetic and summary aggregation
are independent of the generation evaluator. No retraining or resampling.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.publication_pipeline.generator_data import same_binding
from pcontrol.data.scene_pet import scene_occupancy_pet
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import train_time_attention_generator as common_train
from pcontrol.research import train_guarded_terminal_pair as paired_train
from pcontrol.research.refine_natural_direct_p import state_hash
from pcontrol.research.audit_pair_overlap_intervals import pair_overlap_intervals
from pcontrol.research.audit_time_attention_full_pipeline import rank,quantile,midpoint_error_infimum,native_checks,verify_nested


def replay_row(row,case,future):
    n=len(case['dimensions']);ego=int(np.flatnonzero(case['ego_mask'])[0]);requested=float(row['requested_p'])
    if (future.shape!=(175,n,4) or not np.isfinite(future).all() or not np.array_equal(future[0],case['history'][-1])
            or row['num_agents']!=n or not case['agent_mask'].all() or row['K']!=1
            or row['post_sampler_repair'] or row['inference_observed_future_input'] or not row['all_actors_retained']):
        raise ValueError('full-roster/t0/inference/denominator contract mismatch')
    measured=scene_occupancy_pet(future,np.ones(future.shape[:2],bool),case['dimensions'],times=np.arange(175)*.04,
        ego_index=ego,sample_period=.04,window=(0.,6.96),cap_seconds=4.)
    pet=float(measured['pet_value_seconds']);m=case['reference_joint_masses'];nodes=case['reference_row_nodes']
    achieved=rank(m,nodes,pet);canonical=quantile(m,nodes,1.-requested)
    floor=midpoint_error_infimum(m,nodes,requested);target=float(row['control_target_PET_seconds'])
    if not 0<=target<=4:raise ValueError('invalid physical target')
    point=abs(achieved['p_mid']-requested);interval=max(achieved['p_low']-requested,requested-achieved['p_up'],0.)
    errors=dict(PET=abs(pet-row['pet_seconds']),rank=max(abs(achieved[k]-row['estimated_rank'][k]) for k in achieved),
        point=abs(point-row['p_mid_absolute_error']),interval=abs(interval-row['p_interval_error']),
        canonical_target=abs(canonical-row['canonical_target_PET_seconds']),scalar_infimum=abs(floor-row['scalar_error_infimum']),
        control_PET_error=abs(abs(pet-target)-row['PET_control_target_absolute_error_seconds']),
        canonical_PET_error=abs(abs(pet-canonical)-row['canonical_PET_target_absolute_error_seconds']),quality=0.,guidance_rank=0.)
    if point+2e-10<floor:raise ValueError('claimed scalar infimum exceeds realized point error')
    if row['arm']=='canonical':errors['canonical_target']=max(errors['canonical_target'],abs(target-canonical))
    elif row['arm']=='atom_aware':
        target_error=abs(rank(m,nodes,target)['p_mid']-requested)
        if target_error>floor+1e-5+2e-10 or target_error>abs(rank(m,nodes,canonical)['p_mid']-requested)+2e-10:
            raise ValueError('atom-aware physical target fails independent error-floor/canonical bound')
    else:raise ValueError('unknown arm')
    quality=native_checks(future,case['dimensions'],case['road_boundaries'],ego)
    for key,value in quality.items():
        source=row['quality'] if key in row['quality'] else row['scene_quality']
        if key in source:errors['quality']=max(errors['quality'],abs(float(value)-float(source[key])))
    negative=bool((future[...,2]<0).any())
    if negative!=row['quality']['negative_vx_scene']:raise ValueError('negative velocity request count mismatch')
    intervals=pair_overlap_intervals(future,case['dimensions'],ego)
    bg=any(v['background_pair'] for v in intervals);eg=any(not v['background_pair'] for v in intervals)
    if intervals!=row['PL_overlap_intervals'] or bg!=row['background_PL_overlap_scene'] or eg!=row['ego_PL_overlap_scene']:
        raise ValueError('independent PL slab replay mismatch')
    guide=row['guidance']
    if row['network_evaluations']!=100 or guide['callback_invocations']!=50 or not guide['no_best_of_K'] or guide['post_sampler_correction']:
        raise ValueError('sampling call budget or one-path contract mismatch')
    for item in guide['trace']:
        r=rank(m,nodes,float(item['PET']))['p_mid']
        errors['guidance_rank']=max(errors['guidance_rank'],abs(r-item['achieved_P']),abs(abs(r-requested)-item['P_error']),abs(target-item['target_PET']))
    if max(errors.values())>2e-10:raise ValueError('saved numeric values disagree: '+str(errors))
    return dict(scene_id=row['scene_id'],recording_id=row['recording_id'],stratum=row['stratum'],old12=row['old12'],
        noise_index=row['noise_index'],requested_p=requested,arm=row['arm'],N=n,PET=pet,achieved_midrank=achieved['p_mid'],
        P_error=point,interval_error=interval,Fine=point<=.05,scalar_error_infimum=floor,
        scalar_Fine_impossible=floor>.05,control_target=target,canonical_target=canonical,
        control_PET_error=abs(pet-target),canonical_PET_error=abs(pet-canonical),
        native_overlap=quality['all_pair_overlap_scene'],background_PL_overlap=bg,ego_PL_overlap=eg,
        road_outside=quality['road_outside_scene'],negative_vx=negative,
        short_BG_gap=quality['background_pairs_long_gap_under1m_frame_fraction']>0,
        whole_window_critical_actor=measured['critical_other_index'],whole_window_witness=measured['witness'],
        quality=quality,trajectory_artifact=row['trajectory_artifact'],array_key=row['array_key'],maximum_errors=errors)


def own_summary(rows):
    if not rows:raise ValueError('nonempty declared subgroup required')
    count=len(rows);fine=sum(r['Fine'] for r in rows)
    return dict(requests=count,histories=len({r['scene_id'] for r in rows}),Fine_count=fine,Fine_at_0_05=fine/count,
        P_MAE=sum(r['P_error'] for r in rows)/count,
        PET_control_target_MAE_seconds=sum(r['control_PET_error'] for r in rows)/count,
        canonical_PET_target_MAE_seconds=sum(r['canonical_PET_error'] for r in rows)/count,
        scalar_Fine_impossible_requests=sum(r['scalar_Fine_impossible'] for r in rows),
        native_overlap_requests=sum(r['native_overlap'] for r in rows),background_PL_overlap_requests=sum(r['background_PL_overlap'] for r in rows),
        ego_PL_overlap_requests=sum(r['ego_PL_overlap'] for r in rows),road_outside_requests=sum(r['road_outside'] for r in rows),
        negative_vx_requests=sum(r['negative_vx'] for r in rows),short_BG_gap_under1m_requests=sum(r['short_BG_gap'] for r in rows),
        joint_Fine_count=sum(r['Fine'] and not r['native_overlap'] and not r['road_outside'] for r in rows))


def compare_summary(actual,reported):
    for key,value in actual.items():
        if key not in reported or abs(value-reported[key])>2e-10:raise ValueError('summary mismatch: '+key)


def training_audit(freeze,pair_policy):
    results={arm:c.json_file(b) for arm,b in freeze['training_results'].items()}
    common=c.json_file(pair_policy['adaptation']);cp=torch.load(io.verify_binding(common['checkpoint']),map_location='cpu',weights_only=False)
    model=common_train.make_model(cp['architecture'],cp['state_dict'],torch.device('cpu'));initial=state_hash(model);del model
    if initial!=common['selected_state_sha256']:raise ValueError('actual common weights disagree')
    path=io.verify_binding(common['checkpoint']).parent/'epochs.jsonl'
    logs=[json.loads(line) for line in path.read_text().splitlines()]
    choices=[(common['initial_validation']['mean'],0)]+[(r['STOP']['mean'],r['epoch']) for r in logs]
    if (len(logs)!=common['epochs_completed'] or min(choices)[1]!=common['best_epoch']
            or any(r['FIT_rows']!=9913 for r in logs) or common['optimizer_updates']!=78*len(logs)):
        raise ValueError('common adaptation selection/exposure failed')
    actual_states={};traces={}
    for arm,result in results.items():
        if result['status']!='complete' or result['smoke'] or result['epochs_completed']!=3 or result['initial_state_sha256']!=initial:
            raise ValueError('incomplete or unpaired actual training')
        traces[arm]=[json.loads(line) for line in io.verify_binding(result['epochs']).read_text().splitlines()]
        if len(traces[arm])!=3:raise ValueError('missing epoch logs')
        for e,row in enumerate(traces[arm],1):
            if (row['epoch']!=e or row['base_loss_scenes']!=9913 or row['updates']!=78 or row['auxiliary_requests']!=3744
                    or row['forward_NFE']!=7800 or row['backward_NFE']!=7800):raise ValueError('incomplete full-DDIM exposure')
        if not same_binding(result['selected_checkpoint'],result['snapshots']['3']['raw']):raise ValueError('checkpoint selected after generation')
        cp=torch.load(io.verify_binding(result['selected_checkpoint']),map_location='cpu',weights_only=False)
        if not same_binding(cp['reference_manifest'],freeze['reference_manifest']) or cp['arm']!=arm:raise ValueError('checkpoint reference mismatch')
        model=paired_train.model_from_checkpoint(cp,arm,pair_policy['target_policy'],torch.device('cpu'))
        actual_states[arm]=state_hash(model);del model
        if actual_states[arm]!=result['final_state_sha256'] or actual_states[arm]==initial:raise ValueError('actual final parameter replay failed')
    for a,b in zip(traces['canonical'],traces['atom_aware']):
        for key in ('randomness','order_sha256','targets_sha256','p_dropped','auxiliary_histories','present_histories'):
            if a[key]!=b[key]:raise ValueError('unmatched paired streams: '+key)
    return dict(initial_state=initial,final_states=actual_states,common_adaptation_epochs=len(logs),
        terminal_epochs_per_arm=3,updates_per_arm=234,natural_exposures_per_arm=29739,auxiliary_requests_per_arm=11232,
        full50step_forward_backward=True)


def run(root):
    torch.set_num_threads(2)
    freeze_b=c.bind(root/'freeze_before_generation.json');freeze=c.json_file(freeze_b);p=c.json_file(freeze['policy'])
    q=c.json_file(freeze['queue']);pair=c.json_file(p['paired_training_policy']);seen=set();verify_nested(freeze,seen)
    c.verify_sources(freeze['code_sha256']);states=training_audit(freeze,pair)
    cases={r['scene_id']:r for r in q['cases']};all_results={};bindings={};errors={};outrows=[]
    expected={(sid,z,requested) for sid in cases for z in p['noise_indices'] for requested in p['P_grid']}
    if len(expected)!=1440 or len(cases)!=96:raise ValueError('wrong declared primary denominator')
    case_arrays={sid:c.arrays(row['artifact']) for sid,row in cases.items()}
    for arm in p['arms']:
        binding=c.bind(root/arm/'results.json');r=c.json_file(binding);bindings[arm]=binding
        if r['status']!='complete' or r['arm']!=arm or not same_binding(r['freeze'],freeze_b):raise ValueError('completed matching generation required')
        verify_nested(r,seen)
        if (len(r['rows'])!=1440 or {(x['scene_id'],x['noise_index'],x['requested_p']) for x in r['rows']}!=expected
                or len(r['batches'])!=288):raise ValueError('missing/duplicate primary requests')
        sealed=[]
        for b in r['batches']:sealed.extend(c.json_file(b)['rows'])
        if sealed!=r['rows']:raise ValueError('aggregate does not match sealed five-request batches')
        saved={};replayed=[]
        for i,row in enumerate(r['rows']):
            item=cases[row['scene_id']];a=case_arrays[row['scene_id']]
            if not same_binding(row['input_case'],item['artifact']) or not same_binding(row['reference_manifest'],freeze['reference_manifest']):raise ValueError('input/reference version drift')
            if row['noise_sha256']!=hashlib.sha256(a[f'initial_noise_{row["noise_index"]}'].tobytes()).hexdigest():raise ValueError('paired noise changed')
            path=row['trajectory_artifact']['path']
            if path not in saved:saved={path:c.arrays(row['trajectory_artifact'])}
            fresh=replay_row(row,a,saved[path][row['array_key']]);replayed.append(fresh)
            for name,value in fresh['maximum_errors'].items():errors[name]=max(errors.get(name,0.),value)
            if (i+1)%120==0:print(json.dumps(dict(arm=arm,trajectories_replayed=i+1)),flush=True)
        total=own_summary(replayed);compare_summary(total,r['summary'])
        for name,values,key in [('by_P',p['P_grid'],'requested_p'),('by_N',sorted({x['stratum'] for x in replayed}),'stratum')]:
            for value in values:compare_summary(own_summary([x for x in replayed if x[key]==value]),r['summary'][name][str(value)])
        compare_summary(own_summary([x for x in replayed if x['old12'] and x['noise_index']==0 and x['requested_p'] in (.1,.5,.9)]),r['summary']['old36'])
        compare_summary(own_summary([x for x in replayed if not x['old12']]),r['summary']['additional84'])
        all_results[arm]=total;outrows.extend(replayed)
    c.verify_sources(freeze['code_sha256'])
    report=dict(status='pass',freeze=freeze_b,generation_results=bindings,training=states,histories=96,requests_per_arm=1440,
        total_trajectories=2880,maximum_errors=errors,summaries=all_results,rows=outrows,artifact_bindings_verified=len(seen),
        all_requested_P_and_failed_requests_retained=True,same_H_P_z_CDF_across_arms=True,
        risk_and_quality_arithmetic_independent_of_generator_evaluator=True,PET_same_declared_oracle_not_independent_definition=True,
        PL_geometry_not_smooth_spline_certification=True,protected_data_access=False,development_not_blind=True,
        code_sha256=c.source_bindings(('pcontrol/research/audit_guarded_generation.py','pcontrol/research/audit_time_attention_full_pipeline.py',
            'pcontrol/research/audit_pair_overlap_intervals.py','pcontrol/data/scene_pet.py')))
    io.write_json(root/'independent_numeric_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','code_sha256')}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'outputs/natural_percentile/transformer_publication_v1_20260916/guarded_generation_eval_v1')
    run(parser.parse_args().root)
