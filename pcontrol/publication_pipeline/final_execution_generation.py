"""All-request generation, independent replay and failure-aware final summaries."""
from collections import Counter,defaultdict
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from pcontrol.publication_pipeline.final_validation_protocol import (
    verified_json,bind,same_binding,failure_aware_precision)
from pcontrol.publication_pipeline.final_reference_suite import FEATURES
from pcontrol.publication_pipeline.final_generation_adapter import FrozenFinalGenerator,attach_observation_diagnostics
from pcontrol.publication_pipeline.final_execution_data import save_arrays
from pcontrol.publication_pipeline.final_scene_data import write_once
from pcontrol.publication_pipeline.generation_diagnostics import (
    risk_response,pairwise_noise_diversity,failure_label,paired_recording_summary)
from pcontrol.time_attention_pipeline import common as c
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY
from pcontrol.generation.atom_aware_rank_target import select_midrank_target
from pcontrol.research.audit_guarded_generation import replay_row
from pcontrol.research.analyze_guarded_physical_residuals import residuals


def request_key(row):return row['scene_id'],row['noise_index'],row['requested_p']


def expected_keys(queue,policy):
    g=policy['generation']
    return {(r['scene_id'],z,p) for r in queue['cases'] for z in g['noise_indices'] for p in g['P_grid']}


def precision(rows):
    return failure_aware_precision([dict(requested_p=r['requested_p'],
        absolute_error=r['p_mid_absolute_error'] if r['status']=='complete' else None,
        failure_reason=r.get('failure_reason')) for r in rows])


def binary_count_bounds(known_count,total,missing):
    return dict(known_count=int(known_count),unknown_requests=int(missing),denominator=int(total),
        rate_bounds=None if total==0 else [known_count/total,(known_count+missing)/total])


def summarize_requests(rows):
    out=precision(rows);good=[r for r in rows if r['status']=='complete'];missing=len(rows)-len(good)
    out['histories']=len({r['scene_id'] for r in rows})
    out['unavailable_quality_requests']=missing
    fields={
        'native_overlap':lambda r:r['quality']['all_pair_overlap_scene'],
        'road_outside':lambda r:r['quality']['road_outside_scene'],
        'negative_vx':lambda r:r['quality']['negative_vx_scene'],
        'background_PL_overlap':lambda r:r['background_PL_overlap_scene'],
        'ego_PL_overlap':lambda r:r['ego_PL_overlap_scene'],
        'background_gap_under_1m':lambda r:r['scene_quality']['background_pairs_long_gap_under1m_frame_fraction']>0}
    out['binary_quality']={key:binary_count_bounds(sum(fn(r) for r in good),len(rows),missing) for key,fn in fields.items()}
    out['joint_Fine_count']=sum(r['p_mid_absolute_error']<=.05 and not r['quality']['road_outside_scene']
        and not r['quality']['all_pair_overlap_scene'] for r in good)
    out['scalar_Fine_impossible_known']=sum(r['scalar_error_infimum']>.05 for r in rows if 'scalar_error_infimum' in r)
    out['scalar_floor_unavailable_requests']=sum('scalar_error_infimum' not in r for r in rows)
    out['physical_target_MAE_seconds']=None if missing or not good else float(np.mean([r['PET_control_target_absolute_error_seconds'] for r in good]))
    out['canonical_target_MAE_seconds']=None if missing or not good else float(np.mean([r['canonical_PET_target_absolute_error_seconds'] for r in good]))
    return out


def run_generation(queue_binding,contract_binding,output_dir,arm):
    contract=verified_json(contract_binding);policy=verified_json(contract['policy']);queue=verified_json(queue_binding)
    if not same_binding(queue['contract'],contract_binding):raise ValueError('queue contract drift')
    if arm not in policy['generation']['arms']:raise ValueError('unknown generator arm')
    root=Path(output_dir)/arm;root.mkdir(parents=True,exist_ok=True)
    final=root/'result.json';expected=expected_keys(queue,policy)
    if final.exists():
        previous=verified_json(bind(final))
        if (not same_binding(previous['queue'],queue_binding) or not same_binding(previous['contract'],contract_binding)
                or {request_key(r) for r in previous['rows']}!=expected or len(previous['rows'])!=len(expected)):
            raise ValueError('completed generation cannot be reused under changed inputs')
        for b in previous['batches']:
            batch=verified_json(b);c.arrays(batch['trajectories'])
        return bind(final)
    engine=FrozenFinalGenerator(contract_binding,arm);rows=[];batches=[]
    for item in queue['cases']:
        arrays=c.arrays(item['artifact']);features={k:arrays[k] for k in FEATURES};prepared=engine.prepare_case(features)
        ref=prepared['reference']
        if (not np.array_equal(ref._masses[0],arrays['reference_joint_masses'])
                or not np.array_equal(prepared['pieces'][0].numpy(),arrays[PIECES_KEY])):
            raise ValueError('CDF queue replay drift')
        for z in policy['generation']['noise_indices']:
            stem=f'case_{item["case_index"]:03d}_z{z}';record=root/(stem+'.json');claim=root/(stem+'.claim.json')
            if record.exists():
                b=bind(record);saved=verified_json(b)
                if (not same_binding(saved['queue'],queue_binding) or not same_binding(saved['case'],item['artifact'])
                        or saved['arm']!=arm or saved['noise_index']!=z or len(saved['rows'])!=5):
                    raise ValueError('invalid committed generation batch')
                c.arrays(saved['trajectories']);rows.extend(saved['rows']);batches.append(b);continue
            if claim.exists() or (root/(stem+'.npz')).exists():
                raise RuntimeError('uncommitted interrupted batch; no automatic resampling')
            write_once(claim,dict(queue=queue_binding,contract=contract_binding,case=item['artifact'],arm=arm,noise_index=z))
            noise=arrays[f'initial_noise_{z}'];generated={};current=[]
            for p in policy['generation']['P_grid']:
                key=f'generated_p{p:g}'.replace('.','_')
                scalar=select_midrank_target(arrays[PIECES_KEY],p,**engine.training_policy['target_policy'])
                base={k:item[k] for k in ('scene_id','recording_id','role','num_agents','stratum','case_index','old12')}
                base.update(arm=arm,noise_index=z,requested_p=p,K=1,all_actors_retained=True,post_sampler_repair=False,
                    inference_observed_future_input=False,input_case=item['artifact'],
                    noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),
                    scalar_error_infimum=scalar['unrestricted_scalar_error_infimum'])
                start=time.monotonic()
                try:
                    future,value=engine.sample(prepared,p,noise)
                    value=attach_observation_diagnostics(value,future,features,arrays['future_observed'])
                    generated[key]=future;row=dict(base,**value,array_key=key)
                except (RuntimeError,ValueError,FloatingPointError) as error:
                    row=dict(base,status='failed',failure_reason=type(error).__name__+': '+str(error),
                        sampling_seconds=time.monotonic()-start,network_evaluations=None,
                        all_actors_retained=None,full_roster_requested=True)
                current.append(row)
                print(json.dumps(dict(stage='generation_request',scope=queue['scope'],arm=arm,case=item['case_index'],z=z,P=p,status=row['status'])),flush=True)
            trajectory=save_arrays(root/(stem+'.npz'),generated)
            for row in current:
                if row['status']=='complete':row['trajectory_artifact']=trajectory
            b=write_once(record,dict(queue=queue_binding,contract=contract_binding,case=item['artifact'],arm=arm,
                noise_index=z,rows=current,trajectories=trajectory))
            rows.extend(current);batches.append(b)
    engine.assert_unchanged()
    if len(rows)!=len(expected) or {request_key(r) for r in rows}!=expected:raise ValueError('missing/duplicate final requests')
    return write_once(final,dict(status='complete',arm=arm,contract=contract_binding,queue=queue_binding,scope=queue['scope'],
        rows=rows,batches=batches,summary=summarize_requests(rows),
        by_P={str(p):summarize_requests([r for r in rows if r['requested_p']==p]) for p in policy['generation']['P_grid']},
        by_N={name:summarize_requests([r for r in rows if r['stratum']==name]) for name,_,_ in policy['generation']['count_groups']},
        software_rehearsal=queue['software_rehearsal'],all_requested_rows_retained=True))


def audit_generation(queue_binding,contract_binding,result_bindings,output):
    queue=verified_json(queue_binding);contract=verified_json(contract_binding);policy=verified_json(contract['policy'])
    expected=expected_keys(queue,policy);cases={r['scene_id']:r for r in queue['cases']};details=[];maximum=0.;results={}
    for arm,binding in result_bindings.items():
        result=verified_json(binding);results[arm]=result
        if (result['arm']!=arm or not same_binding(result['queue'],queue_binding)
                or not same_binding(result['contract'],contract_binding)):
            raise ValueError('generation result lineage mismatch')
        if len(result['rows'])!=len(expected) or {request_key(r) for r in result['rows']}!=expected:raise ValueError('request denominator changed')
        if summarize_requests(result['rows'])!=result['summary']:raise ValueError('failure-aware summary does not replay')
        cache={};loaded={}
        for row in result['rows']:
            item=cases[row['scene_id']];sid=row['scene_id']
            if sid not in cache:cache={sid:c.arrays(item['artifact'])}
            case=cache[sid];noise=case[f'initial_noise_{row["noise_index"]}']
            if hashlib.sha256(noise.tobytes()).hexdigest()!=row['noise_sha256']:raise ValueError('wrong frozen noise')
            if row['status']=='failed':
                if not row.get('failure_reason') or 'trajectory_artifact' in row or 'p_mid_absolute_error' in row:
                    raise ValueError('failed request hidden or assigned a fabricated error')
                details.append(dict(arm=arm,scene_id=sid,status='failed',P=row['requested_p'],noise_index=row['noise_index']));continue
            path=row['trajectory_artifact']['path']
            if path not in loaded:loaded={path:c.arrays(row['trajectory_artifact'])}
            future=loaded[path][row['array_key']];own=replay_row(row,case,future)
            maximum=max(maximum,max(own['maximum_errors'].values()))
            physical=residuals(future,case['dimensions'],case['road_boundaries'])
            details.append(dict(arm=arm,scene_id=sid,status='complete',P=row['requested_p'],noise_index=row['noise_index'],
                category=failure_label(row),physical=physical,numeric_max=max(own['maximum_errors'].values())))
    if set(results)!=set(policy['generation']['arms']):raise ValueError('both paired arms required')
    left={request_key(r):r for r in results['canonical']['rows']};right={request_key(r):r for r in results['atom_aware']['rows']}
    for key,a in left.items():
        b=right[key]
        if a['noise_sha256']!=b['noise_sha256'] or a['input_case']!=b['input_case']:raise ValueError('unpaired H/P/z')
    report=dict(status='pass',contract=contract_binding,queue=queue_binding,results=result_bindings,
        expected_per_arm=len(expected),replayed_complete=sum(d['status']=='complete' for d in details),
        retained_failures=sum(d['status']=='failed' for d in details),maximum_numeric_error=maximum,
        details=details,all_requests_retained=True,PET_uses_same_declared_oracle=True,
        independent_rank_and_quality_arithmetic=True,software_rehearsal=queue['software_rehearsal'])
    return write_once(output,report)


def describe_generation(queue_binding,result_bindings,audit_binding,output):
    queue=verified_json(queue_binding);contract=verified_json(queue['contract']);policy=verified_json(contract['policy'])
    audit=verified_json(audit_binding)
    if audit['status']!='pass' or audit['results']!=result_bindings:raise ValueError('complete independent audit required')
    summaries={};all_rows={}
    for arm,binding in result_bindings.items():
        result=verified_json(binding);rows=result['rows'];all_rows[arm]=rows;good=[r for r in rows if r['status']=='complete']
        groups=defaultdict(list)
        for row in rows:groups[row['scene_id'],row['noise_index']].append(row)
        complete_sweeps=[v for v in groups.values() if len(v)==5 and all(r['status']=='complete' for r in v)]
        response=[]
        for sweep in complete_sweeps:
            response.append(risk_response(sweep,policy['generation']['P_grid'],[sweep[0]['noise_index']]))
        curves=dict(total_sweeps=len(groups),complete_sweeps=len(response),unavailable_sweeps=len(groups)-len(response),
            nondecreasing_known=sum(r['nondecreasing_sweeps'] for r in response),strictly_increasing_known=sum(r['strictly_increasing_sweeps'] for r in response),
            all_five_Fine=sum(r['all_P_Fine_sweeps'] for r in response))
        lookup={request_key(r):r for r in rows};diversity=[];missing_cells=0
        for item in queue['cases']:
            case=c.arrays(item['artifact'])
            for p in policy['generation']['P_grid']:
                selected=[lookup[item['scene_id'],z,p] for z in policy['generation']['noise_indices']]
                if not all(r['status']=='complete' for r in selected):missing_cells+=1;continue
                futures=np.stack([c.arrays(r['trajectory_artifact'])[r['array_key']] for r in selected])
                values=pairwise_noise_diversity(futures,case['ego_mask'])
                diversity.append(dict(scene_id=item['scene_id'],P=p,**values,
                    achieved_rank_std_across_noise=float(np.std([r['estimated_rank']['p_mid'] for r in selected]))))
        known=[d for d in audit['details'] if d['arm']==arm and d['status']=='complete']
        physical={key:max(d['physical'][key] for d in known) if known else None for key in
            ('maximum_per_side_road_excess_m','maximum_per_actor_reverse_excess_mps')}
        def mean_fields(key):
            fields=set.intersection(*(set(r[key]) for r in good)) if good else set()
            return {field:float(np.mean([r[key][field] for r in good])) for field in fields
                if all(isinstance(r[key][field],(int,float)) and not isinstance(r[key][field],bool) for r in good)}
        summaries[arm]=dict(primary=result['summary'],by_P=result['by_P'],by_N=result['by_N'],response=curves,
            noise_diversity=dict(total_H_P_cells=len(queue['cases'])*5,complete_cells=len(diversity),unavailable_cells=missing_cells,
                mean_available_only={k:float(np.mean([d[k] for d in diversity])) for k in ('all_pairwise_mean_displacement_m','ego_pairwise_mean_displacement_m','background_pairwise_mean_displacement_m','achieved_rank_std_across_noise')} if diversity else {},
                condition_on_Fine=False,not_a_realism_score=True),
            failure_taxonomy=dict(Counter(d.get('category','generation_failure') for d in audit['details'] if d['arm']==arm)),
            physical_residual_maximum_available_only=physical,quality_mean_available_only=mean_fields('quality'),
            scene_quality_mean_available_only=mean_fields('scene_quality'),
            timing=dict(total_seconds=sum(r['sampling_seconds'] for r in rows),
                median_seconds=float(np.median([r['sampling_seconds'] for r in rows])) if rows else None,
                p95_seconds=float(np.quantile([r['sampling_seconds'] for r in rows],.95)) if rows else None,
                known_network_evaluations=sum(r['network_evaluations'] for r in good),unknown_call_count_requests=len(rows)-len(good)))
    left=all_rows['canonical'];right=all_rows['atom_aware']
    if left and all(r['status']=='complete' for r in left+right) and len({r['recording_id'] for r in left})>=2:
        paired=paired_recording_summary(left,right,repetitions=policy['uncertainty']['bootstrap_repetitions'],seed=policy['uncertainty']['seed'])
    else:paired=dict(status='not_estimable_from_complete_multi_recording_pairs',reason='empty_cohort_single_recording_or_failed_output',all_requests_retained=True)
    return write_once(output,dict(status='complete',queue=queue_binding,audit=audit_binding,arms=summaries,
        paired=paired,software_rehearsal=queue['software_rehearsal'],no_Fine_based_filtering=True))
