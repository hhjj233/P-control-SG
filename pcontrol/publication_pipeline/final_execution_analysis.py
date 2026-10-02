"""Whole-scene natural context and failure-aware paired final inference."""
import numpy as np
from pcontrol.publication_pipeline.final_validation_protocol import verified_json,failure_aware_precision
from pcontrol.publication_pipeline.final_execution_generation import request_key
from pcontrol.publication_pipeline.final_scene_data import write_once
from pcontrol.generation.evaluation import quality_metrics
from pcontrol.generation.scene_quality_diagnostics import scene_quality
from pcontrol.time_attention_pipeline import common as c


def paired_fine_with_failures(left,right,*,repetitions=10000,seed=20260916):
    a={request_key(r):r for r in left};b={request_key(r):r for r in right}
    if len(a)!=len(left) or len(b)!=len(right) or set(a)!=set(b):raise ValueError('complete paired request sets required')
    records=sorted({r['recording_id'] for r in left});deltas=[];counts=[]
    for rec in records:
        values=[]
        for key in sorted(a):
            x,y=a[key],b[key]
            if x['recording_id']!=y['recording_id'] or x['noise_sha256']!=y['noise_sha256']:raise ValueError('unpaired request provenance')
            if x['recording_id']!=rec:continue
            hit=lambda r:r['status']=='complete' and r['p_mid_absolute_error']<=.05
            values.append(float(hit(y))-float(hit(x)))
        deltas.append(float(np.mean(values)));counts.append(len(values))
    if not counts:return dict(requests=0,Fine_delta=None,bootstrap_95pct=None)
    means=np.asarray(deltas);weights=np.asarray(counts)
    result=dict(requests=len(left),recordings=len(records),Fine_delta=float(np.average(means,weights=weights)),
        Fine_delta_recording_macro=float(means.mean()),bootstrap_95pct=None,macro_bootstrap_95pct=None,
        failed_request_is_Fine_zero=True,MAE_difference_bounds=None)
    if len(records)>=2:
        rng=np.random.default_rng(seed);indices=rng.integers(0,len(records),(repetitions,len(records)))
        boot=(means[indices]*weights[indices]).sum(1)/weights[indices].sum(1)
        result.update(bootstrap_95pct=np.quantile(boot,[.025,.975]).tolist(),
            macro_bootstrap_95pct=np.quantile(means[indices].mean(1),[.025,.975]).tolist())
    def precision(rows):
        return failure_aware_precision([dict(requested_p=r['requested_p'],absolute_error=r['p_mid_absolute_error']
            if r['status']=='complete' else None,failure_reason=r.get('failure_reason')) for r in rows])
    lower,upper=(precision(rows)['P_MAE_bounds'] for rows in (left,right))
    result['MAE_difference_bounds']=[upper[0]-lower[1],upper[1]-lower[0]]
    result['MAE_incomplete_no_point_estimate']=any(r['status']!='complete' for r in left+right)
    return result


def supplement_analysis(queue_binding,result_bindings,analysis_binding,output):
    queue=verified_json(queue_binding);analysis=verified_json(analysis_binding)
    if analysis['status']!='complete' or analysis['queue']!=queue_binding:raise ValueError('verified complete analysis required')
    contract=verified_json(queue['contract']);policy=verified_json(contract['policy']);observations=[]
    for item in queue['cases']:
        case=c.arrays(item['artifact']);future=case['future_observed']
        quality=quality_metrics(future,dict(case,future=future,future_dt=.04))
        scene=scene_quality(future,case['dimensions'],case['ego_mask'])
        observations.append(dict(scene_id=item['scene_id'],recording_id=item['recording_id'],num_agents=item['num_agents'],
            PET=float(case['natural_PET']),quality=quality,scene_quality=scene))
    def averages(key):
        if not observations:return {}
        fields=set.intersection(*(set(r[key]) for r in observations))
        return {k:float(np.mean([r[key][k] for r in observations])) for k in fields
            if all(isinstance(r[key][k],(int,float)) and not isinstance(r[key][k],bool) for r in observations)}
    left=verified_json(result_bindings['canonical'])['rows'];right=verified_json(result_bindings['atom_aware'])['rows']
    response={}
    for arm,summary in analysis['arms'].items():
        value=summary['response'];n=value['total_sweeps'];missing=value['unavailable_sweeps']
        response[arm]={name:None if n==0 else [value[key]/n,(value[key]+missing)/n]
            for name,key in [('nondecreasing_rate_bounds','nondecreasing_known'),('strict_increase_rate_bounds','strictly_increasing_known')]}
    return write_once(output,dict(status='complete',queue=queue_binding,source_analysis=analysis_binding,
        unique_natural_futures=dict(histories=len(observations),rows=observations,quality_mean=averages('quality'),
            scene_quality_mean=averages('scene_quality'),not_matched_risk_counterfactual_targets=True),
        failure_aware_paired=paired_fine_with_failures(left,right,repetitions=policy['uncertainty']['bootstrap_repetitions'],seed=policy['uncertainty']['seed']),
        response_bounds=response,no_requests_filtered=True,software_rehearsal=queue['software_rehearsal']))
