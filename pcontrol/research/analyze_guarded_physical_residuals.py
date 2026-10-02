#!/usr/bin/env python3
"""Separate inherited t0 violations from per-side generated residuals.

Diagnostic tolerances never replace strict original request flags or remove
requests. No output trajectory, metric definition or sampler is modified.
"""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.research import train_natural_scene_reference as io


def residuals(future,dimensions,bounds):
    f=np.asarray(future,dtype=np.float64);d=np.asarray(dimensions,dtype=np.float64);b=np.asarray(bounds,dtype=np.float64)
    if f.shape!=(175,len(d),4) or not np.isfinite(f).all():raise ValueError('full finite native trajectory required')
    left=np.maximum(b[0]-(f[...,1]-d[None,:,1]/2),0.)
    right=np.maximum(f[...,1]+d[None,:,1]/2-b[-1],0.)
    reverse=np.maximum(-f[...,2],0.)
    extra_road=np.maximum(np.maximum(left-left[0],0.),np.maximum(right-right[0],0.))
    extra_reverse=np.maximum(reverse-reverse[0],0.)
    return dict(initial_road_outside=bool(np.maximum(left[0],right[0]).max()>0),
        initial_negative_vx=bool(reverse[0].max()>0),initial_road_depth_m=float(np.maximum(left[0],right[0]).max()),
        initial_minimum_vx_mps=float(f[0,:,2].min()),maximum_raw_road_depth_m=float(np.maximum(left,right).max()),
        minimum_vx_mps=float(f[...,2].min()),maximum_per_side_road_excess_m=float(extra_road.max()),
        maximum_per_actor_reverse_excess_mps=float(extra_reverse.max()),
        raw_road_outside=bool(np.maximum(left,right).max()>0),raw_negative_vx=bool(reverse.max()>0))


def run():
    root=ROOT/'outputs/natural_percentile/transformer_publication_v1_20260916/guarded_generation_eval_v1'
    ab=c.bind(root/'independent_numeric_audit.json');audit=c.json_file(ab)
    if audit['status']!='pass':raise ValueError('complete generated trajectories must already pass numerical audit')
    freeze=c.json_file(audit['freeze']);queue=c.json_file(freeze['queue']);inputs={q['scene_id']:c.arrays(q['artifact']) for q in queue['cases']}
    thresholds=(0.,1e-8,1e-6,.001);output={};details=[]
    for arm,binding in audit['generation_results'].items():
        result=c.json_file(binding);rows=[];loaded={}
        for row in result['rows']:
            sid=row['scene_id'];case=inputs[sid];path=row['trajectory_artifact']['path']
            if path not in loaded:loaded={path:c.arrays(row['trajectory_artifact'])}
            f=loaded[path][row['array_key']]
            if not np.array_equal(f[0],case['history'][-1]):raise ValueError('fixed initial state drift')
            values=residuals(f,case['dimensions'],case['road_boundaries'])
            if values['raw_road_outside']!=row['quality']['road_outside_scene'] or values['raw_negative_vx']!=row['quality']['negative_vx_scene']:
                raise ValueError('strict original quality flags changed')
            entry=dict(arm=arm,scene_id=sid,P=row['requested_p'],noise_index=row['noise_index'],N=row['num_agents'],**values,
                background_short_gap_span_seconds=row['scene_quality']['background_pairs_long_gap_under1m_longest_sample_span_seconds'])
            rows.append(entry);details.append(entry)
        output[arm]=dict(requests=len(rows),strict_raw_road_requests=sum(r['raw_road_outside'] for r in rows),
            strict_raw_negative_vx_requests=sum(r['raw_negative_vx'] for r in rows),
            inherited_initial_road_histories=len({r['scene_id'] for r in rows if r['initial_road_outside']}),
            inherited_initial_road_requests=sum(r['initial_road_outside'] for r in rows),
            inherited_initial_negative_histories=len({r['scene_id'] for r in rows if r['initial_negative_vx']}),
            inherited_initial_negative_requests=sum(r['initial_negative_vx'] for r in rows),
            strict_road_on_initially_compliant_histories=sum(r['raw_road_outside'] and not r['initial_road_outside'] for r in rows),
            strict_negative_on_initially_compliant_histories=sum(r['raw_negative_vx'] and not r['initial_negative_vx'] for r in rows),
            maximum_raw_road_depth_m=max(r['maximum_raw_road_depth_m'] for r in rows),minimum_vx_mps=min(r['minimum_vx_mps'] for r in rows),
            maximum_per_side_road_excess_m=max(r['maximum_per_side_road_excess_m'] for r in rows),
            maximum_per_actor_reverse_excess_mps=max(r['maximum_per_actor_reverse_excess_mps'] for r in rows),
            background_short_gap_histories=len({r['scene_id'] for r in rows if r['background_short_gap_span_seconds']>0}),
            maximum_background_short_gap_span_seconds=max(r['background_short_gap_span_seconds'] for r in rows),
            tolerance_diagnostics={format(t,'.0e'):dict(road_excess_requests=sum(r['maximum_per_side_road_excess_m']>t for r in rows),
                reverse_excess_requests=sum(r['maximum_per_actor_reverse_excess_mps']>t for r in rows)) for t in thresholds})
        print(json.dumps(dict(arm=arm,summary=output[arm])),flush=True)
    report=dict(status='complete',source_audit=ab,arms=output,rows=details,
        initial_allowances_are_per_actor_and_per_road_side=True,strict_primary_quality_flags_unchanged=True,
        no_requests_excluded=True,diagnostic_tolerances_not_used_to_replace_primary_metrics=True,
        source_trajectories_unchanged=True,protected_data_access=False,code_sha256=io.sha256(__file__))
    io.write_json(root/'physical_residuals.json',report)


if __name__=='__main__':run()
