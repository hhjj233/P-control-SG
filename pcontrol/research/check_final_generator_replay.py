#!/usr/bin/env python3
"""Compare the new generator adapter to two original P5 saved requests exactly."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.publication_pipeline.final_validation_protocol import bind,verified_json,resolve
from pcontrol.publication_pipeline.final_generation_adapter import (
    FrozenFinalGenerator,initialize_runtime,attach_observation_diagnostics)
from pcontrol.publication_pipeline.final_reference_suite import FEATURES
from pcontrol.time_attention_pipeline import common as c
from pcontrol.research.prepare_final_natural_validation import write_once
from pcontrol.research.audit_guarded_generation import replay_row


def run():
    initialize_runtime()
    root=ROOT/'outputs/natural_percentile/transformer_publication_v1_20260916/final_validation_v1_preparation'
    fb=bind(root/'frozen_contract.json');contract=verified_json(fb)
    p5=verified_json(contract['source_generation_freeze']);queue=verified_json(p5['queue'])
    source=queue['cases'][0];case=c.arrays(source['artifact'])
    if source['role']!='STOP':raise PermissionError('known open development history required')
    audited=verified_json(contract['source_generation_audit']);checks=[]
    for arm in ('canonical','atom_aware'):
        results=verified_json(audited['generation_results'][arm])
        old=next(r for r in results['rows'] if r['scene_id']==source['scene_id'] and r['noise_index']==0 and r['requested_p']==.5)
        plugin=FrozenFinalGenerator(fb,arm);features={k:case[k] for k in FEATURES}
        prepared=plugin.prepare_case(features);future,row=plugin.sample(prepared,.5,case['initial_noise_0'])
        row=attach_observation_diagnostics(row,future,features,case['future_observed'])
        previous=c.arrays(old['trajectory_artifact'])[old['array_key']]
        error=float(np.max(abs(future-previous)))
        if error!=0.:raise ValueError('P5 trajectory changed under final adapter: '+str(error))
        for key in ('pet_seconds','p_mid_absolute_error','control_target_PET_seconds','scalar_error_infimum'):
            if row[key]!=old[key]:raise ValueError('P5 score drift: '+key)
        replay=dict(old,**row);own=replay_row(replay,case,future)
        plugin.assert_unchanged()
        checks.append(dict(arm=arm,scene_id=source['scene_id'],requested_p=.5,noise_index=0,
            maximum_trajectory_error=error,maximum_numeric_replay_error=max(own['maximum_errors'].values()),
            original_trajectory=old['trajectory_artifact'],network_evaluations=row['network_evaluations']))
    output=write_once(root/'generator_adapter_replay.json',dict(status='pass_exact_P5_generator_replay',contract=fb,
        source_case=source['artifact'],checks=checks,protected_data_access=False,new_performance_claim=False,
        code={p:bind(ROOT/p) for p in ('pcontrol/research/check_final_generator_replay.py','pcontrol/publication_pipeline/final_generation_adapter.py')}))
    print(json.dumps(dict(status='pass',requests=len(checks),output=output)),flush=True)


if __name__=='__main__':run()
