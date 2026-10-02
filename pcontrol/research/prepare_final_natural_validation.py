#!/usr/bin/env python3
"""Freeze final-evaluation models/rules without opening any protected recordings.

The rehearsal uses only already authorized STOP identity metadata. This script
has no CSV reader and is not yet the final protected-data execution adapter.
"""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from pcontrol.publication_pipeline.final_validation_protocol import (
    PROTOCOL,REFERENCE_ARMS,bind,digest,resolve,verified_json,same_binding,
    validate_policy,select_histories,noise_for_history,require_approval)

CODE = ('pcontrol/research/prepare_final_natural_validation.py',
        'pcontrol/publication_pipeline/final_validation_protocol.py')
DEFAULT_POLICY = 'configs/natural_percentile/final_natural_validation_v1.json'


def verify_file(binding):
    if digest(binding['path']) != binding['sha256']: raise ValueError('binding drift: '+binding['path'])


def codes_unchanged(codes):
    for path,expected in codes.items():
        if digest(path) != expected: raise ValueError('frozen code drift: '+path)


def write_once(path, value):
    path = resolve(path)
    with path.open('x') as handle: json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False)
    return bind(path)


def freeze(policy_binding):
    p = validate_policy(verified_json(policy_binding));sources = p['sources']
    for binding in sources.values(): verify_file(binding)
    guarded = verified_json(sources['guarded_reference_results'])
    capacity = verified_json(sources['capacity_reference_results'])
    prep = verified_json(guarded['prepared']);cap_prep = verified_json(capacity['prepared'])
    prior = verified_json(prep['previous_validation'])
    prior_prep = verified_json(prior['prepared'])
    gen = verified_json(sources['paired_generation_freeze'])
    ga = verified_json(sources['paired_generation_audit'])
    main = verified_json(sources['main_reference'])
    if guarded['status'] != 'complete' or capacity['status'] != 'complete' or ga['status'] != 'pass':
        raise ValueError('completed and audited source branches required')
    if not same_binding(ga['freeze'],sources['paired_generation_freeze']): raise ValueError('generation audit/freeze mismatch')
    if not same_binding(gen['reference_manifest'],sources['main_reference']): raise ValueError('different generated-risk ruler')
    codes_unchanged(gen['code_sha256']);codes_unchanged(guarded['code_sha256']);codes_unchanged(capacity['code_sha256'])
    models = {}
    for arm in REFERENCE_ARMS:
        if arm == 'M2_WideMLP':
            entry = capacity['entries']['group_guarded']
            descriptor = copy.deepcopy(cap_prep['models'][entry['model_id']])
            # The capacity study stores these common bindings at prepared level.
            descriptor.update(normalizer=cap_prep['normalizer'],data=cap_prep['data'],arm=arm)
            evaluation_binding = entry['evaluation']
        else:
            entry = guarded['entries'][arm+'/group_guarded']
            descriptor = copy.deepcopy(prep['unique_models'][entry['model_id']])
            evaluation_binding = entry['evaluation']['result']
        evaluation = verified_json(evaluation_binding)
        choice_binding = evaluation['CAL_selection'];selection = verified_json(choice_binding)
        choice = selection['selections']['count_balanced']
        for name in ('checkpoint','normalizer','data'): verify_file(descriptor[name])
        verify_file(choice['calibration_model'])
        models[arm] = dict(descriptor=descriptor,calibration_model=choice['calibration_model'],
            calibration_selection=choice_binding,selected_family=choice['family'],selected_ridge=choice['ridge'],
            selected_before_final_data=True,source_evaluation=evaluation_binding)
    current = models['M2_TimeAttn']
    if (not same_binding(current['descriptor']['checkpoint'],main['checkpoint'])
            or not same_binding(current['calibration_model'],main['calibration_model'])):
        raise ValueError('primary reference changed during registry construction')
    for binding in gen['checkpoints'].values(): verify_file(binding)
    artifact = dict(protocol=PROTOCOL,status='model_and_analysis_contract_frozen_awaiting_authorization_and_adapters',
        policy=policy_binding,reference_models=models,main_reference=sources['main_reference'],
        generator_checkpoints=gen['checkpoints'],generation_profile=gen['profile'],
        source_generation_audit=sources['paired_generation_audit'],source_generation_freeze=sources['paired_generation_freeze'],
        cohorts=p['cohorts'],current_development_recordings=sorted(set().union(*(set(verified_json(sources['development_roles'])[k]) for k in ('FIT','STOP','CAL','AUDIT')))),
        source_prediction_prepared=prior_prep['source_prepared'],
        code_sha256={path:digest(path) for path in CODE},inherited_generator_code_sha256=gen['code_sha256'],
        execution_ready=False,execution_adapters_pending=['authorized_complete_scene_builder','all_seven_reference_evaluator','final_cohort_paired_generator_and_auditor'],
        authorization_received=False,protected_raw_or_cached_trajectories_read=False,
        raw_directory_stat_performed=False,model_training=False,CAL_fitting=False,
        final_performance_results_exist=False,prior_recording_use_audit_still_required=True)
    root = resolve(p['preparation_root']);root.mkdir(parents=True,exist_ok=False)
    output = write_once(root/'frozen_contract.json',artifact)
    print(json.dumps(dict(status=artifact['status'],reference_models=len(models),generator_models=len(gen['checkpoints']),
        cohorts={k:len(v) for k,v in p['cohorts'].items()},contract=output,protected_data_read=False)),flush=True)
    return output


def rehearse(policy_binding):
    p = validate_policy(verified_json(policy_binding));root = resolve(p['preparation_root'])
    fb = bind(root/'frozen_contract.json');f = verified_json(fb)
    if not same_binding(f['policy'],policy_binding): raise ValueError('different policy')
    codes_unchanged(f['code_sha256'])
    # This reader is restricted to STOP of the existing natural-development
    # manifest; no final CSV/NPZ path is formed or inspected.
    from pcontrol.generation.evaluation import _identity_rows
    from pcontrol.publication_pipeline.final_validation_protocol import IDENTITY_FIELDS
    roles = verified_json(p['sources']['development_roles']);rows = _identity_rows(p['sources']['development_data'],'STOP')
    projected = [{k:int(row[k]) if k=='num_agents' else row[k] for k in IDENTITY_FIELDS} for row in rows]
    selected,summary = select_histories(projected,allowed_recordings=roles['STOP'],role='STOP',
        per_group=p['generation']['history_count_per_stratum_per_cohort'],salt=p['generation']['selection_salt'])
    reverse,reverse_summary = select_histories(list(reversed(projected)),allowed_recordings=list(reversed(roles['STOP'])),
        role='STOP',per_group=32,salt=p['generation']['selection_salt'])
    if selected != reverse or summary != reverse_summary: raise ValueError('selection depends on input ordering')
    noise=[]
    for row in selected:
        for z in p['generation']['noise_indices']:
            value,evidence = noise_for_history(row['scene_id'],z,row['num_agents'],salt=p['generation']['noise_salt'])
            if value.shape != (row['num_agents'],8,2): raise ValueError('full roster noise shape mismatch')
            noise.append(dict(scene_id=row['scene_id'],noise_index=z,**evidence))
    outcome = dict(status='pass_development_selection_rehearsal_only',policy=policy_binding,contract=fb,
        source_data=p['sources']['development_data'],role='STOP',source_histories=len(projected),
        selected=selected,summary=summary,noise=noise,
        hypothetical_requests_per_arm=len(selected)*5*3,actual_generated_trajectories=0,
        no_new_performance_measurement=True,protected_data_read=False,
        input_order_invariance=True,only_identity_N_role_recording_passed_to_selector=True,
        code_sha256={path:digest(path) for path in CODE})
    output = write_once(root/'development_rehearsal.json',outcome)
    print(json.dumps(dict(status=outcome['status'],summary=summary,generated=0,output=output)),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['freeze','rehearse','check-approval'])
    parser.add_argument('--policy',default=DEFAULT_POLICY);parser.add_argument('--policy-sha256',required=True)
    parser.add_argument('--freeze');parser.add_argument('--freeze-sha256')
    parser.add_argument('--approval');parser.add_argument('--approval-sha256')
    parser.add_argument('--scope',choices=['VAL','TEST','R18'])
    args = parser.parse_args();pb = dict(path=str(resolve(args.policy)),sha256=args.policy_sha256)
    if args.command == 'freeze': freeze(pb)
    elif args.command == 'rehearse': rehearse(pb)
    else:
        ab = None if args.approval is None else dict(path=args.approval,sha256=args.approval_sha256)
        fb = dict(path=args.freeze or '',sha256=args.freeze_sha256 or '')
        try: grant = require_approval(pb,fb,ab,scope=args.scope)
        except PermissionError as exc:
            print(json.dumps(dict(status='blocked_before_protected_access',reason=str(exc),protected_data_read=False)),flush=True)
            raise SystemExit(2)
        print(json.dumps(dict(scope=grant.scope,recording_ids=grant.recording_ids,training_permitted=grant.permits_training)),flush=True)
