#!/usr/bin/env python3
"""Approved thread-only continuation; original frozen sources remain intact."""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
from pcontrol.research.orchestrate_frozen_final_validation_v1 import POLICY, FREEZE, APPROVAL, SCOPES
from pcontrol.research.run_final_natural_validation import authorized_context, run_stage
from pcontrol.publication_pipeline.final_validation_protocol import (
    bind, digest, same_binding, verified_json)

PROTOCOL = 'reference_query_thread_only_amendment_v1'
FIX = dict(reference_query_threads=1, generation_threads=2)
BASE = ROOT/'outputs/natural_percentile/transformer_publication_v1_20260916/final_validation_v1'
REPAIR = BASE/'thread_repair_v1'
PROBE = dict(path=str(BASE/'queue_reference_thread_repair_probe_v1.json'),
             sha256='0de3a06f675eaf2df7667d24251f63171dd3e1df7973b2011e19f646a0d72362')
DIAGNOSIS = dict(path=str(BASE/'queue_cdf_replay_diagnosis_v1.json'),
                 sha256='139182cb549533aa0d2094cefff52b7979b295ca8ccc593e1572428e371b61b2')
CODE = ('pcontrol/research/run_final_validation_thread_repair_v1.py',
        'scripts/orchestrate_final_validation_thread_repair_v1.py')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
    return bind(path)


def repair_permission(binding):
    if binding is None:
        raise PermissionError('explicit approval for thread repair required before protected access')
    r = verified_json(binding)
    if (r.get('protocol') != 'explicit_user_reference_thread_repair_permission_v1'
            or r.get('decision') != 'allow_reference_thread_repair_and_resume'
            or r.get('recorded_from_actual_user_reply') is not True
            or not r.get('verbatim_user_reply', '').strip()
            or not r.get('user_message_reference', '').strip()
            or r.get('approved_scopes') != list(SCOPES)
            or r.get('numerical_fix') != FIX
            or r.get('preserve_existing_outputs') is not True
            or r.get('repair_probe_sha256') != PROBE['sha256']):
        raise PermissionError('repair approval does not cover exactly the proposed continuation')
    for key in ('allow_training', 'allow_recalibration', 'allow_reselection',
                'allow_repeat_saved_requests', 'allow_relax_CDF_equality_check'):
        if r.get(key) is not False:
            raise PermissionError('forbidden repair authority: ' + key)
    if not same_binding(r['policy'], POLICY) or not same_binding(r['freeze'], FREEZE):
        raise PermissionError('approval belongs to a different original freeze')
    return r


def preserve_originals(contract):
    for filename, expected in contract['preserved_file_sha256'].items():
        if digest(filename) != expected:
            raise ValueError('original committed artifact changed: ' + filename)


def create_contract(approval):
    repair_permission(approval)
    for s in SCOPES:
        authorized_context(POLICY, FREEZE, APPROVAL, s)
    probe = verified_json(PROBE)
    diagnosis = verified_json(DIAGNOSIS)
    if (probe['status'] != 'pass_exact_reference_thread_repair_probe'
            or probe['histories'] != 262 or probe['generator_sample_calls'] != 0):
        raise ValueError('complete no-sampling repair proof required')
    queues = diagnosis['queue_bindings']
    planned = {}
    for s in SCOPES:
        q = verified_json(queues[s])
        if q['scope'] != s or q['software_rehearsal'] or not same_binding(q['contract'], FREEZE):
            raise ValueError('wrong original queue')
        planned[s] = q['requests_per_arm'] * 2
    if sum(planned.values()) != 7860:
        raise ValueError('original planned denominator changed')
    prior = []
    for f in sorted((BASE/'TEST/generation').glob('*/case_000_z[012].json')):
        prior.extend(verified_json(bind(f))['rows'])
    if len(prior) != 30 or len({(r['arm'], r['scene_id'], r['noise_index'], r['requested_p']) for r in prior}) != 30:
        raise ValueError('expected exactly 30 already committed unique requests')
    contract = dict(protocol=PROTOCOL, created_utc=now(), original_policy=POLICY,
                    original_freeze=FREEZE, original_unsealing_approval=APPROVAL,
                    repair_approval=approval, proof=PROBE, diagnosis=DIAGNOSIS,
                    numerical_fix=FIX, queues=queues, planned_requests=planned,
                    prior_saved_requests=30, remaining_requests=7830,
                    preserved_file_sha256=probe['committed_files_unchanged'],
                    code_sha256={p: digest(p) for p in CODE},
                    all_model_calibration_noise_targets_and_sampler_rules_unchanged=True)
    preserve_originals(contract)
    REPAIR.mkdir(parents=True, exist_ok=False)
    (REPAIR/'stages').mkdir()
    return write_once(REPAIR/'contract.json', contract)


def context(contract_binding, approval, scope):
    repair_permission(approval)
    _, _, _, root = authorized_context(POLICY, FREEZE, APPROVAL, scope)
    c = verified_json(contract_binding)
    if (c['protocol'] != PROTOCOL or c['numerical_fix'] != FIX
            or not same_binding(c['repair_approval'], approval)
            or not same_binding(c['original_freeze'], FREEZE)):
        raise ValueError('wrong repair contract')
    for p, expected in c['code_sha256'].items():
        if digest(p) != expected:
            raise ValueError('repair implementation changed: ' + p)
    if bind(root/'queue/queue.json') != c['queues'][scope]:
        raise ValueError('original queue changed')
    preserve_originals(c)
    return c, root


@contextmanager
def reference_thread_context(reference_class=None):
    """Single Python-thread processes only; never changes the sampler thread count."""
    import torch
    if threading.current_thread() is not threading.main_thread():
        raise RuntimeError('use separate processes, not concurrent Python reference threads')
    if reference_class is None:
        from pcontrol.publication_pipeline.final_reference_suite import FrozenFinalReference
        reference_class = FrozenFinalReference
    original = reference_class.condition_features
    stats = dict(reference_queries=0, reference_threads=1, restored_threads=2)

    def condition(self, features):
        previous = torch.get_num_threads()
        if previous != 2:
            raise RuntimeError('declared two-thread runtime must surround reference query')
        try:
            torch.set_num_threads(1)
            value = original(self, features)
            stats['reference_queries'] += 1
            return value
        finally:
            torch.set_num_threads(previous)

    reference_class.condition_features = condition
    try:
        yield stats
    finally:
        reference_class.condition_features = original


def preflight(contract_binding, approval):
    import numpy as np
    from pcontrol.publication_pipeline.final_reference_suite import FEATURES
    from pcontrol.publication_pipeline.final_generation_adapter import FrozenFinalGenerator, initialize_runtime
    from pcontrol.reference.torch_frozen_inverse import PIECES_KEY
    from pcontrol.generation.cdf_shape_context import CONTEXT_KEY
    c, _ = context(contract_binding, approval, 'TEST')
    initialize_runtime()
    engine = FrozenFinalGenerator(FREEZE, 'canonical')
    rows = []
    with reference_thread_context() as stats:
        for scope in SCOPES:
            context(contract_binding, approval, scope)
            for item in verified_json(c['queues'][scope])['cases']:
                b = item['artifact']
                if digest(b['path']) != b['sha256']:
                    raise ValueError('case artifact drift')
                with np.load(b['path'], allow_pickle=False) as z:
                    prepared = engine.prepare_case({k: z[k].copy() for k in FEATURES})
                    equal = (np.array_equal(prepared['reference']._masses[0], z['reference_joint_masses'])
                             and np.array_equal(prepared['pieces'][0].numpy(), z[PIECES_KEY])
                             and np.array_equal(prepared['features'][CONTEXT_KEY][0].numpy(), z[CONTEXT_KEY]))
                if not equal:
                    raise ValueError('exact frozen CDF replay failed: ' + item['scene_id'])
                rows.append(dict(scope=scope, scene_id=item['scene_id'], exact=True))
    engine.assert_unchanged()
    preserve_originals(c)
    if len(rows) != 262 or stats['reference_queries'] != 262:
        raise ValueError('incomplete repair preflight')
    return write_once(REPAIR/'preflight.json', dict(status='pass', contract=contract_binding,
                      repair_approval=approval, histories=len(rows), rows=rows, runtime=stats,
                      sample_calls=0, original_outputs_unchanged=True))


def run(command, scope, arm, contract_binding, approval):
    c, root = context(contract_binding, approval, scope)
    proof_binding = bind(REPAIR/'preflight.json')
    p = verified_json(proof_binding)
    if p['status'] != 'pass' or p['histories'] != 262 or not same_binding(p['contract'], contract_binding):
        raise ValueError('exact preflight for this repair implementation required')
    key = '_'.join(x for x in (scope, command, arm) if x)
    start = write_once(REPAIR/'stages'/(key+'_start.json'), dict(time_utc=now(), contract=contract_binding,
                       repair_approval=approval, preflight=proof_binding, original_queue=c['queues'][scope]))
    if command == 'generate':
        with reference_thread_context() as stats:
            result = run_stage(command, POLICY, FREEZE, APPROVAL, scope, arm=arm)
    elif command == 'audit':
        stats = dict(reference_queries=0, original_audit_unchanged=True)
        result = run_stage(command, POLICY, FREEZE, APPROVAL, scope)
    else:
        raise ValueError('only original generation and audit stages may be continued')
    preserve_originals(c)
    receipt = write_once(REPAIR/'stages'/(key+'_complete.json'), dict(time_utc=now(), start=start,
                         contract=contract_binding, repair_approval=approval, result=result,
                         runtime=stats, original_saved_files_preserved=True))
    return dict(result=result, execution_amendment=receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('freeze-repair', 'preflight', 'generate', 'audit'))
    parser.add_argument('--repair-approval')
    parser.add_argument('--repair-approval-sha256')
    parser.add_argument('--contract')
    parser.add_argument('--contract-sha256')
    parser.add_argument('--scope', choices=SCOPES)
    parser.add_argument('--arm', choices=('canonical', 'atom_aware'))
    args = parser.parse_args()
    approval = None if args.repair_approval is None else dict(path=args.repair_approval, sha256=args.repair_approval_sha256 or '')
    repair_permission(approval)
    if args.command == 'freeze-repair':
        result = create_contract(approval)
    else:
        contract = dict(path=args.contract or '', sha256=args.contract_sha256 or '')
        result = preflight(contract, approval) if args.command == 'preflight' else run(args.command, args.scope, args.arm, contract, approval)
    print(json.dumps(dict(stage=args.command, scope=args.scope, result=result)), flush=True)


if __name__ == '__main__':
    main()
