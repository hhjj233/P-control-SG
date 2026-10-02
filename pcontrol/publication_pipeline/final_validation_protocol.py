"""Outcome-blind final queue selection and explicit data-access receipt checks.

No CSV or NPZ reader lives in this module. A receipt documents an explicit
decision to open the evaluation data; this is an accidental-execution guard, not an authentication system.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = 'frozen_final_natural_validation_v1'
IDENTITY_FIELDS = {'scene_id','recording_id','num_agents','role'}
COUNT_GROUPS = (('N3_5',3,5),('N6_8',6,8),('N9_plus',9,None))
REFERENCE_ARMS = ('M2_MLP','M2_WideMLP','M2_GRU','M2_TimeAttn_absolute_only',
                  'M2_TimeAttn_relative_only','M1_TimeAttn','M2_TimeAttn')


def resolve(path):
    p = Path(path)
    return (p if p.is_absolute() else ROOT/p).resolve()


def digest(path):
    h = hashlib.sha256()
    with resolve(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def bind(path):
    return {'path':str(resolve(path)),'sha256':digest(path)}


def verified_json(binding):
    if set(binding) != {'path','sha256'} or not re.fullmatch('[0-9a-f]{64}',binding['sha256']):
        raise ValueError('exact SHA256 binding required')
    if digest(binding['path']) != binding['sha256']: raise ValueError('artifact drift: '+binding['path'])
    return json.loads(resolve(binding['path']).read_text())


def same_binding(a,b):
    return resolve(a['path']) == resolve(b['path']) and a['sha256'] == b['sha256']


def validate_policy(policy):
    if policy['protocol'] != PROTOCOL or policy['authorization_required'] is not True:
        raise ValueError('final validation must require explicit authorization')
    if policy['authorization_received'] is not False:
        raise ValueError('approval must be a separate receipt, not an edited policy flag')
    parent = verified_json(policy['sources']['parent_split'])['splits']
    roles = verified_json(policy['sources']['development_roles'])
    cohorts = policy['cohorts']
    if cohorts != {'VAL':parent['val'],'TEST':parent['test'],'R18':['18']}:
        raise ValueError('fixed final recording roster changed')
    current = set().union(*(set(roles[k]) for k in ('FIT','STOP','CAL','AUDIT')))
    if current != set(parent['train']): raise ValueError('development role union changed')
    final = [rec for ids in cohorts.values() for rec in ids]
    if len(final) != len(set(final)) or current.intersection(final):
        raise ValueError('final/development recording overlap')
    reference = policy['reference']
    if (reference['arms'] != list(REFERENCE_ARMS) or reference['refit_or_recalibrate'] is not False
            or reference['views'] != ['raw','frozen_calibrated'] or reference['primary_model'] != 'M2_TimeAttn'):
        raise ValueError('reference comparison scope changed')
    g = policy['generation']
    if (g['P_grid'] != [.1,.3,.5,.7,.9] or g['noise_indices'] != [0,1,2]
            or g['K_per_request'] != 1 or g['Fine_tolerance'] != .05
            or g['history_count_per_stratum_per_cohort'] != 32
            or g['count_groups'] != [list(row) for row in COUNT_GROUPS]):
        raise ValueError('final request geometry or count policy changed')
    for key in ('post_sampler_repair','best_of_K','retry_failed_requests','reselect_checkpoint_or_controller','scalar_floor_replaces_primary_error'):
        if g[key] is not False: raise ValueError('forbidden final optimization: '+key)
    d = policy['data']
    if (d['history_candidates_per_recording'] != 4096 or d['minimum_agents'] != 3
            or d['maximum_agents'] is not None or not d['natural_only']
            or d['replace_incomplete'] or d['fill_missing_futures']
            or d['exclude_collision_or_low_speed_or_initial_violations']):
        raise ValueError('final population/selection changed')
    expected = dict(history_steps=13,history_dt_seconds=.08,native_future_steps_including_t0=175,
        future_dt_seconds=.04,PET_cap_seconds=4.,history_candidate_grid_frames=25,
        freeze_H_cohort_before_future_eligibility_or_PET=True)
    if any(d[k] != v for k,v in expected.items()): raise ValueError('history/future/PET contract changed')
    if g['DDIM_steps'] != 50 or g['CFG_scale'] != 2.5 or g['arms'] != ['canonical','atom_aware']:
        raise ValueError('generation backbone/budget changed')
    return policy


def select_histories(rows, *, allowed_recordings, role, per_group=32, salt):
    """Select only from complete-population identity/N metadata, without labels.

    Completeness defines the population before this call. Its future dependence
    is disclosed, not disguised as a fully history-observable eligibility gate.
    """
    if type(per_group) is not int or per_group < 1 or not salt: raise ValueError('positive count and fixed salt required')
    allowed = set(allowed_recordings)
    if len(allowed) != len(allowed_recordings): raise ValueError('duplicate recording permission')
    clean = []
    for row in rows:
        if set(row) != IDENTITY_FIELDS: raise ValueError('only identity/count/role fields may enter selection')
        if row['recording_id'] not in allowed or row['role'] != role: raise PermissionError('recording/role outside cohort')
        if not isinstance(row['scene_id'],str) or not row['scene_id']: raise ValueError('scene identity required')
        if type(row['num_agents']) is not int or row['num_agents'] < 3: raise ValueError('invalid historical roster size')
        clean.append(dict(row))
    if len({r['scene_id'] for r in clean}) != len(clean): raise ValueError('duplicate history identity')
    def rank(value): return hashlib.sha256((salt+'|'+value).encode()).hexdigest()
    selected = []; groups = {}
    for label,lo,hi in COUNT_GROUPS:
        pool = [r for r in clean if r['num_agents'] >= lo and (hi is None or r['num_agents'] <= hi)]
        buckets = {rec:sorted([r for r in pool if r['recording_id']==rec],key=lambda r:(rank(r['scene_id']),r['scene_id']))
                   for rec in sorted(allowed,key=lambda r:(rank(r),r))}
        chosen = []; target = min(per_group,len(pool))
        while len(chosen) < target:
            for bucket in buckets.values():
                if bucket and len(chosen) < target: chosen.append(bucket.pop(0))
        selected.extend(dict(row,stratum=label) for row in chosen)
        groups[label] = dict(available=len(pool),selected=len(chosen),shortfall=per_group-len(chosen))
    for index,row in enumerate(selected): row['case_index']=index
    return selected, dict(groups=groups,histories=len(selected),by_recording=dict(Counter(r['recording_id'] for r in selected)),
        requested_per_group=per_group,selection_uses_outcomes=False,no_shortfall_borrowing=True)


def noise_for_history(scene_id, noise_index, agents, *, salt):
    import numpy as np
    if noise_index not in (0,1,2) or type(agents) is not int or agents < 3: raise ValueError('invalid noise request')
    seed = int.from_bytes(hashlib.sha256(f'{salt}|{scene_id}|{noise_index}'.encode()).digest()[:8],'little')
    noise = np.random.default_rng(seed).standard_normal((agents,8,2)).astype(np.float32)
    return noise, dict(seed=seed,array_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),arm_or_P_in_seed=False)


def failure_aware_precision(requests):
    """Retain failed requests: report rank-error bounds, never success-only MAE."""
    if not requests: return dict(requests=0,completed=0,failed=0,Fine_count=0,Fine_rate=None,P_MAE=None,P_MAE_bounds=None)
    total = 0.; missing_upper = 0.; fine = 0; completed = 0
    for row in requests:
        p = row['requested_p']; error = row['absolute_error']
        if isinstance(p,bool) or not isinstance(p,(int,float)) or not math.isfinite(p) or not 0 <= p <= 1: raise ValueError('invalid request')
        if error is None:
            if not row.get('failure_reason'): raise ValueError('explicit failed-request reason required')
            missing_upper += max(p,1-p)
        else:
            if not math.isfinite(error) or not 0 <= error <= max(p,1-p)+1e-12: raise ValueError('invalid rank error')
            total += error; completed += 1; fine += error <= .05
    n = len(requests)
    return dict(requests=n,completed=completed,failed=n-completed,Fine_count=fine,Fine_rate=fine/n,
        P_MAE=total/n if completed==n else None,P_MAE_bounds=[total/n,(total+missing_upper)/n])


@dataclass(frozen=True)
class EvaluationPermission:
    scope: str
    recording_ids: tuple
    policy_sha256: str
    freeze_sha256: str
    approval_sha256: str
    permits_training: bool = False


def require_approval(policy_binding, freeze_binding, approval_binding, *, scope):
    # Crucial ordering: no absent-receipt path may read a policy, freeze, raw
    # directory, recording metadata, trajectory or even a raw-file stat.
    if approval_binding is None: raise PermissionError('explicit unsealing approval has not been recorded')
    policy = validate_policy(verified_json(policy_binding))
    freeze = verified_json(freeze_binding); receipt = verified_json(approval_binding)
    if freeze.get('protocol') != PROTOCOL or not same_binding(freeze['policy'],policy_binding): raise ValueError('wrong frozen contract')
    if (receipt.get('protocol') != 'explicit_user_final_validation_permission_v1'
            or receipt.get('decision') != 'allow_one_frozen_evaluation'
            or not receipt.get('verbatim_user_reply','').strip()
            or not receipt.get('user_message_reference','').strip()
            or receipt.get('recorded_from_actual_user_reply') is not True):
        raise PermissionError('approval must record an explicit decision, not a plan or default')
    if not same_binding(receipt['policy'],policy_binding) or not same_binding(receipt['freeze'],freeze_binding):
        raise PermissionError('approval belongs to a different frozen contract')
    approved = receipt.get('approved_recordings_by_scope',{})
    if scope not in policy['cohorts'] or approved.get(scope) != policy['cohorts'][scope]:
        raise PermissionError('requested cohort is not explicitly approved in full')
    if receipt.get('allow_training') is not False or receipt.get('allow_reselection') is not False:
        raise PermissionError('final validation cannot authorize fitting or reselection')
    if freeze.get('execution_ready') is not True:
        raise RuntimeError('execution adapters must be completed and separately frozen before protected reads')
    return EvaluationPermission(scope,tuple(approved[scope]),policy_binding['sha256'],freeze_binding['sha256'],approval_binding['sha256'])
