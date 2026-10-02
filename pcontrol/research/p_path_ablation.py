#!/usr/bin/env python3
"""Versioned P-path ablation; all frozen primary sources stay unchanged."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from pcontrol.time_attention_pipeline import common as c
from pcontrol.publication_pipeline.final_generation_adapter import (
    FrozenFinalGenerator, initialize_runtime, attach_observation_diagnostics)
from pcontrol.publication_pipeline.final_reference_suite import FEATURES
from pcontrol.generation.background_constrained_guidance import BackgroundConstrainedGuidance
from pcontrol.generation.direct_p_cfg import _CFGPrediction
from pcontrol.generation.diffusion import ddim_sample
from pcontrol.generation.scene_quality_diagnostics import scene_quality
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY, quantile_from_pieces
from pcontrol.generation.cdf_shape_context import CONTEXT_KEY
from pcontrol.research.audit_pair_overlap_intervals import pair_overlap_intervals

BASE = ROOT / 'outputs/natural_percentile/transformer_publication_v1_20260916/final_validation_v1'
OUT = ROOT / 'outputs/natural_percentile/canonical_p_ablation_v1_20260925'
PROTOCOL = ROOT / 'docs/plans/CANONICAL_P_ABLATION_PROTOCOL_20260925.md'
GRID = (.1, .3, .5, .7, .9)
ARMS = ('condition_only', 'guidance_only', 'no_p')


def now():
    return datetime.now(timezone.utc).isoformat()


def write_once(path, data):
    with Path(path).open('x') as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
    return c.bind(path)


def load_case(binding, *, observation=False):
    assert c.bind(binding['path']) == binding
    keys = set(FEATURES) | {f'initial_noise_{z}' for z in range(3)} | {
        'reference_joint_masses', 'reference_row_nodes', PIECES_KEY, CONTEXT_KEY}
    if observation:
        keys = {'future_observed'}
    with np.load(binding['path'], allow_pickle=False) as archive:
        return {k: archive[k].copy() for k in keys}


def prepare(engine, arrays):
    assert torch.get_num_threads() == 2
    try:
        torch.set_num_threads(1)
        value = engine.prepare_case({k: arrays[k] for k in FEATURES})
    finally:
        torch.set_num_threads(2)
    if 'reference_joint_masses' in arrays:
        assert np.array_equal(value['reference']._masses[0], arrays['reference_joint_masses'])
        assert np.array_equal(value['pieces'][0].numpy(), arrays[PIECES_KEY])
        assert np.array_equal(value['features'][CONTEXT_KEY][0].numpy(), arrays[CONTEXT_KEY])
    return value


class GeometryOnlyGuidance(BackgroundConstrainedGuidance):
    """Same outer geometry schedule; no P queries, stopping, or risk increments."""
    def __call__(self, x0, timesteps, x_t):
        if x0.shape[0] != 1 or x0.shape[1] != len(self.dimensions) or self.calls >= 50:
            raise ValueError('one fresh all-actor DDIM path required')
        step = self.calls
        self.calls += 1
        self.envelope.background_enabled = step >= 50 - self.background_last_steps
        start = 50 - self.last_steps
        if step < start:
            return x0
        if step < 50 - self.dense_last_steps and (step - start) % self.stride != 0:
            return x0
        current, repair = self.envelope.project(x0[0].detach().double())
        self.road_trace.append(dict(step=step, phase='clean_estimate_envelope', **repair))
        result = current[None].to(x0.dtype)
        if self.envelope.background_enabled:
            corrected, info = self.background.project(result[0].detach().double())
            self.background_trace.append(dict(step=step, phase='before_DDIM_transition', **info))
            result = corrected[None].to(result.dtype)
        return result

    def report(self):
        report = super().report()
        assert report['CDF_rank_queries'] == report['exact_metric_calls'] == report['active_inner_checks'] == 0
        return dict(report, risk_guidance_enabled=False, P_used_in_geometry_callback=False)


def forbidden_rank(_):
    raise AssertionError('risk-disabled callback may not query the risk reference')


def sample(engine, prepared, p, noise, arm):
    if arm not in ('full',) + ARMS:
        raise ValueError(arm)
    condition = arm in ('full', 'condition_only')
    guidance = arm in ('full', 'guidance_only')
    case = prepared['case']
    ref = prepared['reference']
    if noise.shape != (case['num_agents'], 8, 2) or noise.dtype != np.float32:
        raise ValueError('original all-actor float32 noise required')
    # Only G-enabled callbacks receive the requested target. G-disabled constructors
    # get fixed unused placeholders, not P or q_H(P).
    target = float(ref.quantile(1. - p)) if guidance else None
    klass = BackgroundConstrainedGuidance if guidance else GeometryOnlyGuidance
    guide = klass(prepared['decoder'], case['dimensions'], case['history'][-1],
        (lambda y: float(ref.rank(y)['p_mid'])) if guidance else forbidden_rank,
        p if guidance else .5, target if guidance else 2., ego_index=prepared['ego'],
        road_boundaries=case['road_boundaries'], **engine.profile)
    # scale=0 uses the trained null branch and its masked indirect q projection.
    # Never pass the actual request into the null-only denoiser interface.
    model_p = torch.tensor([p if condition else .5], dtype=torch.float32)
    adapter = _CFGPrediction(engine.model, prepared['features'], model_p, 2.5 if condition else 0.)
    tick = time.monotonic()
    coefficients = ddim_sample(adapter, engine.schedule, prepared['features'], torch.tensor(noise[None]),
                               steps=50, x0_callback=guide, prediction_type='v')[0].numpy().astype(np.float64)
    future = engine.basis.decode(coefficients * np.asarray(engine.cn['scale']) + np.asarray(engine.cn['mean']), case['history'][-1])
    assert future.shape == (175, case['num_agents'], 4) and np.isfinite(future).all()
    assert np.array_equal(future[0], case['history'][-1])
    assert adapter.network_evaluations == (100 if condition else 50)
    return future, dict(network_condition_enabled=condition, sampling_risk_guidance_enabled=guidance,
        network_evaluations=adapter.network_evaluations, CFG_scale=2.5 if condition else 0.,
        sampling_seconds=time.monotonic()-tick, guidance=guide.report(),
        all_actors_retained=True, K=1, post_sampler_repair=False, inference_observed_future_input=False)


def score(engine, prepared, future, p):
    ref, case = prepared['reference'], prepared['case']
    measured = ref.score_future(future)
    pet, rank = float(measured['pet_seconds']), measured['estimated_rank']
    target = float(ref.quantile(1.-p))
    e = max(rank['p_low']-p, p-rank['p_up'], 0.)
    overlaps = pair_overlap_intervals(future, case['dimensions'], prepared['ego'])
    return dict(pet_seconds=pet, estimated_rank=rank, p_interval_error=e,
        p_mid_absolute_error=abs(rank['p_mid']-p), canonical_target_PET_seconds=target,
        canonical_PET_target_absolute_error_seconds=abs(pet-target), control_target_PET_seconds=target,
        PET_control_target_absolute_error_seconds=abs(pet-target), target_atom=target in (0., 4.),
        interval_Fine=e <= .05, PL_overlap_intervals=overlaps,
        background_PL_overlap_scene=any(x['background_pair'] for x in overlaps),
        ego_PL_overlap_scene=any(not x['background_pair'] for x in overlaps),
        scene_quality=scene_quality(future, case['dimensions'], case['ego_mask']))


def summarize(rows):
    good = [r for r in rows if r['status'] == 'complete']
    n, missing = len(rows), len(rows)-len(good)
    hits = sum(r['p_interval_error'] <= .05 for r in good)
    return dict(requests=n, completed=len(good), failed=missing, interval_Fine_count=hits,
        interval_Fine_rate=hits/n if n else None,
        interval_P_MAE=None if missing or not n else float(np.mean([r['p_interval_error'] for r in good])),
        midpoint_Fine_count=sum(r['p_mid_absolute_error'] <= .05 for r in good),
        midpoint_P_MAE=None if missing or not n else float(np.mean([r['p_mid_absolute_error'] for r in good])),
        PET_target_MAE_seconds=None if missing or not n else float(np.mean([r['canonical_PET_target_absolute_error_seconds'] for r in good])),
        BG_PL_overlap_known=sum(r['background_PL_overlap_scene'] for r in good),
        ego_PL_overlap_known=sum(r['ego_PL_overlap_scene'] for r in good),
        strict_road_known=sum(r['quality']['road_outside_scene'] for r in good),
        negative_vx_known=sum(r['quality']['negative_vx_scene'] for r in good),
        geometry_unknown=missing,
        mean_ADE_m=None if missing or not n else float(np.mean([r['quality']['ADE_m'] for r in good])))


def engine_and_queue():
    initialize_runtime()
    queue = c.json_file(c.bind(BASE / 'TEST/queue/queue.json'))
    return FrozenFinalGenerator(queue['contract'], 'canonical'), queue


def preflight():
    OUT.mkdir(parents=True, exist_ok=False)
    engine, queue = engine_and_queue()
    source = c.json_file(c.bind(BASE / 'TEST/generation/canonical/result.json'))
    by_key = {(r['case_index'], r['noise_index'], r['requested_p']): r for r in source['rows']}
    selected = [min((x for x in queue['cases'] if x['stratum'] == s), key=lambda x: x['case_index'])
                for s in ('N3_5', 'N6_8', 'N9_plus')]
    checks = []
    for item in selected:
        arrays = load_case(item['artifact'])
        prepared = prepare(engine, arrays)
        for p in (.1, .5, .9):
            future, info = sample(engine, prepared, p, arrays['initial_noise_0'], 'full')
            old = by_key[item['case_index'], 0, p]
            saved = c.arrays(old['trajectory_artifact'])[old['array_key']]
            err = float(abs(future-saved).max())
            assert err == 0., (item['scene_id'], p, err)
            fresh = score(engine, prepared, future, p)
            assert abs(fresh['pet_seconds']-old['pet_seconds']) < 1e-12
            checks.append(dict(scene_id=item['scene_id'], N=item['num_agents'], p=p, exact_full_replay=True))
        null = []
        for p in GRID:
            f, info = sample(engine, prepared, p, arrays['initial_noise_0'], 'no_p')
            assert info['guidance']['CDF_rank_queries'] == info['guidance']['exact_metric_calls'] == 0
            null.append(f)
        assert all(np.array_equal(null[0], f) for f in null)
        for arm in ('condition_only', 'guidance_only'):
            f, info = sample(engine, prepared, .5, arrays['initial_noise_0'], arm)
            assert f.shape == null[0].shape and np.array_equal(f[0], arrays['history'][-1])
        print(json.dumps(dict(stage='preflight', N=item['num_agents'], full_replay=True, no_P_invariant=True)), flush=True)
    # Real large scene from STOP, no decoding of its observed future.
    from pcontrol.research.external_no_p_data import OUT as DATA_OUT
    manifest = c.json_file(c.bind(DATA_OUT / 'data/manifest.json'))
    binding = manifest['packs']['STOP']
    assert c.bind(binding['path']) == binding
    with np.load(binding['path'], allow_pickle=False) as z:
        j = int(np.flatnonzero(z['num_agents'] == 41)[0])
        n = 41
        ego = np.arange(n) == int(z['ego_index'][j])
        features = dict(history=z['history'][j, :, :n].astype(np.float64), dimensions=z['dimensions'][j, :n].astype(np.float64),
            road_boundaries=z['road_boundaries'][j, :z['road_count'][j]].astype(np.float64), ego_mask=ego, agent_mask=np.ones(n, bool))
        features['history'][-1] = z['anchors'][j, :n]
    prepared = prepare(engine, features)
    noise = np.random.default_rng(20260925).standard_normal((41, 8, 2)).astype(np.float32)
    for arm in ARMS:
        f, _ = sample(engine, prepared, .5, noise, arm)
        assert f.shape == (175, 41, 4) and np.array_equal(f[0], features['history'][-1])
    engine.assert_unchanged()
    out = dict(status='pass', time_utc=now(), protocol=c.bind(PROTOCOL), code=c.bind(__file__),
        original_queue=c.bind(BASE / 'TEST/queue/queue.json'), original_full=c.bind(BASE / 'TEST/generation/canonical/result.json'),
        original_contract=queue['contract'], checkpoint=engine.contract['generator_checkpoints']['canonical'],
        state_tensor_sha256=engine.initial_state, full_replay_checks=checks, no_P_all_five_exact_invariance=True,
        N41_all_new_arms=True, full_roster_finite_exact_t0=True, reference_threads=1, generation_threads=2,
        no_new_training=True, primary_metric='atom_compatible_interval_error', tolerance=.05,
        prior_no_ablation_policy_preserved=True,
        all_original_artifacts_unchanged=True)
    write_once(OUT / 'freeze.json', out)
    print(json.dumps(dict(status='preflight_pass', checks=len(checks), output=str(OUT))), flush=True)


def run(arm):
    frozen = c.json_file(c.bind(OUT / 'freeze.json'))
    assert frozen['status'] == 'pass' and c.bind(__file__) == frozen['code']
    assert c.bind(PROTOCOL) == frozen['protocol']
    engine, queue = engine_and_queue()
    assert c.bind(BASE / 'TEST/queue/queue.json') == frozen['original_queue']
    dest = OUT / arm
    dest.mkdir(exist_ok=False)
    rows, batches = [], []
    started = time.monotonic()
    for item in queue['cases']:
        arrays = load_case(item['artifact'])
        prepared = prepare(engine, arrays)
        case_rows = []
        for z in range(3):
            stem = f"case_{item['case_index']:03d}_z{z}"
            write_once(dest / (stem + '.claim.json'), dict(case=item['artifact'], z=z, arm=arm, freeze=c.bind(OUT / 'freeze.json')))
            generated = {}
            for p in ((.5,) if arm == 'no_p' else GRID):
                key = 'generated' if arm == 'no_p' else f'generated_p{p:g}'.replace('.', '_')
                target_ps = GRID if arm == 'no_p' else (p,)
                base = dict(case_index=item['case_index'], scene_id=item['scene_id'], recording_id=item['recording_id'],
                    stratum=item['stratum'], num_agents=item['num_agents'], role='TEST', arm=arm, noise_index=z,
                    input_case=item['artifact'], noise_sha256=hashlib.sha256(arrays[f'initial_noise_{z}'].tobytes()).hexdigest())
                try:
                    future, info = sample(engine, prepared, p, arrays[f'initial_noise_{z}'], arm)
                    generated[key] = future
                    # Only after the sampler has returned may the true future be decoded.
                    observed = load_case(item['artifact'], observation=True)['future_observed']
                    for target_p in target_ps:
                        row = dict(base, status='complete', requested_p=target_p, array_key=key,
                            unique_output_key=f'{stem}/{key}', **info, **score(engine, prepared, future, target_p))
                        row = attach_observation_diagnostics(row, future, prepared['case'], observed)
                        case_rows.append(row)
                except (RuntimeError, ValueError, FloatingPointError) as error:
                    for target_p in target_ps:
                        case_rows.append(dict(base, status='failed', requested_p=target_p,
                            failure_reason=type(error).__name__ + ': ' + str(error)))
            path = dest / (stem + '.npz')
            np.savez_compressed(path, **generated)
            binding = c.bind(path)
            current = [r for r in case_rows if r['noise_index'] == z]
            for row in current:
                if row['status'] == 'complete':
                    row['trajectory_artifact'] = binding
            record = write_once(dest / (stem + '.json'), dict(arm=arm, case=item['artifact'], rows=current, trajectory=binding))
            batches.append(record)
        rows.extend(case_rows)
        print(json.dumps(dict(arm=arm, histories=len(rows)//15, requests=len(rows),
                              elapsed_seconds=time.monotonic()-started, latest=summarize(case_rows))), flush=True)
    engine.assert_unchanged()
    assert len(rows) == 1440 and len({(r['scene_id'], r['noise_index'], r['requested_p']) for r in rows}) == 1440
    result = dict(status='complete', arm=arm, time_utc=now(), freeze=c.bind(OUT / 'freeze.json'),
        rows=rows, batches=batches, summary=summarize(rows),
        by_P={str(p): summarize([r for r in rows if r['requested_p'] == p]) for p in GRID},
        by_N={s: summarize([r for r in rows if r['stratum'] == s]) for s in ('N3_5','N6_8','N9_plus')},
        output_attempts=288 if arm == 'no_p' else 1440, requests=1440, independent_histories=96,
        same_no_P_output_reused_for_scoring=arm == 'no_p', all_failures_retained=True,
        wall_seconds=time.monotonic()-started, checkpoint_unchanged=True, post_primary_inference_ablation=True)
    write_once(dest / 'result.json', result)
    print(json.dumps(dict(status='complete', arm=arm, summary=result['summary'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('preflight','run'))
    parser.add_argument('--arm', choices=ARMS)
    args = parser.parse_args()
    if args.stage == 'preflight':
        preflight()
    else:
        if args.arm is None:
            parser.error('--arm required')
        run(args.arm)
