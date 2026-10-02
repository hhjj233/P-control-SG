#!/usr/bin/env python3
"""RADE reproduction on highD (Wang et al., ITSC 2025) for the comparison in the paper.

RADE conditions a joint multi-agent diffusion model on a scene risk level
r = exp(-k * max(0, PET / T - sigma)), k = 5, sigma = 0.05, T = 3.2 s, and samples
with classifier-free guidance. RADE has no percentile input, so at test time the
requested p is converted to the physical PET target that our reference assigns to
it (the canonical quantile q_H(p)), and that target to r.

For a like-for-like comparison the denoiser, data, trajectory basis, optimizer and
FIT/STOP selection are those of the Standard P-diffusion baseline
(pcontrol/research/run_baseline_generators.py). Only the condition
changes: r from the observed-future PET instead of the percentile label, with
condition dropout for classifier-free guidance. RADE's motion-token dynamics check
targets state-space diffusion; the trajectory basis used here is already smooth,
so that step is omitted.

  train:    python pcontrol/research/rade_baseline.py train --device cuda:0
  evaluate: python pcontrol/research/rade_baseline.py evaluate --scale 2.5
"""
import argparse, json, math, os, sys, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.generation.data import FEATURE_KEYS
from pcontrol.generation.direct_p_cfg import ClassifierFreePercentileDenoiser, cfg_training_loss, cfg_sample
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.evaluation import quality_metrics
from pcontrol.generation.scene_quality_diagnostics import scene_quality
from pcontrol.data.scene_pet import scene_occupancy_pet
from pcontrol.publication_pipeline.final_execution_generation import summarize_requests
from pcontrol.research.run_baseline_generators import load_source
from pcontrol.research.evaluate_natural_diffusion import model_features
from pcontrol.research.audit_time_attention_full_pipeline import rank
from pcontrol.research.audit_pair_overlap_intervals import pair_overlap_intervals
from pcontrol.research import train_natural_scene_reference as io

OUT = ROOT / 'outputs/natural_percentile/experiments_v1/rade'
FINAL = ROOT / 'outputs/natural_percentile/transformer_publication_v1_20260916/final_validation_v1/TEST'
SEED = 20260929
GRID = (.1, .3, .5, .7, .9)
RISK_K, RISK_SIGMA, RISK_T = 5., .05, 3.2
DROP = .1  # condition dropout for classifier-free guidance; RADE does not report its value
COUNTS = {'FIT': 9913, 'STOP': 538}


def risk_level(pet):
    """RADE Eq. (9): r = exp(-k * max(0, PET / T - sigma))."""
    pet = np.asarray(pet, dtype=np.float64)
    return np.exp(-RISK_K * np.maximum(0., pet / RISK_T - RISK_SIGMA))


def batch(source, role, rows, device):
    pack = source['packs'][role]; mask = pack['agent_mask'][rows]; n = int(mask.sum(1).max())
    f = {}
    for k in FEATURE_KEYS:
        a = pack[k][rows]
        if k == 'history': a = a[:, :, :n]
        elif k in ('dimensions', 'agent_mask', 'ego_mask'): a = a[:, :n]
        f[k] = torch.as_tensor(a, device=device)
    clean = torch.as_tensor(pack['coef_clean'][rows, :n], device=device)
    r = torch.tensor(risk_level(source['labels'][role]['pet_seconds'][rows]), dtype=torch.float32, device=device)
    return f, clean, r


def objective(model, schedule, f, clean, r, present, rng):
    noise = torch.randn(clean.shape, generator=rng, dtype=torch.float32).to(clean.device)
    t = torch.randint(100, (len(r),), generator=rng).to(clean.device)
    return cfg_training_loss(model, schedule, clean, f, r, present, timesteps=t, noise=noise)['loss']


@torch.no_grad()
def validation(model, source, schedule, device):
    model.eval(); rng = torch.Generator().manual_seed(SEED + 1); total = 0.
    for start in range(0, COUNTS['STOP'], 128):
        rows = np.arange(start, min(start + 128, COUNTS['STOP'])); f, clean, r = batch(source, 'STOP', rows, device)
        present = torch.ones(len(rows), dtype=torch.bool, device=device)
        total += float(objective(model, schedule, f, clean, r, present, rng)) * len(rows)
    return total / COUNTS['STOP']


def train(device):
    torch.set_num_threads(2); torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    source = load_source(); OUT.mkdir(parents=True, exist_ok=False)
    for role, count in COUNTS.items():
        assert len(source['labels'][role]['pet_seconds']) == count
    freeze = dict(method='RADE reproduction (Wang et al., ITSC 2025) on the P-diffusion backbone',
        source_policy=c.bind(ROOT / 'configs/natural_percentile/guarded_generator_adaptation_v1.json'),
        labels=source['labels_binding'], seed=SEED, FIT=COUNTS['FIT'], STOP=COUNTS['STOP'],
        risk=dict(k=RISK_K, sigma=RISK_SIGMA, T=RISK_T, pet='observed-future ego-SV scene PET, 4 s cap'),
        condition_dropout=DROP, max_epochs=60, patience=12, selection='STOP_fixed_noise_v_MSE_condition_present',
        motion_token_dynamics_check=False, warm_start=False, TEST_selection=False,
        code=c.source_bindings(('pcontrol/research/rade_baseline.py',)))
    io.write_json(OUT / 'freeze.json', freeze)
    model = ClassifierFreePercentileDenoiser(16).to(device); schedule = CosineDiffusionSchedule(100).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    order_rng = np.random.default_rng(SEED); rng = torch.Generator().manual_seed(SEED + 2)
    drop_rng = torch.Generator().manual_seed(SEED + 3)
    best = float('inf'); best_epoch = 0; stale = 0; best_state = None; started = time.monotonic()
    with (OUT / 'epochs.jsonl').open('x') as log:
        for epoch in range(1, 61):
            model.train(); order = order_rng.permutation(COUNTS['FIT']); total = 0.
            for start in range(0, COUNTS['FIT'], 128):
                rows = order[start:start + 128]; f, clean, r = batch(source, 'FIT', rows, device)
                present = (torch.rand(len(rows), generator=drop_rng) >= DROP).to(device)
                opt.zero_grad(set_to_none=True)
                loss = objective(model, schedule, f, clean, r, present, rng)
                if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
                total += float(loss.detach()) * len(rows)
            val = validation(model, source, schedule, device)
            if val < best:
                best = val; best_epoch = epoch; stale = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else: stale += 1
            row = dict(epoch=epoch, FIT_loss=total / COUNTS['FIT'], STOP_loss=val, best_epoch=best_epoch,
                       best_STOP=best, elapsed_seconds=time.monotonic() - started)
            log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)
            if epoch >= 10 and stale >= 12: break
    torch.save(dict(state_dict=best_state, epoch=best_epoch, freeze=c.bind(OUT / 'freeze.json')), OUT / 'checkpoint.pt')
    io.write_json(OUT / 'training_result.json', dict(status='complete', freeze=c.bind(OUT / 'freeze.json'),
        checkpoint=c.bind(OUT / 'checkpoint.pt'), best_epoch=best_epoch, epochs_completed=epoch, STOP_loss=best,
        parameters=sum(p.numel() for p in model.parameters()), wall_seconds=time.monotonic() - started,
        coefficient_normalizer=source['data']['coefficient_normalizer'],
        history_normalizer=source['data']['history_normalizer']))
    print(json.dumps(dict(stage='training_complete', best_epoch=best_epoch)), flush=True)


def evaluate(scale):
    torch.set_num_threads(2); torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False); torch.backends.cuda.enable_math_sdp(True)
    tr = c.json_file(c.bind(OUT / 'training_result.json'))
    checkpoint = torch.load(io.verify_binding(tr['checkpoint']), map_location='cpu', weights_only=False)
    freeze = c.json_file(checkpoint['freeze']); c.verify_sources(freeze['code'])
    model = ClassifierFreePercentileDenoiser(16).eval().requires_grad_(False)
    model.load_state_dict(checkpoint['state_dict'])
    schedule = CosineDiffusionSchedule(100); basis = TrajectoryBasis(8)
    hn = c.json_file(tr['history_normalizer']); cn = c.json_file(tr['coefficient_normalizer'])
    qb = c.bind(FINAL / 'queue/queue.json'); queue = c.json_file(qb)
    old = c.json_file(c.bind(FINAL / 'generation/atom_aware/result.json'))
    oldrows = {(r['scene_id'], r['noise_index'], r['requested_p']): r for r in old['rows']}
    out = OUT / f'TEST_w{scale:g}'; out.mkdir(exist_ok=False); rows = []; start = time.monotonic()
    io.write_json(out / 'freeze.json', dict(training=c.bind(OUT / 'training_result.json'), queue=qb, CFG_scale=scale,
        K=1, noise='same_saved_per_actor16', target='canonical reference quantile q_H(p) converted to RADE risk r',
        code=c.source_bindings(('pcontrol/research/rade_baseline.py',))))
    for ci, item in enumerate(queue['cases']):
        a = c.arrays(item['artifact']); n = item['num_agents']; ego = int(np.flatnonzero(a['ego_mask'])[0])
        case = {k: a[k] for k in ('history', 'dimensions', 'road_boundaries', 'ego_mask', 'agent_mask')}
        case['num_agents'] = n
        f = model_features(case, hn, torch.device('cpu'))
        assert set(f) == FEATURE_KEYS
        for z in range(3):
            artifacts = {}; pending = []
            for p in GRID:
                oldrow = oldrows[item['scene_id'], z, p]
                target = float(oldrow['canonical_target_PET_seconds']); r_target = float(risk_level(target))
                r = dict(scene_id=item['scene_id'], recording_id=item['recording_id'], num_agents=n,
                    stratum=item['stratum'], case_index=ci, noise_index=z, requested_p=p, arm='rade', K=1,
                    CFG_scale=scale, target_PET_seconds=target, risk_condition=r_target,
                    inference_observed_future_input=False, all_actors_retained=True, post_sampler_repair=False)
                tick = time.monotonic()
                try:
                    noise = torch.tensor(a[f'initial_noise_{z}'][None], dtype=torch.float32)
                    with torch.no_grad():
                        coef = cfg_sample(model, schedule, f, torch.tensor([r_target], dtype=torch.float32), noise,
                                          scale=scale, steps=50)[0].numpy()
                    future = basis.decode(coef.astype(np.float64) * np.array(cn['scale']) + np.array(cn['mean']),
                                          case['history'][-1])
                    elapsed = time.monotonic() - tick
                    if not np.isfinite(future).all() or not np.array_equal(future[0], case['history'][-1]):
                        raise ValueError('finite / exact t0')
                    oracle = scene_occupancy_pet(future, np.ones((175, n), bool), case['dimensions'], times=basis.times,
                        ego_index=ego, sample_period=.04, window=(0., 6.96), cap_seconds=4.)
                    pet = float(oracle['pet_value_seconds'])
                    rr = rank(a['reference_joint_masses'], a['reference_row_nodes'], pet)
                    overlaps = pair_overlap_intervals(future, case['dimensions'], ego)
                    quality = quality_metrics(future, dict(case, future=a['future_observed']))
                    r.update(status='complete', pet_seconds=pet, estimated_rank=rr, p_mid_absolute_error=abs(rr['p_mid'] - p),
                        quality=quality, scene_quality=scene_quality(future, case['dimensions'], case['ego_mask']),
                        PL_overlap_intervals=overlaps, background_PL_overlap_scene=any(o['background_pair'] for o in overlaps),
                        ego_PL_overlap_scene=any(not o['background_pair'] for o in overlaps),
                        PET_control_target_absolute_error_seconds=abs(pet - oldrow['control_target_PET_seconds']),
                        canonical_PET_target_absolute_error_seconds=abs(pet - target),
                        scalar_error_infimum=oldrow['scalar_error_infimum'], sampling_seconds=elapsed,
                        critical_other_index=oracle['critical_other_index'])
                    key = 'generated_p' + str(p).replace('.', '_'); artifacts[key] = future; r['array_key'] = key
                except (ValueError, FloatingPointError, RuntimeError) as exc:
                    r.update(status='failed', failure_type=type(exc).__name__, failure_message=str(exc))
                pending.append(r)
            path = out / f'case_{ci:03d}_z{z}.npz'; np.savez_compressed(path, **artifacts); binding = c.bind(path)
            for r in pending: r['trajectory_artifact'] = binding
            io.write_json(out / f'case_{ci:03d}_z{z}.json', dict(rows=pending))
            rows.extend(pending)
        if (ci + 1) % 12 == 0:
            print(json.dumps(dict(completed_histories=ci + 1, seconds=time.monotonic() - start)), flush=True)
    if len(rows) != 1440: raise RuntimeError('denominator changed')
    summary = summarize_requests(rows)
    io.write_json(out / 'result.json', dict(status='complete', arm='rade', CFG_scale=scale, rows=rows, summary=summary,
        queue=qb, training=c.bind(OUT / 'training_result.json'),
        by_N={s: summarize_requests([r for r in rows if r['stratum'] == s]) for s in ('N3_5', 'N6_8', 'N9plus')},
        total_seconds=time.monotonic() - start, all_requested_rows_retained=True))
    print(json.dumps(dict(CFG_scale=scale, summary=summary)), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train'); t.add_argument('--device', default='cuda:0')
    e = sub.add_parser('evaluate'); e.add_argument('--scale', type=float, default=2.5)
    a = ap.parse_args()
    train(a.device) if a.cmd == 'train' else evaluate(a.scale)
