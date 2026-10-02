#!/usr/bin/env python3
"""RADE reproduction v2: longer training and a responsiveness check.

v1 (pcontrol/research/rade_baseline.py) stopped at the 60-epoch cap while still improving
and did not respond to its risk condition. v2 keeps every v1 setting and raises the cap to
400 epochs with a 40-epoch patience on the same early-stopping loss.

devcheck: 96 early-stopping (STOP) histories, three noise draws, three fixed RADE risk levels
          (dangerous: PET 0.5 s, medium: PET 1.5 s, safe: PET 3.0 s) and guidance scales 2.5, 5, 7.5.
          The chosen scale is the one used on TEST. STOP recordings are not used for fitting.
evaluate: the 1,440 percentile requests of Table 2 (targets from the reference), as in v1.
levels:   the three fixed risk levels on the 96 TEST histories, for the planner and realism tables.

  python pcontrol/research/rade_baseline_v2.py train --device cuda:0
  python pcontrol/research/rade_baseline_v2.py devcheck
  python pcontrol/research/rade_baseline_v2.py evaluate --scale S
  python pcontrol/research/rade_baseline_v2.py levels --scale S
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.generation.direct_p_cfg import ClassifierFreePercentileDenoiser, cfg_sample
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.data.scene_pet import scene_occupancy_pet
from pcontrol.research.run_baseline_generators import load_source
from pcontrol.research import rade_baseline as v1
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research.audit_pair_overlap_intervals import pair_overlap_intervals

OUT = ROOT / 'outputs/natural_percentile/experiments_v1/rade_v2'
LEVELS = {'dangerous': .5, 'medium': 1.5, 'safe': 3.0}  # target PET (s) of the fixed RADE risk levels
SCALES = tuple(float(x) for x in os.environ.get('RADE_DEV_SCALES', '2.5,5,7.5').split(','))
DEV_MODELS = os.environ.get('RADE_DEV_MODELS', 'v1,v2').split(',')
SEED = 20260930


def train(device):
    v1.OUT = OUT
    source = v1.load_source
    # Same code path as v1 with a longer cap: patch the loop limits through a local copy.
    import types
    code = Path(v1.__file__).read_text()
    code = code.replace('for epoch in range(1, 61):', 'for epoch in range(1, 401):').replace(
        'if epoch >= 10 and stale >= 12: break', 'if epoch >= 10 and stale >= 40: break').replace(
        "max_epochs=60, patience=12,", "max_epochs=400, patience=40,").replace(
        "motion_token_dynamics_check=False,", "motion_token_dynamics_check=False, variant='v2: max 400 epochs, patience 40 (pcontrol/research/rade_baseline_v2.py)',")
    assert 'range(1, 401)' in code and 'stale >= 40' in code and "variant='v2" in code
    mod = types.ModuleType('rade_v2_train'); mod.__file__ = str(Path(__file__)); exec(compile(code, 'rade_v2_train', 'exec'), mod.__dict__)
    mod.OUT = OUT
    mod.train(device)


def load_model(folder):
    tr = c.json_file(c.bind(folder / 'training_result.json'))
    cp = torch.load(io.verify_binding(tr['checkpoint']), map_location='cpu', weights_only=False)
    m = ClassifierFreePercentileDenoiser(16).eval().requires_grad_(False); m.load_state_dict(cp['state_dict'])
    return m, c.json_file(tr['history_normalizer']), c.json_file(tr['coefficient_normalizer'])


def devcheck():
    source = load_source(); pack = source['packs']['STOP']; basis = TrajectoryBasis(8); schedule = CosineDiffusionSchedule(100)
    n_all = len(pack['scene_id'])
    order = sorted(range(n_all), key=lambda i: hashlib.sha256(f'{SEED}|{pack["scene_id"][i]}'.encode()).hexdigest())[:96]
    report = {}
    for name, folder in [(n, f) for n, f in (('v1', v1.OUT), ('v2', OUT)) if n in DEV_MODELS]:
        model, hn, cn = load_model(folder)
        dscale, hscale = np.asarray(hn['dimension_scale']), np.asarray(hn['history_scale'])
        res = {}
        for scale in SCALES:
            for level, pet_target in LEVELS.items():
                r = float(v1.risk_level(pet_target)); pets = []; overlap = []; road_out = []
                for i in order:
                    mask = pack['agent_mask'][i]; n = int(mask.sum())
                    f = {k: torch.as_tensor(pack[k][i:i + 1][:, :, :n] if k == 'history' else
                                            (pack[k][i:i + 1][:, :n] if k in ('dimensions', 'agent_mask', 'ego_mask') else pack[k][i:i + 1]))
                         for k in ('history', 'dimensions', 'road_boundaries', 'road_boundary_mask', 'ego_mask', 'agent_mask')}
                    anchors = pack['anchors'][i, :n]; dims = pack['dimensions'][i, :n].astype(np.float64) * dscale
                    ego = int(np.flatnonzero(pack['ego_mask'][i, :n])[0])
                    rng = np.random.default_rng(int(hashlib.sha256(f'{SEED}|{pack["scene_id"][i]}'.encode()).hexdigest()[:8], 16))
                    for z in range(3):
                        noise = torch.tensor(rng.standard_normal((1, n, 8, 2)).astype(np.float32))
                        with torch.no_grad():
                            coef = cfg_sample(model, schedule, f, torch.tensor([r], dtype=torch.float32), noise, scale=scale, steps=50)[0].numpy()
                        fut = basis.decode(coef.astype(np.float64) * np.array(cn['scale']) + np.array(cn['mean']), anchors)
                        o = scene_occupancy_pet(fut, np.ones((175, n), bool), dims, times=basis.times, ego_index=ego,
                                                sample_period=.04, window=(0., 6.96), cap_seconds=4.)
                        pets.append(float(o['pet_value_seconds']))
                        overlap.append(any(True for _ in pair_overlap_intervals(fut, dims, ego)))
                        bounds = np.sort(pack['road_boundaries'][i][pack['road_boundary_mask'][i]].astype(np.float64) * hscale[1])
                        road_out.append(bool(np.any((fut[..., 1] - dims[None, :, 1] / 2 < bounds[0]) | (fut[..., 1] + dims[None, :, 1] / 2 > bounds[-1]))))
                pets = np.array(pets)
                res[f'{scale}|{level}'] = dict(risk=r, target_PET=pet_target, median_PET=float(np.median(pets)),
                                               PET_below_1s=float((pets < 1).mean()), PET_below_0_5s=float((pets < .5).mean()), futures=len(pets),
                                               overlap_rate=float(np.mean(overlap)), road_violation_rate=float(np.mean(road_out)))
                print(name, scale, level, json.dumps({k: round(v, 3) for k, v in res[f'{scale}|{level}'].items()}), flush=True)
        report[name] = res
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"devcheck_{'_'.join(str(x) for x in SCALES)}.json").write_text(json.dumps(dict(histories=96, split='STOP', levels=LEVELS, scales=SCALES, results=report,
                                                       code=c.bind(Path(__file__))), indent=2) + '\n')


def evaluate(scale):
    v1.OUT = OUT
    v1.evaluate(scale)


def levels(scale):
    """Three fixed RADE risk levels on the 96 TEST histories (3 noise draws): futures for Tables 4 and 6."""
    model, hn, cn = load_model(OUT); basis = TrajectoryBasis(8); schedule = CosineDiffusionSchedule(100)
    queue = c.json_file(c.bind(v1.FINAL / 'queue/queue.json'))
    out = OUT / f'TEST_levels_w{scale:g}'; out.mkdir(exist_ok=False); rows = []
    from pcontrol.research.evaluate_natural_diffusion import model_features
    for ci, item in enumerate(queue['cases']):
        a = c.arrays(item['artifact']); n = item['num_agents']; ego = int(np.flatnonzero(a['ego_mask'])[0])
        case = {k: a[k] for k in ('history', 'dimensions', 'road_boundaries', 'ego_mask', 'agent_mask')}; case['num_agents'] = n
        f = model_features(case, hn, torch.device('cpu'))
        for z in range(3):
            arrays = {}
            for level, pet_target in LEVELS.items():
                r = float(v1.risk_level(pet_target))
                noise = torch.tensor(a[f'initial_noise_{z}'][None], dtype=torch.float32)
                with torch.no_grad():
                    coef = cfg_sample(model, schedule, f, torch.tensor([r], dtype=torch.float32), noise, scale=scale, steps=50)[0].numpy()
                fut = basis.decode(coef.astype(np.float64) * np.array(cn['scale']) + np.array(cn['mean']), case['history'][-1])
                assert np.isfinite(fut).all() and np.array_equal(fut[0], case['history'][-1])
                o = scene_occupancy_pet(fut, np.ones((175, n), bool), case['dimensions'], times=basis.times, ego_index=ego,
                                        sample_period=.04, window=(0., 6.96), cap_seconds=4.)
                arrays[f'level_{level}'] = fut
                rows.append(dict(case_index=ci, scene_id=item['scene_id'], stratum=item['stratum'], noise_index=z, level=level,
                                 target_PET=pet_target, risk_condition=r, pet_seconds=float(o['pet_value_seconds'])))
            np.savez_compressed(out / f'case_{ci:03d}_z{z}.npz', **arrays)
    summary = {lv: dict(median_PET=float(np.median([r['pet_seconds'] for r in rows if r['level'] == lv])),
                        PET_below_1s=float(np.mean([r['pet_seconds'] < 1 for r in rows if r['level'] == lv]))) for lv in LEVELS}
    io.write_json(out / 'result.json', dict(status='complete', CFG_scale=scale, levels=LEVELS, rows=rows, summary=summary,
                                            code=c.source_bindings(('pcontrol/research/rade_baseline_v2.py',))))
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train'); t.add_argument('--device', default='cuda:0')
    sub.add_parser('devcheck')
    e = sub.add_parser('evaluate'); e.add_argument('--scale', type=float, required=True)
    l = sub.add_parser('levels'); l.add_argument('--scale', type=float, required=True)
    a = ap.parse_args()
    {'train': lambda: train(a.device), 'devcheck': devcheck, 'evaluate': lambda: evaluate(a.scale), 'levels': lambda: levels(a.scale)}[a.cmd]()
