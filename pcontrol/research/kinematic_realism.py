#!/usr/bin/env python3
"""Kinematic realism of generated futures on the 96 TEST histories.

All methods are measured the same way, from positions only: positions are sampled
every 5 native frames (0.2 s), and speed, longitudinal/lateral acceleration and
longitudinal jerk are finite differences on that grid. This does not rely on each
method's own velocity channel and smooths the observed highD futures and the
generated futures alike. Each distribution pools all vehicles and times and is
compared with the observed futures of the same histories (Wasserstein-1 distance
and tail rates).
"""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import wasserstein_distance
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c

NP = ROOT / 'outputs/natural_percentile'
FINAL = NP / 'transformer_publication_v1_20260916/final_validation_v1/TEST'
SUPP = NP / 'baselines_v1'
OUT = NP / 'experiments_v1/kinematic_realism'
GRID = (.1, .3, .5, .7, .9)
EXTERNAL = {'TrafficGen': Path('external_baselines/outputs/trafficgen/analysis/result.json'),
            'CTG++': NP / 'external_no_p_v1_20260922/ctgpp/analysis/result.json',
            'STRIVE': NP / 'legacy_external_no_p_v1_20260922/strive/analysis/result.json'}
REQUEST_SOURCES = {'Our method': FINAL / 'generation/canonical',
                   'P-CVAE': SUPP / 'pcvae/TEST', 'Standard P-diffusion': SUPP / 'p_diffusion/TEST',
                   'RADE': NP / f"experiments_v1/rade_v2/TEST_w{os.environ.get('RADE_V2_SCALE', '10')}"}
STEP, DT = 5, .2
# Tail thresholds: harsh braking (-0.4 g), strong lateral acceleration, implausible total acceleration (> 0.8 g).
TAILS = {'a_long_lt_-4': ('a_long', lambda x: x < -4.), 'abs_a_lat_gt_2': ('a_lat', lambda x: np.abs(x) > 2.),
         'abs_jerk_gt_10': ('jerk_long', lambda x: np.abs(x) > 10.), 'abs_a_gt_8': ('a_norm', lambda x: x > 8.)}


def kinematics(future, ego=None):
    f = np.asarray(future, dtype=np.float64)
    assert f.ndim == 3 and f.shape[0] == 175 and f.shape[2] == 4 and np.isfinite(f).all()
    if ego is not None: f = f[:, ego:ego + 1]
    pos = f[::STEP, :, :2]
    v = np.diff(pos, axis=0) / DT
    a = np.diff(v, axis=0) / DT
    return dict(speed=np.linalg.norm(v, axis=-1).ravel(), a_long=a[..., 0].ravel(), a_lat=a[..., 1].ravel(),
                a_norm=np.linalg.norm(a, axis=-1).ravel(), jerk_long=(np.diff(a[..., 0], axis=0) / DT).ravel())


def pooled(items):
    keys = ('speed', 'a_long', 'a_lat', 'a_norm', 'jerk_long')
    return {k: np.concatenate([i[k] for i in items]) for k in keys}


def compare(sample, observed):
    out = {}
    for k in ('speed', 'a_long', 'a_lat', 'jerk_long'):
        out['W1_' + k] = float(wasserstein_distance(sample[k], observed[k]))
        out['p01_' + k], out['p50_' + k], out['p99_' + k] = map(float, np.percentile(sample[k], [1, 50, 99]))
    for name, (k, rule) in TAILS.items():
        out['rate_' + name] = float(rule(sample[k]).mean())
    out['samples'] = int(len(sample['speed']))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queue = c.json_file(c.bind(FINAL / 'queue/queue.json'))
    cases = [c.arrays(item['artifact']) for item in queue['cases']]
    sid_index = {item['scene_id']: i for i, item in enumerate(queue['cases'])}
    egos = [int(np.flatnonzero(a['ego_mask'])[0]) for a in cases]
    obs_all = pooled([kinematics(a['future_observed']) for a in cases])
    obs_ego = pooled([kinematics(a['future_observed'], e) for a, e in zip(cases, egos)])
    report = dict(observed=dict(all=compare(obs_all, obs_all), ego=compare(obs_ego, obs_ego)), methods={},
                  grid_seconds=DT, tails={k: v[0] for k, v in TAILS.items()})
    for name, folder in REQUEST_SOURCES.items():
        items, ego_items, by_p = [], [], {p: [] for p in GRID}
        for ci, a in enumerate(cases):
            for z in range(3):
                with np.load(folder / f'case_{ci:03d}_z{z}.npz', allow_pickle=False) as d:
                    for p in GRID:
                        f = d['generated_p' + str(p).replace('.', '_')]
                        assert f.shape == a['future_observed'].shape and np.array_equal(f[0], a['history'][-1])
                        k = kinematics(f); items.append(k); by_p[p].append(k); ego_items.append(kinematics(f, egos[ci]))
        assert len(items) == 1440, (name, len(items))
        report['methods'][name] = dict(all=compare(pooled(items), obs_all), ego=compare(pooled(ego_items), obs_ego),
                                       by_p={str(p): compare(pooled(v), obs_all) for p, v in by_p.items()}, futures=len(items))
        print(name, json.dumps({k: round(v, 3) for k, v in report['methods'][name]['all'].items() if k.startswith(('W1', 'rate'))}), flush=True)
    for name, path in EXTERNAL.items():
        rows = json.loads(path.read_text())['rows']; items, ego_items, cache = [], [], {}
        for r in rows:
            ci = sid_index[r['scene_id']]
            art = r['artifact']['path']
            if art not in cache: cache = {art: dict(np.load(art, allow_pickle=False))}
            arr = cache[art][r['array_key']]
            f = arr[r['array_index']] if arr.ndim == 4 else arr
            assert f.shape == cases[ci]['future_observed'].shape, (name, f.shape)
            items.append(kinematics(f)); ego_items.append(kinematics(f, egos[ci]))
        report['methods'][name] = dict(all=compare(pooled(items), obs_all), ego=compare(pooled(ego_items), obs_ego), futures=len(items))
        print(name, json.dumps({k: round(v, 3) for k, v in report['methods'][name]['all'].items() if k.startswith(('W1', 'rate'))}), flush=True)
    report['observed_futures'] = len(cases)
    report['code'] = c.bind(Path(__file__))
    (OUT / 'kinematic_realism.json').write_text(json.dumps(report, indent=2) + '\n')
    o = report['observed']['all']
    print('Observed', json.dumps({k: round(v, 3) for k, v in o.items() if k.startswith('rate')}))


if __name__ == '__main__':
    main()
