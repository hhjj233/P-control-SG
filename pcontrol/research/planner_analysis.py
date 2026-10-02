#!/usr/bin/env python3
"""Detailed analysis of the IDM+MOBIL planner test (Section 5.9), from re-run rollouts.

The stored planner test (pcontrol/research/planner_test.py) kept only summary metrics. This script
re-runs the same simulate() for every scenario, captures the planner path where simulate() hands the
scene to the PET measurement, checks every metric against the stored rows, and adds:
  * the critical encounter: the SV that attains the planner's minimum PET, classified by lane relation,
    lane changes and order of passage (leader ahead, follower closing in, cut-in ahead, lane change
    behind, planner lane change, adjacent lane, or no encounter within the 4 s cap);
  * surrogate safety measures towards the leader and the follower in the planner's lane: time exposed
    and time integrated TTC below 3 s (TET, TIT) and the maximum deceleration rate to avoid a crash (DRAC);
  * time profiles of planner speed and acceleration, and of the front and rear TTC.
Planner variants (driving styles) can be added with --variants; the stored test is the 'calibrated' style.
"""
import argparse, json, sys
from multiprocessing import get_context
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c
import pcontrol.research.planner_test as tm

OUT = ROOT / 'outputs/natural_percentile/experiments_v1/planner_analysis'
GRID = (.1, .3, .5, .7, .9)
TTC_STAR = 3.  # TTC threshold of TET/TIT
# Driving styles. 'calibrated' is the stored test: desired time gap from the observed gap, within 0.6-1.5 s.
VARIANTS = {
    'calibrated': dict(IDM=dict(T=1.5, s0=2., a=1.0, b=2.0, delta=4., a_min=-9.), MOBIL=dict(politeness=.2, threshold=.2, b_safe=4.)),
    'cautious': dict(IDM=dict(T=2.0, s0=3., a=0.8, b=1.5, delta=4., a_min=-9.), MOBIL=dict(politeness=.5, threshold=.3, b_safe=3.), fixed_T=True),
    'assertive': dict(IDM=dict(T=1.0, s0=1.5, a=1.5, b=3.0, delta=4., a_min=-9.), MOBIL=dict(politeness=0., threshold=.1, b_safe=5.), fixed_T=True),
    # Styles that keep the calibrated time gap (no braking transient at the start) and change the rest of the driver.
    'defensive': dict(IDM=dict(T=1.5, s0=3., a=0.8, b=1.5, delta=4., a_min=-9.), MOBIL=dict(politeness=.5, threshold=.3, b_safe=3.)),
    'brisk': dict(IDM=dict(T=1.5, s0=1.5, a=1.5, b=3.0, delta=4., a_min=-9.), MOBIL=dict(politeness=0., threshold=.1, b_safe=5.)),
}
CATEGORIES = ('leader ahead', 'follower closing in', 'cut-in ahead', 'lane change behind', 'planner lane change', 'adjacent lane', 'no encounter')


def lane_of(y, bounds):
    return int(np.clip(np.searchsorted(bounds, y) - 1, 0, len(bounds) - 2))


def classify(scene, bounds, ego, w, pet):
    j = w.get('other_index')
    if j is None or pet is None or pet >= 4. - 1e-9: return 'no encounter'
    te, tw = w['ego_time_seconds'], w['other_time_seconds']
    ke, kw = min(int(round(te / tm.DT)), tm.STEPS - 1), min(int(round(tw / tm.DT)), tm.STEPS - 1)
    witness_lc = lane_of(scene[0, j, 1], bounds) != lane_of(scene[kw, j, 1], bounds)
    planner_lc = lane_of(scene[0, ego, 1], bounds) != lane_of(scene[ke, ego, 1], bounds)
    same0 = lane_of(scene[0, ego, 1], bounds) == lane_of(scene[0, j, 1], bounds)
    behind = tw > te  # the SV reaches the shared point after the planner
    if planner_lc and not witness_lc: return 'planner lane change'
    if witness_lc: return 'lane change behind' if behind else 'cut-in ahead'
    if same0: return 'follower closing in' if behind else 'leader ahead'
    return 'adjacent lane'


def surrogates(scene, dims, bounds, ego):
    """Front/rear TTC series of the planner within its lane, TET/TIT below TTC_STAR and the maximum DRAC."""
    T = scene.shape[0]; front, rear, drac = np.full(T, np.inf), np.full(T, np.inf), 0.
    others = [i for i in range(scene.shape[1]) if i != ego]
    for k in range(T):
        x, y, v = scene[k, ego, 0], scene[k, ego, 1], scene[k, ego, 2]
        lane = lane_of(y, bounds)
        for j in others:
            if not tm.in_lane(scene[k, j, 1], dims[j, 1], lane, bounds): continue
            dx = scene[k, j, 0] - x; gap = abs(dx) - .5 * (dims[ego, 0] + dims[j, 0])
            if gap <= 0: continue
            if dx > 0 and v > scene[k, j, 2]:
                front[k] = min(front[k], gap / (v - scene[k, j, 2])); drac = max(drac, (v - scene[k, j, 2]) ** 2 / (2 * gap))
            if dx < 0 and scene[k, j, 2] > v:
                rear[k] = min(rear[k], gap / (scene[k, j, 2] - v))
    def tet_tit(ttc):
        m = ttc < TTC_STAR
        return float(m.sum() * tm.DT), float(np.sum(TTC_STAR - ttc[m]) * tm.DT)
    (tetf, titf), (tetr, titr) = tet_tit(front), tet_tit(rear)
    return dict(TET_front=tetf, TIT_front=titf, TET_rear=tetr, TIT_rear=titr, DRAC_max=float(drac)), front, rear


class FixedGapNumpy:
    """numpy for the planner module only, with the desired-time-gap calibration replaced by a fixed gap."""
    def __init__(self, T): self.T = T
    def __getattr__(self, name): return getattr(np, name)
    def clip(self, v, lo, hi, *args, **kwargs):
        return float(self.T) if lo == .6 else np.clip(v, lo, hi, *args, **kwargs)


def job(args):
    variant, source, ci, z, p, path, key = args
    spec = VARIANTS[variant]
    tm.IDM.update(spec['IDM']); tm.MOBIL.update(spec['MOBIL'])
    a = ARRAYS[ci]
    future = a['future_observed'] if key is None else np.load(path, allow_pickle=False)[key]
    ego = int(np.flatnonzero(a['ego_mask'])[0]); bounds = np.sort(a['road_boundaries']); dims = a['dimensions']
    captured = {}
    original = tm.scene_occupancy_pet
    def capture(scene, *rest, **kw):
        captured['scene'] = np.array(scene, dtype=np.float64)
        out = original(scene, *rest, **kw); captured['pet'] = out
        return out
    tm.scene_occupancy_pet = capture
    if spec.get('fixed_T'):  # a fixed desired time gap: simulate() calibrates T with np.clip(gap/v, 0.6, T) only
        tm.np = FixedGapNumpy(spec['IDM']['T'])
    try:
        r = tm.simulate(a['history'], np.asarray(future, dtype=np.float64), dims, bounds, ego)
    finally:
        tm.scene_occupancy_pet = original
        tm.np = np
    scene = captured['scene']; w = captured['pet'].get('witness') or {}
    ssm, front, rear = surrogates(scene, dims, bounds, ego)
    v = scene[:, ego, 2]; acc = np.diff(v) / tm.DT
    r.update(variant=variant, source=source, case_index=ci, noise_index=z, requested_p=p, num_agents=int(scene.shape[1]),
             encounter=classify(scene, bounds, ego, w, r['pet']), witness=dict(other_index=w.get('other_index'),
             ego_time=w.get('ego_time_seconds'), other_time=w.get('other_time_seconds')), **ssm)
    profile = dict(speed=v.astype(np.float32), accel=np.concatenate([acc, acc[-1:]]).astype(np.float32),
                   front_ttc=np.minimum(front, 20.).astype(np.float32), rear_ttc=np.minimum(rear, 20.).astype(np.float32))
    return r, profile


def init(arrays):
    global ARRAYS
    ARRAYS = arrays


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--variants', default='calibrated'); args = ap.parse_args()
    variants = args.variants.split(',')
    OUT.mkdir(parents=True, exist_ok=True)
    queue = c.json_file(c.bind(tm.FINAL / 'queue/queue.json'))
    arrays = [dict(c.arrays(item['artifact'])) for item in queue['cases']]
    jobs = []
    for var in variants:
        jobs += [(var, 'Observed', ci, 0, None, None, None) for ci in range(len(arrays))]
        for ci in range(len(arrays)):
            for z in range(3):
                for p in GRID:
                    jobs.append((var, 'Our method', ci, z, p, str(tm.SOURCES['Our method'] / f'case_{ci:03d}_z{z}.npz'), 'generated_p' + str(p).replace('.', '_')))
                for lv in tm.LEVELS:
                    jobs.append((var, 'RADE', ci, z, lv, str(tm.RADE_LEVELS / f'case_{ci:03d}_z{z}.npz'), 'level_' + lv))
    with get_context('fork').Pool(48, initializer=init, initargs=(arrays,)) as pool:
        out = pool.map(job, jobs, chunksize=8)
    rows = [r for r, _ in out]
    stored = {(r['source'], r['case_index'], r['noise_index'], str(r['requested_p'])): r for r in json.loads((tm.OUT / 'rows.json').read_text())}
    checked = 0
    for r in rows:
        if r['variant'] != 'calibrated': continue
        s = stored[(r['source'], r['case_index'], r['noise_index'], str(r['requested_p']))]
        assert (r['pet'] is None) == (s['pet'] is None) and (r['pet'] is None or abs(r['pet'] - s['pet']) < 1e-9), (r['source'], r['case_index'])
        assert r['hard_braking'] == s['hard_braking'] and r['rear_collision'] == s['rear_collision'] and r['front_collision'] == s['front_collision']
        checked += 1
    keys = sorted({(r['variant'], r['source'], str(r['requested_p'])) for r in rows})
    prof = {}
    for var, src, p in keys:
        idx = [i for i, r in enumerate(rows) if (r['variant'], r['source'], str(r['requested_p'])) == (var, src, p)]
        for name in ('speed', 'accel', 'front_ttc', 'rear_ttc'):
            arr = np.stack([out[i][1][name] for i in idx])
            prof[f'{var}|{src}|{p}|{name}'] = np.percentile(arr, [25, 50, 75], 0).astype(np.float32)
            if name.endswith('ttc'):  # share of scenarios below the TET threshold at each time
                prof[f'{var}|{src}|{p}|{name}_below'] = (arr < TTC_STAR).mean(0).astype(np.float32)
        acc = np.stack([out[i][1]['accel'] for i in idx])
        prof[f'{var}|{src}|{p}|hard_braking_so_far'] = (np.minimum.accumulate(acc, 1) < -4.).mean(0).astype(np.float32)
    np.savez_compressed(OUT / 'profiles.npz', **prof)
    (OUT / 'rows.json').write_text(json.dumps(rows) + '\n')
    summary = {}
    for var, src, p in keys:
        rr = [r for r in rows if (r['variant'], r['source'], str(r['requested_p'])) == (var, src, p)]
        n = len(rr); pet = np.array([r['pet'] for r in rr if r['pet'] is not None])
        summary[f'{var}|{src}|{p}'] = dict(
            scenarios=n, hard_braking_rate=sum(r['hard_braking'] for r in rr) / n, rear_collision_rate=sum(r['rear_collision'] for r in rr) / n,
            front_collision_rate=sum(r['front_collision'] for r in rr) / n, PET_below_1s_rate=float((pet < 1.).mean()), median_PET=float(np.median(pet)),
            encounter_share={k: sum(r['encounter'] == k for r in rr) / n for k in CATEGORIES},
            **{f'mean_{m}': float(np.mean([r[m] for r in rr])) for m in ('TET_front', 'TIT_front', 'TET_rear', 'TIT_rear', 'DRAC_max')})
    (OUT / 'summary.json').write_text(json.dumps(dict(variants={k: VARIANTS[k] for k in variants}, TTC_threshold_TET=TTC_STAR,
                                                       checked_against_stored_rows=checked, summary=summary, code=c.bind(Path(__file__))), indent=2) + '\n')
    for k, s in summary.items():
        print(k, json.dumps({m: round(v, 3) for m, v in s.items() if isinstance(v, float)}), json.dumps({e: round(v, 2) for e, v in s['encounter_share'].items() if v}), flush=True)
    print('checked against stored rows:', checked)


if __name__ == '__main__':
    main()
