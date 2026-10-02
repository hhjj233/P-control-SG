#!/usr/bin/env python3
"""Downstream AV test: an IDM+MOBIL ego against generated SV behavior.

For every scenario the generated ego is discarded. An IDM+MOBIL vehicle starts from the
ego's observed state at t0 and drives for 6.96 s at 25 Hz, while the SVs replay their
trajectories without reacting (log-replay testing). IDM (Treiber et al., 2000) gives the
longitudinal acceleration and MOBIL (Kesting et al., 2007) decides lane changes; each
lane change moves the ego laterally with a smooth 4 s profile. The planner's outcome is
measured with the same minimum ego--SV occupancy PET as the paper, plus collisions,
minimum time to collision with the leader, hard braking and lane changes.

Scenario sources on the 96 TEST histories: our method (5 requested percentiles, 3 noise
draws), RADE (same requests) and the observed futures.
"""
import json, os, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.data.scene_pet import scene_occupancy_pet

NP = ROOT / 'outputs/natural_percentile'
FINAL = NP / 'transformer_publication_v1_20260916/final_validation_v1/TEST'
OUT = NP / 'experiments_v1/idm_mobil'
SOURCES = {'Our method': FINAL / 'generation/canonical'}
# RADE v2 at three fixed physical risk levels (target PET 0.5, 1.5 and 3.0 s), pcontrol/research/rade_baseline_v2.py.
RADE_LEVELS = NP / f"experiments_v1/rade_v2/TEST_levels_w{os.environ.get('RADE_V2_SCALE', '10')}"
LEVELS = ('safe', 'medium', 'dangerous')
GRID = (.1, .3, .5, .7, .9)
DT, STEPS = .04, 175
# IDM highway parameters (Treiber and Kesting) and MOBIL parameters (Kesting et al., 2007).
IDM = dict(T=1.5, s0=2., a=1.0, b=2.0, delta=4., a_min=-9.)
MOBIL = dict(politeness=.2, threshold=.2, b_safe=4.)
LC_SECONDS = 4.


def idm_accel(v, v0, gap, dv, T=None):
    """dv = v - v_leader (closing speed); gap is bumper to bumper; T defaults to IDM['T']."""
    free = 1. - (v / max(v0, .1)) ** IDM['delta']
    if gap is None:
        return max(IDM['a_min'], IDM['a'] * free)
    T = IDM['T'] if T is None else T
    s_star = IDM['s0'] + max(0., v * T + v * dv / (2. * np.sqrt(IDM['a'] * IDM['b'])))
    return float(max(IDM['a_min'], IDM['a'] * (free - (s_star / max(gap, .1)) ** 2)))


def lane_of(y, bounds):
    return int(np.searchsorted(bounds, y) - 1)


def in_lane(y, width, lane, bounds, margin=.3):
    """An SV occupies a lane when its lateral footprint overlaps the lane by more than `margin`."""
    lo, hi = bounds[lane], bounds[lane + 1]
    return min(hi, y + width / 2) - max(lo, y - width / 2) > margin


def neighbours(k, ex, lane, svs, dims_e, dims, bounds):
    """Nearest SV ahead and behind the ego in `lane` at frame k: (index, gap, speed) or None.
    Encroaching SVs count, so a cut-in is perceived once part of its body enters the lane."""
    ahead = behind = None
    for j in range(svs.shape[1]):
        if not in_lane(svs[k, j, 1], dims[j, 1], lane, bounds): continue
        dx = svs[k, j, 0] - ex
        gap = abs(dx) - .5 * (dims_e[0] + dims[j, 0])
        if dx >= 0 and (ahead is None or gap < ahead[1]): ahead = (j, gap, svs[k, j, 2])
        if dx < 0 and (behind is None or gap < behind[1]): behind = (j, gap, svs[k, j, 2])
    return ahead, behind


def simulate(history, future, dims, bounds, ego):
    others = [i for i in range(future.shape[1]) if i != ego]
    svs, dims_sv = future[:, others], dims[others]
    de = dims[ego]; lanes = len(bounds) - 1
    centres = .5 * (bounds[:-1] + bounds[1:])
    x, y, v = float(history[-1, ego, 0]), float(history[-1, ego, 1]), float(history[-1, ego, 2])
    v0 = float(np.max(np.linalg.norm(history[:, ego, 2:4], axis=-1)))
    lane = lane_of(y, bounds); lane = min(max(lane, 0), lanes - 1)
    # Desired time gap calibrated to the ego's observed time gap at t0 (within 0.6-1.5 s),
    # so the planner does not start with an artificial braking transient.
    lead0, _ = neighbours(0, x, lane, svs, de, dims_sv, bounds)
    T_ego = float(np.clip(lead0[1] / max(v, .1), .6, IDM['T'])) if lead0 and lead0[1] > 0 else IDM['T']
    ego_path = np.zeros((STEPS, 4)); ego_path[0] = history[-1, ego]
    collision_partner = None
    lc_start, lc_from, lc_to, lane_changes, collision_time, accels = None, y, y, 0, None, []
    for k in range(STEPS - 1):
        ahead, _ = neighbours(k, x, lane, svs, de, dims_sv, bounds)
        acc = idm_accel(v, v0, ahead[1] if ahead else None, v - ahead[2] if ahead else 0., T_ego)
        # MOBIL, evaluated every 0.4 s when no lane change is in progress.
        if lc_start is None and k % 10 == 0:
            best = None
            for target in (lane - 1, lane + 1):
                if not 0 <= target < lanes: continue
                new_ahead, new_behind = neighbours(k, x, target, svs, de, dims_sv, bounds)
                if (new_ahead and new_ahead[1] < 0) or (new_behind and new_behind[1] < 0): continue
                acc_new = idm_accel(v, v0, new_ahead[1] if new_ahead else None, v - new_ahead[2] if new_ahead else 0., T_ego)
                follower_penalty = 0.
                if new_behind is not None:
                    j, gap_b, vb = new_behind
                    after = idm_accel(vb, vb, gap_b, vb - v)  # new follower behind the ego
                    if after < -MOBIL['b_safe']: continue
                    before_gap = (new_ahead[1] + gap_b + de[0]) if new_ahead else None
                    before = idm_accel(vb, vb, before_gap, vb - new_ahead[2] if new_ahead else 0.)
                    follower_penalty = before - after
                gain = acc_new - acc - MOBIL['politeness'] * follower_penalty
                if gain > MOBIL['threshold'] and (best is None or gain > best[0]): best = (gain, target)
            if best is not None:
                lc_start, lc_from, lc_to, lane = k, y, float(centres[best[1]]), best[1]; lane_changes += 1
        v = max(0., v + acc * DT); x += v * DT; accels.append(acc)
        if lc_start is not None:
            u = min(1., (k + 1 - lc_start) * DT / LC_SECONDS)
            y_new = lc_from + (lc_to - lc_from) * (1 - np.cos(np.pi * u)) / 2
            vy = (y_new - y) / DT; y = y_new
            if u >= 1.: lc_start = None
        else:
            vy = 0.
        ego_path[k + 1] = (x, y, v, vy)
        if collision_time is None:
            sep = np.abs(svs[k + 1, :, :2] - np.array([x, y]))
            half = .5 * (de[None] + dims_sv)
            hit = np.flatnonzero(np.all(sep <= half, axis=1))
            if hit.size:
                collision_time = (k + 1) * DT; j = int(hit[0])
                collision_partner = dict(front=bool(svs[k + 1, j, 0] > x),
                                         partner_changed_lane=lane_of(svs[k + 1, j, 1], bounds) != lane_of(svs[0, j, 1], bounds))
    scene = future.copy(); scene[:, ego] = ego_path
    o = scene_occupancy_pet(scene, np.ones(scene.shape[:2], bool), dims, times=np.arange(STEPS) * DT, ego_index=int(ego),
                            sample_period=DT, window=(0., 6.96), cap_seconds=4.)
    ttc = []
    for k in range(STEPS):
        ahead, _ = neighbours(k, ego_path[k, 0], lane_of(ego_path[k, 1], bounds), svs, de, dims_sv, bounds)
        if ahead and ego_path[k, 2] > ahead[2]: ttc.append(max(ahead[1], 0.) / (ego_path[k, 2] - ahead[2]))
    return dict(collision=collision_time is not None, collision_time=collision_time, collision_partner=collision_partner,
                front_collision=bool(collision_partner and collision_partner['front']),
                rear_collision=bool(collision_partner and not collision_partner['front']), desired_time_gap=T_ego, pet=o['pet_value_seconds'],
                min_ttc=float(min(ttc)) if ttc else None, min_accel=float(min(accels)),
                hard_braking=bool(min(accels) < -4.), lane_changes=lane_changes)


def job(args):
    source, ci, z, p, path, key = args
    a = c_arrays[ci]
    future = a['future_observed'] if key is None else np.load(path, allow_pickle=False)[key]
    ego = int(np.flatnonzero(a['ego_mask'])[0])
    r = simulate(a['history'], np.asarray(future, dtype=np.float64), a['dimensions'], np.sort(a['road_boundaries']), ego)
    r.update(source=source, case_index=ci, noise_index=z, requested_p=p, num_agents=int(future.shape[1]))
    return r


def init(arrays):
    global c_arrays
    c_arrays = arrays


def summarize(rows):
    n = len(rows)
    pet = np.array([r['pet'] for r in rows if r['pet'] is not None])
    ttc = np.array([r['min_ttc'] for r in rows if r['min_ttc'] is not None])
    return dict(scenarios=n, collision_rate=sum(r['collision'] for r in rows) / n,
                front_collision_rate=sum(r['front_collision'] for r in rows) / n,
                rear_collision_rate=sum(r['rear_collision'] for r in rows) / n,
                PET_below_1s_rate=float((pet < 1.).mean()), PET_below_2s_rate=float((pet < 2.).mean()),
                median_PET=float(np.median(pet)), hard_braking_rate=sum(r['hard_braking'] for r in rows) / n,
                TTC_below_3s_rate=float((ttc < 3.).sum() / n), mean_lane_changes=sum(r['lane_changes'] for r in rows) / n,
                mean_min_accel=float(np.mean([r['min_accel'] for r in rows])))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    queue = c.json_file(c.bind(FINAL / 'queue/queue.json'))
    arrays = [dict(c.arrays(item['artifact'])) for item in queue['cases']]
    jobs = [('Observed', ci, 0, None, None, None) for ci in range(len(arrays))]
    for name, folder in SOURCES.items():
        for ci in range(len(arrays)):
            for z in range(3):
                for p in GRID:
                    jobs.append((name, ci, z, p, str(folder / f'case_{ci:03d}_z{z}.npz'), 'generated_p' + str(p).replace('.', '_')))
    for ci in range(len(arrays)):
        for z in range(3):
            for lv in LEVELS:
                jobs.append(('RADE', ci, z, lv, str(RADE_LEVELS / f'case_{ci:03d}_z{z}.npz'), 'level_' + lv))
    with Pool(48, initializer=init, initargs=(arrays,)) as pool:
        rows = pool.map(job, jobs, chunksize=8)
    strata = {ci: ('N3_5' if a['history'].shape[1] <= 5 else 'N6_8' if a['history'].shape[1] <= 8 else 'N9_plus') for ci, a in enumerate(arrays)}
    report = dict(parameters=dict(IDM=IDM, MOBIL=MOBIL, lane_change_seconds=LC_SECONDS, dt=DT, steps=STEPS),
                  observed=summarize([r for r in rows if r['source'] == 'Observed']), methods={})
    for name in SOURCES:
        mine = [r for r in rows if r['source'] == name]
        report['methods'][name] = dict(all=summarize(mine), by_p={str(p): summarize([r for r in mine if r['requested_p'] == p]) for p in GRID},
            by_p_and_N={f'{p}|{s}': summarize([r for r in mine if r['requested_p'] == p and strata[r['case_index']] == s])
                        for p in GRID for s in ('N3_5', 'N6_8', 'N9_plus')})
        print(name, json.dumps({p: {k: round(v, 3) for k, v in s.items() if k in ('collision_rate', 'front_collision_rate', 'rear_collision_rate', 'PET_below_1s_rate', 'median_PET', 'hard_braking_rate', 'TTC_below_3s_rate')}
                                for p, s in report['methods'][name]['by_p'].items()}), flush=True)
    rade = [r for r in rows if r['source'] == 'RADE']
    report['methods']['RADE'] = dict(all=summarize(rade), by_level={lv: summarize([r for r in rade if r['requested_p'] == lv]) for lv in LEVELS},
                                     levels_folder=str(RADE_LEVELS))
    print('RADE', json.dumps({lv: {k: round(v, 3) for k, v in s.items() if k in ('rear_collision_rate', 'front_collision_rate', 'PET_below_1s_rate', 'median_PET', 'hard_braking_rate')}
                              for lv, s in report['methods']['RADE']['by_level'].items()}), flush=True)
    print('Observed', json.dumps({k: round(v, 3) for k, v in report['observed'].items() if isinstance(v, float)}), flush=True)
    report['code'] = c.bind(Path(__file__))
    (OUT / 'idm_mobil.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUT / 'rows.json').write_text(json.dumps(rows) + '\n')


if __name__ == '__main__':
    main()
