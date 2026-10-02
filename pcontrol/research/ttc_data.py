#!/usr/bin/env python3
"""Pure car-following clips with a minimum time-to-collision label.

A clip is pure car-following when, at t0, the ego has a leader directly ahead in its
lane, the ego and that leader keep their lanes over the 13 history frames and the 175
future frames, and no other vehicle is ever between them in that lane. Lanes come from
the carriageway lane markings in the canonical ego frame (x forward, y left).

TTC(t) = gap(t) / (v_ego(t) - v_leader(t)) while the ego closes in (bumper-to-bumper gap
along x). The label is the minimum over the 6.96 s window, 0 on overlap, and it is
capped at TTC_CAP. The CDF model works on [0, 4], so the model target is
4 * min(TTC, TTC_CAP) / TTC_CAP.

Sources: FIT complete scenes (complete_scene_expanded_v1_20260910/FIT) for training and
early stopping, and the final TEST complete scenes for evaluation and generation.
"""
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c

NP = ROOT / 'outputs/natural_percentile'
FIT_DIR = NP / 'complete_scene_expanded_v1_20260910/FIT'
TEST_DATASET = NP / 'transformer_publication_v1_20260916/final_validation_v1/TEST/dataset.json'
OUT = NP / 'experiments_v1/ttc_data'
TTC_CAP = 10.


def lanes(y, bounds):
    return np.searchsorted(bounds, y) - 1


def min_ttc(future, dims, ego, lead):
    gap = future[:, lead, 0] - future[:, ego, 0] - .5 * (dims[ego, 0] + dims[lead, 0])
    closing = future[:, ego, 2] - future[:, lead, 2]
    if np.any(gap <= 0): return 0.
    ttc = np.where(closing > 0, gap / np.where(closing > 0, closing, 1.), np.inf)
    return float(ttc.min())


def car_following(history, future, dims, bounds, ego):
    """Leader index when the clip is pure car-following, else None."""
    le0 = lanes(history[-1, ego, 1], bounds)
    if not 0 <= le0 < len(bounds) - 1: return None
    ahead = [j for j in range(history.shape[1]) if j != ego and lanes(history[-1, j, 1], bounds) == le0
             and history[-1, j, 0] > history[-1, ego, 0]]
    if not ahead: return None
    lead = min(ahead, key=lambda j: history[-1, j, 0])
    for track in (history, future):
        if np.any(lanes(track[:, ego, 1], bounds) != le0) or np.any(lanes(track[:, lead, 1], bounds) != le0):
            return None
    for j in range(future.shape[1]):
        if j in (ego, lead): continue
        inside = (lanes(future[:, j, 1], bounds) == le0) & (future[:, j, 0] > future[:, ego, 0]) & (future[:, j, 0] < future[:, lead, 0])
        if inside.any(): return None
    gap0 = history[-1, lead, 0] - history[-1, ego, 0] - .5 * (dims[ego, 0] + dims[lead, 0])
    return lead if gap0 > 0 else None


def scan(args):
    role, path = args
    out = []
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files if k != 'metadata_json'}
    has_mask = 'carriageway_boundary_mask' in d
    for i, sid in enumerate(d['scene_id'].tolist()):
        if not (d['complete_horizon'][i] and d['point_identified'][i]): continue
        lo, hi = d['offsets'][i], d['offsets'][i + 1]
        ids = d['agent_ids'][lo:hi]; ego = int(np.flatnonzero(ids == d['ego_id'][i])[0])
        history = np.transpose(d['history_agents'][lo:hi], (1, 0, 2))
        future = np.transpose(d['future_native_agents'][lo:hi], (1, 0, 2))
        if not np.transpose(d['future_observed_mask_agents'][lo:hi]).all(): continue
        dims = d['dimensions_agents'][lo:hi]
        bounds = d['carriageway_boundaries'][i]
        bounds = np.sort(bounds[d['carriageway_boundary_mask'][i]] if has_mask else np.unique(bounds))
        if len(bounds) < 2 or history.shape[1] < 3: continue
        lead = car_following(history, future, dims, bounds, ego)
        if lead is None: continue
        ttc = min_ttc(future, dims, ego, lead)
        out.append(dict(role=role, scene_id=sid, recording_id=str(d['recording_id'][i]), num_agents=int(history.shape[1]),
                        ego=ego, leader=int(lead), ttc_seconds=ttc, history=history, future=future, dimensions=dims, road_boundaries=bounds))
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [('FIT', p) for p in sorted(FIT_DIR.glob('*.npz'))]
    test = json.loads(TEST_DATASET.read_text())
    jobs += [('TEST', Path(e['data']['path'])) for e in test['recordings'].values()]
    with Pool(min(48, len(jobs))) as pool:
        clips = [r for rows in pool.map(scan, jobs) for r in rows]
    summary = {}
    for role in ('FIT', 'TEST'):
        rows = [r for r in clips if r['role'] == role]
        ttc = np.array([r['ttc_seconds'] for r in rows])
        summary[role] = dict(clips=len(rows), recordings=len({r['recording_id'] for r in rows}),
            TTC_quantiles={str(q): float(np.quantile(np.minimum(ttc, 1e6), q)) for q in (.05, .1, .25, .5, .75, .9)},
            share_no_closing=float(np.isinf(ttc).mean()), share_ttc_le_cap=float((ttc <= TTC_CAP).mean()),
            share_ttc_le_4=float((ttc <= 4).mean()), share_zero=float((ttc == 0).mean()),
            by_N={s: int(sum(1 for r in rows if (r['num_agents'] <= 5 if s == 'N3_5' else 6 <= r['num_agents'] <= 8 if s == 'N6_8' else r['num_agents'] >= 9)))
                  for s in ('N3_5', 'N6_8', 'N9_plus')})
        np.savez_compressed(OUT / f'{role}.npz', scene_id=np.array([r['scene_id'] for r in rows]),
            recording_id=np.array([r['recording_id'] for r in rows]), num_agents=np.array([r['num_agents'] for r in rows]),
            ego=np.array([r['ego'] for r in rows]), leader=np.array([r['leader'] for r in rows]),
            ttc_seconds=ttc, offsets=np.concatenate([[0], np.cumsum([r['num_agents'] for r in rows])]),
            history=np.concatenate([np.transpose(r['history'], (1, 0, 2)) for r in rows]),
            future=np.concatenate([np.transpose(r['future'], (1, 0, 2)) for r in rows]),
            dimensions=np.concatenate([r['dimensions'] for r in rows]),
            road_offsets=np.concatenate([[0], np.cumsum([len(r['road_boundaries']) for r in rows])]),
            road_boundaries=np.concatenate([r['road_boundaries'] for r in rows]))
    summary.update(TTC_cap_seconds=TTC_CAP, code=c.bind(Path(__file__)))
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=1))


if __name__ == '__main__':
    main()
