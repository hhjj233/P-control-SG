#!/usr/bin/env python3
"""How TTC adversity arises in the generated car-following futures.

For every future of the TTC TEST run (96 histories, three noise draws), this records the largest
speed drop of the ego and of its leader within the window (speed channel, relative to the start
of the future) and the time at which the minimum TTC occurs. Medians are reported per request,
for the unguided generator and for the observed futures of the same histories.
"""
import json, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c

REV = ROOT / 'outputs/natural_percentile/experiments_v1'
RUN = REV / 'ttc_generation_cap20_test_inverse_soft_L30_I3_S1.0_C1.0_B200_bounded_M0.002'
OUT = REV / 'ttc_behavior_cap20/ttc_behavior.json'
GRID, DT = (.1, .3, .5, .7, .9), .04


def describe(f, ego, lead, dims):
    f = np.asarray(f, dtype=np.float64)
    gap = f[:, lead, 0] - f[:, ego, 0] - .5 * (dims[ego, 0] + dims[lead, 0]); closing = f[:, ego, 2] - f[:, lead, 2]
    ttc = np.where(closing > 0, gap / np.where(closing > 0, closing, 1.), np.inf)
    return dict(ego_largest_speed_drop=float(f[0, ego, 2] - f[:, ego, 2].min()),
                leader_largest_speed_drop=float(f[0, lead, 2] - f[:, lead, 2].min()),
                time_of_minimum_TTC=float(np.argmin(ttc) * DT))


def medians(items):
    return {k: float(np.median([i[k] for i in items])) for k in items[0]} | dict(futures=len(items))


def main():
    rows = json.loads((RUN / 'rows.json').read_text())
    z = dict(np.load(REV / 'ttc_data/TEST.npz', allow_pickle=False)); index = {str(s): i for i, s in enumerate(z['scene_id'])}
    cases = sorted({(r['case_index'], r['scene_id']) for r in rows})
    guided, unguided, observed = {p: [] for p in GRID}, [], []
    for ci, sid in cases:
        i = index[sid]; lo, hi = z['offsets'][i], z['offsets'][i + 1]
        dims, ego, lead = z['dimensions'][lo:hi], int(z['ego'][i]), int(z['leader'][i])
        with np.load(RUN / f'case_{ci:03d}.npz', allow_pickle=False) as d:
            for zi in range(3):
                for p in GRID: guided[p].append(describe(d[f'TTC_guidance_z{zi}_p{p}'], ego, lead, dims))
                unguided.append(describe(d[f'No_guidance_z{zi}'], ego, lead, dims))
        observed.append(describe(np.transpose(z['future'][lo:hi], (1, 0, 2)), ego, lead, dims))
    report = dict(status='complete', histories=len(cases), run=c.bind(RUN / 'result.json'),
                  guided_by_p={str(p): medians(v) for p, v in guided.items()}, unguided=medians(unguided),
                  observed=medians(observed), code=c.bind(Path(__file__)))
    OUT.parent.mkdir(parents=True, exist_ok=True); OUT.write_text(json.dumps(report, indent=2) + '\n')
    for name, m in [('observed', report['observed']), ('unguided', report['unguided'])] + [(f'p={p}', v) for p, v in report['guided_by_p'].items()]:
        print(f"{name:10s}", {k: round(v, 2) for k, v in m.items()})


if __name__ == '__main__':
    main()
