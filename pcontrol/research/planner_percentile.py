#!/usr/bin/env python3
"""Does the percentile request reach the planner?

Checks whether the planner test received the request p and realized it.
  1. The planner test runs on the generated futures of the main experiment (generation/canonical), whose
     realized percentile is the stored estimated_rank of the evidence rows; the share within 0.05 of the
     request is reported per request.
  2. With the IDM and MOBIL planner in place of the ego, the scene the planner meets is scored on the same
     scale: its minimum PET (stored planner rows, calibrated style) is placed in the reference CDF of its own
     history with the rank function of the evaluation (scripts/audit_time_attention_full_pipeline.rank), which
     reproduces the stored ranks exactly. The center of the compatible interval summarizes each scene.
  3. Within each history and noise draw, the Spearman correlation between the request and this percentile.
Output: outputs/natural_percentile/experiments_v1/planner_analysis/percentile.json
"""
import json, sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.research.audit_time_attention_full_pipeline import rank
import pcontrol.research.planner_test as tm

ANALYSIS = ROOT / 'outputs/natural_percentile/experiments_v1/planner_analysis'
EVIDENCE = ROOT / 'outputs/paper/canonical_evidence_v1/canonical_evidence.json'
GRID = (.1, .3, .5, .7, .9)


def nearest(p, iv):
    return min(max(p, iv['p_low']), iv['p_up'])


def main():
    queue = c.json_file(c.bind(tm.FINAL / 'queue/queue.json'))
    arrays = [dict(c.arrays(it['artifact'])) for it in queue['cases']]
    rows = [r for r in json.loads((ANALYSIS / 'rows.json').read_text()) if r['variant'] == 'calibrated']
    ev = {(r['case_index'], r['noise_index'], r['requested_p']): r for r in json.loads(EVIDENCE.read_text())['models']['canonical_TEST']['rows']}
    # The rank function reproduces the stored ranks of the generated futures.
    for r in list(ev.values())[:50]:
        a = arrays[r['case_index']]
        got = rank(a['reference_joint_masses'], a['reference_row_nodes'], r['pet_seconds'])
        assert all(abs(got[k] - r['estimated_rank'][k]) < 1e-12 for k in ('p_low', 'p_up'))
    planner, observed = {}, []
    for r in rows:
        a = arrays[r['case_index']]
        iv = rank(a['reference_joint_masses'], a['reference_row_nodes'], min(r['pet'], 4.))
        if r['source'] == 'Our method':
            planner[(r['case_index'], r['noise_index'], r['requested_p'])] = iv
        elif r['source'] == 'Observed':
            observed.append(iv['p_mid'])
    by_p = {}
    for p in GRID:
        keys = [k for k in planner if k[2] == p]
        gen = np.array([nearest(p, ev[k]['estimated_rank']) for k in keys])
        mid = np.array([planner[k]['p_mid'] for k in keys])
        near = np.array([nearest(p, planner[k]) for k in keys])
        by_p[str(p)] = dict(scenarios=len(keys), generated_within_005=float(np.mean(np.abs(gen - p) <= .05 + 1e-12)),
                            planner_median_percentile=float(np.median(mid)), planner_mean_percentile=float(mid.mean()),
                            planner_IQR=[float(np.percentile(mid, 25)), float(np.percentile(mid, 75))],
                            planner_within_005=float(np.mean(np.abs(near - p) <= .05 + 1e-12)))
    rho = []
    for ci in range(len(arrays)):
        for z in range(3):
            v = [planner[(ci, z, p)]['p_mid'] for p in GRID]
            if len(set(np.round(v, 12))) > 1: rho.append(spearmanr(GRID, v).correlation)
    out = dict(status='pass', code=c.bind(Path(__file__)), rows=c.bind(ANALYSIS / 'rows.json'), by_request=by_p,
               observed=dict(scenarios=len(observed), planner_median_percentile=float(np.median(observed)),
                             planner_mean_percentile=float(np.mean(observed))),
               within_history=dict(pairs=len(rho), median_spearman=float(np.median(rho)), share_positive=float(np.mean(np.array(rho) > 0))))
    (ANALYSIS / 'percentile.json').write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(out['by_request'], indent=1)); print(out['observed'], out['within_history'])


if __name__ == '__main__':
    main()
