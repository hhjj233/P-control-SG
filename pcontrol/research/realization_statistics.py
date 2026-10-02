#!/usr/bin/env python3
"""Uncertainty and tolerance analysis of percentile realization.

Per-request interval errors come from the same scoring as Table 2. Requests of one
history are correlated, so confidence intervals use a cluster bootstrap over the 96
TEST histories (2,000 resamples). Paired differences against our method resample the
same histories for both methods. Tolerance curves report Fine at 0.01, 0.02, 0.05, 0.10.
"""
import json, os, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c

EVIDENCE = ROOT / 'outputs/paper/canonical_evidence_v1/canonical_evidence.json'
SCORED = ROOT / 'outputs/natural_percentile/experiments_v1/scored_baselines.json'
ABLATION = ROOT / 'outputs/natural_percentile/canonical_p_ablation_v1_20260925/comparison.json'
LEDGER = Path('external_baselines/outputs/retention_ledger.json')
OUT = ROOT / 'outputs/natural_percentile/experiments_v1/statistics.json'
REQUESTS = (.1, .3, .5, .7, .9)
TOLERANCES = (.01, .02, .05, .10)
B, SEED = 2000, 20260929


def rows_errors(rows):
    """(scene_id, error) pairs from scored request rows."""
    return [(r['scene_id'], float(r['p_interval_error'])) for r in rows]


def external_errors(rows):
    out = []
    for r in rows:
        lo, up = r['rank_interval']['p_low'], r['rank_interval']['p_up']
        out.extend((r['scene_id'], max(lo - p, p - up, 0.)) for p in REQUESTS)
    return out


def collect():
    ev = json.loads(EVIDENCE.read_text()); sc = json.loads(SCORED.read_text())
    methods = {'Our method': rows_errors(ev['models']['canonical_TEST']['rows']),
               'P-CVAE': rows_errors(ev['models']['pcvae']['rows']),
               'Standard P-diffusion': rows_errors(ev['models']['p_diffusion']['rows']),
               'RADE': rows_errors(sc['models']['rade_v2_w' + os.environ.get('RADE_V2_SCALE', '10')]['rows'])}
    ab = json.loads(ABLATION.read_text())
    for arm, label in (('condition_only', 'Condition only'), ('guidance_only', 'Guidance only'), ('no_p', 'No P path')):
        src = json.loads(Path(ab['results'][arm]['source']['path']).read_text())
        rows = src['rows']
        if arm == 'no_p':  # one unconditioned output per history and noise draw, scored against every request
            methods[label] = [(r['scene_id'], float(e)) for r in rows for e in r['interval_errors_by_request'].values()] \
                if rows and 'interval_errors_by_request' in rows[0] else rows_errors(rows)
        else:
            methods[label] = rows_errors(rows)
    ledger = json.loads(LEDGER.read_text())
    for arm, entry in ledger['retained'].items():
        methods[entry['label']] = external_errors(json.loads(Path(entry['score']['path']).read_text())['rows'])
    return methods


def by_history(pairs):
    groups = {}
    for sid, e in pairs: groups.setdefault(sid, []).append(e)
    return {k: np.array(v) for k, v in groups.items()}


def stats(groups, histories, idx):
    errs = np.concatenate([groups[histories[i]] for i in idx])
    return float((errs <= .05).mean()), float(errs.mean())


def main():
    methods = collect()
    grouped = {m: by_history(p) for m, p in methods.items()}
    histories = sorted(grouped['Our method'])
    for m, g in grouped.items():
        assert set(g) == set(histories), (m, len(g))
    rng = np.random.default_rng(SEED)
    boot = rng.integers(0, len(histories), size=(B, len(histories)))
    full = np.arange(len(histories))
    report = dict(bootstrap=dict(resamples=B, cluster='TEST history', seed=SEED), methods={}, paired_vs_ours={})
    samples = {}
    for m, g in grouped.items():
        fine, mae = stats(g, histories, full)
        s = np.array([stats(g, histories, idx) for idx in boot]); samples[m] = s
        errs = np.concatenate(list(g.values()))
        report['methods'][m] = dict(requests=int(errs.size), Fine=fine, P_MAE=mae,
            Fine_CI95=[float(np.percentile(s[:, 0], 2.5)), float(np.percentile(s[:, 0], 97.5))],
            P_MAE_CI95=[float(np.percentile(s[:, 1], 2.5)), float(np.percentile(s[:, 1], 97.5))],
            Fine_at_tolerance={str(t): float((errs <= t).mean()) for t in TOLERANCES})
    ours = samples['Our method']
    for m, s in samples.items():
        if m == 'Our method': continue
        d = ours - s
        report['paired_vs_ours'][m] = dict(
            Fine_difference=report['methods']['Our method']['Fine'] - report['methods'][m]['Fine'],
            Fine_difference_CI95=[float(np.percentile(d[:, 0], 2.5)), float(np.percentile(d[:, 0], 97.5))],
            P_MAE_difference=report['methods']['Our method']['P_MAE'] - report['methods'][m]['P_MAE'],
            P_MAE_difference_CI95=[float(np.percentile(d[:, 1], 2.5)), float(np.percentile(d[:, 1], 97.5))],
            share_of_resamples_ours_better_Fine=float((d[:, 0] > 0).mean()))
    report['code'] = c.bind(Path(__file__))
    OUT.write_text(json.dumps(report, indent=2) + '\n')
    for m, s in report['methods'].items():
        print(f"{m:28s} n={s['requests']:5d} Fine={100*s['Fine']:6.2f} [{100*s['Fine_CI95'][0]:.2f},{100*s['Fine_CI95'][1]:.2f}] "
              f"P-MAE={s['P_MAE']:.4f} [{s['P_MAE_CI95'][0]:.4f},{s['P_MAE_CI95'][1]:.4f}] tol={ {k: round(100*v, 1) for k, v in s['Fine_at_tolerance'].items()} }")


if __name__ == '__main__':
    main()
