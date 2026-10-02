#!/usr/bin/env python3
"""Summarize generated futures with the metrics of the paper.

A generated future is placed in the reference distribution of its own history. Its minimum PET y has the compatible
percentile interval [p_low, p_up]: a single point at continuous outcomes and an interval at the point masses (PET = 0,
the 4 s cap). The interval error of a request p is the distance from p to that interval.

    Fine       share of requests with interval error <= 0.05
    P-MAE      mean interval error
    PET-MAE    mean |y - q_H(p)| (seconds)
    BG / Ego   share of futures with background-background / ego-SV footprint overlap
    Road       share of futures with a strict road violation (any exceedance of the outer road boundary)

Usage:
    python pcontrol/tools/evaluate.py --rows outputs/generation/rows.jsonl
"""
import argparse
from collections import defaultdict
import json

import numpy as np


def summarize(rows, tolerance=.05):
    good = [r for r in rows if r['status'] == 'complete']
    n = len(rows)
    error = np.array([r['p_interval_error'] for r in good])
    # A request without a complete future counts as not realized within the tolerance.
    return dict(requests=n, failed=n - len(good), Fine=float(np.sum(error <= tolerance) / n) if n else None,
                P_MAE=float(error.mean()) if len(good) == n and n else None,
                PET_MAE=float(np.mean([r['canonical_PET_target_absolute_error_seconds'] for r in good])) if good else None,
                BG=float(np.mean([r['background_PL_overlap_scene'] for r in good])) if good else None,
                Ego=float(np.mean([r['ego_PL_overlap_scene'] for r in good])) if good else None,
                Road=float(np.mean([r['quality']['road_outside_scene'] for r in good])) if good else None)


def table(groups):
    head = f'{"":12s} {"requests":>8s} {"Fine (%)":>9s} {"P-MAE":>8s} {"PET-MAE":>8s} {"BG (%)":>7s} {"Ego (%)":>7s} {"Road (%)":>8s}'
    lines = [head]
    cell = lambda v, width, fmt, scale=1: f'{scale * v:{width}{fmt}}' if v is not None else f'{"n/a":>{width}s}'
    for name, s in groups:
        lines.append(f'{name:12s} {s["requests"]:8d} {cell(s["Fine"], 9, ".2f", 100)} {cell(s["P_MAE"], 8, ".5f")} '
                     f'{cell(s["PET_MAE"], 8, ".5f")} {cell(s["BG"], 7, ".2f", 100)} {cell(s["Ego"], 7, ".2f", 100)} '
                     f'{cell(s["Road"], 8, ".2f", 100)}')
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--rows', nargs='+', required=True, help='rows.jsonl files written by pcontrol/tools/generate.py')
    parser.add_argument('--tolerance', type=float, default=.05)
    parser.add_argument('--json', help='also write the summary to this file')
    args = parser.parse_args()
    rows = [json.loads(line) for f in args.rows for line in open(f) if line.strip()]
    by_p, by_n = defaultdict(list), defaultdict(list)
    for r in rows:
        by_p[r['requested_p']].append(r)
        by_n[r.get('stratum', '')].append(r)
    groups = [('all', summarize(rows, args.tolerance))]
    groups += [(f'p = {p:g}', summarize(v, args.tolerance)) for p, v in sorted(by_p.items())]
    groups += [(f'{k}', summarize(v, args.tolerance)) for k, v in sorted(by_n.items()) if k]
    print(table(groups))
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(dict(groups), f, indent=2)


if __name__ == '__main__':
    main()
