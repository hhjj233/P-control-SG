#!/usr/bin/env python3
"""Generate futures at requested risk percentiles with the released reference and generator.

Each request p is a risk percentile in (0, 1) of the history's own distribution of the minimum ego--SV PET:
p = 0.9 asks for a future more critical than 90% of the natural futures of that history. The reference maps p to the
physical PET target q_H(p), and the generator samples one joint future of all vehicles per request and noise draw.

Usage:
    # the paper's evaluation: 32 histories per vehicle-count group, five requests, the paper's three noise draws
    python pcontrol/tools/generate.py --scenes data/scenes --paper-histories --out outputs/generation
    # a few scenes with your own requests and random noise
    python pcontrol/tools/generate.py --scenes data/scenes/35 --limit 5 --requests 0.2 0.8 --noise random --draws 2 --out outputs/demo

Writes <out>/rows.jsonl (one record per request; summarize it with pcontrol/tools/evaluate.py) and
<out>/futures/<scene_id>_z<draw>.npz with one (175, N, 4) array per request (x, y, vx, vy at 25 Hz, ego first).
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pcontrol.api import (PAPER_REQUESTS, attach_observation_diagnostics, load_generator, load_reference,  # noqa: E402
                          load_scenes, paper_noise, random_noise)
from pcontrol.publication_pipeline.final_validation_protocol import select_histories  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
PAPER_HISTORY_SALT = 'final_natural_validation_H_20260916_v1'


def scene_files(paths):
    files = []
    for p in map(Path, paths):
        files += sorted(p.rglob('complete_scenes.npz')) if p.is_dir() else [p]
    return files


def json_safe(value):
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, dict): return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [json_safe(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--scenes', nargs='+', required=True, help='complete_scenes.npz files or folders containing them')
    parser.add_argument('--out', required=True, help='output folder')
    parser.add_argument('--reference', default=str(ROOT / 'checkpoints/reference.pt'))
    parser.add_argument('--generator', default=str(ROOT / 'checkpoints/generator.pt'))
    parser.add_argument('--requests', nargs='+', type=float, default=list(PAPER_REQUESTS), help='risk percentiles in (0, 1)')
    parser.add_argument('--noise', choices=('paper', 'random'), default='paper',
                        help="'paper': the noise of the paper's evaluation (draws 0-2), 'random': seeded normal noise")
    parser.add_argument('--draws', type=int, default=3, help='noise draws per history')
    parser.add_argument('--seed', type=int, default=0, help='seed of the random noise')
    parser.add_argument('--paper-histories', action='store_true',
                        help='select 32 histories per vehicle-count group (3-5, 6-8, >=9) as in the paper')
    parser.add_argument('--per-group', type=int, default=32)
    parser.add_argument('--scene-ids', nargs='+', help='generate only these scenes')
    parser.add_argument('--limit', type=int, help='generate for the first N scenes only')
    args = parser.parse_args()
    if args.noise == 'paper' and args.draws > 3:
        parser.error('the paper used three noise draws per history; use --noise random for more')

    scenes = [s for f in scene_files(args.scenes) for s in load_scenes(f)]
    if args.paper_histories:
        identities = [{k: s[k] for k in ('scene_id', 'recording_id', 'num_agents', 'role')} for s in scenes]
        recordings = sorted({s['recording_id'] for s in scenes})
        chosen, _ = select_histories(identities, allowed_recordings=recordings, role=scenes[0]['role'],
                                     per_group=args.per_group, salt=PAPER_HISTORY_SALT)
        lookup = {s['scene_id']: s for s in scenes}
        scenes = [dict(lookup[c['scene_id']], case_index=c['case_index']) for c in chosen]
    if args.scene_ids:
        scenes = [s for s in scenes if s['scene_id'] in set(args.scene_ids)]
    if args.limit:
        scenes = scenes[:args.limit]
    print(f'{len(scenes)} histories, {len(args.requests)} requests, {args.draws} noise draws', flush=True)

    reference = load_reference(args.reference)
    generator = load_generator(args.generator, reference)
    out = Path(args.out)
    (out / 'futures').mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    with (out / 'rows.jsonl').open('w') as log:
        for k, scene in enumerate(scenes):
            n = scene['num_agents']
            group = 'N3_5' if n <= 5 else 'N6_8' if n <= 8 else 'N9_plus'
            prepared = generator.prepare(scene['features'])
            for z in range(args.draws):
                noise = paper_noise(scene['scene_id'], z, n) if args.noise == 'paper' else random_noise(n, rng.integers(2**63))
                futures = {}
                for p in args.requests:
                    start = time.monotonic()
                    future, row = generator.sample(prepared, p, noise)
                    row = attach_observation_diagnostics(row, future, scene['features'], scene['future_observed'])
                    key = f'p{p:g}'
                    futures[key] = future
                    row.update(scene_id=scene['scene_id'], recording_id=scene['recording_id'], num_agents=n, stratum=group,
                               noise_index=z, noise=args.noise, array=f'futures/{scene["scene_id"]}_z{z}.npz:{key}',
                               natural_PET=scene['natural_PET'], seconds=time.monotonic() - start)
                    log.write(json.dumps(json_safe(row)) + '\n')
                    log.flush()
                np.savez_compressed(out / 'futures' / f'{scene["scene_id"]}_z{z}.npz', **futures)
            print(f'[{k + 1}/{len(scenes)}] {scene["scene_id"]} ({n} vehicles) done', flush=True)
    generator.assert_unchanged()


if __name__ == '__main__':
    main()
