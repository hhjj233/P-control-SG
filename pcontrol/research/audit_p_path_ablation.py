#!/usr/bin/env python3
"""Replay saved P-path ablations and rescore the unchanged Canonical full arm."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.research.p_path_ablation import BASE, OUT, GRID, ARMS, write_once, summarize, now
from pcontrol.time_attention_pipeline import common as c
from pcontrol.data.scene_pet import scene_occupancy_pet
from pcontrol.research.audit_time_attention_full_pipeline import rank, quantile
from pcontrol.research.audit_pair_overlap_intervals import pair_overlap_intervals
from pcontrol.generation.evaluation import quality_metrics


def main():
    torch.set_num_threads(1)
    queue = c.json_file(c.bind(BASE / 'TEST/queue/queue.json'))
    cases = {r['scene_id']: r for r in queue['cases']}
    expected = {(s, z, p) for s in cases for z in range(3) for p in GRID}
    sources = {'full': c.bind(BASE / 'TEST/generation/canonical/result.json')}
    sources.update({arm: c.bind(OUT / arm / 'result.json') for arm in ARMS})
    results, checks = {}, {}
    for arm, binding in sources.items():
        doc = c.json_file(binding)
        rows = doc['rows']
        assert len(rows) == 1440 and {(r['scene_id'], r['noise_index'], r['requested_p']) for r in rows} == expected
        maximum = 0.
        unique, case_cache, traj_cache = {}, {}, {}
        for row in rows:
            sid, p = row['scene_id'], row['requested_p']
            if row['status'] != 'complete':
                assert row.get('failure_reason')
                continue
            if sid not in case_cache:
                case_cache = {sid: c.arrays(cases[sid]['artifact'])}
                traj_cache = {}
            a = case_cache[sid]
            trajectory = row['trajectory_artifact']
            key = (trajectory['path'], row['array_key'])
            if key not in unique:
                if trajectory['path'] not in traj_cache:
                    traj_cache[trajectory['path']] = c.arrays(trajectory)
                f = traj_cache[trajectory['path']][row['array_key']]
                n = len(a['dimensions']); e = int(np.flatnonzero(a['ego_mask'])[0])
                assert f.shape == (175, n, 4) and np.isfinite(f).all() and np.array_equal(f[0], a['history'][-1])
                metric = scene_occupancy_pet(f, np.ones((175, n), bool), a['dimensions'], times=np.arange(175)*.04,
                    ego_index=e, sample_period=.04, window=(0., 6.96), cap_seconds=4.)
                y = float(metric['pet_value_seconds'])
                r = rank(a['reference_joint_masses'], a['reference_row_nodes'], y)
                overlap = pair_overlap_intervals(f, a['dimensions'], e)
                q = quality_metrics(f, dict(future=a['future_observed'], dimensions=a['dimensions'],
                    road_boundaries=a['road_boundaries'], ego_mask=a['ego_mask']))
                unique[key] = dict(PET=y, rank=r, BG=any(x['background_pair'] for x in overlap),
                    ego=any(not x['background_pair'] for x in overlap), quality=q)
            v = unique[key]
            target = quantile(a['reference_joint_masses'], a['reference_row_nodes'], 1.-p)
            interval = max(v['rank']['p_low']-p, p-v['rank']['p_up'], 0.)
            errors = [abs(v['PET']-row['pet_seconds']), abs(interval-row['p_interval_error']),
                      abs(target-row['canonical_target_PET_seconds'])]
            errors += [abs(v['rank'][k]-row['estimated_rank'][k]) for k in v['rank']]
            maximum = max(maximum, *errors)
            assert v['BG'] == row['background_PL_overlap_scene'] and v['ego'] == row['ego_PL_overlap_scene']
            for k in ('road_outside_scene','negative_vx_scene','all_pair_overlap_scene'):
                assert v['quality'][k] == row['quality'][k]
            if arm in ('condition_only','no_p'):
                assert row['guidance']['CDF_rank_queries'] == row['guidance']['exact_metric_calls'] == 0
            if arm == 'no_p':
                assert row['network_condition_enabled'] is False and row['sampling_risk_guidance_enabled'] is False
            row['target_atom'] = target in (0., 4.)
        assert maximum < 2e-9, (arm, maximum)
        if arm == 'no_p':
            # Five requests for one H,z must bind to the identical output array.
            for sid in cases:
                for z in range(3):
                    group = [r for r in rows if r['scene_id']==sid and r['noise_index']==z and r['status']=='complete']
                    assert len({(r['trajectory_artifact']['sha256'],r['array_key']) for r in group}) <= 1
        results[arm] = dict(source=binding, summary=summarize(rows), unique_saved_outputs=len(unique),
            by_P={str(p): summarize([r for r in rows if r['requested_p']==p]) for p in GRID},
            by_target={kind: summarize([r for r in rows if r['status']=='complete' and r['target_atom']==isatom])
                for kind,isatom in (('atom',True),('continuous',False))},
            by_N={s: summarize([r for r in rows if r['stratum']==s]) for s in ('N3_5','N6_8','N9_plus')})
        checks[arm] = dict(maximum_replay_error=maximum, full_roster_exact_t0=True, unique_replays=len(unique))
        print(json.dumps(dict(arm=arm, **results[arm]['summary'])), flush=True)
    write_once(OUT / 'comparison.json', dict(status='complete_replayed', time_utc=now(),
        protocol=c.bind(ROOT/'docs/plans/CANONICAL_P_ABLATION_PROTOCOL_20260925.md'),
        code=c.bind(__file__), freeze=c.bind(OUT/'freeze.json'), results=results, checks=checks,
        no_model_selection=True, metric_revision_not_training_improvement=True,
        all_evaluation_requests=5760, original_full_arm_reused=True, independent_histories=96,
        inferential_scope='fixed-checkpoint inference information paths, not four independently trained models',
        paper_updated=False))


if __name__ == '__main__':
    main()
