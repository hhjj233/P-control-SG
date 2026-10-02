#!/usr/bin/env python3
"""Uniform atom-compatible rescoring; never modify saved models or trajectories."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.research.audit_time_attention_full_pipeline import rank, quantile
from pcontrol.data.scene_pet import scene_occupancy_pet
from pcontrol.research.audit_pair_overlap_intervals import pair_overlap_intervals
from pcontrol.generation.evaluation import quality_metrics
from pcontrol.research.p_path_ablation import summarize

FINAL = ROOT / 'outputs/natural_percentile/transformer_publication_v1_20260916/final_validation_v1'
SUPP = ROOT / 'outputs/natural_percentile/baselines_v1'
OUT = ROOT / 'outputs/paper/canonical_evidence_v1'


def score_rows(source, scope, *, replay):
    doc = c.json_file(c.bind(source))
    queue = c.json_file(c.bind(FINAL / scope / 'queue/queue.json'))
    cases = {v['scene_id']: v for v in queue['cases']}
    expected = {(sid, z, p) for sid in cases for z in range(3) for p in (.1,.3,.5,.7,.9)}
    rows = []
    loaded = {}
    targets = {}
    worst = 0.
    for original in doc['rows']:
        r = dict(original)
        assert r['status'] == 'complete'
        sid, p = r['scene_id'], r['requested_p']
        if sid not in loaded:
            loaded = {sid: c.arrays(cases[sid]['artifact'])}
            trajectory_cache = {}
        a = loaded[sid]
        key = (sid, p)
        if key not in targets:
            targets[key] = quantile(a['reference_joint_masses'], a['reference_row_nodes'], 1.-p)
        target = targets[key]
        y = r['pet_seconds']
        ranks = rank(a['reference_joint_masses'], a['reference_row_nodes'], y)
        worst = max(worst, *(abs(ranks[k]-r['estimated_rank'][k]) for k in ranks),
                    abs(abs(y-target)-r['canonical_PET_target_absolute_error_seconds']))
        if replay:
            path = r['trajectory_artifact']['path']
            if path not in trajectory_cache:
                trajectory_cache[path] = c.arrays(r['trajectory_artifact'])
            f = trajectory_cache[path][r['array_key']]
            n = len(a['dimensions']); e = int(np.flatnonzero(a['ego_mask'])[0])
            assert f.shape == (175,n,4) and np.isfinite(f).all() and np.array_equal(f[0],a['history'][-1])
            metric = scene_occupancy_pet(f,np.ones((175,n),bool),a['dimensions'],times=np.arange(175)*.04,
                ego_index=e,sample_period=.04,window=(0.,6.96),cap_seconds=4.)
            worst = max(worst,abs(float(metric['pet_value_seconds'])-y))
            overlaps = pair_overlap_intervals(f,a['dimensions'],e)
            assert any(x['background_pair'] for x in overlaps)==r['background_PL_overlap_scene']
            assert any(not x['background_pair'] for x in overlaps)==r['ego_PL_overlap_scene']
            q = quality_metrics(f,dict(future=a['future_observed'],dimensions=a['dimensions'],
                road_boundaries=a['road_boundaries'],ego_mask=a['ego_mask']))
            assert q['road_outside_scene']==r['quality']['road_outside_scene']
        interval = max(ranks['p_low']-p,p-ranks['p_up'],0.)
        if 'p_interval_error' in r:
            worst=max(worst,abs(interval-r['p_interval_error']))
        r.update(p_interval_error=interval,interval_Fine=interval<=.05,
                 canonical_target_PET_seconds=target,target_atom=target in (0.,4.))
        rows.append(r)
    assert len(rows)==len(expected) and {(r['scene_id'],r['noise_index'],r['requested_p']) for r in rows}==expected
    assert worst<2e-9,worst
    return dict(source=c.bind(source),summary=summarize(rows),rows=rows,
        by_target={kind:summarize([r for r in rows if r['target_atom']==v]) for kind,v in (('atom',True),('continuous',False))},
        by_N={s:summarize([r for r in rows if r['stratum']==s]) for s in ('N3_5','N6_8','N9_plus')},
        by_P={str(p):summarize([r for r in rows if r['requested_p']==p]) for p in (.1,.3,.5,.7,.9)},
        maximum_replay_error=worst,new_trajectory_replay=replay)


def main():
    torch.set_num_threads(1)
    dest = OUT / 'canonical_evidence.json'
    assert not dest.exists()
    models = {}
    for scope in ('TEST','VAL','R18'):
        models['canonical_'+scope] = score_rows(FINAL/scope/'generation/canonical/result.json',scope,replay=False)
        print(scope,json.dumps(models['canonical_'+scope]['summary']),flush=True)
    for arm in ('pcvae','p_diffusion'):
        models[arm] = score_rows(SUPP/arm/'TEST/result.json','TEST',replay=True)
        print(arm,json.dumps(models[arm]['summary']),flush=True)
    # All complete natural TEST clips, not the count-balanced generation subset.
    dataset = c.json_file(c.bind(FINAL/'TEST/dataset.json'))
    pet, counts = [], []
    for entry in dataset['recordings'].values():
        assert c.bind(entry['data']['path']) == entry['data']
        with np.load(entry['data']['path'],allow_pickle=False) as z:
            assert z['scene_id'].tolist()==entry['scene_ids']
            pet.extend(z['pet_value'].tolist());counts.extend(z['num_agents'].tolist())
    pet, counts=np.array(pet),np.array(counts)
    assert len(pet)==11649 and np.all((pet>=0)&(pet<=4))
    np.savez_compressed(OUT/'observed_test_pet.npz',PET=pet,num_agents=counts)
    context={}
    for label,lo,hi in [('N3_5',3,5),('N6_8',6,8),('N9_plus',9,10000)]:
        y=pet[(counts>=lo)&(counts<=hi)]
        context[label]=dict(scenes=len(y),PET_le_1s=float((y<=1).mean()),median_PET=float(np.median(y)),cap_fraction=float((y==4).mean()))
    report=dict(status='complete',models=models,natural_context=context,observed=c.bind(OUT/'observed_test_pet.npz'),
        dataset=c.bind(FINAL/'TEST/dataset.json'),code=c.bind(__file__),new_training=False,
        primary_error='distance_to_atom_compatible_rank_interval',tolerance=.05,
        old_midpoint_fields_retained=True,source_trajectories_unchanged=True)
    dest.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status='complete',context=context,output=str(dest))),flush=True)


if __name__=='__main__':main()
