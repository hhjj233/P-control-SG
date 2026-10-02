#!/usr/bin/env python3
"""All-request TEST evaluation, no tuning, no candidate selection/repair."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pcontrol.research.run_baseline_generators import OUT,new_model
from pcontrol.time_attention_pipeline import common as c
from pcontrol.generation.data import FEATURE_KEYS
from pcontrol.generation.direct_p import direct_p_sample
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.evaluation import quality_metrics
from pcontrol.generation.scene_quality_diagnostics import scene_quality
from pcontrol.data.scene_pet import scene_occupancy_pet
from pcontrol.publication_pipeline.final_execution_generation import summarize_requests
from pcontrol.research.evaluate_natural_diffusion import model_features
from pcontrol.research.audit_time_attention_full_pipeline import rank
from pcontrol.research.audit_pair_overlap_intervals import pair_overlap_intervals
from pcontrol.research import train_natural_scene_reference as io

FINAL=ROOT/'outputs/natural_percentile/transformer_publication_v1_20260916/final_validation_v1/TEST'
GRID=(.1,.3,.5,.7,.9)

def evaluate(arm):
    torch.set_num_threads(2);torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    tr=c.json_file(c.bind(OUT/arm/'training_result.json'))
    checkpoint=torch.load(io.verify_binding(tr['checkpoint']),map_location='cpu',weights_only=False)
    freeze=c.json_file(checkpoint['freeze']);c.verify_sources(freeze['code'])
    model=new_model(arm).eval().requires_grad_(False);model.load_state_dict(checkpoint['state_dict'])
    schedule=CosineDiffusionSchedule(100);basis=TrajectoryBasis(8)
    hn=c.json_file(tr['history_normalizer']);cn=c.json_file(tr['coefficient_normalizer'])
    qb=c.bind(FINAL/'queue/queue.json');queue=c.json_file(qb)
    old=c.json_file(c.bind(FINAL/'generation/atom_aware/result.json'))
    oldrows={(r['scene_id'],r['noise_index'],r['requested_p']):r for r in old['rows']}
    out=OUT/arm/'TEST';out.mkdir(exist_ok=False);rows=[];start=time.monotonic()
    io.write_json(out/'freeze.json',dict(training=c.bind(OUT/arm/'training_result.json'),queue=qb,
        comparison='post_primary_model_family_baselines_no_constraint_refinement',K=1,noise='same_saved_per_actor16',
        code=c.source_bindings(('pcontrol/research/evaluate_baseline_generators.py',))))
    for ci,item in enumerate(queue['cases']):
        a=c.arrays(item['artifact']);n=item['num_agents'];ego=int(np.flatnonzero(a['ego_mask'])[0])
        # Physical model input excludes observed futures and derived test targets.
        case={k:a[k] for k in ('history','dimensions','road_boundaries','ego_mask','agent_mask')}
        case['num_agents']=n
        f=model_features(case,hn,torch.device('cpu'))
        assert set(f)==FEATURE_KEYS
        for z in range(3):
            artifacts={};pending=[]
            for p in GRID:
                r=dict(scene_id=item['scene_id'],recording_id=item['recording_id'],num_agents=n,
                    stratum=item['stratum'],case_index=ci,noise_index=z,requested_p=p,arm=arm,K=1,
                    inference_observed_future_input=False,all_actors_retained=True,post_sampler_repair=False)
                tick=time.monotonic()
                try:
                    noise=torch.tensor(a[f'initial_noise_{z}'][None],dtype=torch.float32)
                    with torch.no_grad():
                        if arm=='pcvae':coef=model.sample(f,torch.tensor([p]),noise)[0].numpy()
                        else:coef=direct_p_sample(model,schedule,f,torch.tensor([p]),noise,steps=50)[0].numpy()
                    future=basis.decode(coef.astype(np.float64)*np.array(cn['scale'])+np.array(cn['mean']),case['history'][-1])
                    elapsed=time.monotonic()-tick
                    if not np.isfinite(future).all() or not np.array_equal(future[0],case['history'][-1]):raise ValueError('finite / exact t0')
                    # Ground-truth future is first used here, after the output is fixed.
                    oracle=scene_occupancy_pet(future,np.ones((175,n),bool),case['dimensions'],times=basis.times,
                        ego_index=ego,sample_period=.04,window=(0.,6.96),cap_seconds=4.)
                    pet=float(oracle['pet_value_seconds']);rr=rank(a['reference_joint_masses'],a['reference_row_nodes'],pet)
                    overlaps=pair_overlap_intervals(future,case['dimensions'],ego)
                    oldrow=oldrows[item['scene_id'],z,p]
                    quality=quality_metrics(future,dict(case,future=a['future_observed']))
                    r.update(status='complete',pet_seconds=pet,estimated_rank=rr,p_mid_absolute_error=abs(rr['p_mid']-p),
                        quality=quality,scene_quality=scene_quality(future,case['dimensions'],case['ego_mask']),
                        PL_overlap_intervals=overlaps,background_PL_overlap_scene=any(o['background_pair'] for o in overlaps),
                        ego_PL_overlap_scene=any(not o['background_pair'] for o in overlaps),
                        PET_control_target_absolute_error_seconds=abs(pet-oldrow['control_target_PET_seconds']),
                        canonical_PET_target_absolute_error_seconds=abs(pet-oldrow['canonical_target_PET_seconds']),
                        scalar_error_infimum=oldrow['scalar_error_infimum'],sampling_seconds=elapsed,
                        network_evaluations=1 if arm=='pcvae' else 50,
                        critical_other_index=oracle['critical_other_index'])
                    key='generated_p'+str(p).replace('.','_');artifacts[key]=future;r['array_key']=key
                except (ValueError,FloatingPointError,RuntimeError) as exc:
                    r.update(status='failed',failure_type=type(exc).__name__,failure_message=str(exc))
                pending.append(r)
            path=out/f'case_{ci:03d}_z{z}.npz';np.savez_compressed(path,**artifacts);binding=c.bind(path)
            for r in pending:r['trajectory_artifact']=binding
            io.write_json(out/f'case_{ci:03d}_z{z}.json',dict(rows=pending))
            rows.extend(pending)
        if (ci+1)%12==0:print(json.dumps(dict(arm=arm,completed_histories=ci+1,seconds=time.monotonic()-start)),flush=True)
    if len(rows)!=1440:raise RuntimeError('denominator changed')
    summary=summarize_requests(rows)
    result=dict(status='complete',arm=arm,rows=rows,summary=summary,queue=qb,training=c.bind(OUT/arm/'training_result.json'),
        by_N={s:summarize_requests([r for r in rows if r['stratum']==s]) for s in ('N3_5','N6_8','N9plus')},
        total_seconds=time.monotonic()-start,all_requested_rows_retained=True)
    io.write_json(out/'result.json',result)
    print(json.dumps(dict(arm=arm,summary=summary)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm',required=True,choices=('pcvae','p_diffusion'));evaluate(p.parse_args().arm)
