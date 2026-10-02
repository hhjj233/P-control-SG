#!/usr/bin/env python3
"""Fresh K=1 STOP36 generation under the Transformer reference; no rescoring-only shortcut."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research.train_time_attention_generator import make_model
from pcontrol.research.evaluate_natural_diffusion import model_features
from pcontrol.research.pilot_relative_direct_p import existing_stop_cases
from pcontrol.research.refine_natural_direct_p import state_hash
from pcontrol.research.audit_pair_overlap_intervals import pair_overlap_intervals
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.risk_guidance import TorchTrajectoryDecoder
from pcontrol.generation.cdf_shape_context import CONTEXT_KEY,shape_from_reference
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY
from pcontrol.generation.percentile_sampling_guidance import percentile_guided_sample
from pcontrol.generation.background_constrained_guidance import BackgroundConstrainedGuidance
from pcontrol.generation.evaluation import quality_metrics
from pcontrol.generation.scene_quality_diagnostics import scene_quality
from pcontrol.reference.scene_models import SceneCDFReference
from pcontrol.reference.scene_calibration import SceneCDFWarp
from pcontrol.plugins.risk_plugin import FrozenRiskPlugin

PROTOCOL="time_attention_complete_pipeline_generation_v1"
OLD_PROFILE=dict(path="configs/natural_percentile/background_constrained_guidance_v1.json",
                 sha256="cf3c92cb41542ccef0f433e337dc6fa3a35fa8315d61b2352eb532d5a9444b2e")
OLD_REFERENCE=dict(path="outputs/natural_percentile/scene_refinement_v1_20260910/development_evaluation/results.json",
                   sha256="2c14d29a2ea252bc46e6e858f68872d6c7a6bb2d1581571f0e233ff3bebdbdcf")
CODE=("pcontrol/research/evaluate_time_attention_pipeline.py","pcontrol/time_attention_pipeline/common.py",
      "pcontrol/generation/percentile_sampling_guidance.py","pcontrol/generation/background_constrained_guidance.py",
      "pcontrol/generation/background_envelope.py","pcontrol/generation/road_constrained_guidance.py",
      "pcontrol/generation/road_envelope.py","pcontrol/generation/directional_percentile_guidance.py",
      "pcontrol/generation/directional_sampling_geometry.py","pcontrol/generation/sampling_geometry.py",
      "pcontrol/generation/scene_quality_diagnostics.py","pcontrol/research/audit_pair_overlap_intervals.py")


def legacy_reference():
    result=c.json_file(OLD_REFERENCE);prep=c.json_file(result["parent_prepared"]);selection=c.json_file(result["base_selection"])
    checkpoint=torch.load(io.verify_binding(selection["selected"]["checkpoint"]),map_location="cpu",weights_only=False)
    if selection["selected"]["checkpoint"]["sha256"]!="c214085f32754612d083e508dc17603c31b178640b16343ae1fc2ebd36fc8c6e":raise ValueError("old reference changed")
    model=SceneCDFReference('M2',torch.linspace(0,4,65,dtype=torch.float64),hidden_dim=64,heads=4)
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return FrozenRiskPlugin(model,c.json_file(prep['normalizer']),SceneCDFWarp.from_dict(c.json_file(result['calibration_model'])),
                            provenance=dict(legacy_result=OLD_REFERENCE))


def score_cell(errors):
    e=np.asarray(errors,dtype=float)
    return dict(requests=len(e),Fine_count=int((e<=.05).sum()),Fine_at_0_05=float((e<=.05).mean()),P_MAE=float(e.mean()))


def summary(rows):
    result=score_cell([r['p_mid_absolute_error'] for r in rows])
    result.update(histories=len({r['scene_id'] for r in rows}),
        PET_target_MAE_seconds=float(np.mean([r['PET_target_absolute_error_seconds'] for r in rows])),
        interval_MAE=float(np.mean([r['p_interval_error'] for r in rows])),
        native_overlap_requests=sum(r['quality']['all_pair_overlap_scene'] for r in rows),
        background_PL_overlap_requests=sum(r['background_PL_overlap_scene'] for r in rows),
        ego_PL_overlap_requests=sum(r['ego_PL_overlap_scene'] for r in rows),
        native_road_requests=sum(r['quality']['road_outside_scene'] for r in rows),
        negative_vx_fraction=float(np.mean([r['quality']['negative_vx_frame_actor_fraction'] for r in rows])),
        joint_Fine_count=sum(r['p_mid_absolute_error']<=.05 and not r['quality']['all_pair_overlap_scene'] and not r['quality']['road_outside_scene'] for r in rows),
        short_BG_gap_under1m_requests=sum(r['scene_quality']['background_pairs_long_gap_under1m_frame_fraction']>0 for r in rows),
        by_P={str(p):score_cell([r['p_mid_absolute_error'] for r in rows if r['requested_p']==p]) for p in (.1,.5,.9)})
    return result


def run(pb):
    p=c.policy(pb);training_binding=c.bind(c.OUTPUT/'generator/terminal/result.json');training=c.json_file(training_binding)
    if training['status']!='complete' or training['smoke'] or training['policy']!=pb or training['epochs_completed']!=p['generator']['terminal_epochs']:
        raise ValueError('completed new-reference generator training required')
    c.verify_sources(training['code_sha256'])
    checkpoint=torch.load(io.verify_binding(training['selected_checkpoint']),map_location='cpu',weights_only=False)
    if checkpoint['reference_manifest']!=training['reference_manifest'] or checkpoint['labels_manifest']!=training['labels_manifest']:
        raise ValueError('generator was not trained under the evaluated reference')
    model=make_model(checkpoint['architecture'],checkpoint['state_dict'],torch.device('cpu')).eval().requires_grad_(False)
    initial=state_hash(model);schedule=CosineDiffusionSchedule(100);basis=TrajectoryBasis(8)
    cn=c.json_file(checkpoint['coefficient_normalizer']);hn=c.json_file(checkpoint['history_normalizer'])
    new_plugin=c.TimeAttentionRiskPlugin.from_manifest(training['reference_manifest'],device='cpu');old_plugin=legacy_reference()
    old_result=c.json_file(p['old_generation_result']);old_child=c.json_file(old_result['candidates']['background_guarded']['result'])
    cases=existing_stop_cases(old_child);old_rows={(r['scene_id'],r['requested_p']):r for r in old_child['rows']}
    old_policy=c.json_file(OLD_PROFILE);profile=dict(old_policy['profiles']['background_guarded'])
    if (profile['last_steps']!=p['sampling']['risk_last_steps'] or profile['inner_steps']!=p['sampling']['inner_steps']
            or profile['background_last_steps']!=p['sampling']['background_last_steps'] or profile['background_clearance_m']!=p['sampling']['background_clearance_m']):
        raise ValueError('do not change the retained sampling profile')
    root=c.OUTPUT/'generation';root.mkdir(exist_ok=False)
    codes={**training['code_sha256'],**c.source_bindings(CODE)}
    io.write_json(root/'freeze_before_generation.json',dict(protocol=PROTOCOL,policy=pb,training=training_binding,
        checkpoint=training['selected_checkpoint'],reference_manifest=training['reference_manifest'],old_result=p['old_generation_result'],
        profile=profile,K=1,CFG_scale=p['sampling']['CFG_scale'],model_state_sha256=initial,code_sha256=codes,
        no_per_request_selection=True,no_post_sampler_repair=True,requested_count=36,STOP_reused_development=True))
    rows=[];cross={k:[] for k in ('old_output_old_CDF','old_output_new_CDF','new_output_old_CDF','new_output_new_CDF')}
    old_rank_replay=0.;started=time.monotonic()
    for number,case in enumerate(cases):
        ref=new_plugin.condition(case['history'],case['dimensions'],case['road_boundaries'],case['ego_mask'],case['agent_mask'])
        old_ref=old_plugin.condition(case['history'],case['dimensions'],case['road_boundaries'],case['ego_mask'],case['agent_mask'])
        features=model_features(case,hn,torch.device('cpu'))
        features[CONTEXT_KEY]=torch.tensor(np.asarray(shape_from_reference(ref))[None],dtype=torch.float32)
        features[PIECES_KEY]=c.StableTorchInverse(ref._masses,[ref.num_agents],warp=ref._warp,base_knots=ref._knots).compiled_pieces()
        z=torch.tensor(case['initial_noise'][None]);decoder=TorchTrajectoryDecoder(basis,cn,case['history'][-1])
        ego=int(np.flatnonzero(case['ego_mask'])[0]);futures={};infos={}
        for requested in p['sampling']['p_grid']:
            before=time.monotonic()
            guide=BackgroundConstrainedGuidance(decoder,case['dimensions'],case['history'][-1],
                lambda y:float(ref.rank(y)['p_mid']),requested,float(ref.target_pet(requested)),
                ego_index=ego,road_boundaries=case['road_boundaries'],**profile)
            info=percentile_guided_sample(model,schedule,features,torch.tensor([requested]),z,guide,scale=p['sampling']['CFG_scale'])
            coefficients=info.pop('sample')[0].numpy().astype(np.float64)
            future=basis.decode(coefficients*np.asarray(cn['scale'])+np.asarray(cn['mean']),case['history'][-1])
            if not np.array_equal(future[0],case['history'][-1]):raise ValueError('initial state changed')
            info['seconds']=time.monotonic()-before;infos[requested]=info;futures[requested]=future
        arrays={k:case[k] for k in ('history','dimensions','road_boundaries','ego_mask','agent_ids','initial_noise')}
        arrays.update(future_observed=case['future'],new_reference_joint_masses=ref._masses[0],
            new_reference_row_nodes=ref._warp.row_nodes([ref.num_agents])[0],old_reference_joint_masses=old_ref._masses[0],
            old_reference_row_nodes=old_ref._warp.row_nodes([old_ref.num_agents])[0],
            **{CONTEXT_KEY:features[CONTEXT_KEY][0].numpy(),PIECES_KEY:features[PIECES_KEY][0].numpy()})
        arrays.update({f"generated_p{str(q).replace('.','_')}":v for q,v in futures.items()})
        artifact=io.save_pack(root/f'case_{number:02d}.npz',arrays)
        for requested,future in futures.items():
            info=infos[requested];measured=ref.score_future(future);rank=measured['estimated_rank'];target=ref.target_spec(requested)
            pet=measured['pet_seconds'];err=abs(rank['p_mid']-requested);interval=max(rank['p_low']-requested,requested-rank['p_up'],0.)
            overlap=pair_overlap_intervals(future,case['dimensions'],ego)
            previous=old_rows[case['scene_id'],requested];old_y=previous['pet_seconds']
            recomputed_old=float(old_ref.rank(old_y)['p_mid']);old_rank_replay=max(old_rank_replay,abs(recomputed_old-previous['estimated_rank']['p_mid']))
            cross['old_output_old_CDF'].append(abs(recomputed_old-requested))
            cross['old_output_new_CDF'].append(abs(float(ref.rank(old_y)['p_mid'])-requested))
            cross['new_output_old_CDF'].append(abs(float(old_ref.rank(pet)['p_mid'])-requested));cross['new_output_new_CDF'].append(err)
            row={k:case[k] for k in ('scene_id','recording_id','role','num_agents','stratum','noise_seed')}
            row.update(requested_p=requested,pet_seconds=pet,estimated_rank=rank,target_spec=target,p_mid_absolute_error=err,
                p_interval_error=interval,PET_target_absolute_error_seconds=abs(pet-target['target_pet_seconds']),
                quality=quality_metrics(future,case),scene_quality=scene_quality(future,case['dimensions'],case['ego_mask']),
                PL_overlap_intervals=overlap,background_PL_overlap_scene=any(x['background_pair'] for x in overlap),
                ego_PL_overlap_scene=any(not x['background_pair'] for x in overlap),guidance=info['guidance'],
                network_evaluations=info['network_evaluations'],sampling_seconds=info['seconds'],K=1,CFG_scale=p['sampling']['CFG_scale'],
                trajectory_artifact=artifact,array_key=f"generated_p{str(requested).replace('.','_')}",
                reference_manifest=training['reference_manifest'],new_generator_trained_on_Transformer_labels=True,
                post_sampler_repair=False,all_actors_retained=True,inference_observed_future_input=False)
            rows.append(row);print(json.dumps(dict(scene=case['scene_id'],P=requested,PET=pet,error=err,
                BG_overlap=row['background_PL_overlap_scene'],short_BG=row['scene_quality']['background_pairs_long_gap_under1m_frame_fraction'])),flush=True)
    if len(rows)!=36 or len({(r['scene_id'],r['requested_p']) for r in rows})!=36:raise ValueError('all36 requests required')
    if state_hash(model)!=initial or any(q.grad is not None for q in model.parameters()):raise ValueError('inference mutated generator')
    if old_rank_replay>2e-5:raise ValueError('old reference numerical replay inconsistent; do not compare silently')
    total=summary(rows);timing=dict(total_seconds=time.monotonic()-started,sampling_seconds=sum(r['sampling_seconds'] for r in rows),
        network_evaluations=sum(r['network_evaluations'] for r in rows),PET_queries=sum(r['guidance']['exact_metric_calls'] for r in rows),
        road_QP=sum(r['guidance']['road_envelope']['QP_solves'] for r in rows),
        background_QP=sum(r['guidance']['background_envelope']['QP_solves'] for r in rows))
    result=dict(protocol=PROTOCOL,status='complete',policy=pb,training=training_binding,checkpoint=training['selected_checkpoint'],
        reference_manifest=training['reference_manifest'],rows=rows,summary=total,timing=timing,code_sha256=codes,
        old_reference_rank_replay_max_error=old_rank_replay,cross_reference_matrix={k:score_cell(v) for k,v in cross.items()},
        cross_reference_matrix_is_sensitivity_not_ground_truth=True,old_pipeline_original_Fine_count=28,
        all36_retained=True,old_outputs_not_overwritten=True,models_trained_and_futures_regenerated=True,
        P_condition_guidance_ablation=False)
    c.verify_sources(codes);io.write_json(root/'results.json',result)
    print(json.dumps(dict(stage='generation_complete',summary=total,cross_reference=result['cross_reference_matrix'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',required=True);parser.add_argument('--policy-sha256',required=True)
    a=parser.parse_args();torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False);torch.backends.cuda.enable_math_sdp(True)
    run(dict(path=str(io.resolve(a.policy)),sha256=a.policy_sha256))
