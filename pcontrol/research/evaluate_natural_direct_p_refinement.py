#!/usr/bin/env python3
"""STOP-only global selection of continued/CFG direct-P; frozen AUDIT follow-up."""
import argparse
import json
import hashlib
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from pcontrol.data.complete_scene_view import resolve,verify_binding,sha256
from pcontrol.generation.direct_p_cfg import ClassifierFreePercentileDenoiser,cfg_sample
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.evaluation import select_role_cases,quality_metrics
from pcontrol.plugins.risk_plugin import FrozenRiskPlugin
from pcontrol.research.evaluate_natural_diffusion import bound_json,write_json,model_features,clean_json
from pcontrol.research.evaluate_natural_direct_p import summarize,interval_error

POLICY_SHA="78e67912b8de8890e4cb703f0738f3ec17670d12b0e13fa4cb2fbd673cda1f3a"
PROTOCOL="natural_direct_P_refinement_global_setting_evaluation_v1"
CODE=("pcontrol/research/evaluate_natural_direct_p_refinement.py","pcontrol/research/evaluate_natural_direct_p.py",
      "pcontrol/research/evaluate_natural_diffusion.py","pcontrol/generation/direct_p_cfg.py",
      "pcontrol/generation/direct_p.py","pcontrol/generation/diffusion.py",
      "pcontrol/generation/evaluation.py","pcontrol/generation/trajectory_basis.py",
      "pcontrol/plugins/risk_plugin.py")


def check_codes(codes):
    for name,digest in codes.items(): verify_binding(dict(path=name,sha256=digest))


def quality_guards(summary,parent,policy):
    checks={}
    guards=policy["evaluation"]["quality_guards"]
    for key in ("all_pair_overlap_scene_rate","road_outside_scene_rate"):
        bound=parent[key]*guards[key+"_ratio"]
        checks[key]=dict(value=summary[key],maximum=bound,passed=summary[key]<=bound+1e-12)
    for key,ratio_key in (("negative_vx_frame_actor_fraction","negative_vx_frame_actor_fraction_ratio"),
                          ("acceleration_vector_rms_mps2","acceleration_vector_rms_ratio"),
                          ("jerk_vector_rms_mps3","jerk_vector_rms_ratio")):
        bound=parent["mean_quality"][key]*guards[ratio_key]
        value=summary["mean_quality"][key]
        checks[key]=dict(value=value,maximum=bound,passed=value<=bound+1e-12)
    return dict(eligible=all(c["passed"] for c in checks.values()),checks=checks)


def select_settings(candidates,parent,policy):
    def best(recipe):
        eligible=[key for key,row in candidates.items() if row["recipe"]==recipe and row["guard"]["eligible"]]
        return min(eligible,key=lambda key:(candidates[key]["summary"]["p_mid_MAE"],candidates[key]["scale"],
                                            candidates[key]["epoch"],key)) if eligible else None
    per_recipe={recipe:best(recipe) for recipe in ("continue","cfg")}
    eligible=[key for key in per_recipe.values() if key is not None]
    winner=min(eligible,key=lambda key:(candidates[key]["summary"]["p_mid_MAE"],candidates[key]["scale"],
                                        candidates[key]["epoch"],key)) if eligible else None
    if winner is not None and candidates[winner]["summary"]["p_mid_MAE"]>=parent["p_mid_MAE"]-policy["evaluation"]["minimum_improvement_to_select_new"]:
        winner=None
    audit=[]
    for key in per_recipe.values():
        if key is not None and key not in audit: audit.append(key)
    cfg_key=per_recipe["cfg"]
    if cfg_key is not None:
        raw="cfg_e%d_s1"%candidates[cfg_key]["epoch"]
        if raw not in candidates: raise ValueError("CFG comparison is missing same-checkpoint scale1")
        if raw not in audit: audit.append(raw)
    return dict(selected=winner or "parent",best_by_recipe=per_recipe,AUDIT_candidate_ids=audit,
                same_checkpoint_scale1_is_a_diagnostic_not_an_extra_selected_model=True)


def validate_stop_selection(selected,stop,grid,training,policy,policy_binding,parent_summary):
    """Replay the full fixed candidate grid and guards before AUDIT access."""
    if (selected.get("protocol")!=PROTOCOL or stop.get("protocol")!=PROTOCOL
            or selected["policy"]!=policy_binding or stop["policy"]!=policy_binding
            or selected["training_results"]!=training or stop["training_results"]!=training
            or selected["AUDIT_decoded_before_selection"] is not False
            or stop["role"]!="STOP" or stop["status"]!="complete"
            or stop["parent_result"]!=policy["parent_STOP"] or stop["parent_summary"]!=parent_summary
            or stop["evaluation_code_sha256"]!=selected["evaluation_code_sha256"]
            or set(stop["candidates"])!=set(grid)):
        raise ValueError("STOP/AUDIT selection provenance or full candidate grid changed")
    for key,candidate in stop["candidates"].items():
        if any(candidate[k]!=v for k,v in grid[key].items()):
            raise ValueError("a frozen STOP candidate checkpoint or setting changed")
        if candidate["guard"]!=quality_guards(candidate["summary"],parent_summary,policy):
            raise ValueError("stored STOP eligibility does not replay the declared quality guards")
        source=bound_json(candidate["result"])
        if (source["status"]!="complete" or source["candidate_id"]!=key
                or source["candidate"]!=grid[key] or source["summary"]!=candidate["summary"]
                or summarize(source["rows"])!=candidate["summary"]):
            raise ValueError("candidate summary does not replay its frozen STOP rows")
    expected=select_settings(stop["candidates"],parent_summary,policy)
    if selected["choice"]!=expected:raise ValueError("global setting choice differs from fixed STOP search")
    check_codes(selected["evaluation_code_sha256"])
    return expected["AUDIT_candidate_ids"]


def candidate_grid(policy,training):
    candidates={}
    for recipe in policy["recipes"]:
        report=bound_json(training[recipe]);check_codes(report["code_sha256"])
        if (report.get("status")!="complete" or report["recipe"]!=recipe
                or report["policy"]["sha256"]!=POLICY_SHA
                or report["parent_training"]!=policy["parent_training"]
                or set(report["checkpoints"])!={"0","120","240"}
                or report["epochs_completed"]!=policy["training"]["epochs"]
                or report["labels_changed"] is not False
                or report["risk_or_geometry_loss_added"] is not False
                or report["AUDIT_decoded"] is not False):
            raise ValueError("only completed same-policy refinement recipes allowed")
        scales=policy["sampling"]["continue_scales"] if recipe=="continue" else policy["sampling"]["CFG_scales"]
        for epoch in policy["training"]["candidate_epochs"]:
            checkpoint=report["checkpoints"][str(epoch)];verify_binding(checkpoint)
            for scale in scales:
                key=recipe+"_e"+str(epoch)+"_s"+format(scale,"g")
                candidates[key]=dict(recipe=recipe,epoch=epoch,scale=float(scale),checkpoint=checkpoint,training_result=training[recipe])
    return candidates


def load_candidate(candidate,policy,data,device):
    report=bound_json(candidate["training_result"]);check_codes(report["code_sha256"])
    checkpoint=torch.load(verify_binding(candidate["checkpoint"]),map_location="cpu")
    if (checkpoint["recipe"]!=candidate["recipe"] or int(checkpoint["epoch"])!=candidate["epoch"]
            or checkpoint["policy"]["sha256"]!=POLICY_SHA
            or checkpoint["code_sha256"]!=report["code_sha256"] or checkpoint["basis"]!=data["basis"]
            or checkpoint["prediction_type"]!="v" or checkpoint["parent_training"]!=policy["parent_training"]
            or checkpoint["data"]!=report["data"] or checkpoint["labels_manifest"]!=report["labels_manifest"]):
        raise ValueError("refinement checkpoint does not match the frozen candidate")
    config={k:checkpoint["architecture"][k] for k in ("coefficient_dim","hidden_dim","heads","layers","feedforward_dim")}
    if config["coefficient_dim"]!=2*data["basis"]["modes"]:raise ValueError("refinement basis mismatch")
    with torch.random.fork_rng(devices=[]):model=ClassifierFreePercentileDenoiser(**config)
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    model.to(device).eval().requires_grad_(False)
    schedule=CosineDiffusionSchedule(100).to(device)
    if schedule.as_dict()!=checkpoint["schedule"]:raise ValueError("forward process changed")
    return model,schedule


def evaluate_candidate(key,candidate,cases,plugin,policy,data,output,device):
    target=output/key;target.mkdir(exist_ok=False)
    model,schedule=load_candidate(candidate,policy,data,device)
    cnorm,hnorm=bound_json(data["coefficient_normalizer"]),bound_json(data["history_normalizer"])
    basis=TrajectoryBasis(int(data["basis"]["modes"]))
    mean,scale=np.asarray(cnorm["mean"]),np.asarray(cnorm["scale"])
    rows=[];started=time.perf_counter();p_values=policy["sampling"]["p_grid"]
    for number,case in enumerate(cases):
        seed=int.from_bytes(hashlib.sha256(("natural_diffusion_K1_noise_v1|"+case["scene_id"]).encode()).digest()[:8],"little")%(2**32)
        z=np.random.default_rng(seed).standard_normal((1,case["num_agents"],basis.modes,2)).astype(np.float32)
        features=model_features(case,hnorm,device);noise=torch.from_numpy(z).to(device)
        arrays={k:case[k] for k in ("history","dimensions","road_boundaries","ego_mask","agent_ids")}
        arrays.update(initial_noise=z[0],future_observed=case["future"])
        futures={}
        for p in p_values:
            condition=torch.tensor([p],device=device,dtype=torch.float32)
            c=cfg_sample(model,schedule,features,condition,noise,scale=candidate["scale"],steps=50)
            physical=c[0].detach().cpu().numpy().astype(np.float64)*scale+mean
            f=basis.decode(physical,case["history"][-1]);futures[p]=f
            arrays["generated_p"+format(p,"g").replace(".","_")]=f
        # Independent evaluation only: all raw futures were generated before
        # constructing this history's reference or looking up target PET.
        reference=plugin.condition(case["history"],case["dimensions"],case["road_boundaries"],case["ego_mask"],case["agent_mask"])
        current=[]
        for p in p_values:
            f=futures[p];scored=reference.score_future(f);quality=quality_metrics(f,case)
            rank,spec=scored["estimated_rank"],reference.target_spec(p)
            current.append(dict(scene_id=case["scene_id"],recording_id=case["recording_id"],num_agents=case["num_agents"],
                role=case["role"],stratum=case["stratum"],candidate_id=key,recipe=candidate["recipe"],epoch=candidate["epoch"],
                CFG_scale=candidate["scale"],requested_p=p,noise_seed=seed,pet_seconds=scored["pet_seconds"],
                pet_raw_seconds=scored["pet_raw_seconds"],estimated_rank=rank,target_spec=spec,
                p_mid_absolute_error=abs(rank["p_mid"]-p),p_interval_error=interval_error(p,rank),
                PET_target_absolute_error_seconds=abs(scored["pet_seconds"]-spec["target_pet_seconds"]),
                critical_container_index=scored["critical_container_index"],witness=scored["metric"]["witness"],quality=quality,
                method="GP_direct_P",array_key="generated_p"+format(p,"g").replace(".","_"),
                K=1,network_evaluations=50 if candidate["scale"] in (0.,1.) else 100,
                post_correction=False,external_risk_gradient_guidance=False,
                candidate_origin="generated_not_natural_observation"))
        path=target/("case_%02d.npz"%number)
        with path.open("xb") as handle:np.savez_compressed(handle,**arrays)
        binding=dict(path=str(path),sha256=sha256(path))
        for row in current:row["trajectory_artifact"]=binding
        rows.extend(current)
        write_json(target/("case_%02d.json"%number),dict(case_id=case["scene_id"],rows=current))
    summary=summarize(rows)
    result=dict(protocol=PROTOCOL,status="complete",candidate_id=key,candidate=candidate,rows=rows,summary=summary,
                wall_seconds=time.perf_counter()-started,post_correction=False,external_risk_gradient_guidance=False)
    binding=write_json(target/"results.json",result)
    print(json.dumps(clean_json(dict(candidate_complete=key,result=binding,p_mid_MAE=summary["p_mid_MAE"],
        p_response=summary["mean_p90_minus_p10_response"],overlap=summary["all_pair_overlap_scene_rate"],
        road=summary["road_outside_scene_rate"],wall_seconds=result["wall_seconds"]))),flush=True)
    return dict(candidate,summary=summary,result=binding)


def run(policy_binding,role,selection_binding=None):
    if policy_binding["sha256"]!=POLICY_SHA or role not in ("STOP","AUDIT"):raise ValueError("registered refinement policy/role required")
    policy=bound_json(policy_binding);parent_training=bound_json(policy["parent_training"])
    check_codes(parent_training["code_sha256"])
    parent_policy=bound_json(parent_training["policy"]);data=bound_json(parent_training["data"])
    source=resolve(policy["output_root"])
    training={recipe:dict(path=str(source/recipe/"result.json"),sha256=sha256(source/recipe/"result.json")) for recipe in policy["recipes"]}
    grid=candidate_grid(policy,training)
    if role=="AUDIT":
        if selection_binding is None:raise PermissionError("freeze global STOP settings before any AUDIT observations")
        selected=bound_json(selection_binding);stop=bound_json(selected["STOP_result"])
        parent_stop=bound_json(policy["parent_STOP"])
        keys=validate_stop_selection(selected,stop,grid,training,policy,policy_binding,parent_stop["summaries"]["GP_direct_P"])
    else:
        if selection_binding is not None:raise ValueError("STOP may not consume a future choice")
        keys=list(grid)
    parent=bound_json(policy["parent_STOP" if role=="STOP" else "parent_AUDIT"])
    if parent["role"]!=role or parent["training_result"]!=policy["parent_training"]:raise ValueError("parent comparison changed")
    parent_summary=parent["summaries"]["GP_direct_P"]
    parent_rows=[r for r in parent["rows"] if r["method"]=="GP_direct_P"]
    torch.set_num_threads(policy["runtime"]["threads"]);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    device=torch.device("cuda:1")
    output=source/(role+"_selection_evaluation");output.mkdir(exist_ok=False)
    codes={name:sha256(ROOT/name) for name in CODE}
    write_json(output/"freeze_before_sampling.json",dict(protocol=PROTOCOL,policy=policy_binding,role=role,
        training_results=training,evaluation_code_sha256=codes,settings={k:grid[k] for k in keys},
        selection=selection_binding,shared_noise_across_settings_and_p=True,K=1,post_correction=False))
    cases=select_role_cases(parent_policy["data_manifest"],role=role,per_stratum=policy["evaluation"]["per_N_stratum"],
                            salt=policy["evaluation"]["hash_salt"])
    if {c["scene_id"] for c in cases}!={r["scene_id"] for r in parent_rows}:raise ValueError("same parent paired history set required")
    write_json(output/"case_selection.json",dict(cases=[{k:c[k] for k in ("scene_id","recording_id","num_agents","stratum","selection_hash")} for c in cases]))
    plugin=FrozenRiskPlugin.from_refinement_binding(parent_policy["risk_plugin_result"],device="cpu")
    candidates={}
    for key in keys:
        candidates[key]=evaluate_candidate(key,grid[key],cases,plugin,policy,data,output,device)
        candidates[key]["guard"]=quality_guards(candidates[key]["summary"],parent_summary,policy)
    check_codes(codes)
    result=dict(protocol=PROTOCOL,status="complete",policy=policy_binding,role=role,training_results=training,
        evaluation_code_sha256=codes,candidates=candidates,parent_summary=parent_summary,
        parent_result=policy["parent_STOP" if role=="STOP" else "parent_AUDIT"],
        post_correction=False,external_risk_gradient_guidance=False,setting_selection_on_STOP_only=True,
        internal_development_not_final_blind=True)
    binding=write_json(output/"results.json",result)
    if role=="STOP":
        choice=select_settings(candidates,parent_summary,policy)
        selected=write_json(output/"selection_before_AUDIT.json",dict(protocol=PROTOCOL,policy=policy_binding,STOP_result=binding,
            choice=choice,training_results=training,evaluation_code_sha256=codes,AUDIT_decoded_before_selection=False))
        print(json.dumps(dict(STOP_complete=True,result=binding,selection=selected,choice=choice)),flush=True)
    else:print(json.dumps(dict(AUDIT_complete=True,result=binding)),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy",required=True);parser.add_argument("--policy-sha256",required=True)
    parser.add_argument("--role",choices=("STOP","AUDIT"),required=True)
    parser.add_argument("--selection");parser.add_argument("--selection-sha256")
    args=parser.parse_args()
    selection=None if args.selection is None else dict(path=args.selection,sha256=args.selection_sha256)
    run(dict(path=args.policy,sha256=args.policy_sha256),args.role,selection)
