#!/usr/bin/env python3
"""Freeze the chosen Transformer, CAL-only recalibrate, then evaluate AUDIT."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from pcontrol.time_attention_pipeline import common as c
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import pilot_time_attention_cdf as training
from pcontrol.reference.scene_calibration import fit_scene_calibration
from pcontrol.reference.scene_context_calibration import (
    diagnostic_arrays,summarize_diagnostics,interval_quadratic,
)
from pcontrol.reference.scene_calibration import precompute_crps_quadratic

PROTOCOL="time_attention_reference_validation_v1"
CODE=("pcontrol/research/validate_time_attention_reference.py","pcontrol/time_attention_pipeline/common.py",
      "pcontrol/reference/scene_context_plugin.py","pcontrol/reference/scene_context_calibration.py",
      "pcontrol/reference/scene_calibration.py","pcontrol/reference/time_attention_cdf.py",*io.CODE)


def model_and_inputs(pb):
    policy=c.policy(pb);prepared,_=io.read_prepared(policy["reference_prepared"])
    result=c.json_file(policy["reference_training_result"])
    if result["checkpoint"]["sha256"]!=policy["reference_checkpoint"]["sha256"] or result["arm"]!="M2_TimeAttn":
        raise ValueError("wrong Transformer training result/checkpoint")
    cp=torch.load(io.verify_binding(policy["reference_checkpoint"]),map_location="cpu",weights_only=False)
    if cp["data"]!=prepared["data"] or cp["normalizer"]!=prepared["normalizer"] or cp["arm"]!="M2_TimeAttn":
        raise ValueError("reference source population or normalization mismatch")
    c.verify_sources(result["code_sha256"])
    model=c.make_reference_model();model.load_state_dict(cp["state_dict"],strict=True);model.eval().requires_grad_(False)
    return policy,prepared,model


def get_predictions(model,prepared,role,policy):
    purpose={"CAL":"calibrate","AUDIT":"audit"}[role]
    examples=list(io.make_view(prepared["data"],purpose))
    pack=io.pack_examples(examples,io.load_json(prepared["normalizer"]))
    if len(pack["target"])!=dict(CAL=504,AUDIT=487)[role]:raise ValueError("unchanged role denominator required")
    device=torch.device(policy["runtime"]["device"]);model=model.to(device)
    resident=training.resident(pack,device)
    evaluation=dict(quantile_levels=policy["calibration"]["quantile_levels"],
                    thresholds_seconds=policy["calibration"]["thresholds_seconds"],PIT_seed=20260915)
    raw=training.prediction_arrays(model,resident,pack,dict(training=dict(batch_size=128),evaluation=evaluation))
    # Physical contexts are not needed for a count-only calibration family.
    context=c.count_context(raw["num_agents"])
    return raw,context


def metric_policy(policy):
    return dict(policy["calibration"],diagnostic_groups=dict(minimum_rows=20))


def score(raw,nodes,policy):
    masses,y,n=raw["joint_masses"],raw["target"],raw["num_agents"]
    quadratic=dict(CRPS_seconds=precompute_crps_quadratic(masses,n,y,family="global"),
                   twCRPS_1s=interval_quadratic(masses,n,y,1.),twCRPS_2s=interval_quadratic(masses,n,y,2.))
    mp=metric_policy(policy);context=c.count_context(n)
    arrays=diagnostic_arrays(masses,context,y,nodes,mp,quadratic)
    full=summarize_diagnostics(arrays,y,context,raw["recording_id"],mp)
    # The fake unused speed/gap columns above are NOT diagnostic strata. Only
    # global/count/recording summaries are retained or used for selection.
    full.pop("context_selection_score");full.pop("groups_used_for_selection")
    full["groups"]={"N":full["groups"]["N"]}
    full["selection_score"]=full["overall"]["expected_PIT_grid_KS"]+2*full["overall"]["threshold_MAE"]
    arrays.update(scene_id=raw["scene_id"],recording_id=raw["recording_id"],num_agents=n,target=y,
                  joint_masses=masses,effective_nodes=nodes)
    return arrays,full


def calibrate(pb):
    policy,prepared,model=model_and_inputs(pb)
    root=c.OUTPUT/"reference";root.mkdir(parents=True,exist_ok=False)
    freeze=io.write_json(root/"freeze_before_CAL.json",dict(protocol=PROTOCOL,policy=pb,
        checkpoint=policy["reference_checkpoint"],normalizer=prepared["normalizer"],data=prepared["data"],
        source_codes=c.source_bindings(CODE),base_weights_frozen=True,AUDIT_decoded=False,
        no_same_dataset_fit_and_calibration=True))
    raw,context=get_predictions(model,prepared,"CAL",policy)
    raw_binding=io.save_pack(root/"CAL_raw_predictions.npz",raw)
    identity=c.StableCountWarp.identity();nodes=identity.row_nodes(raw["num_agents"])
    identity_arrays,baseline=score(raw,nodes,policy)
    candidates=dict(identity=dict(eligible=True,family="identity",ridge=None,metrics=baseline,
                                  predictions=io.save_pack(root/"identity_CAL_OOF.npz",identity_arrays)))
    records=np.unique(raw["recording_id"]);cfg=policy["calibration"]
    for family in ("global","count"):
        for ridge in cfg["ridge_grid"]:
            name=f"{family}_ridge_{ridge:g}";row_nodes=np.empty_like(nodes);folds=[]
            for rec in records:
                held=raw["recording_id"]==rec
                fit=fit_scene_calibration(raw["joint_masses"][~held],raw["num_agents"][~held],raw["target"][~held],family=family,ridge=ridge)
                if not fit.report["success"]:raise RuntimeError("calibration fit did not converge")
                warp=c.StableCountWarp.from_dict(fit.warp.as_dict());row_nodes[held]=warp.row_nodes(raw["num_agents"][held])
                folds.append(dict(held_recording=str(rec),train_recordings=records[records!=rec].tolist(),
                                  fitting_rows=int((~held).sum()),held_rows=int(held.sum()),fit=fit.report))
            a,m=score(raw,row_nodes,policy)
            eligible=(m["overall"]["CRPS_seconds"]<=baseline["overall"]["CRPS_seconds"]*cfg["CRPS_guard_ratio"]
                      and m["groups"]["N"]["N9_plus"]["CRPS_seconds"]<=baseline["groups"]["N"]["N9_plus"]["CRPS_seconds"]*cfg["highN_CRPS_guard_ratio"])
            candidates[name]=dict(family=family,ridge=ridge,eligible=bool(eligible),metrics=m,folds=folds,
                                  predictions=io.save_pack(root/(name+"_CAL_OOF.npz"),a))
            print(json.dumps(dict(candidate=name,eligible=bool(eligible),score=m["selection_score"],CRPS=m["overall"]["CRPS_seconds"])),flush=True)
    chosen=min((k for k,v in candidates.items() if v["eligible"]),key=lambda k:(candidates[k]["metrics"]["selection_score"],
                0 if k=="identity" else 1 if candidates[k]["family"]=="global" else 2,-(candidates[k]["ridge"] or 0.)))
    item=candidates[chosen]
    if chosen=="identity":warp=identity;fit_report=dict(identity=True,success=True)
    else:
        fit=fit_scene_calibration(raw["joint_masses"],raw["num_agents"],raw["target"],family=item["family"],ridge=item["ridge"])
        if not fit.report["success"]:raise RuntimeError("final calibration fit failed")
        warp=c.StableCountWarp.from_dict(fit.warp.as_dict());fit_report=fit.report
    calibration=io.write_json(root/"calibration_model.json",warp.as_dict())
    report=dict(protocol=PROTOCOL,status="calibration_frozen",policy=pb,freeze=freeze,
        checkpoint=policy["reference_checkpoint"],normalizer=prepared["normalizer"],data=prepared["data"],
        CAL_raw_predictions=raw_binding,candidates=candidates,selected_name=chosen,selected_family=item["family"],
        selected_ridge=item["ridge"],calibration_model=calibration,final_fit=fit_report,
        calibration_selected_on="CAL_record_LOO_only",AUDIT_decoded=False,code_sha256=c.source_bindings(CODE))
    c.verify_sources(c.json_file(freeze)["source_codes"])
    io.write_json(root/"selection_before_AUDIT.json",report)
    print(json.dumps(dict(stage="calibration_frozen",selected=chosen)),flush=True)


def validate(pb):
    policy,prepared,model=model_and_inputs(pb);root=c.OUTPUT/"reference"
    sb=c.bind(root/"selection_before_AUDIT.json");selection=c.json_file(sb)
    if selection["policy"]!=pb or selection["status"]!="calibration_frozen":raise ValueError("CAL selection must already freeze")
    c.verify_sources(selection["code_sha256"])
    output=root/"development_evaluation";output.mkdir(exist_ok=False)
    io.write_json(output/"freeze.json",dict(selection=sb,AUDIT_used_for_selection=False,
        AUDIT_previously_used_in_project=True,not_a_new_blind_test=True))
    raw,_=get_predictions(model,prepared,"AUDIT",policy)
    warp=c.StableCountWarp.from_dict(c.json_file(selection["calibration_model"]))
    models={};predictions={}
    for name,one in (("raw_TimeAttn",c.StableCountWarp.identity()),("calibrated_TimeAttn",warp)):
        a,m=score(raw,one.row_nodes(raw["num_agents"]),policy);models[name]=m
        predictions[name]=io.save_pack(output/(name+"_predictions.npz"),a)
    report=dict(protocol=PROTOCOL,status="complete",architecture="M2_TimeAttn",policy=pb,
        checkpoint=policy["reference_checkpoint"],normalizer=prepared["normalizer"],data=prepared["data"],
        calibration_model=selection["calibration_model"],calibration_selection=sb,
        calibration_selected_on="CAL_record_LOO_only",selected_family=selection["selected_family"],
        selected_ridge=selection["selected_ridge"],models=models,predictions=predictions,
        code_sha256=c.source_bindings(CODE),AUDIT_used_for_selection=False,old_reference_overwritten=False,
        evaluation_scope="reused_development487_not_final_blind_test",continue_full_Transformer_pipeline=True)
    io.write_json(root/"manifest.json",report)
    print(json.dumps(dict(stage="reference_validated",family=selection["selected_family"],ridge=selection["selected_ridge"],
        metrics={k:v["overall"] for k,v in models.items()})),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["calibrate","validate"])
    parser.add_argument("--policy",required=True);parser.add_argument("--policy-sha256",required=True)
    args=parser.parse_args();torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    {"calibrate":calibrate,"validate":validate}[args.command](dict(path=str(io.resolve(args.policy)),sha256=args.policy_sha256))
