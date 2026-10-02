#!/usr/bin/env python3
"""Five recording-excluded Transformer teachers with own CAL-fitted warps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

from pcontrol.time_attention_pipeline import common as c
from pcontrol.reference import direct_p_crossfit as old
from pcontrol.reference.scene_calibration import fit_scene_calibration
from pcontrol.reference.scores import crps_from_params
from pcontrol.generation.cdf_shape_context import GRID_SECONDS,CONTEXT_KEY
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY
from pcontrol.research import pilot_time_attention_cdf as train_utils
from pcontrol.research import train_natural_scene_reference as io

PROTOCOL="time_attention_recording_crossfit_v1"
PHYSICAL_PREPARED=dict(path="outputs/natural_percentile/natural_direct_p_labels_v1_20260910/prepared.json",
                       sha256="b65c5afc88c4ab97310dd6c1b1078e1abd0a339fc6e52fa67eeebda95b2bfed6")
CODE=("pcontrol/research/crossfit_time_attention_reference.py","pcontrol/time_attention_pipeline/common.py",
      "pcontrol/reference/time_attention_cdf.py","pcontrol/reference/direct_p_crossfit.py",
      "pcontrol/reference/scene_context_plugin.py","pcontrol/reference/scene_context_calibration.py",
      "pcontrol/reference/scene_calibration.py","pcontrol/reference/torch_frozen_cdf.py",
      "pcontrol/reference/torch_frozen_inverse.py",*io.CODE)


def prepare(pb):
    p=c.policy(pb);reference=c.bind(c.OUTPUT/"reference/manifest.json");ref=c.json_file(reference)
    if ref["status"]!="complete" or ref["policy"]!=pb:raise ValueError("validate the new Transformer reference first")
    physical=c.json_file(PHYSICAL_PREPARED);prepared=c.json_file(p["reference_prepared"])
    if (physical["source_data"]["sha256"]!=prepared["data"]["sha256"]
            or io.resolve(physical["source_data"]["path"])!=io.resolve(prepared["data"]["path"])
            or physical["counts"]!={"CAL":504,"FIT":9913,"STOP":538}):
        raise ValueError("physical cache source or population differs")
    folds=p["oof"]["heldout_recordings"]
    if physical["folds"]!=folds:raise ValueError("do not change recording fold identities")
    c.verify_sources(physical["code_sha256"])
    for role,b in physical["physical_packs"].items():io.verify_binding(b)
    root=c.OUTPUT/"crossfit";root.mkdir(exist_ok=False)
    report=dict(protocol=PROTOCOL,status="prepared",policy=pb,reference_manifest=reference,
        source_data=physical["source_data"],physical_prepared=PHYSICAL_PREPARED,physical_packs=physical["physical_packs"],
        role_recordings=physical["role_recordings"],folds=folds,selected_family=ref["selected_family"],
        selected_ridge=ref["selected_ridge"],fixed_base_epochs=p["oof"]["base_epochs"],
        fixed_highN_epochs=p["oof"]["highN_epochs"],epoch_selection_uses_held_or_STOP=False,
        code_sha256=c.source_bindings(CODE),generator_changed=False)
    io.write_json(root/"prepared.json",report)
    print(json.dumps(dict(stage="crossfit_prepared",folds=folds,epochs=[report["fixed_base_epochs"],report["fixed_highN_epochs"]])),flush=True)


def inputs(pb):
    p=c.policy(pb);b=c.bind(c.OUTPUT/"crossfit/prepared.json");r=c.json_file(b)
    if r["policy"]!=pb or r["status"]!="prepared":raise ValueError("wrong crossfit preparation")
    c.verify_sources(r["code_sha256"])
    return p,r,b


def train_fixed(train,cfg,device,log,fold):
    if not np.all(train["role"]=="FIT"):raise PermissionError("only fold FIT may optimize a teacher")
    torch.manual_seed(cfg["seed"]);model=c.make_reference_model().to(device)
    initial=train_utils.tensor_hash(model.state_dict());resident=train_utils.resident(train,device)
    counts=train["agent_mask"].sum(1);trace=[];started=time.monotonic()
    for stage,epochs,lr,offset in (("base",cfg["base_epochs"],cfg["base_learning_rate"],0),
                                 ("highN",cfg["highN_epochs"],cfg["highN_learning_rate"],1000)):
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=cfg["weight_decay"])
        rng=np.random.default_rng(cfg["seed"]+offset)
        weights=np.ones(len(counts)) if stage=="base" else np.where(counts>=9,cfg["highN_weight_multiplier"],1.)
        weights=torch.as_tensor(weights/weights.mean(),dtype=torch.float64,device=device)
        for epoch in range(1,epochs+1):
            model.train();order=rng.permutation(len(counts));total=0.
            for start in range(0,len(order),cfg["batch_size"]):
                rows=order[start:start+cfg["batch_size"]];features,y=train_utils.batch(resident,rows)
                optimizer.zero_grad(set_to_none=True)
                loss=(crps_from_params(model(features),y,normalized=True)*weights[torch.as_tensor(rows,device=device)]).mean()
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite teacher loss")
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"])
                if not torch.isfinite(norm):raise FloatingPointError("nonfinite teacher gradient")
                optimizer.step();total+=float(loss.detach())*len(rows)
            row=dict(fold=fold,stage=stage,epoch=epoch,training_rows=len(counts),normalized_training_CRPS=total/len(counts),
                     order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),seconds=time.monotonic()-started)
            trace.append(row);log.write(json.dumps(row)+"\n");log.flush()
            if epoch%5==0 or epoch==epochs:print(json.dumps(row),flush=True)
    model.eval().requires_grad_(False)
    return model,dict(initial_state_sha256=initial,final_state_sha256=train_utils.tensor_hash(model.state_dict()),
                      initialization="fresh_random_no_checkpoint_loaded",same_seed_across_folds=True,
                      held_or_STOP_selection=False,epochs=trace,wall_seconds=time.monotonic()-started)


def payload(masses,physical,warp,fold):
    n=physical["agent_mask"].sum(1);y=physical["target"]
    labels=old.rank_labels(masses,n,y,{k:physical[k] for k in ("scene_id","recording_id","role")},warp,fold)
    nodes=warp.row_nodes(n)
    shape=warp.cdf(masses,n,GRID_SECONDS[None]);shape[:,-1]=warp.cdf(masses,n,4.,side="left")
    inverse=c.StableTorchInverse(masses,n,row_nodes=nodes)
    context={k:physical[k] for k in ("scene_id","recording_id","role")}
    context.update(joint_masses=masses,row_nodes=nodes,num_agents=n,pet_seconds=y,fold=np.full(len(y),fold,np.int64),
                   **{CONTEXT_KEY:shape.astype(np.float32),PIECES_KEY:inverse.compiled_pieces().cpu().numpy()})
    # All values used as conditions derive from H; observed y enters labels only.
    return labels,context


def run_fold(pb,fold):
    p,prep,prep_b=inputs(pb);cfg=p["oof"]
    if str(fold) not in prep["folds"]:raise ValueError("fold must be 0..4")
    root=c.OUTPUT/"crossfit"/f"fold_{fold}";root.mkdir(exist_ok=False)
    fit=old._load_pack(prep["physical_packs"]["FIT"],"FIT")
    held=prep["folds"][str(fold)];mask=np.isin(fit["recording_id"],held)
    train_rows,held_rows=np.flatnonzero(~mask),np.flatnonzero(mask)
    norm=old.fit_fold_normalizer(fit,train_rows,held,fold);norm_b=io.write_json(root/"normalizer.json",norm)
    normalized=old.normalized_subset(fit,train_rows,norm)
    device=torch.device(p["runtime"]["device"])
    with (root/"epochs.jsonl").open("x") as log:model,training=train_fixed(normalized,cfg,device,log,fold)
    header=dict(protocol=PROTOCOL,policy=pb,prepared=prep_b,fold=fold,normalizer=norm_b,
                train_recordings=norm["training_recordings"],held_recordings=held,training_rows=len(train_rows),held_rows=len(held_rows),
                fixed_base_epochs=cfg["base_epochs"],fixed_highN_epochs=cfg["highN_epochs"],seed=cfg["seed"],
                initialization=training["initialization"],initial_state_sha256=training["initial_state_sha256"],
                held_or_STOP_model_selection=False,architecture=model.architecture_config(),code_sha256=prep["code_sha256"])
    with (root/"teacher.pt").open("xb") as handle:torch.save(dict(header=header,state_dict={k:v.cpu().clone() for k,v in model.state_dict().items()}),handle)
    cp_b=c.bind(root/"teacher.pt")
    io.write_json(root/"training_report.json",training)
    # Teacher weights and normalization freeze BEFORE its own CAL warp is fitted.
    cal=old._load_pack(prep["physical_packs"]["CAL"],"CAL")
    cal_mass=old.predict_masses(model,old.normalized_subset(cal,np.arange(len(cal["target"])),norm),cfg["batch_size"])
    if prep["selected_family"]=="identity":warp=c.StableCountWarp.identity();fit_report=dict(success=True,identity=True)
    else:
        fitted=fit_scene_calibration(cal_mass,cal["agent_mask"].sum(1),cal["target"],family=prep["selected_family"],ridge=prep["selected_ridge"])
        if not fitted.report["success"]:raise RuntimeError("fold CAL optimizer failed")
        warp=c.StableCountWarp.from_dict(fitted.warp.as_dict());fit_report=fitted.report
    warp_b=io.write_json(root/"calibration_model.json",warp.as_dict())
    io.save_pack(root/"CAL_raw_predictions.npz",dict(joint_masses=cal_mass,target=cal["target"],num_agents=cal["agent_mask"].sum(1),scene_id=cal["scene_id"],recording_id=cal["recording_id"]))
    cal_report=io.write_json(root/"calibration_report.json",dict(role="CAL",rows=504,recordings=sorted(set(cal["recording_id"])),
        family=prep["selected_family"],ridge=prep["selected_ridge"],fit=fit_report,fullFIT_nodes_reused=False))
    held_physical={k:v[held_rows] for k,v in fit.items()}
    held_mass=old.predict_masses(model,old.normalized_subset(fit,held_rows,norm),cfg["batch_size"])
    labels,context=payload(held_mass,held_physical,warp,fold)
    lb=io.save_pack(root/"labels.npz",labels);cb=io.save_pack(root/"context.npz",context)
    result=dict(header,status="complete",checkpoint=cp_b,labels=lb,context=cb,calibration_model=warp_b,calibration_report=cal_report,
                training_trace=c.bind(root/"epochs.jsonl"),native_future_trajectories_decoded=False,CAL_not_generator_data=True)
    c.verify_sources(prep["code_sha256"]);io.write_json(root/"result.json",result)
    print(json.dumps(dict(stage="fold_complete",fold=fold,training_rows=len(train_rows),held_rows=len(held_rows))),flush=True)


def aggregate(pb):
    p,prep,prep_b=inputs(pb);root=c.OUTPUT/"crossfit"
    physical=old._load_pack(prep["physical_packs"]["FIT"],"FIT")
    lookup={str(s):i for i,s in enumerate(physical["scene_id"])};seen=np.zeros(len(lookup),bool)
    assembled_labels=None;assembled_context=None;fold_bindings={};initials=[]
    for fold in range(5):
        rb=c.bind(root/f"fold_{fold}/result.json");r=c.json_file(rb);c.verify_sources(r["code_sha256"])
        if r["policy"]!=pb or r["prepared"]!=prep_b or r["status"]!="complete" or r["held_recordings"]!=prep["folds"][str(fold)]:
            raise ValueError("mismatched teacher completion")
        if set(r["held_recordings"])&set(r["train_recordings"]):raise ValueError("teacher recording leakage")
        labels=c.arrays(r["labels"]);context=c.arrays(r["context"])
        if not np.array_equal(labels["scene_id"],context["scene_id"]):raise ValueError("labels and CDF context must align")
        rows=np.array([lookup[str(s)] for s in labels["scene_id"]]);old._validate_arrays(labels)
        if seen[rows].any() or not np.all(np.isin(labels["recording_id"],r["held_recordings"])):raise ValueError("held coverage changed")
        if not np.array_equal(labels["pet_seconds"],physical["target"][rows]):raise ValueError("natural PET label changed")
        if assembled_labels is None:
            assembled_labels={k:np.empty((len(lookup),)+v.shape[1:],dtype=v.dtype) for k,v in labels.items()}
            assembled_context={k:np.empty((len(lookup),)+v.shape[1:],dtype=v.dtype) for k,v in context.items()}
        for k,v in labels.items():assembled_labels[k][rows]=v
        for k,v in context.items():assembled_context[k][rows]=v
        seen[rows]=True;fold_bindings[str(fold)]=rb;initials.append(r["initial_state_sha256"])
    if not seen.all() or len(set(initials))!=1:raise ValueError("complete same-seed fresh-teacher coverage required")
    old._validate_arrays(assembled_labels)
    fit_labels=io.save_pack(root/"FIT_labels.npz",assembled_labels);fit_context=io.save_pack(root/"FIT_context.npz",assembled_context)
    # STOP gets full-FIT Transformer reference, never an OOF label substitution.
    stop=old._load_pack(prep["physical_packs"]["STOP"],"STOP");reference=c.json_file(prep["reference_manifest"])
    normalizer=c.json_file(reference["normalizer"]);model=c.make_reference_model()
    cp=torch.load(io.verify_binding(reference["checkpoint"]),map_location="cpu",weights_only=False)
    model.load_state_dict(cp["state_dict"],strict=True);model.to(p["runtime"]["device"]).eval().requires_grad_(False)
    mass=old.predict_masses(model,old.normalized_subset(stop,np.arange(len(stop["target"])),normalizer),p["oof"]["batch_size"])
    warp=c.StableCountWarp.from_dict(c.json_file(reference["calibration_model"]))
    labels,context=payload(mass,stop,warp,-1)
    sb=io.save_pack(root/"STOP_labels.npz",labels);sc=io.save_pack(root/"STOP_context.npz",context)
    manifest=dict(protocol=PROTOCOL,status="complete",policy=pb,prepared=prep_b,reference_manifest=prep["reference_manifest"],
        data=prep["source_data"],physical_packs=prep["physical_packs"],fold_artifacts=fold_bindings,
        roles=dict(FIT=dict(rows=9913,labels=fit_labels,context=fit_context,label_source="recording_excluded_TimeAttn_teacher_with_own_CAL_warp"),
                   STOP=dict(rows=538,labels=sb,context=sc,label_source="fullFIT_TimeAttn_reference")),
        code_sha256=prep["code_sha256"],fullFIT_in_sample_label_substitution=False,random_within_atom_labels=False,
        CAL_generator_examples=False,native_future_trajectories_decoded=False)
    io.write_json(root/"manifest.json",manifest)
    print(json.dumps(dict(stage="crossfit_complete",FIT=9913,STOP=538,reference=prep["reference_manifest"])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["prepare","fold","aggregate"])
    parser.add_argument("--policy",required=True);parser.add_argument("--policy-sha256",required=True);parser.add_argument("--fold",type=int)
    a=parser.parse_args();os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8");torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    pb=dict(path=str(io.resolve(a.policy)),sha256=a.policy_sha256)
    if a.command=="fold":run_fold(pb,a.fold)
    else:{"prepare":prepare,"aggregate":aggregate}[a.command](pb)
