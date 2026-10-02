#!/usr/bin/env python3
"""Two-arm scratch history-encoder study, FIT/STOP only and no promotion."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcontrol.research import train_natural_scene_reference as io
from pcontrol.reference.scene_models import SceneCDFReference
from pcontrol.reference.time_attention_cdf import TimeAttentionSceneCDF
from pcontrol.reference.mixed_cdf import cdf_from_params, quantile_from_params
from pcontrol.reference.scores import crps_from_params

PROTOCOL = "natural_M2_time_attention_scratch_pilot_v1"
ARMS = ("M2_MLP", "M2_TimeAttn")
ENCODERS = ("vehicle_encoder.", "relative_history_encoder.")
CODE = ("pcontrol/research/pilot_time_attention_cdf.py", "pcontrol/reference/time_attention_cdf.py", *io.CODE)


def codes():
    return {name: io.sha256(ROOT/name) for name in CODE}


def binding(path):
    path = io.resolve(path)
    return dict(path=str(path), sha256=io.sha256(path))


def validate_policy(p):
    if (p["protocol"] != PROTOCOL or p["arms"] != list(ARMS)
            or p["scope"] != "Stage_A_history_encoder_only_FIT_STOP_development"
            or p["evaluation"]["allowed_roles"] != ["FIT", "STOP"]
            or p["evaluation"]["CAL_decoded"] or p["evaluation"]["AUDIT_decoded"]
            or not all(p["evaluation"][k] for k in ("no_calibration_refit", "no_OOF_relabeling", "no_generator_change_or_sampling", "no_production_promotion"))
            or p["training"]["from_scratch"] is not True or p["training"]["pretrained_weights"]
            or p["training"]["multiseed"] or p["model"] != dict(hidden_dim=64, heads=4, bins=64,
                cap_seconds=4., zero_atom_enabled=True, temporal_layers=2, temporal_feedforward_dim=128, dropout=0.)
            or p["training"]["loss"] != "analytic_CRPS_over_cap; stage2_positive_H_only_weights_with_fixed_FIT_mean"
            or p["runtime"]["AMP"] or p["runtime"]["TF32"] or not p["runtime"]["deterministic"]):
        raise ValueError("policy changes the bounded history-encoder-only study")
    return p


def inputs(policy_path, checksum):
    pb = dict(path=str(io.resolve(policy_path)), sha256=checksum)
    policy = validate_policy(io.load_json(pb))
    prepared, _ = io.read_prepared(policy["reference_prepared"])
    archived = io.load_json(policy["archived_reference_result"])
    if prepared["counts"] != dict(FIT=9913, STOP=538) or set(prepared["packs"]) != {"FIT", "STOP"}:
        raise ValueError("only frozen complete FIT9913/STOP538 allowed")
    if (archived["status"] != "complete" or archived["recipe"] != "highN"
            or archived["best_epoch"] != 7):
        raise ValueError("frozen high-N reference required as the historical fallback")
    io.verify_binding(archived["checkpoint"])
    for item in prepared["packs"].values(): io.verify_binding(item)
    return policy, pb, prepared, archived


def seed_runtime(policy):
    cfg = policy["runtime"]; seed = policy["training"]["seed"]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(cfg["threads"]); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.mha.set_fastpath_enabled(False)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("declared CUDA device unavailable; do not silently change runtime")
    return device


def make_model(arm, policy):
    if arm not in ARMS: raise ValueError("undeclared arm")
    torch.manual_seed(policy["training"]["seed"])
    cfg = policy["model"]; knots = torch.linspace(0,4,65,dtype=torch.float64)
    if arm == "M2_MLP": return SceneCDFReference("M2",knots,hidden_dim=64,heads=4)
    return TimeAttentionSceneCDF(knots,hidden_dim=64,heads=4,temporal_layers=cfg["temporal_layers"],
                                 temporal_feedforward_dim=cfg["temporal_feedforward_dim"])


def tensor_hash(state):
    h = hashlib.sha256()
    for name,value in sorted(state.items()):
        value=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(value.dtype).encode());h.update(str(value.shape).encode());h.update(value.tobytes())
    return h.hexdigest()


def prepare(policy_path, checksum):
    policy, pb, prepared, archived = inputs(policy_path,checksum)
    root = io.resolve(policy["output_root"]);root.mkdir(parents=True,exist_ok=False)
    a=make_model("M2_MLP",policy);b=make_model("M2_TimeAttn",policy)
    sa={k:v for k,v in a.state_dict().items() if not k.startswith(ENCODERS)}
    sb={k:v for k,v in b.state_dict().items() if not k.startswith(ENCODERS)}
    if sa.keys()!=sb.keys() or any(not torch.equal(sa[k],sb[k]) for k in sa):
        raise ValueError("nonencoder scratch initialization differs")
    report=dict(protocol=PROTOCOL,status="prepared",policy=pb,code_sha256=codes(),
        data=prepared["data"],normalizer=prepared["normalizer"],packs=prepared["packs"],counts=prepared["counts"],
        FIT_recordings=prepared["FIT_recordings"],STOP_recordings=prepared["STOP_recordings"],
        initial_nonencoder_sha256=tensor_hash(sa),
        architectures={"M2_MLP":a.architecture_config(),"M2_TimeAttn":b.architecture_config()},
        archived_reference_checkpoint=archived["checkpoint"],archived_raw_STOP=archived["best_STOP"],
        arrays_decoded=False,old_weights_loaded_for_initialization=False,CAL_decoded=False,AUDIT_decoded=False,
        generator_touched=False,calibration_changed=False,production_promotion=False,
        environment=dict(python=platform.python_version(),numpy=np.__version__,torch=torch.__version__,
                         cuda=torch.version.cuda,device=policy["runtime"]["device"]))
    out=io.write_json(root/"prepared.json",report)
    print(json.dumps(dict(stage="prepared",binding=out,parameters={k:v["parameter_count"] for k,v in report["architectures"].items()})),flush=True)


def read_frozen(policy_path,checksum):
    policy,pb,prepared,archived=inputs(policy_path,checksum)
    root=io.resolve(policy["output_root"]);fb=binding(root/"prepared.json");frozen=io.load_json(fb)
    if frozen["policy"]!=pb or frozen["code_sha256"]!=codes() or frozen["packs"]!=prepared["packs"]:
        raise ValueError("code/policy/input changed since preparation")
    return policy,pb,prepared,archived,root,fb,frozen


def load_allowed_pack(item,role,prepared):
    if role not in ("FIT","STOP"):raise PermissionError("only FIT/STOP may be decoded")
    pack=io.load_pack(item,role)
    if (len(pack["target"])!=prepared["counts"][role]
            or sorted(set(pack["recording_id"]))!=prepared[role+"_recordings"]
            or len(set(pack["scene_id"]))!=len(pack["target"])):
        raise ValueError("natural complete scene identities/roles changed")
    return pack


def resident(pack,device):
    return {k:torch.from_numpy(pack[k]).to(device) for k in (*io.FEATURES,"target")}


def batch(pack,rows):
    if isinstance(rows,slice):return {k:pack[k][rows] for k in io.FEATURES},pack["target"][rows]
    index=torch.as_tensor(rows,device=pack["target"].device,dtype=torch.long)
    return {k:pack[k].index_select(0,index) for k in io.FEATURES},pack["target"].index_select(0,index)


def stop_scores(model,data,counts,batch_size):
    model.eval();values=[]
    with torch.no_grad():
        for start in range(0,len(counts),batch_size):
            features,y=batch(data,slice(start,start+batch_size))
            values.append(crps_from_params(model(features),y).cpu().numpy())
    values=np.concatenate(values)
    if not np.isfinite(values).all():raise FloatingPointError("nonfinite STOP CRPS")
    overall=float(values.mean());high=float(values[counts>=9].mean())
    return dict(overall=overall,highN=high,selection=.5*(overall+high))


def stage_selection(scores,initial,stage,cfg):
    if stage=="stage1":return True,scores["overall"]
    valid=(scores["overall"]<=initial["overall"]*cfg["overall_guard_vs_stage1"]
           and scores["highN"]<=initial["highN"]*cfg["highN_guard_vs_stage1"])
    return bool(valid),scores["selection"]


def save_checkpoint(path,model,header):
    value=dict(header,state_dict={k:v.detach().cpu().clone() for k,v in model.state_dict().items()})
    with path.open("xb") as handle:torch.save(value,handle)
    return binding(path)


def train_stage(model,fit,stop,fit_counts,stop_counts,policy,arm,stage,output,header):
    output.mkdir(exist_ok=False)
    cfg=policy["training"][stage];global_cfg=policy["training"];bs=global_cfg["batch_size"]
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=global_cfg["weight_decay"])
    raw=np.where(fit_counts>=9,cfg["highN_weight"],1.).astype(np.float64)
    weights=torch.as_tensor(raw/raw.mean(),device=fit["target"].device)
    rng=np.random.default_rng(global_cfg["seed"]+(0 if stage=="stage1" else 1000))
    initial=stop_scores(model,stop,stop_counts,bs)
    best=dict(initial);best_epoch=0;best_value=stage_selection(best,initial,stage,cfg)[1]
    significant=best_value;stale=0;best_state=copy.deepcopy(model.state_dict())
    started=time.perf_counter();steps=0;all_order_digest=hashlib.sha256()
    with (output/"epochs.jsonl").open("x") as log:
        for epoch in range(1,cfg["epochs"]+1):
            model.train();order=rng.permutation(len(fit_counts));order_hash=hashlib.sha256(order.tobytes()).hexdigest()
            all_order_digest.update(order.tobytes());total=0.
            for start in range(0,len(order),bs):
                rows=order[start:start+bs];features,y=batch(fit,rows)
                optimizer.zero_grad(set_to_none=True)
                scores=crps_from_params(model(features),y,normalized=True)
                indices=torch.as_tensor(rows,device=weights.device)
                loss=(scores*weights[indices]).mean()
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite training CRPS")
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),global_cfg["gradient_clip_norm"])
                if not torch.isfinite(norm):raise FloatingPointError("nonfinite gradients")
                optimizer.step();total+=float(loss.detach())*len(rows)*4.;steps+=1
            observed=stop_scores(model,stop,stop_counts,bs)
            valid,value=stage_selection(observed,initial,stage,cfg)
            if valid and value<best_value:
                best,best_value,best_epoch,best_state=dict(observed),value,epoch,copy.deepcopy(model.state_dict())
            if valid and value<significant-cfg["minimum_improvement_seconds"]:
                significant,stale=value,0
            else:stale+=1
            row=dict(arm=arm,stage=stage,epoch=epoch,STOP=observed,eligible=valid,best_STOP=best,best_epoch=best_epoch,
                     stale=stale,online_train_CRPS_seconds=total/len(fit_counts),order_sha256=order_hash,
                     FIT_observations=len(order),optimizer_steps=steps,wall_seconds=time.perf_counter()-started)
            log.write(json.dumps(row,sort_keys=True)+"\n");log.flush()
            print(json.dumps(row,sort_keys=True),flush=True)
            if stale>=cfg["patience"]:break
    model.load_state_dict(best_state,strict=True)
    replay=stop_scores(model,stop,stop_counts,bs)
    if any(abs(replay[k]-best[k])>1e-10 for k in best):raise ValueError("best checkpoint does not replay")
    cp=save_checkpoint(output/"best.pt",model,dict(header,stage=stage,best_epoch=best_epoch,STOP=best))
    result=dict(status="complete",arm=arm,stage=stage,initial_STOP=initial,best_STOP=best,best_epoch=best_epoch,
                epochs_completed=epoch,optimizer_steps=steps,checkpoint=cp,epochs=binding(output/"epochs.jsonl"),
                order_sha256=all_order_digest.hexdigest(),wall_seconds=time.perf_counter()-started,
                weight_min=float(weights.min()),weight_max=float(weights.max()),stage1_or_epoch0_fallback=best_epoch==0)
    io.write_json(output/"result.json",result)
    return result


def tail_crps(params,y,upper):
    widths=params.knots[1:]-params.knots[:-1]
    masses=params.continuous_masses;zero=params.zero_mass
    end=zero[:,None]+masses.cumsum(-1);start=torch.cat((zero[:,None],end[:,:-1]),-1)
    length=torch.minimum((upper-params.knots[:-1]).clamp_min(0.),widths)
    below=torch.minimum((y[:,None]-params.knots[:-1]).clamp_min(0.),length)
    split=start+masses*below/widths;hi=start+masses*length/widths
    a,b=split-1.,hi-1.
    return (below*(start.square()+start*split+split.square())/3.+(length-below)*(a.square()+a*b+b.square())/3.).sum(-1)


def prediction_arrays(model,data,pack,policy):
    model.eval();chunks={k:[] for k in ("joint_masses","crps_seconds","twCRPS_1s","twCRPS_2s","cdf_left","cdf_right","threshold_cdf","quantiles")}
    device=data["target"].device;ev=policy["evaluation"]
    thresholds=torch.tensor(ev["thresholds_seconds"],dtype=torch.float64,device=device)[None]
    levels=torch.tensor(ev["quantile_levels"],dtype=torch.float64,device=device)[None]
    with torch.no_grad():
        for start in range(0,len(pack["target"]),policy["training"]["batch_size"]):
            f,y=batch(data,slice(start,start+policy["training"]["batch_size"]));p=model(f)
            values=dict(joint_masses=p.joint_masses,crps_seconds=crps_from_params(p,y),
                        twCRPS_1s=tail_crps(p,y,1.),twCRPS_2s=tail_crps(p,y,2.),
                        cdf_left=cdf_from_params(p,y,side="left"),cdf_right=cdf_from_params(p,y),
                        threshold_cdf=cdf_from_params(p,thresholds),quantiles=quantile_from_params(p,levels))
            for key,value in values.items():chunks[key].append(value.cpu().numpy())
    a={k:np.concatenate(v) for k,v in chunks.items()}
    a.update(target=pack["target"],scene_id=pack["scene_id"],recording_id=pack["recording_id"],num_agents=pack["agent_mask"].sum(1))
    a["pit_uniform"]=np.random.default_rng(ev["PIT_seed"]).uniform(size=len(a["target"]))
    a["randomized_pit"]=a["cdf_left"]+a["pit_uniform"]*(a["cdf_right"]-a["cdf_left"])
    return a


def diagnostics(a,policy):
    summary=io.summarize_predictions(a,policy)
    high=a["num_agents"]>=9
    summary.update(highN_scenes=int(high.sum()),highN_CRPS_seconds=float(a["crps_seconds"][high].mean()),
                   twCRPS_1s=float(a["twCRPS_1s"].mean()),twCRPS_2s=float(a["twCRPS_2s"].mean()),
                   highN_twCRPS_1s=float(a["twCRPS_1s"][high].mean()))
    grid=np.linspace(.05,.95,19);lo=a["cdf_left"];hi=a["cdf_right"];width=hi-lo
    pit=np.where((width>0)[:,None],np.clip((grid[None]-lo[:,None])/np.where(width>0,width,1.)[:,None],0.,1.),hi[:,None]<=grid[None]).mean(0)
    levels=np.asarray(policy["evaluation"]["quantile_levels"]);res=a["target"][:,None]-a["quantiles"]
    summary.update(expected_PIT_grid_MAE=float(abs(pit-grid).mean()),expected_PIT_grid_max=float(abs(pit-grid).max()),
                   pinball=np.maximum(levels*res,(levels-1)*res).mean(0).tolist(),
                   by_N={name:dict(scenes=int(mask.sum()),CRPS_seconds=float(a["crps_seconds"][mask].mean()))
                         for name,mask in (("N3_5",a["num_agents"]<6),("N6_8",(a["num_agents"]>=6)&(a["num_agents"]<9)),("N9plus",high))})
    return summary


def train(policy_path,checksum,arm):
    if arm not in ARMS:raise ValueError("undeclared arm")
    policy,pb,prepared,archived,root,fb,frozen=read_frozen(policy_path,checksum)
    device=seed_runtime(policy);out=root/arm;out.mkdir(exist_ok=False)
    fit_pack=load_allowed_pack(prepared["packs"]["FIT"],"FIT",prepared)
    stop_pack=load_allowed_pack(prepared["packs"]["STOP"],"STOP",prepared)
    fit,stop=resident(fit_pack,device),resident(stop_pack,device)
    fit_counts=fit_pack["agent_mask"].sum(1);stop_counts=stop_pack["agent_mask"].sum(1)
    m=make_model(arm,policy)
    common={k:v for k,v in m.state_dict().items() if not k.startswith(ENCODERS)}
    if tensor_hash(common)!=frozen["initial_nonencoder_sha256"]:raise ValueError("initial shared readout tensors differ")
    m=m.to(device);header=dict(protocol=PROTOCOL,arm=arm,prepared=fb,policy=pb,data=prepared["data"],
                              normalizer=prepared["normalizer"],code_sha256=frozen["code_sha256"],seed=policy["training"]["seed"],
                              architecture=m.architecture_config())
    torch.cuda.reset_peak_memory_stats(device);start=time.perf_counter();stages={};preds={};metrics={}
    for stage in ("stage1","stage2"):
        stages[stage]=train_stage(m,fit,stop,fit_counts,stop_counts,policy,arm,stage,out/stage,header)
        a=prediction_arrays(m,stop,stop_pack,policy);preds[stage]=io.save_pack(out/(stage+"_STOP_predictions.npz"),a)
        metrics[stage]=diagnostics(a,policy)
    # The archived reference is evaluated only as a fixed comparator, never as
    # scratch initialization or training supervision. Hash-authenticated local
    # checkpoint permits legacy numpy metadata during deserialization.
    old_cp=torch.load(io.verify_binding(archived["checkpoint"]),map_location="cpu",weights_only=False)
    old=SceneCDFReference("M2",torch.linspace(0,4,65,dtype=torch.float64),hidden_dim=64,heads=4)
    if old_cp["data"]!=prepared["data"] or old_cp["normalizer"]!=prepared["normalizer"]:
        raise ValueError("archived reference data/normalizer changed")
    old.load_state_dict(old_cp["state_dict"],strict=True);old=old.to(device)
    original=prediction_arrays(old,stop,stop_pack,policy);old_metrics=diagnostics(original,policy)
    if abs(old_metrics["CRPS_seconds"]-archived["best_STOP"]["overall"])>1e-6:
        raise ValueError("archived raw STOP score not numerically compatible")
    old_pred=io.save_pack(out/"archived_raw_M2_STOP_predictions.npz",original)
    if codes()!=frozen["code_sha256"]:raise ValueError("source changed during training")
    result=dict(protocol=PROTOCOL,status="complete",arm=arm,prepared=fb,policy=pb,code_sha256=codes(),
        architecture=header["architecture"],initial_nonencoder_sha256=frozen["initial_nonencoder_sha256"],
        stages=stages,STOP_metrics=metrics,predictions=preds,archived_raw_M2=old_metrics,archived_predictions=old_pred,
        checkpoint=stages["stage2"]["checkpoint"],wall_seconds=time.perf_counter()-start,
        peak_allocated_gpu_MiB=torch.cuda.max_memory_allocated(device)/2**20,
        CAL_decoded=False,AUDIT_decoded=False,calibration_fitted=False,generator_changed=False,
        production_reference_changed=False,scope="STOP_development_single_seed_not_blind_or_capacity_matched")
    rb=io.write_json(out/"result.json",result)
    print(json.dumps(dict(stage="arm_complete",arm=arm,result=rb,STOP=metrics["stage2"])),flush=True)


def summarize(policy_path,checksum):
    policy,pb,prepared,archived,root,fb,frozen=read_frozen(policy_path,checksum)
    results={arm:io.load_json(binding(root/arm/"result.json")) for arm in ARMS}
    for arm,r in results.items():
        if r["status"]!="complete" or r["arm"]!=arm or r["policy"]!=pb or r["prepared"]!=fb or r["code_sha256"]!=codes():
            raise ValueError("completed matched frozen arms required")
    paired_orders={}
    for stage in ("stage1","stage2"):
        logs=[[json.loads(line) for line in io.verify_binding(results[a]["stages"][stage]["epochs"]).read_text().splitlines()] for a in ARMS]
        common=min(map(len,logs))
        if any(logs[0][i]["order_sha256"]!=logs[1][i]["order_sha256"] for i in range(common)):
            raise ValueError("matching stage/epochs did not visit FIT in the same order")
        paired_orders[stage]=common
    baseline=results["M2_MLP"]["STOP_metrics"]["stage2"];candidate=results["M2_TimeAttn"]["STOP_metrics"]["stage2"]
    historical=results["M2_MLP"]["archived_raw_M2"]
    checks=dict(overall_1pct_vs_scratch=candidate["CRPS_seconds"]<=.99*baseline["CRPS_seconds"],
                highN_vs_scratch=candidate["highN_CRPS_seconds"]<=baseline["highN_CRPS_seconds"],
                lowPET_vs_scratch=candidate["twCRPS_1s"]<=baseline["twCRPS_1s"],
                overall_vs_archived=candidate["CRPS_seconds"]<=historical["CRPS_seconds"],
                highN_vs_archived=candidate["highN_CRPS_seconds"]<=historical["highN_CRPS_seconds"])
    success=all(checks.values())
    result=dict(protocol=PROTOCOL,status="complete",policy=pb,prepared=fb,code_sha256=codes(),
        arms={a:binding(root/a/"result.json") for a in ARMS},final_STOP={a:r["STOP_metrics"]["stage2"] for a,r in results.items()},
        archived_raw_M2=historical,gate_checks=checks,candidate_passes_development_gate=success,
        recommendation="candidate_ready_for_separate_reference_validation" if success else "retain_archived_reference; do_not_promote_candidate",
        matching_order_epochs=paired_orders,same_data_and_loss_and_budget_caps=True,equal_parameters_or_flops_claimed=False,
        CAL_decoded=False,AUDIT_decoded=False,OOF_labels_changed=False,generator_changed=False,production_promotion=False)
    rb=io.write_json(root/"selection_development_only.json",result)
    print(json.dumps(dict(stage="study_complete",result=rb,checks=checks,gate=success,
                          CRPS={a:r["STOP_metrics"]["stage2"]["CRPS_seconds"] for a,r in results.items()})),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["prepare","train","summarize"])
    parser.add_argument("--policy",required=True);parser.add_argument("--policy-sha256",required=True)
    parser.add_argument("--arm",choices=ARMS)
    args=parser.parse_args()
    if args.command=="train":train(args.policy,args.policy_sha256,args.arm)
    else:{"prepare":prepare,"summarize":summarize}[args.command](args.policy,args.policy_sha256)
