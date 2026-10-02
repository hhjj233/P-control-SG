#!/usr/bin/env python3
"""Adapt the existing natural generator to new Transformer OOF coordinates.

Warm weights are explicit; all labels, shape conditions, inverse targets and
CDF-dependent tangents are new. The unchanged full-DDIM terminal objectives
run after natural v-MSE adaptation. Old optimizers and old p labels are NOT
silently carried over. No sampling-quality-driven checkpoint search is used.
"""
import argparse
import copy
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
from pcontrol.time_attention_pipeline import generator_data as gd
from pcontrol.generation.trainable_risk_generator import TrainableRiskGenerator
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.direct_p_cfg import cfg_training_loss
from pcontrol.generation.wide_risk_sampling import WideRiskHistoryCycle
from pcontrol.generation.slow_history_sampling import speed_strata
from pcontrol.generation.cdf_shape_context import CONTEXT_KEY
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import train_natural_direct_p as direct
from pcontrol.research import train_wide_coupled_risk_generator as legacy
from pcontrol.research.refine_natural_direct_p import PairedStreams,state_hash
from pcontrol.research.finetune_joint_risk_generator_v2 import update_adapter_ema

PROTOCOL="time_attention_conditioned_generator_training_v1"
CODE=("pcontrol/research/train_time_attention_generator.py","pcontrol/time_attention_pipeline/generator_data.py",
      "pcontrol/time_attention_pipeline/common.py","pcontrol/reference/time_attention_cdf.py",
      "pcontrol/generation/broadphase_geometry.py","pcontrol/data/scene_pet_broadphase.py",*legacy.CODE)


def runtime(p):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    torch.set_num_threads(p["runtime"]["threads"]);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    # Natural p-Jacobian losses require higher-order autograd. Math attention
    # implements it; fused flash/memory-efficient attention need not do so.
    torch.backends.cuda.enable_flash_sdp(False);torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.manual_seed(p["generator"]["seed"])
    return torch.device(p["runtime"]["device"])


def active_terminal_settings(p):
    original=legacy.read_policy(dict(path="configs/natural_percentile/wide_coupled_risk_continuation_v1.json",sha256=legacy.POLICY_SHA))
    keep=("batch_size","gradient_clip_norm","EMA_decay","p_dropout_probability","loss_weights","min_jacobian_sigma",
          "terminal_DDIM_steps","grad_last_steps","terminal_CFG_scale","terminal_beta","minimum_density","numeric_weight",
          "PET_target_weight","PET_target_beta","Fine_band_weight","Fine_band_temperature","encounter_weight","encounter_margin_m")
    cfg={k:copy.deepcopy(original["training"][k]) for k in keep}
    if cfg["terminal_DDIM_steps"]!=50 or cfg["grad_last_steps"]!=50:raise ValueError("retain full DDIM forward/backward")
    coverage={k:copy.deepcopy(original["coverage"][k]) for k in ("slots_per_history","grid","interior_margin","atom_tolerance")}
    return cfg,copy.deepcopy(original["physics"]),coverage


def prepare(pb):
    source=gd.load_inputs(pb);p=source["policy"]
    root=c.OUTPUT/"generator_data";root.mkdir(exist_ok=False)
    code=c.source_bindings(CODE)
    io.write_json(root/"freeze_before_cache.json",dict(policy=pb,labels_manifest=source["labels_binding"],
        generator_data=p["generator_data"],reference_manifest=source["labels_manifest"]["reference_manifest"],code_sha256=code))
    cache,summary=gd.regenerate_tangent(source)
    cb=io.save_pack(root/"FIT_tangents.npz",cache)
    joins=io.write_json(root/"label_join.json",source["joins"])
    cfg,physics,coverage=active_terminal_settings(p)
    manifest=dict(protocol=PROTOCOL,status="prepared",policy=pb,labels_manifest=source["labels_binding"],
        reference_manifest=source["labels_manifest"]["reference_manifest"],generator_data=p["generator_data"],
        tangent_cache=cb,tangent_summary=summary,label_joins=joins,terminal_settings=cfg,physics=physics,coverage=coverage,
        counts=dict(FIT=9913,STOP=538),code_sha256=code,calibration_or_AUDIT_examples_in_generator=False,
        natural_coefficients_unchanged=True,new_Transformer_labels_and_conditions=True,old_density_reused=False)
    c.verify_sources(code);io.write_json(root/"manifest.json",manifest)
    print(json.dumps(dict(stage="generator_data_prepared",tangent_summary=summary,label_joins=source["joins"])),flush=True)


def load_source(pb):
    mb=c.bind(c.OUTPUT/"generator_data/manifest.json");m=c.json_file(mb)
    if m["policy"]!=pb or m["status"]!="prepared":raise ValueError("new complete generator preparation required")
    c.verify_sources(m["code_sha256"]);s=gd.load_inputs(pb)
    if s["labels_binding"]!=m["labels_manifest"]:raise ValueError("generator labels changed")
    s.update(preparation=m,preparation_binding=mb,cache=c.arrays(m["tangent_cache"]))
    return s


def make_model(architecture,state,device):
    kw={k:architecture[k] for k in ("coefficient_dim","hidden_dim","heads","layers","feedforward_dim","context_hidden_dim","dynamic_bottleneck")}
    with torch.random.fork_rng(devices=[]):model=TrainableRiskGenerator(**kw)
    model.load_state_dict(state,strict=True)
    if model.architecture_config()!=architecture or sum(p.numel() for p in model.parameters())!=980368:
        raise ValueError("generator architecture changed")
    return model.to(device)


def model_batch(source,role,rows,device):
    f,clean,p=direct.tensor_batch(source["packs"][role],source["pvalues"][role],rows,device)
    ctx=source["contexts"][role]
    f[CONTEXT_KEY]=torch.tensor(ctx[CONTEXT_KEY][rows],dtype=torch.float32,device=device)
    f[PIECES_KEY]=torch.tensor(ctx[PIECES_KEY][rows],dtype=torch.float64,device=device)
    return f,clean,p


def validation(model,schedule,source,device):
    model.eval();pack=source["packs"]["STOP"];rng=np.random.default_rng(20261015)
    values=[]
    with torch.no_grad():
        for step in (0,10,25,50,75,99):
            noise=rng.standard_normal(pack["coef_clean"].shape).astype(np.float32);chunks=[]
            for start in range(0,538,128):
                rows=slice(start,start+128);f,clean,p=model_batch(source,"STOP",rows,device)
                t=torch.full((len(clean),),step,dtype=torch.long,device=device)
                eps=torch.tensor(noise[rows,:clean.shape[1]],device=device)
                result=cfg_training_loss(model,schedule,clean,f,p,torch.ones(len(clean),dtype=torch.bool,device=device),timesteps=t,noise=eps)
                chunks.append(result["per_scene_loss"].cpu().numpy())
            values.append(float(np.concatenate(chunks).mean()))
    if not np.isfinite(values).all():raise FloatingPointError("nonfinite generator validation")
    return dict(mean=float(np.mean(values)),by_timestep=dict(zip(map(str,(0,10,25,50,75,99)),values)),
                rows=538,new_fullFIT_Transformer_p_and_context=True)


def cpu_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def checkpoint(path,model,source,pb,stage,epoch,extra=None):
    m=source["preparation"];data=source["data"]
    value=dict(protocol=PROTOCOL,policy=pb,stage=stage,epoch=epoch,state_dict=cpu_state(model),
        architecture=model.architecture_config(),generator_data=m["generator_data"],labels_manifest=m["labels_manifest"],
        reference_manifest=m["reference_manifest"],generator_preparation=source["preparation_binding"],
        basis=data["basis"],coefficient_normalizer=data["coefficient_normalizer"],history_normalizer=data["history_normalizer"],
        code_sha256=m["code_sha256"],**(extra or {}))
    with path.open("xb") as f:torch.save(value,f)
    return c.bind(path)


def adapt(pb):
    s=load_source(pb);p=s["policy"];cfg=p["generator"];device=runtime(p)
    root=c.OUTPUT/"generator"/"adaptation";root.mkdir(parents=True,exist_ok=False)
    cp=torch.load(io.verify_binding(p["generator_warm_start"]),map_location="cpu",weights_only=False)
    c.verify_sources(cp["code_sha256"])
    model=make_model(cp["architecture"],cp["model_state"],device)
    initial=state_hash(model)
    groups=model.finetuning_groups(cfg["core_learning_rate"],cfg["adapter_learning_rate"])
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4)
    ema=copy.deepcopy(model).requires_grad_(False);schedule=CosineDiffusionSchedule(100).to(device)
    freeze=io.write_json(root/"freeze_before_training.json",dict(policy=pb,preparation=s["preparation_binding"],
        warm_start=p["generator_warm_start"],initial_state_sha256=initial,old_optimizer_reused=False,
        old_p_labels_reused=False,source_codes=s["preparation"]["code_sha256"],fixed_validation_noise_seed=20261015))
    best=validation(ema,schedule,s,device);initial_validation=copy.deepcopy(best)
    best_state=cpu_state(ema);best_epoch=0;stale=0;significant=best["mean"]
    streams=PairedStreams(cfg["seed"],cfg["seed"]+1);started=time.monotonic();updates=0
    with (root/"epochs.jsonl").open("x") as logfile:
        for epoch in range(1,cfg["adaptation_max_epochs"]+1):
            model.train();order=streams.order(9913);total=0.;dropped=0
            for start in range(0,9913,cfg["batch_size"]):
                rows=order[start:start+cfg["batch_size"]];f,clean,labels=model_batch(s,"FIT",rows,device)
                times,noise,_,present=streams.draw(clean.shape,100,cfg["p_dropout_probability"])
                optimizer.zero_grad(set_to_none=True)
                result=cfg_training_loss(model,schedule,clean,f,labels,torch.tensor(present,device=device),timesteps=times.to(device),noise=noise.to(device))
                loss=result["loss"]
                if not torch.isfinite(loss):raise FloatingPointError("nonfinite adaptation loss")
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise FloatingPointError("nonfinite adaptation gradients")
                optimizer.step();update_adapter_ema(ema,model,cfg["EMA_decay"])
                total+=float(loss.detach())*len(rows);dropped+=int((~present).sum());updates+=1
            score=validation(ema,schedule,s,device)
            if score["mean"]<best["mean"]:best,best_epoch,best_state=score,epoch,cpu_state(ema)
            if score["mean"]<significant-1e-5:significant,stale=score["mean"],0
            else:stale+=1
            row=dict(epoch=epoch,FIT_rows=9913,mean_train_v_MSE=total/9913,STOP=score,best_epoch=best_epoch,
                     best_STOP=best,stale=stale,optimizer_updates=updates,p_dropped=dropped,
                     order_sha256=hashlib.sha256(order.tobytes()).hexdigest(),seconds=time.monotonic()-started)
            logfile.write(json.dumps(row)+"\n");logfile.flush();print(json.dumps(row),flush=True)
            if stale>=cfg["adaptation_patience"]:break
    ema.load_state_dict(best_state,strict=True)
    chosen=checkpoint(root/"best_ema.pt",ema,s,pb,"adaptation",best_epoch,dict(EMA=True,warm_start=p["generator_warm_start"]))
    c.verify_sources(s["preparation"]["code_sha256"])
    result=dict(protocol=PROTOCOL,status="complete",stage="adaptation",policy=pb,preparation=s["preparation_binding"],
        checkpoint=chosen,initial_validation=initial_validation,best_validation=best,best_epoch=best_epoch,
        epochs_completed=epoch,optimizer_updates=updates,warm_start=p["generator_warm_start"],freeze=freeze,
        labels_manifest=s["labels_binding"],reference_manifest=s["preparation"]["reference_manifest"],
        old_optimizer_reused=False,new_labels_and_context_used=True,all_FIT_rows_each_epoch=True,
        wall_seconds=time.monotonic()-started,code_sha256=s["preparation"]["code_sha256"])
    io.write_json(root/"result.json",result);print(json.dumps(dict(stage="adaptation_complete",best_epoch=best_epoch,STOP=best)),flush=True)


def terminal(pb,smoke=False):
    s=load_source(pb);p=s["policy"];cfg=p["generator"];device=runtime(p)
    adaptation=c.bind(c.OUTPUT/"generator/adaptation/result.json");r=c.json_file(adaptation)
    if r["policy"]!=pb or r["preparation"]!=s["preparation_binding"] or r["status"]!="complete":raise ValueError("complete new-label adaptation required")
    cp=torch.load(io.verify_binding(r["checkpoint"]),map_location="cpu",weights_only=False)
    model=make_model(cp["architecture"],cp["state_dict"],device)
    groups=model.finetuning_groups(cfg["core_learning_rate"],cfg["adapter_learning_rate"])
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-4);ema=copy.deepcopy(model).requires_grad_(False)
    schedule=CosineDiffusionSchedule(100).to(device);teacher=gd.TransformerFITTeacher(s,device)
    speed,strata=speed_strata(s["physical"]["history"],s["packs"]["FIT"]["agent_mask"],s["packs"]["FIT"]["role"])
    slow_cycle=WideRiskHistoryCycle(s["packs"]["FIT"]["recording_id"],strata,seed=cfg["seed"]+4)
    streams=PairedStreams(cfg["seed"]+10,cfg["seed"]+11);aux=torch.Generator().manual_seed(cfg["seed"]+12)
    settings=s["preparation"]["terminal_settings"];physics=s["preparation"]["physics"];coverage=s["preparation"]["coverage"]
    root=c.OUTPUT/"generator"/("terminal_smoke" if smoke else "terminal");root.mkdir(exist_ok=False)
    initial=state_hash(model);io.write_json(root/"freeze.json",dict(policy=pb,adaptation=adaptation,preparation=s["preparation_binding"],
        smoke=smoke,initial_state_sha256=initial,objective_implementation="unchanged_wide_coupled_train_epoch",
        terminal_settings=settings,physics=physics,coverage=coverage,auxiliary_noise="shared_within_H_across_3_P",
        CDF_source="own_Transformer_OOF",new_optimizer=True,full_generator_trainable=True))
    shape=dict(shape=s["contexts"]["FIT"][CONTEXT_KEY],pieces=s["contexts"]["FIT"][PIECES_KEY])
    started=time.monotonic();snapshots={};traces=[]
    with (root/"epochs.jsonl").open("x") as log:
        for epoch in range(1,(1 if smoke else cfg["terminal_epochs"])+1):
            row=legacy.train_epoch(model,ema,schedule,optimizer,s["packs"]["FIT"],s["pvalues"]["FIT"],s["cache"],teacher,
                streams,aux,settings,physics,coverage,"wide_coupled",device,shape_values=shape,slow_cycle=slow_cycle,
                max_updates=1 if smoke else None,progress=lambda x:print(json.dumps(dict(stage="terminal",epoch=epoch,**x)),flush=True))
            row.update(epoch=epoch,smoke=smoke);traces.append(row);log.write(json.dumps(row)+"\n");log.flush()
            raw=checkpoint(root/f"raw_epoch_{epoch:03d}.pt",model,s,pb,"terminal",epoch,dict(EMA=False,smoke=smoke))
            averaged=checkpoint(root/f"ema_epoch_{epoch:03d}.pt",ema,s,pb,"terminal",epoch,dict(EMA=True,smoke=smoke))
            snapshots[str(epoch)]=dict(raw=raw,EMA=averaged)
            print(json.dumps(dict(stage="terminal_epoch_complete",epoch=epoch,statistics=row)),flush=True)
    if not smoke:
        if any(x["base_loss_scenes"]!=9913 or x["updates"]!=78 or x["auxiliary_requests"]!=3744 for x in traces):
            raise ValueError("incomplete natural-base or auxiliary exposure")
        if initial==state_hash(model):raise ValueError("new-reference terminal training did not update generator")
    c.verify_sources(s["preparation"]["code_sha256"])
    result=dict(protocol=PROTOCOL,status="smoke_complete" if smoke else "complete",policy=pb,preparation=s["preparation_binding"],
        adaptation=adaptation,smoke=smoke,epochs_completed=len(traces),snapshots=snapshots,
        selected_checkpoint=snapshots[str(len(traces))]["raw"],selection_rule="predeclared_last_raw_checkpoint_not_STOP_Fine_search",
        labels_manifest=s["labels_binding"],reference_manifest=s["preparation"]["reference_manifest"],
        initial_state_sha256=initial,final_state_sha256=state_hash(model),
        wall_seconds=time.monotonic()-started,code_sha256=s["preparation"]["code_sha256"],
        CAL_AUDIT_examples_used=False,new_reference_all_training_interfaces_consistent=True)
    io.write_json(root/"result.json",result)
    print(json.dumps(dict(stage="terminal_complete",smoke=smoke,checkpoint=result["selected_checkpoint"])),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("command",choices=["prepare","adapt","terminal"])
    parser.add_argument("--policy",required=True);parser.add_argument("--policy-sha256",required=True);parser.add_argument("--smoke",action="store_true")
    a=parser.parse_args();pb=dict(path=str(io.resolve(a.policy)),sha256=a.policy_sha256)
    if a.command=="terminal":terminal(pb,a.smoke)
    else:{"prepare":prepare,"adapt":adapt}[a.command](pb)
