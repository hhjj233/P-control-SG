#!/usr/bin/env python3
"""Paired direct-P continuation versus p-only classifier-free training.

All old data, labels and weights remain immutable. Both recipes warm from the
same parent EMA and retain H/static inputs. Only the independent p-presence
mask differs. Fixed epoch120/240 snapshots are candidates; STOP v-MSE is a
diagnostic, not an automatic checkpoint choice or an early stopping signal.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.research import train_natural_direct_p as direct
from pcontrol.generation.direct_p import JointPercentileDenoiser
from pcontrol.generation.direct_p_cfg import ClassifierFreePercentileDenoiser,cfg_training_loss
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.data.complete_scene_view import resolve,verify_binding,sha256

PROTOCOL="natural_direct_P_paired_continuation_and_CFG_training_v1"
POLICY_SHA="78e67912b8de8890e4cb703f0738f3ec17670d12b0e13fa4cb2fbd673cda1f3a"
PARENT_SHA="0a43a9603629997d16c9376a3c8a7994bace102c33582784a7124febbe82e86e"
CHECKPOINT_SHA="c47f337b2d329541a465d1c4836338014498e6ff46fb76eae5f1b96d9b01b168"
CODE=("pcontrol/research/refine_natural_direct_p.py","pcontrol/generation/direct_p_cfg.py")


def code_hashes(parent):
    code=dict(parent["code_sha256"])
    for path,digest in code.items():verify_binding(dict(path=path,sha256=digest))
    code.update({path:sha256(ROOT/path) for path in CODE})
    return code


def state_hash(model):
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_policy(policy,recipe):
    if (policy.get("protocol")!="natural_direct_P_condition_strength_refinement_v1"
            or policy.get("recipes")!=["continue","cfg"] or recipe not in policy["recipes"]
            or policy["parent_training"]["sha256"]!=PARENT_SHA
            or policy["parent_checkpoint_sha256"]!=CHECKPOINT_SHA):
        raise ValueError("only the two registered parent-EMA refinement recipes are allowed")
    model,cfg=policy["model"],policy["training"]
    if (model["dropout_probability"]!={"continue":0.,"cfg":.15}
            or model["conditional_parent_replay_required"] is not True
            or model["drop_only_p_not_history_or_geometry"] is not True or model["new_risk_or_PET_input"] is not False):
        raise ValueError("the only ablation is p-presence, never history/static dropout or PET input")
    fixed=dict(seed=20260910,optimizer_state_restored=False,epochs=240,snapshot_epochs=[0,120,240],
        candidate_epochs=[120,240],validation_every=20,batch_size=128,learning_rate=.0003,weight_decay=.0001,
        gradient_clip_norm=1.,EMA_decay=.995,p_dropout_seed=20261021,record_epoch_randomness_hashes=True,
        validation_timesteps=[0,10,25,50,75,99],validation_noise_seed=20261010,labels_unchanged=True,
        extra_risk_or_geometry_loss=False,multiseed_replication=False)
    if any(cfg.get(k)!=v for k,v in fixed.items()):raise ValueError("fixed refinement budget/randomness/selection policy changed")
    if (cfg["diffusion_randomness"]!="dedicated_CPU_torch_generator_seed_plus_1_then_device_transfer"
            or cfg["scene_order_randomness"]!="dedicated_NumPy_generator_seed"
            or cfg["p_dropout_randomness"]!="dedicated_NumPy_generator_independent_of_order_and_diffusion_noise"
            or cfg["loss"]!="v_prediction_MSE_equal_scene_mean_over_actual_agents_and_coefficients"
            or policy["runtime"]!={"devices":{"continue":"cuda:0","cfg":"cuda:1"},"threads":2,"TF32":False}):
        raise ValueError("paired noise, standard loss and registered devices must remain fixed")


def read_inputs(policy_binding,recipe):
    if policy_binding["sha256"]!=POLICY_SHA:raise ValueError("unregistered refinement policy")
    policy=direct.prior.load_json(policy_binding);validate_policy(policy,recipe)
    parent=direct.prior.load_json(policy["parent_training"])
    if (parent.get("status")!="complete" or parent.get("protocol")!=direct.PROTOCOL
            or parent["checkpoint"]["sha256"]!=CHECKPOINT_SHA
            or parent.get("p_is_actual_denoiser_condition") is not True
            or parent.get("simulated_training_futures") is not False
            or parent.get("primary_sampling_post_correction") is not False):
        raise ValueError("parent is not the unchanged completed natural direct-P EMA")
    source=direct.prepare_inputs(parent["data"],parent["labels_manifest"],parent["policy"])
    if source["label_join"]!=direct.prior.load_json(parent["label_join"]):
        raise ValueError("refinement joined labels differ from the parent training labels")
    code_hashes(parent)
    # Only an internally produced exact hash-authorized checkpoint is unpickled.
    checkpoint=torch.load(verify_binding(parent["checkpoint"]),map_location="cpu")
    for key in ("protocol","data","policy","labels_manifest","label_join","p_definition","architecture",
                "basis","coefficient_normalizer","history_normalizer","seed","best_epoch","code_sha256"):
        if checkpoint[key]!=parent[key]:raise ValueError("parent checkpoint/header differs: "+key)
    if checkpoint["prediction_type"]!="v":raise ValueError("the parent prediction parameterization changed")
    return policy,parent,checkpoint,source


class PairedStreams:
    """Three independent streams; both recipes consume identical t/noise/order."""
    def __init__(self,seed,drop_seed):
        self.order_rng=np.random.default_rng(seed)
        self.diffusion_rng=torch.Generator(device="cpu").manual_seed(seed+1)
        self.drop_rng=np.random.default_rng(drop_seed)

    def order(self,rows):return self.order_rng.permutation(rows)

    def draw(self,shape,steps,drop_probability):
        if not 0<=drop_probability<=1:raise ValueError("invalid p-drop probability")
        timesteps=torch.randint(steps,(shape[0],),generator=self.diffusion_rng,dtype=torch.long)
        noise=torch.randn(tuple(shape),generator=self.diffusion_rng,dtype=torch.float32)
        uniforms=self.drop_rng.random(shape[0])
        present=uniforms>=drop_probability
        return timesteps,noise,uniforms,present


def warmed_models(checkpoint,device):
    cfg={k:checkpoint["architecture"][k] for k in ("coefficient_dim","hidden_dim","heads","layers","feedforward_dim")}
    original=JointPercentileDenoiser(**cfg)
    original.load_state_dict(checkpoint["state_dict"],strict=True);original.eval();original.requires_grad_(False)
    model=ClassifierFreePercentileDenoiser(**cfg)
    report=model.load_from_direct_p_state_dict(checkpoint["state_dict"])
    with torch.no_grad():expected=original.percentile_embedding(torch.tensor([.5],dtype=torch.float32))[0]
    if not torch.equal(model.null_p_embedding,expected):raise ValueError("null was not initialized from loaded parent p=.5")
    report.update(initial_state_sha256=state_hash(model),null_initialization_device="cpu",
        initial_null_sha256=hashlib.sha256(model.null_p_embedding.detach().numpy().tobytes()).hexdigest())
    return original.to(device),model.to(device),report


def parent_replay(original,model,schedule,pack,p_values,device):
    """Read-only check using actual FIT H/p and an isolated software-noise draw."""
    features,clean,p=direct.tensor_batch(pack,p_values,slice(0,4),device)
    noise=torch.randn(tuple(clean.shape),generator=torch.Generator().manual_seed(20261022),dtype=torch.float32).to(device)
    t=torch.tensor([0,25,50,99][:len(clean)],device=device,dtype=torch.long)
    noisy=schedule.q_sample(clean,t,noise,features["agent_mask"])
    original.eval();model.eval()
    with torch.no_grad():
        parent_output=original(noisy,t,features,p)
        child_output=model(noisy,t,features,p,torch.ones(len(clean),dtype=torch.bool,device=device))
    if not torch.equal(parent_output,child_output):raise ValueError("all-present initialized model does not bitwise replay parent")
    return dict(all_present_parent_bitwise_replay=True,probe_scenes=len(clean),probe_role="FIT",
        probe_timesteps=t.cpu().tolist(),p_dropped=False,target_PET_or_oracle_used=False)


def train_epoch(model,ema,schedule,optimizer,pack,p_values,streams,cfg,drop_probability,device):
    model.train();order=streams.order(len(pack["scene_id"]))
    order_hash=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest()
    diffusion_hash,drop_hash,present_hash=hashlib.sha256(),hashlib.sha256(),hashlib.sha256()
    total=0.;updates=0;dropped=0
    for start in range(0,len(order),cfg["batch_size"]):
        rows=order[start:start+cfg["batch_size"]]
        features,clean,p=direct.tensor_batch(pack,p_values,rows,device)
        t,noise,uniforms,presence=streams.draw(clean.shape,schedule.steps,drop_probability)
        diffusion_hash.update(np.asarray(clean.shape,dtype=np.int64).tobytes())
        diffusion_hash.update(t.numpy().tobytes());diffusion_hash.update(noise.numpy().tobytes())
        drop_hash.update(uniforms.tobytes());present_hash.update(presence.tobytes())
        dropped+=int((~presence).sum())
        optimizer.zero_grad(set_to_none=True)
        answer=cfg_training_loss(model,schedule,clean,features,p,torch.from_numpy(presence).to(device),
            timesteps=t.to(device),noise=noise.to(device),prediction_type="v")
        loss=answer["loss"]
        if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite refinement v-MSE")
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"])
        if not bool(torch.isfinite(norm)):raise FloatingPointError("nonfinite refinement gradient")
        optimizer.step();direct.prior.update_ema(ema,model,cfg["EMA_decay"])
        total+=float(loss.detach())*len(rows);updates+=1
    return dict(online_train_v_MSE=total/len(order),updates=updates,scenes=len(order),p_dropped=dropped,
        p_present=len(order)-dropped,p_drop_fraction=dropped/len(order),H_or_static_dropped=False,
        randomness=dict(order_sha256=order_hash,diffusion_t_and_noise_sha256=diffusion_hash.hexdigest(),
            p_dropout_uniform_sha256=drop_hash.hexdigest(),effective_p_presence_mask_sha256=present_hash.hexdigest(),
            diffusion_rng_end_state_sha256=hashlib.sha256(streams.diffusion_rng.get_state().numpy().tobytes()).hexdigest()))


def save_snapshot(output,epoch,ema,schedule,policy_binding,policy,parent,source,recipe,code,initial_state):
    checkpoint=dict(protocol=PROTOCOL,state_dict={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},
        epoch=epoch,recipe=recipe,policy=policy_binding,parent_training=policy["parent_training"],
        parent_checkpoint=parent["checkpoint"],data=parent["data"],labels_manifest=parent["labels_manifest"],
        label_join=parent["label_join"],p_definition=parent["p_definition"],code_sha256=code,
        architecture=ema.architecture_config(),schedule=schedule.as_dict(),prediction_type="v",
        basis=source["data"]["basis"],coefficient_normalizer=source["data"]["coefficient_normalizer"],
        history_normalizer=source["data"]["history_normalizer"],seed=policy["training"]["seed"],
        p_dropout_probability=policy["model"]["dropout_probability"][recipe],
        initial_model_state_sha256=initial_state,EMA=True,optimizer_state_restored=False,
        checkpoint_is_preregistered_candidate=epoch in policy["training"]["candidate_epochs"],
        CAL_generator_examples=False,AUDIT_decoded=False,extra_risk_or_geometry_loss=False)
    path=output/("ema_epoch_%03d.pt"%epoch)
    with path.open("xb") as handle:torch.save(checkpoint,handle)
    return dict(path=str(path),sha256=sha256(path))


def run(policy_binding,recipe):
    policy,parent,checkpoint,source=read_inputs(policy_binding,recipe)
    cfg=policy["training"];device=torch.device(policy["runtime"]["devices"][recipe])
    torch.set_num_threads(policy["runtime"]["threads"]);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.manual_seed(cfg["seed"]);np.random.seed(cfg["seed"]);random.seed(cfg["seed"])
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("registered CUDA device unavailable")
    code=code_hashes(parent);original,model,warm=warmed_models(checkpoint,device)
    schedule=CosineDiffusionSchedule(source["policy"]["generator"]["diffusion_steps"]).to(device)
    if schedule.as_dict()!=checkpoint["schedule"]:raise ValueError("forward diffusion schedule changed")
    replay=parent_replay(original,model,schedule,source["fit"],source["fit_p"],device)
    del original,checkpoint
    output=resolve(policy["output_root"])/recipe;output.mkdir(parents=True,exist_ok=False)
    freeze=direct.prior.write_json(output/"freeze_before_training.json",dict(protocol=PROTOCOL,recipe=recipe,
        policy=policy_binding,parent_training=policy["parent_training"],parent_checkpoint=parent["checkpoint"],
        data=parent["data"],labels_manifest=parent["labels_manifest"],joined_labels=source["label_join"],
        code_sha256=code,warm_start=warm,parent_replay=replay,epochs_fixed=240,candidate_epochs=[120,240],
        STOP_v_MSE_not_used_for_early_stop_or_candidate_selection=True,
        CAL_generator_examples=False,AUDIT_decoded=False,parent_STOP_or_AUDIT_outcomes_opened=False,
        runtime=dict(device=str(device),torch=torch.__version__,numpy=np.__version__,threads=cfg.get("threads",2))))
    ema=copy.deepcopy(model);ema.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    streams=PairedStreams(cfg["seed"],cfg["p_dropout_seed"])
    initial=direct.fixed_validation(ema,schedule,source["stop"],source["stop_p"],policy,device)
    checkpoints={"0":save_snapshot(output,0,ema,schedule,policy_binding,policy,parent,source,recipe,code,warm["initial_state_sha256"])}
    best_value,best_epoch=initial["mean"],0;diagnostics={"0":initial};updates=0;started=time.perf_counter()
    with (output/"epochs.jsonl").open("x",encoding="utf-8") as log:
        for epoch in range(1,cfg["epochs"]+1):
            row=train_epoch(model,ema,schedule,optimizer,source["fit"],source["fit_p"],streams,cfg,
                policy["model"]["dropout_probability"][recipe],device)
            updates+=row["updates"];row.update(epoch=epoch,total_updates=updates,wall_seconds=time.perf_counter()-started)
            if epoch%cfg["validation_every"]==0:
                scores=direct.fixed_validation(ema,schedule,source["stop"],source["stop_p"],policy,device)
                diagnostics[str(epoch)]=scores;row["STOP_fixed_all_present_v_MSE"]=scores
                if scores["mean"]<best_value:best_value,best_epoch=scores["mean"],epoch
            if epoch in cfg["snapshot_epochs"]:
                checkpoints[str(epoch)]=save_snapshot(output,epoch,ema,schedule,policy_binding,policy,parent,source,recipe,code,warm["initial_state_sha256"])
                row["snapshot"]=checkpoints[str(epoch)]
            log.write(json.dumps(row,sort_keys=True)+"\n");log.flush();print(json.dumps(row,sort_keys=True),flush=True)
    if set(checkpoints)!={"0","120","240"}:raise ValueError("a registered fixed-budget snapshot is missing")
    if code_hashes(parent)!=code:raise ValueError("refinement source changed after freeze")
    for binding in (policy_binding,policy["parent_training"],parent["checkpoint"],parent["data"],parent["labels_manifest"],
                    source["data"]["packs"]["FIT"],source["data"]["packs"]["STOP"]):verify_binding(binding)
    for role in ("FIT","STOP"):verify_binding(source["label_manifest"]["roles"][role]["artifact"])
    result=dict(protocol=PROTOCOL,status="complete",recipe=recipe,policy=policy_binding,parent_training=policy["parent_training"],
        parent_checkpoint=parent["checkpoint"],data=parent["data"],labels_manifest=parent["labels_manifest"],
        code_sha256=code,architecture=ema.architecture_config(),prediction_type="v",checkpoints=checkpoints,
        initial_state_sha256=warm["initial_state_sha256"],warm_start=warm,parent_replay=replay,freeze=freeze,
        epochs_completed=cfg["epochs"],candidate_epochs=cfg["candidate_epochs"],all_fixed_epochs_completed=True,
        STOP_v_MSE_diagnostics=diagnostics,best_STOP_v_MSE_diagnostic_only=dict(value=best_value,epoch=best_epoch),
        checkpoint_automatically_selected=False,early_stopping=False,optimizer_state_restored=False,
        total_updates=updates,wall_seconds=time.perf_counter()-started,
        epochs=dict(path=str(output/"epochs.jsonl"),sha256=sha256(output/"epochs.jsonl")),
        labels_changed=False,H_or_static_dropout=False,CAL_generator_examples=False,AUDIT_decoded=False,
        risk_or_geometry_loss_added=False,post_correction=False,control_success_claimed=False)
    binding=direct.prior.write_json(output/"result.json",result)
    print(json.dumps(dict(training_complete=True,recipe=recipe,result=binding,checkpoints=checkpoints)),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--policy",required=True);parser.add_argument("--policy-sha256",required=True)
    parser.add_argument("--recipe",choices=("continue","cfg"),required=True);parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args();binding=dict(path=args.policy,sha256=args.policy_sha256)
    if args.validate_only:
        policy,parent,checkpoint,source=read_inputs(binding,args.recipe)
        print(json.dumps(dict(status="validated_no_training",recipe=args.recipe,parent=policy["parent_training"],label_join=source["label_join"]),indent=2))
    else:run(binding,args.recipe)


if __name__=="__main__":main()
