#!/usr/bin/env python3
"""Paired warm-start training with offline natural risk-tangent supervision.

No CDF or geometric functional is evaluated per batch. The cache is joined by
natural FIT scene identity and enters only optional losses, never the denoiser
or sampler. All recipes retain ordinary v-MSE on every original FIT example.
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
from pcontrol.research import refine_natural_direct_p as paired
from pcontrol.generation.direct_p_cfg import ClassifierFreePercentileDenoiser,cfg_training_loss
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.risk_tangent_loss import natural_tangent_losses
from pcontrol.data.complete_scene_view import verify_binding,resolve,sha256

PROTOCOL="natural_direct_P_offline_risk_tangent_training_v1"
POLICY_SHA="237ec4f3d1645d0dda94810822ab9b9facea59146fa760e32732f5f89eb1c342"
PARENT_CHECKPOINT_SHA="526d3062eb1a9f025e097490f646fdb277d940f1a97d049ef9ee6577f5cae52b"
WEIGHTS={"continue":(0.,0.),"projection":(1.,0.),"tangent":(1.,.1)}
CODE=("pcontrol/research/train_natural_risk_tangent.py","pcontrol/generation/risk_tangent_loss.py",
      "pcontrol/generation/risk_tangent.py")


def validate_policy(policy,recipe):
    if (policy.get("protocol")!="natural_direct_P_risk_tangent_training_v1"
            or policy.get("recipes")!=list(WEIGHTS) or recipe not in WEIGHTS
            or policy["parent_checkpoint"]["sha256"]!=PARENT_CHECKPOINT_SHA):
        raise ValueError("unregistered parent/recipe for natural tangent training")
    for name,(projection,jacobian) in WEIGHTS.items():
        if policy["loss"]["weights"][name]!={"projection":projection,"jacobian":jacobian}:
            raise ValueError("predeclared risk-loss weights changed")
    expected=dict(seed=20260910,epochs=80,snapshot_epochs=[0,40,80],candidate_epochs=[40,80],validation_every=20,
        batch_size=128,learning_rate=.0001,weight_decay=.0001,gradient_clip_norm=1.,EMA_decay=.995,
        p_dropout_probability=.15,p_dropout_seed=20261021,validation_timesteps=[0,10,25,50,75,99],
        validation_noise_seed=20261010,optimizer_state_restored=False,labels_changed=False,multiseed_replication=False,early_stopping=False)
    if any(policy["training"].get(k)!=v for k,v in expected.items()):raise ValueError("fixed tangent-training budget changed")
    if (policy["loss"]["min_jacobian_sigma"]!=.5 or policy["loss"]["projection_requires_p_present"] is not True
            or policy["loss"]["jacobian_create_graph"] is not True or policy["loss"]["unsupported_rows_keep_base_loss"] is not True
            or policy["cache"]["selected_scenes"]!=2048 or policy["cache"]["role"]!="FIT"
            or policy["cache"]["risk_rank_tangent_sign"]!=-1):raise ValueError("support/sign/second-order contract changed")
    if policy["runtime"]!={"devices":{name:"cuda:0" for name in WEIGHTS},"threads":2,"TF32":False}:
        raise ValueError("use only registered devices; do not interfere with other GPU workloads")
    for key in ("post_correction","risk_oracle_or_gradient_in_sampling","risk_tangent_cache_in_sampling","target_PET_input","Best_of_K"):
        if policy["sampling"][key] is not False:raise ValueError("offline training cache must not become inference control")


def join_risk_cache(pack,p_mid,arrays,*,expected_selected=2048):
    required={"scene_id","recording_id","role","agent_mask","selected_subset_mask","valid_geom","valid_density",
              "g","unit_g","h","density","p_mid"}
    if not required<=set(arrays):raise ValueError("risk cache lacks declared identity/direction/mask fields")
    n=len(pack["scene_id"])
    if not np.all(pack["role"]=="FIT") or not np.all(arrays["role"]=="FIT"):
        raise PermissionError("only FIT risk-cache rows may enter generator training")
    def keys(values):return list(zip(*(np.asarray(values[k]).astype(str).tolist() for k in ("scene_id","recording_id","role"))))
    target,source=keys(pack),keys(arrays)
    if (len(source)!=n or len(set(target))!=n or len(set(source))!=n or set(target)!=set(source)):
        raise ValueError("risk cache is not an exact three-key FIT identity join")
    lookup={k:i for i,k in enumerate(source)};order=np.asarray([lookup[k] for k in target])
    aligned={k:(np.asarray(arrays[k]) if np.array_equal(order,np.arange(n)) else np.asarray(arrays[k])[order]) for k in required}
    mask=aligned["agent_mask"]
    if mask.dtype!=np.bool_ or not np.array_equal(mask,pack["agent_mask"]):raise ValueError("cache changed actual agent masks/order")
    subset,geom,valid_density=[aligned[k] for k in ("selected_subset_mask","valid_geom","valid_density")]
    if (any(x.shape!=(n,) or x.dtype!=np.bool_ for x in (subset,geom,valid_density))
            or int(subset.sum())!=expected_selected or np.any(geom&~subset) or np.any(valid_density&~subset)):
        raise ValueError("only the fixed FIT subset may support additional losses")
    if aligned["p_mid"].dtype!=np.float64 or not np.array_equal(aligned["p_mid"],p_mid):
        raise ValueError("offline cache relabelled the original OOF percentiles")
    shape=pack["coef_clean"].shape
    for key in ("g","unit_g","h"):
        a=aligned[key]
        if a.shape!=shape or a.dtype!=np.float64 or not np.isfinite(a).all() or np.any(a[~mask]!=0) or np.any(a[~subset]!=0):
            raise ValueError("cache direction shape/padding/subset/precision mismatch: "+key)
    g,unit,h,density=(aligned[k] for k in ("g","unit_g","h","density"))
    if density.shape!=(n,) or density.dtype!=np.float64 or not np.isfinite(density).all() or np.any(density<0):
        raise ValueError("finite nonnegative detached OOF density required")
    if np.any(g[~geom]!=0) or np.any(unit[~geom]!=0) or np.any(density[~valid_density]!=0):
        raise ValueError("unsupported training gradients/densities must remain zero")
    norms=np.linalg.norm(g.reshape(n,-1),axis=1)
    if np.any(norms[geom]<=0) or not np.allclose(unit[geom],g[geom]/norms[geom,None,None,None],rtol=1e-10,atol=1e-12):
        raise ValueError("unit_g does not normalize the geometry gradient")
    if not np.allclose(h,density[:,None,None,None]*g,rtol=1e-10,atol=1e-12):
        raise ValueError("h must be the positive CDF tangent f*g, not the negative rank tangent")
    used={k:aligned[k] for k in ("g","density","selected_subset_mask","valid_geom","valid_density")}
    evidence=dict(rows=n,selected_scenes=int(subset.sum()),valid_geometry=int(geom.sum()),valid_density=int(valid_density.sum()),
        join_key=["scene_id","recording_id","role"],identity_sha256=hashlib.sha256("\n".join("|".join(k) for k in target).encode()).hexdigest(),
        p_mid_float64_sha256=hashlib.sha256(np.ascontiguousarray(p_mid).tobytes()).hexdigest(),
        agent_mask_exact=True,original_OOF_labels_exact=True,h_equals_positive_CDF_density_times_g=True,
        cached_fields_are_model_inputs=False)
    return used,evidence


def read_inputs(policy_binding,risk_cache_binding,recipe):
    if policy_binding["sha256"]!=POLICY_SHA:raise ValueError("unregistered tangent policy")
    policy=direct.prior.load_json(policy_binding);validate_policy(policy,recipe)
    base=direct.prior.load_json(policy["base_training"]);parent=direct.prior.load_json(policy["parent_training"])
    if (parent.get("status")!="complete" or parent.get("recipe")!="cfg"
            or not direct.same_binding(parent["checkpoints"]["120"],policy["parent_checkpoint"])
            or not direct.same_binding(base["data"],policy["generator_data"])
            or not direct.same_binding(base["labels_manifest"],policy["labels_manifest"])):
        raise ValueError("risk-tangent warm parent or natural source changed")
    source=direct.prepare_inputs(base["data"],base["labels_manifest"],base["policy"])
    if source["label_join"]!=direct.prior.load_json(base["label_join"]):raise ValueError("original OOF-label join changed")
    for path,digest in parent["code_sha256"].items():verify_binding(dict(path=path,sha256=digest))
    checkpoint=torch.load(verify_binding(policy["parent_checkpoint"]),map_location="cpu")
    if (checkpoint["epoch"]!=120 or checkpoint["recipe"]!="cfg" or checkpoint["prediction_type"]!="v"
            or checkpoint["code_sha256"]!=parent["code_sha256"] or checkpoint["architecture"]!=parent["architecture"]
            or not direct.same_binding(checkpoint["data"],base["data"])
            or not direct.same_binding(checkpoint["labels_manifest"],base["labels_manifest"])):
        raise ValueError("the exact selected learned-null CFG checkpoint is required")
    from pcontrol.generation.risk_tangent import validate_tangent_manifest,load_tangent_cache
    manifest=validate_tangent_manifest(risk_cache_binding)
    for field in ("policy","generator_data","labels_manifest","physical_prepared"):
        expected=policy_binding if field=="policy" else policy[field]
        if not direct.same_binding(manifest[field],expected):raise ValueError("offline risk-cache source mismatch: "+field)
    cache,evidence=join_risk_cache(source["fit"],source["fit_p"],load_tangent_cache(risk_cache_binding))
    return policy,parent,base,checkpoint,source,manifest,cache,evidence


def cache_batch(cache,rows,width,device):
    return dict(natural_gradient=torch.from_numpy(np.ascontiguousarray(cache["g"][rows,:width])).to(device),
        density=torch.from_numpy(np.ascontiguousarray(cache["density"][rows])).to(device),
        valid_geometry=torch.from_numpy(np.ascontiguousarray(cache["valid_geom"][rows]&cache["selected_subset_mask"][rows])).to(device),
        valid_density=torch.from_numpy(np.ascontiguousarray(cache["valid_density"][rows]&cache["selected_subset_mask"][rows])).to(device))


def train_epoch(model,ema,schedule,optimizer,pack,p_values,cache,streams,cfg,weights,device,min_sigma=.5):
    model.train();order=streams.order(len(pack["scene_id"]));diffusion_hash=hashlib.sha256();drop_hash=hashlib.sha256();present_hash=hashlib.sha256()
    base_total=total=proj_total=jac_total=gain_total=residual_total=0.;proj_count=jac_count=evaluated_count=evaluated_batches=disconnected=updates=dropped=0
    for start in range(0,len(order),cfg["batch_size"]):
        rows=order[start:start+cfg["batch_size"]]
        features,clean,p=direct.tensor_batch(pack,p_values,rows,device)
        if weights["jacobian"]>0:p=p.detach().requires_grad_(True)
        t,noise,u,present=streams.draw(clean.shape,schedule.steps,cfg["p_dropout_probability"])
        diffusion_hash.update(np.asarray(clean.shape,dtype=np.int64).tobytes());diffusion_hash.update(t.numpy().tobytes());diffusion_hash.update(noise.numpy().tobytes())
        drop_hash.update(u.tobytes());present_hash.update(present.tobytes());dropped+=int((~present).sum())
        presence=torch.from_numpy(present).to(device);times=t.to(device)
        optimizer.zero_grad(set_to_none=True)
        base=cfg_training_loss(model,schedule,clean,features,p,presence,timesteps=times,noise=noise.to(device),prediction_type="v")
        extra=natural_tangent_losses(base["model_prediction"],base["prediction_target"],base["noisy_coefficients"],p,
            schedule.alpha_bar(times,base["model_prediction"]),features["agent_mask"],
            condition_present=presence,projection_weight=weights["projection"],jacobian_weight=weights["jacobian"],
            min_jacobian_sigma=min_sigma,**cache_batch(cache,rows,clean.shape[1],device))
        if extra["diagnostics"]["jacobian_evaluated"] and extra["diagnostics"]["p_graph_connected"] is False:
            raise RuntimeError("Jacobian supervision is disconnected from the p tensor used by the model")
        loss=base["loss"]+extra["loss"]
        if not bool(torch.isfinite(loss)):raise FloatingPointError("nonfinite natural tangent training loss")
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"])
        if not bool(torch.isfinite(norm)):raise FloatingPointError("nonfinite natural tangent parameter gradient")
        optimizer.step();direct.prior.update_ema(ema,model,cfg["EMA_decay"])
        count=len(rows);base_total+=float(base["loss"].detach())*count;total+=float(loss.detach())*count;updates+=1
        pc,jc=extra["support_counts"]["projection"],extra["support_counts"]["jacobian"]
        proj_total+=float(extra["projection_loss"].detach())*pc;proj_count+=pc;jac_count+=jc
        if extra["diagnostics"]["jacobian_evaluated"]:
            evaluated_batches+=1;evaluated_count+=jc;jac_total+=float(extra["jacobian_loss"].detach())*jc
            gain_total+=float(extra["diagnostics"]["mean_adversity_gain"])*jc
            residual_total+=float(extra["diagnostics"]["mean_absolute_gain_residual"])*jc
            disconnected+=int(extra["diagnostics"]["p_graph_connected"] is False)
    return dict(online_train_v_MSE=base_total/len(order),optimized_total_batch_mean_scene_weighted=total/len(order),
        projection_loss_eligible_mean=proj_total/max(proj_count,1),jacobian_loss_eligible_mean=None if evaluated_count==0 else jac_total/evaluated_count,
        mean_adversity_gain=None if evaluated_count==0 else gain_total/evaluated_count,
        mean_absolute_gain_residual=None if evaluated_count==0 else residual_total/evaluated_count,
        projection_support_events=proj_count,potential_jacobian_support_events=jac_count,
        jacobian_evaluated_events=evaluated_count,jacobian_evaluated_batches=evaluated_batches,p_disconnected_batches=disconnected,
        base_loss_scenes=len(order),updates=updates,p_dropped=dropped,p_drop_fraction=dropped/len(order),
        H_or_static_dropped=False,per_batch_CDF_or_geometry_evaluations=0,
        randomness=dict(order_sha256=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest(),
            diffusion_t_and_noise_sha256=diffusion_hash.hexdigest(),p_dropout_uniform_sha256=drop_hash.hexdigest(),
            effective_p_presence_mask_sha256=present_hash.hexdigest(),
            diffusion_rng_end_state_sha256=hashlib.sha256(streams.diffusion_rng.get_state().numpy().tobytes()).hexdigest()))


def warmed_model(checkpoint,device):
    kwargs={key:checkpoint["architecture"][key] for key in ("coefficient_dim","hidden_dim","heads","layers","feedforward_dim")}
    model=ClassifierFreePercentileDenoiser(**kwargs)
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    if model.architecture_config()!=checkpoint["architecture"]:raise ValueError("no model parameters/architecture may be added")
    if not torch.equal(model.null_p_embedding.detach(),checkpoint["state_dict"]["null_p_embedding"]):raise ValueError("learned parent null was reset")
    initial=paired.state_hash(model)
    return model.to(device),initial


def code_hashes(parent,cache_manifest):
    code=dict(parent["code_sha256"])
    for path,digest in code.items():verify_binding(dict(path=path,sha256=digest))
    code.update(cache_manifest["code_sha256"])
    for path,digest in code.items():verify_binding(dict(path=path,sha256=digest))
    code.update({path:sha256(ROOT/path) for path in CODE})
    return code


def snapshot(output,epoch,ema,schedule,policy_binding,risk_cache_binding,policy,source,parent,code,recipe,initial):
    value=dict(protocol=PROTOCOL,state_dict={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},
        epoch=epoch,recipe=recipe,policy=policy_binding,risk_cache=risk_cache_binding,parent_training=policy["parent_training"],
        parent_checkpoint=policy["parent_checkpoint"],base_training=policy["base_training"],data=policy["generator_data"],
        labels_manifest=policy["labels_manifest"],basis=source["data"]["basis"],
        coefficient_normalizer=source["data"]["coefficient_normalizer"],history_normalizer=source["data"]["history_normalizer"],
        architecture=ema.architecture_config(),schedule=schedule.as_dict(),prediction_type="v",code_sha256=code,
        initial_model_state_sha256=initial,loss_weights=policy["loss"]["weights"][recipe],min_jacobian_sigma=.5,
        cache_used_only_in_training_loss=True,model_parameters_added=False,learned_parent_null_retained=True,
        labels_changed=False,AUDIT_decoded=False,checkpoint_is_candidate=epoch in policy["training"]["candidate_epochs"])
    path=output/("ema_epoch_%03d.pt"%epoch)
    with path.open("xb") as handle:torch.save(value,handle)
    return dict(path=str(path),sha256=sha256(path))


def run(policy_binding,risk_cache_binding,recipe):
    policy,parent,base,checkpoint,source,cache_manifest,cache,cache_join=read_inputs(policy_binding,risk_cache_binding,recipe)
    cfg=policy["training"];device=torch.device(policy["runtime"]["devices"][recipe])
    torch.set_num_threads(policy["runtime"]["threads"]);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.manual_seed(cfg["seed"]);np.random.seed(cfg["seed"]);random.seed(cfg["seed"])
    if device.type=="cuda" and not torch.cuda.is_available():raise RuntimeError("registered CUDA unavailable")
    code=code_hashes(parent,cache_manifest);model,initial=warmed_model(checkpoint,device)
    schedule=CosineDiffusionSchedule(source["policy"]["generator"]["diffusion_steps"]).to(device)
    if schedule.as_dict()!=checkpoint["schedule"]:raise ValueError("frozen forward diffusion schedule changed")
    del checkpoint
    output=resolve(policy["output_root"])/recipe;output.mkdir(parents=True,exist_ok=False)
    freeze=direct.prior.write_json(output/"freeze_before_training.json",dict(protocol=PROTOCOL,policy=policy_binding,recipe=recipe,
        risk_cache=risk_cache_binding,cache_join=cache_join,parent_checkpoint=policy["parent_checkpoint"],
        parent_training=policy["parent_training"],base_training=policy["base_training"],label_join=source["label_join"],
        code_sha256=code,initial_model_state_sha256=initial,model_parameters_added=False,learned_null_retained=True,
        loss=policy["loss"],CDF_geometry_evaluated_per_batch=False,CAL_generator_examples=False,AUDIT_decoded=False,
        parent_AUDIT_or_STOP_result_content_opened=False,epochs_fixed=80,candidate_epochs=[40,80],runtime=dict(device=str(device),threads=2)))
    ema=copy.deepcopy(model);ema.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"],weight_decay=cfg["weight_decay"])
    streams=paired.PairedStreams(cfg["seed"],cfg["p_dropout_seed"])
    diagnostics={"0":direct.fixed_validation(ema,schedule,source["stop"],source["stop_p"],policy,device)}
    checkpoints={"0":snapshot(output,0,ema,schedule,policy_binding,risk_cache_binding,policy,source,parent,code,recipe,initial)}
    updates=0;started=time.perf_counter()
    with (output/"epochs.jsonl").open("x",encoding="utf-8") as log:
        for epoch in range(1,cfg["epochs"]+1):
            row=train_epoch(model,ema,schedule,optimizer,source["fit"],source["fit_p"],cache,streams,cfg,policy["loss"]["weights"][recipe],device,
                min_sigma=policy["loss"]["min_jacobian_sigma"])
            updates+=row["updates"];row.update(epoch=epoch,total_updates=updates,wall_seconds=time.perf_counter()-started)
            if epoch%cfg["validation_every"]==0:
                diagnostics[str(epoch)]=direct.fixed_validation(ema,schedule,source["stop"],source["stop_p"],policy,device)
                row["STOP_all_present_v_MSE_diagnostic"]=diagnostics[str(epoch)]
            if epoch in cfg["snapshot_epochs"]:
                checkpoints[str(epoch)]=snapshot(output,epoch,ema,schedule,policy_binding,risk_cache_binding,policy,source,parent,code,recipe,initial)
                row["snapshot"]=checkpoints[str(epoch)]
            log.write(json.dumps(row,sort_keys=True)+"\n");log.flush();print(json.dumps(row,sort_keys=True),flush=True)
    if set(checkpoints)!={"0","40","80"}:raise ValueError("fixed-budget snapshot missing")
    if code_hashes(parent,cache_manifest)!=code:raise ValueError("risk-tangent source changed after freeze")
    for binding in (policy_binding,risk_cache_binding,policy["parent_checkpoint"],policy["generator_data"],policy["labels_manifest"],cache_manifest["FIT_array"]):verify_binding(binding)
    result=dict(protocol=PROTOCOL,status="complete",recipe=recipe,policy=policy_binding,parent_training=policy["parent_training"],
        parent_checkpoint=policy["parent_checkpoint"],base_training=policy["base_training"],data=policy["generator_data"],
        labels_manifest=policy["labels_manifest"],risk_cache=risk_cache_binding,cache_join=cache_join,code_sha256=code,
        architecture=ema.architecture_config(),prediction_type="v",checkpoints=checkpoints,initial_model_state_sha256=initial,
        epochs_completed=cfg["epochs"],candidate_epochs=cfg["candidate_epochs"],all_fixed_epochs_completed=True,
        early_stopping=False,checkpoint_automatically_selected=False,loss_weights=policy["loss"]["weights"][recipe],
        STOP_v_MSE_diagnostics=diagnostics,epochs=dict(path=str(output/"epochs.jsonl"),sha256=sha256(output/"epochs.jsonl")),
        total_updates=updates,wall_seconds=time.perf_counter()-started,freeze=freeze,model_parameters_added=False,
        labels_changed=False,CAL_generator_examples=False,AUDIT_decoded=False,H_or_static_dropout=False,
        cache_used_only_in_training_loss=True,risk_cache_in_sampling=False,post_correction=False,control_success_claimed=False)
    binding=direct.prior.write_json(output/"result.json",result)
    print(json.dumps(dict(training_complete=True,recipe=recipe,result=binding)),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("policy","policy-sha256","risk-cache","risk-cache-sha256"):parser.add_argument("--"+name,required=True)
    parser.add_argument("--recipe",choices=list(WEIGHTS),required=True);parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args();policy=dict(path=args.policy,sha256=args.policy_sha256);cache=dict(path=args.risk_cache,sha256=args.risk_cache_sha256)
    if args.validate_only:
        loaded=read_inputs(policy,cache,args.recipe);print(json.dumps(dict(status="validated_no_training",cache_join=loaded[-1]),indent=2))
    else:run(policy,cache,args.recipe)


if __name__=="__main__":main()
