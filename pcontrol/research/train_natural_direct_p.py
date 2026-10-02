#!/usr/bin/env python3
"""Fresh direct-p natural diffusion training with recording-OOF labels.

The frozen natural coefficient packs remain unchanged. Only their immutable
scene identities join frozen p_mid labels. PET, IDs, fold membership and risk
networks never enter the denoiser. The first pilot uses only standard v-MSE.
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

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pcontrol.research import train_natural_diffusion as prior
from pcontrol.data.complete_scene_view import resolve, verify_binding, sha256
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.direct_p import JointPercentileDenoiser, direct_p_training_loss

PROTOCOL = "natural_recording_OOF_direct_P_generator_training_v1"
POLICY_SHA = "c5d52262816e9e7b298d7108960735009359e4abb57e26adacf73b29ac7d3ab7"
DATA_SHA = "b10d412b54cf6a94c82639362515b4fff62305d8e448c5c6dbcd3eaa00fad408"
P_DEFINITION = "midpoint_of_estimated_adversity_rank_interval_no_atom_randomization"
FEATURES = prior.FEATURES
LABEL_KEYS = {"scene_id", "recording_id", "role", "p_mid", "p_low", "p_up", "atom_width", "pet_seconds", "fold"}
CODE = ("pcontrol/research/train_natural_direct_p.py", "pcontrol/generation/direct_p.py",
        "pcontrol/generation/diffusion.py", "pcontrol/reference/direct_p_crossfit.py",
        "pcontrol/research/train_natural_diffusion.py", "pcontrol/generation/trajectory_basis.py",
        "pcontrol/generation/data.py", "pcontrol/research/prepare_natural_diffusion.py")


def code_hashes():
    return {name: sha256(ROOT/name) for name in CODE}


def same_binding(left, right):
    return left["sha256"] == right["sha256"] and resolve(left["path"]) == resolve(right["path"])


def validate_policy(policy, data_binding):
    if (policy.get("protocol") != "natural_recording_OOF_direct_P_diffusion_pilot_v1"
            or not same_binding(policy["generator_data"], data_binding) or data_binding["sha256"] != DATA_SHA):
        raise ValueError("direct-P policy must bind the unchanged natural generation data")
    g, t = policy["generator"], policy["training"]
    expected_model = dict(prediction_type="v", hidden_dim=128, heads=4, layers=3, feedforward_dim=256,
                          diffusion_steps=100, basis_modes=8, schedule="cosine", initialization="fresh_random")
    if any(g.get(k) != value for k, value in expected_model.items()) or g.get("risk_condition_required") is not True:
        raise ValueError("registered fresh continuous-p v-denoiser architecture required")
    for field in ("pretrained_weights", "target_PET_input", "focal_or_future_semantics_input", "reference_network_imported_by_denoiser"):
        if g.get(field) is not False:
            raise ValueError("prohibited generator initialization/input: " + field)
    fixed_training = dict(seed=20260910, batch_size=128, epochs=80, patience=12, minimum_improvement=.00001,
        learning_rate=.0003, weight_decay=.0001, gradient_clip_norm=1., EMA_decay=.995,
        validation_timesteps=[0,10,25,50,75,99], validation_noise_seed=20261010)
    if any(t.get(k) != value for k, value in fixed_training.items()):
        raise ValueError("direct-P optimization/STOP-selection budget differs from preregistration")
    if (t.get("auxiliary_risk_or_ranking_loss") is not False
            or t.get("risk_label_jitter_or_arbitrary_p_augmentation") is not False
            or t.get("loss") != "v_prediction_MSE_equal_scene_mean_over_real_agents_and_coefficients"
            or policy["runtime"] != dict(device="cuda:1", threads=2)):
        raise ValueError("first direct-P pilot uses only fixed standard v-MSE and registered runtime")
    for name in ("guidance", "post_correction", "best_of_K_or_rejection", "physical_PET_target_supplied_to_sampler"):
        if policy["sampling"].get(name) is not False:
            raise ValueError("primary direct-P outputs cannot use external inference control")
    if (policy["oof"]["folds"] != 5 or policy["oof"]["random_within_atom_labels"] is not False
            or policy["oof"]["FIT_relabelled_by_full_reference"] is not False
            or policy["oof"]["CAL_generator_examples"] is not False or policy["oof"]["AUDIT_decoded"] is not False):
        raise ValueError("recording-OOF midpoint labels and protected roles required")


def join_percentile_labels(pack, labels, *, role):
    """Exact set join; a matching row count or accidental order is insufficient."""
    if role not in ("FIT", "STOP"):
        raise PermissionError("generator labels may contain only FIT/STOP examples")
    if set(labels) != LABEL_KEYS:
        raise ValueError("frozen direct-P label archive schema differs")
    n = len(pack["scene_id"])
    if any(np.asarray(labels[key]).shape != (n,) for key in LABEL_KEYS):
        raise ValueError("exactly one scalar label row per generator scene required")
    if not np.all(pack["role"] == role) or not np.all(labels["role"] == role):
        raise PermissionError("role mismatch rejected before label use")
    fields = ("scene_id", "recording_id", "role")
    def keys(arrays):
        return list(zip(*(np.asarray(arrays[name]).astype(str).tolist() for name in fields)))
    target_keys, label_keys = keys(pack), keys(labels)
    if (len(set(target_keys)) != n or len(set(label_keys)) != n
            or len(set(pack["scene_id"].astype(str))) != n or len(set(labels["scene_id"].astype(str))) != n):
        raise ValueError("duplicate generator/label scene identity")
    if set(target_keys) != set(label_keys):
        raise ValueError("scene_id/recording_id/role label coverage differs from exact generator population")
    lookup = {key: i for i, key in enumerate(label_keys)}
    order = np.asarray([lookup[key] for key in target_keys], dtype=np.int64)
    low, mid, up, width = [np.asarray(labels[key])[order] for key in ("p_low", "p_mid", "p_up", "atom_width")]
    if (any(x.dtype != np.float64 or not np.isfinite(x).all() for x in (low,mid,up,width))
            or np.any(low < 0) or np.any(up > 1) or np.any(mid < 0) or np.any(mid > 1)
            or np.any(mid < low-2e-15) or np.any(mid > up+2e-15)
            or not np.allclose(mid, (low+up)/2, rtol=0, atol=2e-15)
            or not np.allclose(width, up-low, rtol=0, atol=2e-15)):
        raise ValueError("labels must be the unchanged legal estimated interval midpoint, not jittered/rerolled p")
    fold = np.asarray(labels["fold"])[order]
    if not np.issubdtype(fold.dtype, np.integer) or (np.any((fold < 0) | (fold > 4)) if role == "FIT" else np.any(fold != -1)):
        raise ValueError("FIT needs five OOF-fold identifiers; STOP uses its frozen full-reference tag -1")
    pet = np.asarray(labels["pet_seconds"])[order]
    if pet.dtype != np.float64 or not np.isfinite(pet).all() or np.any((pet < 0) | (pet > 4)):
        raise ValueError("label provenance PET must remain a valid capped natural observation")
    identity_text = "\n".join("|".join(key) for key in target_keys)
    evidence = dict(role=role, rows=n, recording_ids=sorted(set(pack["recording_id"].astype(str))),
        join_key=list(fields), joined_identity_sha256=hashlib.sha256(identity_text.encode()).hexdigest(),
        joined_p_mid_float64_sha256=hashlib.sha256(np.ascontiguousarray(mid).tobytes()).hexdigest(),
        source_row_order_equal=bool(np.array_equal(order, np.arange(n))),
        fold_counts={str(i): int(np.sum(fold == i)) for i in sorted(set(fold.tolist()))},
        p_definition=P_DEFINITION, model_condition_dtype="float32", stored_label_dtype="float64",
        p_clipped_jittered_or_resampled=False, PET_passed_as_model_input=False)
    return np.ascontiguousarray(mid), evidence


def tensor_batch(pack, p_mid, rows, device):
    features, clean = prior.tensor_batch(pack, rows, device)
    p = torch.from_numpy(np.ascontiguousarray(p_mid[rows])).to(device=device, dtype=torch.float32)
    if p.shape != (len(clean),):
        raise ValueError("one joined percentile per scene, separate from history/static features")
    return features, clean, p


def make_model(policy, modes, device):
    g = policy["generator"]
    return JointPercentileDenoiser(2*modes, hidden_dim=g["hidden_dim"], heads=g["heads"],
        layers=g["layers"], feedforward_dim=g["feedforward_dim"]).to(device)


def fixed_validation(model, schedule, pack, p_mid, policy, device):
    model.eval(); cfg = policy["training"]
    rng = np.random.default_rng(cfg["validation_noise_seed"])
    scores = []
    with torch.no_grad():
        for step in cfg["validation_timesteps"]:
            noise = rng.standard_normal(pack["coef_clean"].shape).astype(np.float32)
            chunks = []
            for start in range(0, len(pack["scene_id"]), cfg["batch_size"]):
                rows = slice(start, start+cfg["batch_size"])
                features, clean, p = tensor_batch(pack, p_mid, rows, device)
                eps = torch.from_numpy(np.ascontiguousarray(noise[rows, :clean.shape[1]])).to(device)
                t = torch.full((len(clean),), step, dtype=torch.long, device=device)
                answer = direct_p_training_loss(model, schedule, clean, features, p, timesteps=t, noise=eps,
                                               prediction_type="v")
                chunks.append(answer["per_scene_loss"].cpu().numpy())
            scores.append(float(np.concatenate(chunks).mean()))
    if not np.isfinite(scores).all():
        raise FloatingPointError("direct-P STOP loss became nonfinite")
    return dict(mean=float(np.mean(scores)), by_timestep=dict(zip(map(str,cfg["validation_timesteps"]),scores)),
                fixed_STOP_p_mid=True, p_augmented=False)


def prepare_inputs(data_binding, labels_binding, policy_binding):
    if policy_binding["sha256"] != POLICY_SHA:
        raise ValueError("unregistered direct-P generator policy")
    policy = prior.load_json(policy_binding); validate_policy(policy, data_binding)
    data = prior.load_json(data_binding)
    # The old prepared data correctly retain their OLD history-only data policy.
    # Validate against that original policy, not by rewriting its frozen source.
    old_policy = prior.load_json(data["policy"])
    coefficient = prior.validate_prepared_metadata(data, old_policy, data["policy"])
    if (not same_binding(data["dataset_manifest"], policy["data_manifest"])
            or not same_binding(data["reference_prepared"], policy["reference_prepared"])
            or data["basis"]["modes"] != policy["generator"]["basis_modes"]):
        raise ValueError("direct-P changes neither the natural source nor basis/normalization")
    from pcontrol.reference.direct_p_crossfit import validate_label_manifest, load_role_labels
    label_manifest = validate_label_manifest(labels_binding)
    if (label_manifest.get("p_definition") != P_DEFINITION
            or not same_binding(label_manifest["policy"], policy_binding)):
        raise ValueError("recording-OOF estimated midpoint label definition differs")
    fit = prior.load_pack(data["packs"]["FIT"], "FIT")
    stop = prior.load_pack(data["packs"]["STOP"], "STOP")
    if (len(fit["scene_id"]) != 9913 or len(stop["scene_id"]) != 538
            or len(set(fit["recording_id"])) != 13 or len(set(stop["recording_id"])) != 4
            or set(fit["recording_id"]) & set(stop["recording_id"])
            or hashlib.sha256("\n".join(fit["scene_id"].tolist()).encode()).hexdigest() != coefficient["FIT_scene_id_sha256"]):
        raise ValueError("immutable FIT9913/STOP538 and their FIT normalizer identities required")
    labels = {role: load_role_labels(labels_binding, role) for role in ("FIT", "STOP")}
    fold_mapping = {rec: int(fold) for fold, records in policy["oof"]["heldout_recordings"].items() for rec in records}
    if any(int(fold) != fold_mapping.get(str(rec), -99)
           for rec,fold in zip(labels["FIT"]["recording_id"],labels["FIT"]["fold"])):
        raise ValueError("a FIT p label is assigned to a teacher that did not hold out its recording")
    fit_p, fit_join = join_percentile_labels(fit, labels["FIT"], role="FIT")
    stop_p, stop_join = join_percentile_labels(stop, labels["STOP"], role="STOP")
    if fit["coef_clean"].shape[-2] != data["basis"]["modes"] or stop["coef_clean"].shape[-2] != data["basis"]["modes"]:
        raise ValueError("unchanged basis and coefficient dimensions disagree")
    return dict(policy=policy, data=data, label_manifest=label_manifest, fit=fit, stop=stop, fit_p=fit_p, stop_p=stop_p,
        label_join=dict(FIT=fit_join, STOP=stop_join))


def run(data_binding, labels_binding, policy_binding):
    source = prepare_inputs(data_binding, labels_binding, policy_binding)
    policy, data = source["policy"], source["data"]
    fit, stop, fit_p, stop_p = (source[k] for k in ("fit", "stop", "fit_p", "stop_p"))
    cfg = policy["training"]
    torch.set_num_threads(policy["runtime"]["threads"]); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    device = torch.device(policy["runtime"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered CUDA unavailable; do not silently change run protocol")
    code = code_hashes(); output = resolve(policy["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    join_binding = prior.write_json(output/"label_join.json", source["label_join"])
    freeze = prior.write_json(output/"freeze_before_training.json", dict(protocol=PROTOCOL, data=data_binding,
        labels_manifest=labels_binding, policy=policy_binding, code_sha256=code, label_join=join_binding,
        p_definition=P_DEFINITION, label_role_artifacts=source["label_manifest"]["roles"],
        label_fold_artifacts=source["label_manifest"]["fold_artifacts"],
        initialization="fresh_random_no_history_only_or_simulation_weights", PET_input=False,
        label_validation="crossfit_validator_checks_all_folds_and_train_normalizer_held_recording_exclusion",
        upstream_CAL_used_only_for_teacher_calibration=True, CAL_generator_examples=False,
        CAL_arrays_decoded_by_trainer=False, AUDIT_decoded=False, auxiliary_risk_loss=False,
        python=sys.version, torch=torch.__version__, numpy=np.__version__, device=str(device)))
    model = make_model(policy, int(data["basis"]["modes"]), device)
    ema = copy.deepcopy(model)
    for parameter in ema.parameters(): parameter.requires_grad_(False)
    schedule = CosineDiffusionSchedule(policy["generator"]["diffusion_steps"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"])
    generator = torch.Generator(device=device).manual_seed(cfg["seed"]+1)
    initial = fixed_validation(ema, schedule, stop, stop_p, policy, device)
    best = meaningful = initial["mean"]
    best_epoch = stale = updates = 0
    best_state = {k: v.detach().cpu().clone() for k,v in ema.state_dict().items()}
    started = time.perf_counter()
    with (output/"epochs.jsonl").open("x",encoding="utf-8") as log:
        for epoch in range(1,cfg["epochs"]+1):
            model.train(); order = rng.permutation(len(fit["scene_id"])); total = 0.
            for start in range(0,len(order),cfg["batch_size"]):
                rows = order[start:start+cfg["batch_size"]]
                features, clean, p = tensor_batch(fit,fit_p,rows,device)
                optimizer.zero_grad(set_to_none=True)
                answer = direct_p_training_loss(model,schedule,clean,features,p,generator=generator,prediction_type="v")
                loss = answer["loss"]
                if not bool(torch.isfinite(loss)): raise FloatingPointError("nonfinite direct-P v loss")
                loss.backward(); grad = torch.nn.utils.clip_grad_norm_(model.parameters(),cfg["gradient_clip_norm"])
                if not bool(torch.isfinite(grad)): raise FloatingPointError("nonfinite direct-P gradient")
                optimizer.step(); prior.update_ema(ema,model,cfg["EMA_decay"])
                total += float(loss.detach())*len(rows); updates += 1
            scores = fixed_validation(ema,schedule,stop,stop_p,policy,device)
            if scores["mean"] < best:
                best,best_epoch = scores["mean"],epoch
                best_state = {k:v.detach().cpu().clone() for k,v in ema.state_dict().items()}
            if scores["mean"] < meaningful-cfg["minimum_improvement"]: meaningful,stale = scores["mean"],0
            else: stale += 1
            row = dict(epoch=epoch,online_train_v_MSE=total/len(order),STOP_fixed_v_MSE=scores,
                best_STOP_v_MSE=best,best_epoch=best_epoch,stale=stale,updates=updates,
                direct_p_condition_used=True,wall_seconds=time.perf_counter()-started)
            log.write(json.dumps(row,sort_keys=True)+"\n");log.flush();print(json.dumps(row,sort_keys=True),flush=True)
            if stale >= cfg["patience"]: break
    ema.load_state_dict(best_state,strict=True)
    replay = fixed_validation(ema,schedule,stop,stop_p,policy,device)
    if abs(replay["mean"]-best)>1e-9: raise ValueError("selected conditional EMA does not replay fixed STOP loss")
    if code_hashes()!=code: raise ValueError("direct-P source changed after freeze")
    for binding in (data_binding,labels_binding,policy_binding,join_binding,freeze,
                    data["packs"]["FIT"],data["packs"]["STOP"],data["history_normalizer"],data["coefficient_normalizer"]):
        verify_binding(binding)
    for role in ("FIT","STOP"):verify_binding(source["label_manifest"]["roles"][role]["artifact"])
    shared = dict(protocol=PROTOCOL,data=data_binding,policy=policy_binding,labels_manifest=labels_binding,
        label_role_artifacts=source["label_manifest"]["roles"],label_join=join_binding,p_definition=P_DEFINITION,
        label_fold_artifacts=source["label_manifest"]["fold_artifacts"],
        code_sha256=code,architecture=ema.architecture_config(),prediction_type="v",basis=data["basis"],
        coefficient_normalizer=data["coefficient_normalizer"],history_normalizer=data["history_normalizer"],
        seed=cfg["seed"],best_epoch=best_epoch,p_is_actual_denoiser_condition=True,
        estimated_OOF_midrank_not_known_true_percentile=True,reference_network_in_generator_graph=False,
        natural_supervision_only=True,simulated_training_futures=False,pretrained_generator_weights_loaded=False,
        CAL_generator_examples=False,AUDIT_decoded=False,primary_sampling_guidance=False,
        primary_sampling_post_correction=False,auxiliary_risk_or_ranking_loss=False)
    checkpoint = dict(shared,state_dict=best_state,schedule=schedule.as_dict(),best_STOP_v_MSE=best)
    with (output/"best_ema.pt").open("xb") as handle:torch.save(checkpoint,handle)
    result = dict(shared,status="complete",initial_STOP_v_MSE=initial,best_STOP_v_MSE=replay,
        epochs_completed=epoch,updates=updates,wall_seconds=time.perf_counter()-started,freeze=freeze,
        checkpoint=dict(path=str(output/"best_ema.pt"),sha256=sha256(output/"best_ema.pt")),
        epochs=dict(path=str(output/"epochs.jsonl"),sha256=sha256(output/"epochs.jsonl")),
        counts=dict(FIT=len(fit_p),STOP=len(stop_p)),risk_control_success_claimed=False,
        same_capacity_as_unconditional_baseline_claimed=False)
    result_binding = prior.write_json(output/"result.json",result)
    print(json.dumps(dict(training_complete=True,result=result_binding,best_STOP_v_MSE=best,best_epoch=best_epoch)),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("data-manifest","data-sha256","labels-manifest","labels-sha256","policy","policy-sha256"):
        parser.add_argument("--"+name,required=True)
    parser.add_argument("--validate-only",action="store_true")
    args=parser.parse_args()
    bindings=(dict(path=args.data_manifest,sha256=args.data_sha256),dict(path=args.labels_manifest,sha256=args.labels_sha256),
              dict(path=args.policy,sha256=args.policy_sha256))
    if args.validate_only:
        prepared=prepare_inputs(*bindings)
        print(json.dumps(dict(status="validated_no_training",label_join=prepared["label_join"]),indent=2))
    else:run(*bindings)


if __name__=="__main__":main()
