#!/usr/bin/env python3
"""Train the independent natural history-conditioned diffusion prior.

Risk plugins/ranks are deliberately absent from this training program.
Only FIT coefficients train the network. STOP fixed-noise v-MSE selects EMA.
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
from pcontrol.data.complete_scene_view import resolve, sha256, verify_binding
from pcontrol.generation.diffusion import (
    JointHistoryDenoiser, CosineDiffusionSchedule, diffusion_training_loss,
)

POLICY_SHA = "5477744cd4a71eef90d0fd55882185cda50256e74f0eeb8703f55d4ed31d4997"
PROTOCOL = "natural_history_only_diffusion_prior_pilot_v1"
FEATURES = ("history", "dimensions", "road_boundaries", "road_boundary_mask", "ego_mask", "agent_mask")
CODE = ("pcontrol/research/train_natural_diffusion.py", "pcontrol/generation/diffusion.py",
        "pcontrol/generation/trajectory_basis.py", "pcontrol/generation/data.py",
        "pcontrol/research/prepare_natural_diffusion.py")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def load_json(binding):
    return json.loads(verify_binding(binding).read_text())


def code_hashes():
    return {name: sha256(ROOT / name) for name in CODE}


def validate_prepared_metadata(data, policy, policy_binding):
    if (data.get("protocol") != "natural_diffusion_training_data_v1" or data.get("status") != "prepared"
            or data["policy"]["sha256"] != policy_binding["sha256"]
            or data["reference_prepared"] != policy["reference_prepared"]
            or data["dataset_manifest"] != policy["data_manifest"]
            or data.get("counts") != {"FIT": 9913, "STOP": 538}
            or set(data["packs"]) != {"FIT", "STOP"}):
        raise ValueError("prepared natural generation source, policy, status or roles changed")
    verify_binding(data["policy"])
    for key in ("CAL_decoded", "AUDIT_decoded", "raw_CSV_read", "simulated_futures_used", "risk_labels_used_for_training"):
        if data.get(key) is not False:
            raise ValueError("invalid preparation scope: " + key)
    for name, digest in data["code_sha256"].items():
        verify_binding(dict(path=name, sha256=digest))
    diagnostic = load_json(data["basis_diagnostic"])
    modes = int(data["basis"]["modes"])
    if (diagnostic["role"] != "FIT" or diagnostic["selected_K"] != modes
            or diagnostic["dataset_manifest"] != data["dataset_manifest"]
            or diagnostic["CAL_AUDIT_STOP_used_for_basis_selection"] is not False
            or diagnostic["per_K"][str(modes)]["gate_passed"] is not True
            or modes != min(int(k) for k, v in diagnostic["per_K"].items() if v["gate_passed"])
            or data["basis"]["frames"] != 175 or data["basis"]["sample_period"] != .04):
        raise ValueError("selected trajectory representation did not pass the frozen FIT-only gate")
    reference = load_json(data["reference_prepared"])
    if data["history_normalizer"] != reference["normalizer"]:
        raise ValueError("history normalization differs from the frozen FIT-only reference")
    coefficient, history = (load_json(data[key]) for key in ("coefficient_normalizer", "history_normalizer"))
    if (coefficient["roles_used"] != ["FIT"] or coefficient["STOP_CAL_AUDIT_used"] is not False
            or coefficient["labels_or_risk_used"] is not False or coefficient["fit_scene_count"] != 9913
            or coefficient["FIT_scene_id_sha256"] != history["FIT_scene_id_sha256"]):
        raise ValueError("coefficient normalization did not use the same FIT9913 identities")
    return coefficient


def load_pack(binding, role):
    if role not in ("FIT", "STOP"):
        raise PermissionError("training may decode FIT/STOP packs only")
    expected = set(FEATURES) | {"coef_clean", "anchors", "scene_id", "recording_id", "role"}
    path = verify_binding(binding)
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("generator pack must contain only permitted context, coefficients and metadata")
        roles = archive["role"]
        if not np.all(roles == role):
            raise ValueError("wrong role rejected before target/context decode")
        pack = {key: archive[key] for key in expected if key != "role"}
        pack["role"] = roles
    c, mask = pack["coef_clean"], pack["agent_mask"]
    if c.ndim != 4 or c.shape[:2] != mask.shape or c.shape[-1] != 2 or not np.isfinite(c[mask]).all():
        raise ValueError("valid per-actor normalized coefficient supervision required")
    return pack


def tensor_batch(pack, rows, device):
    arrays = {key: pack[key][rows] for key in FEATURES}
    width = int(np.flatnonzero(arrays["agent_mask"].any(0))[-1]) + 1
    road_width = int(np.flatnonzero(arrays["road_boundary_mask"].any(0))[-1]) + 1
    arrays["history"] = arrays["history"][:, :, :width]
    for key in ("dimensions", "agent_mask", "ego_mask"):
        arrays[key] = arrays[key][:, :width]
    for key in ("road_boundaries", "road_boundary_mask"):
        arrays[key] = arrays[key][:, :road_width]
    features = {key: torch.from_numpy(np.ascontiguousarray(value)).to(device) for key, value in arrays.items()}
    clean = torch.from_numpy(np.ascontiguousarray(pack["coef_clean"][rows, :width])).to(device=device, dtype=torch.float32)
    return features, clean


def make_model(policy, modes, device):
    m = policy["generator"]
    return JointHistoryDenoiser(2 * modes, hidden_dim=m["hidden_dim"], heads=m["heads"],
        layers=m["layers"], feedforward_dim=m["feedforward_dim"]).to(device)


def fixed_validation(model, schedule, pack, policy, device):
    """Fixed targets/noise per epoch, scene means then timestep means."""
    model.eval()
    cfg = policy["training"]
    scores = []
    # Generate role-wide random arrays on CPU; mask/padding never enter loss.
    rng = np.random.default_rng(cfg["validation_noise_seed"])
    with torch.no_grad():
        for timestep in cfg["validation_timesteps"]:
            noise = rng.standard_normal(pack["coef_clean"].shape).astype(np.float32)
            chunks = []
            for start in range(0, len(pack["scene_id"]), cfg["batch_size"]):
                rows = slice(start, start + cfg["batch_size"])
                features, clean = tensor_batch(pack, rows, device)
                eps = torch.from_numpy(np.ascontiguousarray(noise[rows, :clean.shape[1]])).to(device)
                t = torch.full((len(clean),), timestep, device=device, dtype=torch.long)
                result = diffusion_training_loss(model, schedule, clean, features, timesteps=t, noise=eps,
                    prediction_type=policy["generator"]["prediction_type"])
                chunks.append(result["per_scene_loss"].detach().cpu().numpy())
            scores.append(float(np.concatenate(chunks).mean()))
    if not np.isfinite(scores).all():
        raise FloatingPointError("validation loss is nonfinite")
    return dict(mean=float(np.mean(scores)), by_timestep=dict(zip(map(str, cfg["validation_timesteps"]), scores)))


def update_ema(ema, model, decay):
    with torch.no_grad():
        for target, source in zip(ema.parameters(), model.parameters()):
            target.mul_(decay).add_(source, alpha=1. - decay)
        for target, source in zip(ema.buffers(), model.buffers()):
            target.copy_(source)


def run(data_binding, policy_binding):
    if policy_binding["sha256"] != POLICY_SHA:
        raise ValueError("unregistered generator training policy")
    policy = load_json(policy_binding)
    data = load_json(data_binding)
    coefficient = validate_prepared_metadata(data, policy, policy_binding)
    if (data["dataset_manifest"]["sha256"] != policy["data_manifest"]["sha256"]
            or set(data["packs"]) != {"FIT", "STOP"}):
        raise ValueError("training source/roles must match the frozen natural dataset")
    modes = int(data["basis"]["modes"])
    if modes not in (8, 16, 24):
        raise ValueError("basis dimensionality must pass the registered FIT reconstruction gate")
    fit, stop = load_pack(data["packs"]["FIT"], "FIT"), load_pack(data["packs"]["STOP"], "STOP")
    if hashlib.sha256("\n".join(fit["scene_id"].tolist()).encode()).hexdigest() != coefficient["FIT_scene_id_sha256"]:
        raise ValueError("training FIT identities differ from normalization fit identities")
    if (len(fit["scene_id"]) != 9913 or len(stop["scene_id"]) != 538
            or set(fit["recording_id"]) & set(stop["recording_id"])
            or len(set(fit["recording_id"])) != 13 or len(set(stop["recording_id"])) != 4):
        raise ValueError("immutable FIT9913/STOP538 recording split required")
    if fit["coef_clean"].shape[-2] != modes or stop["coef_clean"].shape[-2] != modes:
        raise ValueError("basis and coefficient dimensions disagree")
    for key in ("coefficient_normalizer", "history_normalizer"):
        verify_binding(data[key])
    cfg = policy["training"]
    torch.set_num_threads(policy["runtime"]["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(cfg["seed"]); np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    device = torch.device(policy["runtime"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("registered CUDA device unavailable; do not silently change run protocol")
    code = code_hashes()
    output = resolve(policy["output_root"])
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "freeze_before_training.json", dict(protocol=PROTOCOL, data=data_binding,
        policy=policy_binding, code_sha256=code, modes=modes, risk_plugin_used=False,
        risk_or_percentile_labels_used=False, CAL_decoded=False, AUDIT_decoded=False,
        python=sys.version, torch=torch.__version__, numpy=np.__version__, device=str(device)))
    model = make_model(policy, modes, device)
    ema = copy.deepcopy(model)
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    schedule = CosineDiffusionSchedule(policy["generator"]["diffusion_steps"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"])
    generator = torch.Generator(device=device).manual_seed(cfg["seed"] + 1)
    initial = fixed_validation(ema, schedule, stop, policy, device)
    best, meaningful = initial["mean"], initial["mean"]
    best_epoch, stale, updates = 0, 0, 0
    best_state = {key: value.detach().cpu().clone() for key, value in ema.state_dict().items()}
    started = time.perf_counter()
    with (output / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            order = rng.permutation(len(fit["scene_id"]))
            total = 0.
            for start in range(0, len(order), cfg["batch_size"]):
                rows = order[start:start + cfg["batch_size"]]
                features, clean = tensor_batch(fit, rows, device)
                optimizer.zero_grad(set_to_none=True)
                answer = diffusion_training_loss(model, schedule, clean, features, generator=generator,
                    prediction_type=policy["generator"]["prediction_type"])
                loss = answer["loss"]
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite diffusion training loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("nonfinite denoiser gradient")
                optimizer.step(); update_ema(ema, model, cfg["EMA_decay"])
                total += float(loss.detach()) * len(rows); updates += 1
            scores = fixed_validation(ema, schedule, stop, policy, device)
            if scores["mean"] < best:
                best, best_epoch = scores["mean"], epoch
                best_state = {key: value.detach().cpu().clone() for key, value in ema.state_dict().items()}
            if scores["mean"] < meaningful - cfg["minimum_improvement"]:
                meaningful, stale = scores["mean"], 0
            else:
                stale += 1
            row = dict(epoch=epoch, online_train_v_MSE=total / len(order), STOP_fixed_v_MSE=scores,
                best_STOP_v_MSE=best, best_epoch=best_epoch, stale=stale, updates=updates,
                wall_seconds=time.perf_counter() - started)
            log.write(json.dumps(row, sort_keys=True) + "\n"); log.flush()
            print(json.dumps(row, sort_keys=True), flush=True)
            if stale >= cfg["patience"]:
                break
    ema.load_state_dict(best_state, strict=True)
    replay = fixed_validation(ema, schedule, stop, policy, device)
    if abs(replay["mean"] - best) > 1e-9:
        raise ValueError("selected EMA checkpoint does not replay STOP selection loss")
    if code_hashes() != code:
        raise ValueError("training source changed after run freeze")
    checkpoint = dict(protocol=PROTOCOL, state_dict=best_state, policy=policy_binding, data=data_binding,
        code_sha256=code, basis=data["basis"], coefficient_normalizer=data["coefficient_normalizer"],
        history_normalizer=data["history_normalizer"], architecture=ema.architecture_config(),
        prediction_type=policy["generator"]["prediction_type"], schedule=schedule.as_dict(), best_epoch=best_epoch,
        best_STOP_v_MSE=best, seed=cfg["seed"], risk_plugin_in_training=False)
    with (output / "best_ema.pt").open("xb") as handle:
        torch.save(checkpoint, handle)
    result = dict(protocol=PROTOCOL, status="complete", data=data_binding, policy=policy_binding,
        code_sha256=code, architecture=ema.architecture_config(), prediction_type=policy["generator"]["prediction_type"],
        initial_STOP_v_MSE=initial, best_STOP_v_MSE=replay, best_epoch=best_epoch, epochs_completed=epoch,
        updates=updates, seed=cfg["seed"], wall_seconds=time.perf_counter() - started,
        checkpoint=dict(path=str(output / "best_ema.pt"), sha256=sha256(output / "best_ema.pt")),
        epochs=dict(path=str(output / "epochs.jsonl"), sha256=sha256(output / "epochs.jsonl")),
        CAL_decoded=False, AUDIT_decoded=False, risk_labels_in_training=False,
        simulated_training_futures=False, generator_risk_control_success_claimed=False)
    binding = write_json(output / "result.json", result)
    print(json.dumps(dict(training_complete=True, result=binding, best_STOP_v_MSE=best, best_epoch=best_epoch)), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("data-manifest", "data-sha256", "policy", "policy-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(dict(path=args.data_manifest, sha256=args.data_sha256), dict(path=args.policy, sha256=args.policy_sha256))
