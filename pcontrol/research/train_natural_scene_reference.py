#!/usr/bin/env python3
"""Bounded, source-bound natural scene CRPS pilot: prepare, train, then audit.

No raw CSV, generator, simulation, OOF labels, or calibration fitting. CAL is
never decoded. AUDIT is only opened after all predeclared models are frozen.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pcontrol.data.complete_scene_view import collate_reference_examples, resolve, sha256, verify_binding
from pcontrol.reference.mixed_cdf import distribution_from_logits, cdf_from_params, quantile_from_params
from pcontrol.reference.scores import crps_from_params

POLICY_SHA = "de977c5819e758c947767ce2bfbb7924d932ef58613867fda07a03a8fe01d1cf"
PROTOCOL = "natural_ego_environment_scene_CRPS_pilot_v1"
FEATURES = ("history", "dimensions", "road_boundaries", "road_boundary_mask", "ego_mask", "agent_mask")
CODE = ("pcontrol/research/train_natural_scene_reference.py", "pcontrol/reference/scene_models.py",
        "pcontrol/reference/mixed_cdf.py", "pcontrol/reference/scores.py",
        "pcontrol/data/complete_scene_view.py", "pcontrol/data/expanded_scene_view.py")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def load_json(binding):
    return json.loads(verify_binding(binding).read_text())


def code_bindings():
    return {name: sha256(ROOT / name) for name in CODE}


def verify_code(expected):
    if expected != code_bindings():
        raise ValueError("training/model/reader/scoring code changed after preparation")


def read_policy(path, checksum):
    if checksum != POLICY_SHA:
        raise ValueError("only preregistered scene CRPS training policy allowed")
    return load_json(dict(path=str(path), sha256=checksum))


def fit_normalizer(examples):
    if not examples or any(x["metadata"]["role"] != "FIT" for x in examples):
        raise ValueError("normalizer requires nonempty FIT-only observations")
    history = np.concatenate([x["features"]["history"].reshape(-1, 4) for x in examples])
    dimensions = np.concatenate([x["features"]["dimensions"] for x in examples])
    if not np.isfinite(history).all() or not np.isfinite(dimensions).all():
        raise ValueError("normalizer cannot use nonfinite true observations")
    return dict(protocol="natural_scene_FIT_scale_only_v1", fit_scene_count=len(examples),
                history_valid_state_count=len(history), dimension_actor_occurrences=len(dimensions),
                history_scale=np.maximum(history.std(0), 1.).tolist(),
                dimension_scale=np.maximum(np.sqrt(np.mean(dimensions ** 2, axis=0)), 1.).tolist(),
                history_centering=False, road_scale_source="history_scale_y", targets_used=False,
                FIT_scene_id_sha256=hashlib.sha256("\n".join(x["metadata"]["scene_id"] for x in examples).encode()).hexdigest(),
                roles_used=["FIT"], future_or_CAL_or_AUDIT_used=False)


def pack_examples(examples, normalizer):
    batch = collate_reference_examples(examples)
    f = batch["features"]
    f["history"] = (f["history"] / np.asarray(normalizer["history_scale"])).astype(np.float32)
    f["dimensions"] = (f["dimensions"] / np.asarray(normalizer["dimension_scale"])).astype(np.float32)
    f["road_boundaries"] = (f["road_boundaries"] / normalizer["history_scale"][1]).astype(np.float32)
    result = dict(f, target=batch["target"],
                  scene_id=np.array([x["scene_id"] for x in batch["metadata"]]),
                  recording_id=np.array([x["recording_id"] for x in batch["metadata"]]),
                  role=np.array([x["role"] for x in batch["metadata"]]))
    if not np.isfinite(result["target"]).all() or np.any((result["target"] < 0) | (result["target"] > 4)):
        raise ValueError("training requires exact finite [0,4] scene labels")
    return result


def save_pack(path, pack):
    with Path(path).open("xb") as handle:
        np.savez_compressed(handle, **pack)
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def load_pack(binding, role):
    path = verify_binding(binding)
    with np.load(path, allow_pickle=False) as archive:
        expected = set(FEATURES) | {"target", "scene_id", "recording_id", "role"}
        if set(archive.files) != expected:
            raise ValueError("packed data feature allowlist mismatch before member decode")
        declared_role = archive["role"]
        if not np.all(declared_role == role):
            raise ValueError("packed data role mismatch before feature/target decode")
        pack = {key: archive[key] for key in expected if key != "role"}
        pack["role"] = declared_role
    return pack


def _batch(pack, rows):
    # A literal allowlist prevents metadata or supervision from reaching model.
    arrays = {key: pack[key][rows] for key in FEATURES}
    # Storage packs share a role-wide width; discard only trailing columns
    # masked false for EVERY row in this batch. No actual vehicle is removed.
    n = int(np.flatnonzero(arrays["agent_mask"].any(0))[-1]) + 1
    r = int(np.flatnonzero(arrays["road_boundary_mask"].any(0))[-1]) + 1
    arrays["history"] = arrays["history"][:, :, :n]
    for key in ("dimensions", "ego_mask", "agent_mask"):
        arrays[key] = arrays[key][:, :n]
    for key in ("road_boundaries", "road_boundary_mask"):
        arrays[key] = arrays[key][:, :r]
    features = {key: torch.from_numpy(arrays[key]) for key in FEATURES}
    target = torch.from_numpy(pack["target"][rows])
    return features, target


def make_view(binding, purpose):
    from pcontrol.data.expanded_scene_view import ExpandedSceneReferenceView
    return ExpandedSceneReferenceView(binding["path"], binding["sha256"], purpose=purpose)


def prepare(data_binding, policy_binding, output_root):
    policy = read_policy(policy_binding["path"], policy_binding["sha256"])
    verify_binding(data_binding)
    output = resolve(output_root)
    if output.exists():
        raise FileExistsError("never overwrite a previous scene CRPS pilot")
    code = code_bindings()
    fit = list(make_view(data_binding, "fit"))
    stop = list(make_view(data_binding, "stop"))
    fit_records = {x["metadata"]["recording_id"] for x in fit}
    stop_records = {x["metadata"]["recording_id"] for x in stop}
    if len(fit_records) != 13 or len(stop_records) != 4 or fit_records & stop_records or len(fit) <= 1256 or len(stop) != 538:
        raise ValueError("expected expanded FIT13 and unchanged STOP4/538 with no record overlap")
    normalizer = fit_normalizer(fit)
    fit_pack, stop_pack = pack_examples(fit, normalizer), pack_examples(stop, normalizer)
    output.mkdir(parents=True)
    norm_binding = write_json(output / "normalizer.json", normalizer)
    packs = dict(FIT=save_pack(output / "FIT_predictor_pack.npz", fit_pack),
                 STOP=save_pack(output / "STOP_predictor_pack.npz", stop_pack))
    verify_code(code)
    prepared = dict(protocol=PROTOCOL, status="prepared", data=data_binding, policy=policy_binding,
                    code_sha256=code, normalizer=norm_binding, packs=packs,
                    counts=dict(FIT=len(fit), STOP=len(stop)), seed=policy["training"]["seed"],
                    environment=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
                                     device=policy["runtime"]["device"], threads=policy["runtime"]["threads_per_worker"]),
                    FIT_recordings=sorted(fit_records), STOP_recordings=sorted(stop_records),
                    CAL_decoded=False, AUDIT_decoded=False, generator_training=False,
                    estimand=policy["estimand"], target_or_future_is_feature=False)
    binding = write_json(output / "prepared.json", prepared)
    print(json.dumps(dict(stage="prepared", binding=binding, counts=prepared["counts"])), flush=True)
    return binding


class GlobalSceneCDF(nn.Module):
    """Context-free learnable proper-score baseline, no reference lookup."""
    def __init__(self, knots, zero_atom_enabled=True):
        super().__init__()
        self.register_buffer("knots", knots.to(torch.float64))
        self.zero_atom_enabled = zero_atom_enabled
        self.logits = nn.Parameter(torch.zeros(len(knots) + int(zero_atom_enabled)))

    def forward(self, features):
        if set(features) != set(FEATURES):
            raise ValueError("global baseline receives the same allowed features")
        logits = self.logits[None].expand(features["history"].shape[0], -1).to(torch.float64)
        return distribution_from_logits(self.knots, logits, zero_atom_enabled=self.zero_atom_enabled,
                                        model_version="natural_scene_global_CDF_v1")


def make_model(variant, policy):
    from pcontrol.reference.scene_models import SceneCDFReference
    m = policy["model"]
    knots = torch.linspace(0., m["cap_seconds"], m["bins"] + 1, dtype=torch.float64)
    if variant == "Global":
        return GlobalSceneCDF(knots, m["zero_atom_enabled"])
    return SceneCDFReference(variant, knots, zero_atom_enabled=m["zero_atom_enabled"],
                             hidden_dim=m["hidden_dim"], heads=m["heads"])


def set_runtime(policy):
    torch.set_num_threads(policy["runtime"]["threads_per_worker"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    seed = policy["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def mean_crps(model, pack, batch_size):
    model.eval()
    scores = []
    with torch.no_grad():
        for start in range(0, len(pack["target"]), batch_size):
            features, target = _batch(pack, slice(start, start + batch_size))
            scores.append(crps_from_params(model(features), target).cpu().numpy())
    values = np.concatenate(scores)
    if not np.isfinite(values).all():
        raise FloatingPointError("nonfinite evaluation CRPS")
    return float(values.mean())


def read_prepared(binding):
    prepared = load_json(binding)
    if prepared.get("protocol") != PROTOCOL or prepared.get("status") != "prepared":
        raise ValueError("wrong prepared pilot")
    verify_code(prepared["code_sha256"])
    policy = read_policy(prepared["policy"]["path"], prepared["policy"]["sha256"])
    verify_binding(prepared["data"])
    verify_binding(prepared["normalizer"])
    return prepared, policy


def train_cell(prepared_binding, variant):
    prepared, policy = read_prepared(prepared_binding)
    if variant not in policy["model"]["variants"]:
        raise ValueError("unregistered model variant")
    set_runtime(policy)
    output = resolve(prepared_binding["path"]).parent / variant
    output.mkdir(exist_ok=False)
    fit = load_pack(prepared["packs"]["FIT"], "FIT")
    stop = load_pack(prepared["packs"]["STOP"], "STOP")
    model = make_model(variant, policy)
    count = sum(p.numel() for p in model.parameters())
    if variant in ("M1", "M2"):
        other = make_model("M2" if variant == "M1" else "M1", policy)
        if count != sum(p.numel() for p in other.parameters()):
            raise ValueError("M1/M2 capacity matching failed")
        del other
    cfg = policy["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"])
    initial_stop = mean_crps(model, stop, cfg["batch_size"])
    best = initial_stop
    significant_best = best
    best_epoch, bad_epochs = 0, 0
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    with (output / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            order = rng.permutation(len(fit["target"]))
            accumulated = 0.
            for start in range(0, len(order), cfg["batch_size"]):
                rows = order[start:start + cfg["batch_size"]]
                features, target = _batch(fit, rows)
                optimizer.zero_grad(set_to_none=True)
                losses = crps_from_params(model(features), target, normalized=True)
                loss = losses.mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite CRPS training loss")
                loss.backward()
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
                if not bool(torch.isfinite(grad)):
                    raise FloatingPointError("nonfinite model gradient")
                optimizer.step()
                accumulated += float(loss.detach()) * len(rows) * 4.
            score = mean_crps(model, stop, cfg["batch_size"])
            if score < best:
                best, best_epoch, best_state = score, epoch, copy.deepcopy(model.state_dict())
            if score < significant_best - cfg["minimum_improvement_seconds"]:
                significant_best, bad_epochs = score, 0
            else:
                bad_epochs += 1
            row = dict(epoch=epoch, variant=variant, online_train_mean_CRPS_seconds=accumulated / len(order),
                       STOP_mean_CRPS_seconds=score, best_STOP_CRPS_seconds=best, best_epoch=best_epoch,
                       bad_epochs=bad_epochs, wall_seconds=time.perf_counter() - started)
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            print(json.dumps(row, sort_keys=True), flush=True)
            if bad_epochs >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    checked_stop = mean_crps(model, stop, cfg["batch_size"])
    if abs(checked_stop - best) > 1e-12:
        raise ValueError("best checkpoint does not replay selected STOP score")
    verify_code(prepared["code_sha256"])
    checkpoint = dict(protocol=PROTOCOL, variant=variant, state_dict=best_state,
                      prepared=prepared_binding, normalizer=prepared["normalizer"], data=prepared["data"],
                      code_sha256=prepared["code_sha256"], seed=cfg["seed"], best_epoch=best_epoch,
                      best_STOP_CRPS_seconds=best, parameters=count)
    with (output / "best.pt").open("xb") as handle:
        torch.save(checkpoint, handle)
    result = dict(protocol=PROTOCOL, status="completed", variant=variant, prepared=prepared_binding,
                  code_sha256=prepared["code_sha256"], parameters=count, seed=cfg["seed"],
                  architecture=model.architecture_config() if hasattr(model, "architecture_config") else
                      dict(version="natural_scene_global_CDF_v1", context_conditioned=False, parameters=count,
                           knots=model.knots.tolist(), zero_atom_enabled=True, pretrained_weights_loaded=False),
                  initial_STOP_CRPS_seconds=initial_stop, best_STOP_CRPS_seconds=best, best_epoch=best_epoch,
                  epochs_completed=epoch, wall_seconds=time.perf_counter() - started,
                  checkpoint=dict(path=str(output / "best.pt"), sha256=sha256(output / "best.pt")),
                  epochs=dict(path=str(output / "epochs.jsonl"), sha256=sha256(output / "epochs.jsonl")),
                  CAL_decoded=False, AUDIT_decoded=False, calibration_training=False, generator_training=False)
    binding = write_json(output / "result.json", result)
    print(json.dumps(dict(stage="training_completed", variant=variant, binding=binding, best_STOP_CRPS_seconds=best)), flush=True)
    return result


def prediction_arrays(model, pack, policy):
    model.eval()
    levels = torch.tensor(policy["evaluation"]["quantile_levels"], dtype=torch.float64)[None]
    thresholds = torch.tensor(policy["evaluation"]["thresholds_seconds"], dtype=torch.float64)[None]
    chunks = {key: [] for key in ("crps_seconds", "cdf_left", "cdf_right", "quantiles", "threshold_cdf", "joint_masses")}
    with torch.no_grad():
        for start in range(0, len(pack["target"]), policy["training"]["batch_size"]):
            features, target = _batch(pack, slice(start, start + policy["training"]["batch_size"]))
            params = model(features)
            values = dict(crps_seconds=crps_from_params(params, target), cdf_left=cdf_from_params(params, target, side="left"),
                          cdf_right=cdf_from_params(params, target), quantiles=quantile_from_params(params, levels),
                          threshold_cdf=cdf_from_params(params, thresholds), joint_masses=params.joint_masses)
            for key, value in values.items():
                chunks[key].append(value.cpu().numpy())
    arrays = {key: np.concatenate(value) for key, value in chunks.items()}
    arrays.update(target=pack["target"], scene_id=pack["scene_id"], recording_id=pack["recording_id"],
                  num_agents=pack["agent_mask"].sum(1))
    uniforms = np.random.default_rng(policy["evaluation"]["pit_seed"]).uniform(size=len(pack["target"]))
    arrays["pit_uniform"] = uniforms
    arrays["randomized_pit"] = arrays["cdf_left"] + uniforms * (arrays["cdf_right"] - arrays["cdf_left"])
    return arrays


def summarize_predictions(a, policy, mask=None):
    if mask is None:
        mask = np.ones(len(a["target"]), bool)
    p = {key: value[mask] for key, value in a.items()}
    n = len(p["target"])
    if not n:
        return dict(scenes=0)
    by_record = {str(rec): float(p["crps_seconds"][p["recording_id"] == rec].mean()) for rec in np.unique(p["recording_id"])}
    levels = np.asarray(policy["evaluation"]["quantile_levels"])
    less = (p["target"][:, None] < p["quantiles"]).mean(0)
    leq = (p["target"][:, None] <= p["quantiles"]).mean(0)
    thresholds = np.asarray(policy["evaluation"]["thresholds_seconds"])
    obs = (p["target"][:, None] <= thresholds).astype(float)
    ordered = np.sort(p["randomized_pit"])
    ks = max(np.max(np.arange(1, n + 1) / n - ordered), np.max(ordered - np.arange(n) / n))
    return dict(scenes=n, CRPS_seconds=float(p["crps_seconds"].mean()), normalized_CRPS=float(p["crps_seconds"].mean() / 4),
                recording_macro_CRPS_seconds=float(np.mean(list(by_record.values()))), by_recording_CRPS_seconds=by_record,
                cap_Brier=float(np.mean((p["joint_masses"][:, -1] - (p["target"] == 4)) ** 2)),
                zero_Brier=float(np.mean((p["joint_masses"][:, 0] - (p["target"] == 0)) ** 2)),
                randomized_PIT_KS=float(ks), randomized_PIT_mean=float(ordered.mean()),
                quantile_levels=levels.tolist(), quantile_strict_coverage=less.tolist(), quantile_nonstrict_coverage=leq.tolist(),
                atom_aware_coverage_gap_mean=float(np.maximum.reduce((less - levels, levels - leq, np.zeros_like(levels))).mean()),
                atom_aware_coverage_gap_max=float(np.maximum.reduce((less - levels, levels - leq, np.zeros_like(levels))).max()),
                thresholds_seconds=thresholds.tolist(), threshold_observed_frequency=obs.mean(0).tolist(),
                threshold_predicted_mean=p["threshold_cdf"].mean(0).tolist(),
                threshold_mean_absolute_calibration_error=float(np.abs(p["threshold_cdf"].mean(0) - obs.mean(0)).mean()),
                threshold_Brier=((p["threshold_cdf"] - obs) ** 2).mean(0).tolist())


def paired_recording_bootstrap(left, right, policy):
    if any(not np.array_equal(left[key], right[key]) for key in ("scene_id", "recording_id", "target")):
        raise ValueError("paired comparisons require identical scenes, recordings, targets and order")
    delta = left["crps_seconds"] - right["crps_seconds"]
    records = np.unique(left["recording_id"])
    totals = np.array([delta[left["recording_id"] == rec].sum() for rec in records])
    counts = np.array([(left["recording_id"] == rec).sum() for rec in records])
    rng = np.random.default_rng(policy["evaluation"]["bootstrap_seed"])
    draws = rng.integers(len(records), size=(policy["evaluation"]["recording_bootstrap_repetitions"], len(records)))
    scene_weighted = totals[draws].sum(1) / counts[draws].sum(1)
    macro = (totals[draws] / counts[draws]).mean(1)
    return dict(delta_scene_mean_seconds=float(delta.mean()), delta_recording_macro_seconds=float(np.mean(totals / counts)),
                recording_bootstrap_scene_weighted_95pct=np.quantile(scene_weighted, [.025, .975]).tolist(),
                recording_bootstrap_macro_95pct=np.quantile(macro, [.025, .975]).tolist(),
                recordings=len(records), repeats=len(draws), lower_delta_favors_left=True,
                interval_is_exploratory_few_recordings=True, single_training_seed=True)


def audit_models(prepared_binding):
    prepared, policy = read_prepared(prepared_binding)
    set_runtime(policy)
    root = resolve(prepared_binding["path"]).parent
    output = root / "AUDIT_once"
    if output.exists():
        raise FileExistsError("no repeat/tuning on this frozen AUDIT comparison")
    bindings, results, models = {}, {}, {}
    for variant in policy["model"]["variants"]:
        result_path = root / variant / "result.json"
        binding = dict(path=str(result_path), sha256=sha256(result_path))
        result = load_json(binding)
        if (result.get("protocol") != PROTOCOL or result.get("status") != "completed"
                or result.get("variant") != variant or result.get("prepared") != prepared_binding
                or result.get("code_sha256") != prepared["code_sha256"]):
            raise ValueError("all predeclared training cells must complete against one frozen preparation")
        checkpoint_path = verify_binding(result["checkpoint"], root / variant / "best.pt")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        expected_header = dict(protocol=PROTOCOL, variant=variant, prepared=prepared_binding,
            data=prepared["data"], normalizer=prepared["normalizer"], code_sha256=prepared["code_sha256"],
            seed=policy["training"]["seed"], parameters=result["parameters"], best_epoch=result["best_epoch"],
            best_STOP_CRPS_seconds=result["best_STOP_CRPS_seconds"])
        if any(checkpoint.get(key) != value for key, value in expected_header.items()):
            raise ValueError("checkpoint header lineage mismatch before AUDIT access")
        model = make_model(variant, policy)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        if sum(p.numel() for p in model.parameters()) != result["parameters"]:
            raise ValueError("checkpoint/model parameter count mismatch before AUDIT access")
        models[variant] = model
        bindings[variant], results[variant] = binding, result
    if results["M1"]["parameters"] != results["M2"]["parameters"]:
        raise ValueError("primary model pair differs in parameter count")
    output.mkdir()
    freeze = write_json(output / "freeze_before_AUDIT_decode.json", dict(protocol=PROTOCOL, prepared=prepared_binding,
                        training_results=bindings, checkpoint_bindings={v: r["checkpoint"] for v, r in results.items()},
                        primary_comparison=policy["model"]["primary_comparison"], code_sha256=prepared["code_sha256"],
                        all_checkpoint_headers_and_state_dicts_prevalidated=True,
                        CAL_decoded=False, AUDIT_decoded=False, retuning_authorized=False))
    examples = list(make_view(prepared["data"], "audit"))
    if len(examples) != 487 or len({e["metadata"]["recording_id"] for e in examples}) != 6:
        raise ValueError("unchanged AUDIT6/487 required")
    pack = pack_examples(examples, load_json(prepared["normalizer"]))
    summaries, predictions = {}, {}
    for variant, result in results.items():
        verify_binding(result["checkpoint"])
        arrays = prediction_arrays(models[variant], pack, policy)
        predictions[variant] = arrays
        binding = save_pack(output / f"{variant}_predictions.npz", arrays)
        summaries[variant] = dict(overall=summarize_predictions(arrays, policy), predictions=binding,
            by_N_group={name: summarize_predictions(arrays, policy, (arrays["num_agents"] >= lo) & (arrays["num_agents"] <= hi))
                        for name, lo, hi in (("N3_5", 3, 5), ("N6_8", 6, 8), ("N9_plus", 9, 100000))})
    comparisons = {f"M2_minus_{v}": paired_recording_bootstrap(predictions["M2"], predictions[v], policy)
                   for v in ("M1", "M0", "Global")}
    verify_code(prepared["code_sha256"])
    report = dict(protocol=PROTOCOL, status="complete", prepared=prepared_binding, freeze=freeze,
                  code_sha256=prepared["code_sha256"], models=summaries, paired_comparisons=comparisons,
                  primary_comparison="M2_minus_M1", CAL_decoded=False, calibration_training=False,
                  generator_training=False, retuning_on_AUDIT=False,
                  scope="raw_CDF_internal_development_AUDIT_one_seed_not_final_blind_test",
                  estimand=policy["estimand"])
    binding = write_json(output / "results.json", report)
    print(json.dumps(dict(stage="AUDIT_complete", binding=binding, scores={v: r["overall"]["CRPS_seconds"] for v, r in summaries.items()},
                          comparisons=comparisons)), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    for key in ("manifest", "manifest-sha256", "policy", "policy-sha256", "output-root"):
        p.add_argument("--" + key, required=True)
    for name in ("train", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--prepared", required=True)
        p.add_argument("--prepared-sha256", required=True)
        if name == "train":
            p.add_argument("--variant", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(dict(path=args.manifest, sha256=args.manifest_sha256), dict(path=args.policy, sha256=args.policy_sha256), args.output_root)
    elif args.command == "train":
        train_cell(dict(path=args.prepared, sha256=args.prepared_sha256), args.variant)
    else:
        audit_models(dict(path=args.prepared, sha256=args.prepared_sha256))


if __name__ == "__main__":
    main()
