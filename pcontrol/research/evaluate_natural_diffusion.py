#!/usr/bin/env python3
"""Paired K=1 STOP selection then locked internal-development AUDIT pilot."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from pcontrol.data.complete_scene_view import verify_binding, sha256
from pcontrol.generation.diffusion import JointHistoryDenoiser, CosineDiffusionSchedule, ddim_sample
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.risk_guidance import TorchTrajectoryDecoder, ActivePETGuidance, interval_p_error
from pcontrol.generation.evaluation import select_role_cases, quality_metrics
from pcontrol.plugins.risk_plugin import FrozenRiskPlugin

PROTOCOL = "natural_decoupled_diffusion_K1_development_pilot_v1"
POLICY_SHA = "5477744cd4a71eef90d0fd55882185cda50256e74f0eeb8703f55d4ed31d4997"
CODE = ("pcontrol/research/evaluate_natural_diffusion.py", "pcontrol/generation/risk_guidance.py",
        "pcontrol/generation/evaluation.py", "pcontrol/plugins/risk_plugin.py")


def clean_json(value):
    if isinstance(value, dict): return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray): return clean_json(value.tolist())
    if isinstance(value, np.generic): return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "positive_infinity" if np.isposinf(value) else "negative_infinity" if np.isneginf(value) else "NaN"
    return value


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(clean_json(value), handle, sort_keys=True, indent=2, allow_nan=False)
    return dict(path=str(Path(path).resolve()), sha256=sha256(path))


def bound_json(binding): return json.loads(verify_binding(binding).read_text())


def model_features(case, normalizer, device):
    n = case["num_agents"]
    hscale, dscale = np.asarray(normalizer["history_scale"]), np.asarray(normalizer["dimension_scale"])
    arrays = dict(history=(case["history"][None] / hscale).astype(np.float32),
                  dimensions=(case["dimensions"][None] / dscale).astype(np.float32),
                  road_boundaries=(case["road_boundaries"][None] / hscale[1]).astype(np.float32),
                  road_boundary_mask=np.ones((1, len(case["road_boundaries"])), bool),
                  ego_mask=case["ego_mask"][None], agent_mask=np.ones((1, n), bool))
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in arrays.items()}


def summarize(rows):
    summary = dict(rows=len(rows), scenes=len({r["scene_id"] for r in rows}),
                   mean_interval_p_error=float(np.mean([r["interval_p_error"] for r in rows])),
                   mean_midrank_p_absolute_error=float(np.mean([r["midrank_p_absolute_error"] for r in rows])),
                   mean_PET_target_absolute_error_seconds=float(np.mean([r["PET_target_absolute_error_seconds"] for r in rows])))
    for name in ("all_pair_overlap_scene", "road_outside_scene"):
        summary[name + "_rate"] = float(np.mean([r["quality"][name] for r in rows]))
    numeric_keys = set.intersection(*(set(r["quality"]) for r in rows))
    summary["mean_quality"] = {k: float(np.mean([r["quality"][k] for r in rows])) for k in sorted(numeric_keys)
                               if all(isinstance(r["quality"][k], (bool, int, float, np.number)) for r in rows)}
    pairs = []
    for sid in sorted({r["scene_id"] for r in rows}):
        ordered = sorted((r for r in rows if r["scene_id"] == sid), key=lambda r: r["requested_p"])
        pairs.append(dict(scene_id=sid, requested_p=[r["requested_p"] for r in ordered],
                          achieved_midrank=[r["estimated_rank"]["p_mid"] for r in ordered],
                          PET=[r["pet_seconds"] for r in ordered],
                          strictly_ordered=bool(np.all(np.diff([r["pet_seconds"] for r in ordered]) < -1e-6)),
                          high_minus_low_p=float(ordered[-1]["estimated_rank"]["p_mid"] - ordered[0]["estimated_rank"]["p_mid"])))
    summary["paired_response"] = pairs
    summary["strict_order_fraction"] = float(np.mean([r["strictly_ordered"] for r in pairs]))
    summary["mean_p90_minus_p10_response"] = float(np.mean([r["high_minus_low_p"] for r in pairs]))
    summary["guidance_supported_steps"] = sum(r["guidance_supported_steps"] for r in rows)
    summary["guidance_attempted_steps"] = sum(r["guidance_attempted_steps"] for r in rows)
    return summary


def choose_strength(summaries):
    base = summaries["0"]
    eligible = []
    for key, value in summaries.items():
        if all(value[name + "_rate"] <= base[name + "_rate"] + 1e-12
               for name in ("all_pair_overlap_scene", "road_outside_scene")):
            eligible.append(float(key))
    return min(eligible, key=lambda s: (summaries[format(s, "g")]["mean_interval_p_error"], s))


def run(training_binding, role, selection_binding=None):
    if role not in ("STOP", "AUDIT"): raise PermissionError("pilot role must be STOP or internal AUDIT")
    training = bound_json(training_binding)
    if (training.get("status") != "complete" or training["policy"]["sha256"] != POLICY_SHA
            or training.get("risk_labels_in_training") is not False or training.get("simulated_training_futures") is not False):
        raise ValueError("only the frozen completed natural-history-only prior is allowed")
    policy = bound_json(training["policy"])
    data = bound_json(training["data"])
    for path, digest in training["code_sha256"].items(): verify_binding(dict(path=path, sha256=digest))
    if role == "AUDIT":
        if selection_binding is None: raise PermissionError("AUDIT requires locked STOP selection before any case access")
        selected = bound_json(selection_binding)
        if (selected["training_result"] != training_binding or selected["selected_on"] != "STOP"
                or selected["AUDIT_decoded_before_selection"] is not False):
            raise ValueError("invalid STOP strength selection")
        for name, digest in selected["evaluation_code_sha256"].items(): verify_binding(dict(path=name, sha256=digest))
        stop_result = bound_json(selected["STOP_result"])
        if (stop_result["role"] != "STOP" or stop_result["status"] != "complete"
                or stop_result["training_result"] != training_binding
                or stop_result["evaluation_code_sha256"] != selected["evaluation_code_sha256"]
                or set(map(float, stop_result["summaries"])) != set(policy["sampling"]["guidance_strength_candidates_STOP_only"])
                or float(selected["selected_strength"]) != choose_strength(stop_result["summaries"])):
            raise ValueError("AUDIT selection does not replay the hash-bound STOP candidate comparison")
        strengths = sorted(set((0., float(selected["selected_strength"]))))
    else:
        if selection_binding is not None: raise ValueError("STOP cannot consume a later selection")
        strengths = policy["sampling"]["guidance_strength_candidates_STOP_only"]
    cfg = policy["runtime"]
    torch.set_num_threads(cfg["threads"]); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(cfg["device"])
    checkpoint = torch.load(verify_binding(training["checkpoint"]), map_location="cpu")
    if (checkpoint["data"] != training["data"] or checkpoint["policy"] != training["policy"]
            or checkpoint["prediction_type"] != policy["generator"]["prediction_type"]
            or checkpoint["architecture"] != training["architecture"]
            or checkpoint["architecture"]["coefficient_dim"] != 2 * data["basis"]["modes"]
            or checkpoint["code_sha256"] != training["code_sha256"]
            or checkpoint["seed"] != policy["training"]["seed"] or checkpoint["best_epoch"] != training["best_epoch"]
            or checkpoint["basis"] != data["basis"]
            or any(checkpoint["architecture"][k] != policy["generator"][k]
                   for k in ("hidden_dim", "heads", "layers", "feedforward_dim"))):
        raise ValueError("checkpoint and completed training result disagree")
    architecture = {k: checkpoint["architecture"][k] for k in
                    ("coefficient_dim", "hidden_dim", "heads", "layers", "feedforward_dim")}
    model = JointHistoryDenoiser(**architecture).to(device)
    model.load_state_dict(checkpoint["state_dict"], strict=True); model.eval()
    for parameter in model.parameters(): parameter.requires_grad_(False)
    schedule = CosineDiffusionSchedule(policy["generator"]["diffusion_steps"]).to(device)
    if checkpoint["schedule"] != schedule.as_dict():
        raise ValueError("sampling schedule differs from the trained forward process")
    basis = TrajectoryBasis(int(data["basis"]["modes"]))
    cnorm, hnorm = bound_json(data["coefficient_normalizer"]), bound_json(data["history_normalizer"])
    plugin = FrozenRiskPlugin.from_refinement_binding(policy["risk_plugin_result"], device="cpu")
    output = verify_binding(training["checkpoint"]).parent / (role + "_pilot")
    output.mkdir(parents=True, exist_ok=False)
    code = {name: sha256(ROOT / name) for name in CODE}
    write_json(output / "freeze_before_sampling.json", dict(protocol=PROTOCOL, role=role, training=training_binding,
        policy=training["policy"], strengths=strengths, evaluation_code_sha256=code,
        selection_binding=selection_binding, K=1, shared_noise_across_p_and_strength=True, plugin=plugin.describe()))
    cases = select_role_cases(policy["data_manifest"], role=role, per_stratum=4,
                             salt="natural_decoupled_diffusion_eval_v1")
    declarations = [{k: c[k] for k in ("scene_id", "recording_id", "num_agents", "role", "stratum", "selection_hash")} for c in cases]
    write_json(output / "case_selection.json", dict(cases=declarations, selected_by_identity_and_N_only=True,
                                                  no_score_or_PET_selection=True, random_population_sample=False))
    rows, started = [], time.perf_counter()
    for number, case in enumerate(cases):
        features = model_features(case, hnorm, device)
        reference = plugin.condition(case["history"], case["dimensions"], case["road_boundaries"],
                                     case["ego_mask"], np.ones(case["num_agents"], bool))
        decoder = TorchTrajectoryDecoder(basis, cnorm, case["history"][-1], device=device)
        seed = int.from_bytes(hashlib.sha256(("natural_diffusion_K1_noise_v1|" + case["scene_id"]).encode()).digest()[:8], "little") % (2**32)
        rng = np.random.default_rng(seed)
        z = torch.from_numpy(rng.standard_normal((1, case["num_agents"], basis.modes, 2)).astype(np.float32)).to(device)
        arrays = {k: case[k] for k in ("history", "dimensions", "road_boundaries", "ego_mask", "agent_ids")}
        arrays["future_observed"] = case["future"]
        arrays["initial_noise"] = z[0].cpu().numpy()
        observed_score = reference.score_future(case["future"])
        observed_quality = quality_metrics(case["future"], case)
        baseline = None
        for strength in strengths:
            for p in policy["sampling"]["p_grid"]:
                guide = ActivePETGuidance(reference, decoder, p, strength=float(strength), total_steps=50,
                                          last_steps=20, rms_cap=.15)
                if strength == 0 and baseline is not None:
                    future, score, quality = baseline
                else:
                    coefficients = ddim_sample(model, schedule, features, z, steps=50,
                                               prediction_type=checkpoint["prediction_type"],
                                               x0_callback=None if strength == 0 else guide)
                    with torch.no_grad(): future = decoder(coefficients[0]).cpu().numpy()
                    score, quality = reference.score_future(future), quality_metrics(future, case)
                    if strength == 0: baseline = (future, score, quality)
                label = "s" + format(strength, "g").replace(".", "_") + "_p" + format(p, "g").replace(".", "_")
                arrays["generated_" + label] = future
                rank = score["estimated_rank"]
                row = dict(scene_id=case["scene_id"], recording_id=case["recording_id"], num_agents=case["num_agents"],
                    role=role, stratum=case["stratum"], strength=float(strength), requested_p=float(p), noise_seed=seed,
                    pet_seconds=score["pet_seconds"], pet_raw_seconds=score["pet_raw_seconds"], estimated_rank=rank,
                    target=reference.target_spec(p), interval_p_error=interval_p_error(p, rank),
                    midrank_p_absolute_error=abs(float(rank["p_mid"])-p),
                    PET_target_absolute_error_seconds=abs(score["pet_seconds"]-float(reference.target_pet(p))),
                    critical_container_index=score["critical_container_index"], witness=score["metric"]["witness"],
                    quality=quality, guidance_attempted_steps=len(guide.trace),
                    guidance_supported_steps=sum(t["supported"] for t in guide.trace), guidance_trace=guide.trace,
                    guidance_reasons=dict(Counter(t["reason"] for t in guide.trace)),
                    observed_PET=observed_score["pet_seconds"], observed_quality=observed_quality,
                    array_key="generated_"+label, candidate_origin="generated_not_natural_observation", K=1)
                rows.append(row)
                print(json.dumps(clean_json(dict(stage="sample_scored", role=role, case=number+1, total=len(cases),
                    N=case["num_agents"], strength=strength, p=p, PET=row["pet_seconds"],
                    p_error=row["interval_p_error"], supported=row["guidance_supported_steps"], wall_seconds=time.perf_counter()-started))), flush=True)
        with (output / ("case_%02d.npz" % number)).open("xb") as handle: np.savez_compressed(handle, **arrays)
        for row in rows:
            if row["scene_id"] == case["scene_id"]:
                row["trajectory_artifact"] = dict(path=str(output / ("case_%02d.npz" % number)), sha256=sha256(output / ("case_%02d.npz" % number)))
        write_json(output / ("case_%02d.json" % number), dict(case=declarations[number], rows=[r for r in rows if r["scene_id"] == case["scene_id"]]))
    summaries = {format(s, "g"): summarize([r for r in rows if r["strength"] == s]) for s in strengths}
    for name, digest in code.items(): verify_binding(dict(path=name, sha256=digest))
    result = dict(protocol=PROTOCOL, status="complete", role=role, training_result=training_binding,
                  summaries=summaries, rows=rows, wall_seconds=time.perf_counter()-started, evaluation_code_sha256=code,
                  role_is_internal_development_not_final_blind_test=True, K=1, CAL_decoded=False,
                  estimator_true_percentiles_known=False, generated_outputs_asserted_natural=False,
                  quality_measures_discrete_box_overlap_not_verified_collisions=True)
    result_binding = write_json(output / "results.json", result)
    if role == "STOP":
        strength = choose_strength(summaries)
        selection = write_json(output / "selected_strength_before_AUDIT.json", dict(protocol=PROTOCOL,
            training_result=training_binding, STOP_result=result_binding, selected_on="STOP", selected_strength=strength,
            evaluation_code_sha256=code, AUDIT_decoded_before_selection=False,
            rule="min mean interval-p-error subject to no increase in scene overlap/road-outside; tie smaller strength"))
        print(json.dumps(dict(STOP_complete=True, result=result_binding, selection=selection, selected_strength=strength)), flush=True)
    else: print(json.dumps(dict(AUDIT_complete=True, result=result_binding)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", required=True); parser.add_argument("--training-sha256", required=True)
    parser.add_argument("--role", choices=("STOP", "AUDIT"), required=True)
    parser.add_argument("--selection"); parser.add_argument("--selection-sha256")
    args = parser.parse_args()
    selected = None if args.selection is None else dict(path=args.selection, sha256=args.selection_sha256)
    run(dict(path=args.training_result, sha256=args.training_sha256), args.role, selected)
