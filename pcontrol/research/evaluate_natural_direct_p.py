#!/usr/bin/env python3
"""Raw direct-P versus history-only, paired H/z, with no guidance/correction."""
import argparse
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
from pcontrol.generation.direct_p import JointPercentileDenoiser, direct_p_sample
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.prior import NaturalDiffusionPrior
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.evaluation import select_role_cases, quality_metrics
from pcontrol.plugins.risk_plugin import FrozenRiskPlugin
from pcontrol.research.evaluate_natural_diffusion import bound_json, write_json, clean_json, model_features

POLICY_SHA = "c5d52262816e9e7b298d7108960735009359e4abb57e26adacf73b29ac7d3ab7"
PROTOCOL = "natural_raw_direct_P_paired_K1_development_v1"
CODE = ("pcontrol/research/evaluate_natural_direct_p.py", "pcontrol/research/evaluate_natural_diffusion.py",
        "pcontrol/generation/direct_p.py", "pcontrol/generation/diffusion.py",
        "pcontrol/generation/prior.py", "pcontrol/generation/trajectory_basis.py",
        "pcontrol/generation/evaluation.py", "pcontrol/plugins/risk_plugin.py")


def interval_error(p, rank):
    return float(max(rank["p_low"]-p, p-rank["p_up"], 0.))


def summarize(rows):
    if not rows: raise ValueError("cannot summarize empty raw generation results")
    result = dict(requests=len(rows), histories=len({r["scene_id"] for r in rows}),
        p_mid_MAE=float(np.mean([r["p_mid_absolute_error"] for r in rows])),
        p_interval_MAE=float(np.mean([r["p_interval_error"] for r in rows])),
        Fine_at_0_05=float(np.mean([r["p_mid_absolute_error"] <= .05 for r in rows])),
        PET_target_MAE_seconds=float(np.mean([r["PET_target_absolute_error_seconds"] for r in rows])))
    for key in ("all_pair_overlap_scene", "road_outside_scene", "negative_vx_scene"):
        result[key+"_rate"] = float(np.mean([r["quality"][key] for r in rows]))
    quality_keys = ("ADE_m", "FDE_m", "ego_ADE_m", "acceleration_vector_rms_mps2", "acceleration_max_mps2",
                    "jerk_vector_rms_mps3", "negative_vx_frame_actor_fraction", "road_outside_frame_actor_fraction",
                    "all_pair_overlap_pair_frame_fraction")
    result["mean_quality"] = {key: float(np.mean([r["quality"][key] for r in rows])) for key in quality_keys}
    pairs = []
    for sid in sorted({r["scene_id"] for r in rows}):
        group = sorted((r for r in rows if r["scene_id"] == sid), key=lambda r: r["requested_p"])
        values = np.asarray([r["pet_seconds"] for r in group])
        pairs.append(dict(scene_id=sid, requested_p=[r["requested_p"] for r in group], PET_seconds=values.tolist(),
            achieved_p_mid=[r["estimated_rank"]["p_mid"] for r in group],
            PET_strictly_decreasing=bool(np.all(np.diff(values) < -1e-6)),
            p90_minus_p10=float(group[-1]["estimated_rank"]["p_mid"] - group[0]["estimated_rank"]["p_mid"])))
    result["paired_response"] = pairs
    result["strict_PET_order_fraction"] = float(np.mean([r["PET_strictly_decreasing"] for r in pairs]))
    result["mean_p90_minus_p10_response"] = float(np.mean([r["p90_minus_p10"] for r in pairs]))
    for name, group in [(s, [r for r in rows if r["stratum"] == s]) for s in ("N3_5", "N6_8", "N9_plus")]:
        if group:
            result.setdefault("by_N", {})[name] = dict(requests=len(group), actual_N=sorted({r["num_agents"] for r in group}),
                p_mid_MAE=float(np.mean([r["p_mid_absolute_error"] for r in group])),
                p_interval_MAE=float(np.mean([r["p_interval_error"] for r in group])))
    continuous = [r for r in rows if r["target_spec"]["exact_point_rank_identified"]]
    result["requested_nonatom_target_subset"] = dict(requests=len(continuous),
        p_mid_MAE=None if not continuous else float(np.mean([r["p_mid_absolute_error"] for r in continuous])))
    return result


def fixed_p05_diagnostic(rows):
    """Reuse GP(p=.5) outputs for every requested p; no extra generation/selection."""
    gp = [r for r in rows if r["method"] == "GP_direct_P"]
    midpoint = {r["scene_id"]: r for r in gp if r["requested_p"] == .5}
    frozen = []
    for row in gp:
        source, p = midpoint[row["scene_id"]], row["requested_p"]
        item = dict(source, method="GP_fixed_p05_inference_diagnostic", requested_p=p,
                    actual_conditioning_p=.5, target_spec=row["target_spec"])
        item.update(p_mid_absolute_error=abs(source["estimated_rank"]["p_mid"]-p),
                    p_interval_error=interval_error(p,source["estimated_rank"]),
                    PET_target_absolute_error_seconds=abs(source["pet_seconds"]-row["target_spec"]["target_pet_seconds"]))
        frozen.append(item)
    return dict(summary=summarize(frozen), reused_existing_p05_outputs=True,
                additional_model_forward_calls=0, same_weights_and_capacity_as_GP=True,
                purpose="inference_condition_usage_not_a_separately_trained_ablation")


def run(training_binding, role, stop_binding=None):
    if role not in ("STOP", "AUDIT"): raise PermissionError("only explicit STOP/AUDIT pilot roles")
    result = bound_json(training_binding)
    if result.get("status") != "complete" or result["policy"]["sha256"] != POLICY_SHA:
        raise ValueError("completed direct-P training and registered policy required")
    policy = bound_json(result["policy"])
    for name, digest in result["code_sha256"].items(): verify_binding(dict(path=name, sha256=digest))
    if role == "AUDIT":
        if stop_binding is None: raise PermissionError("finish fixed raw STOP evaluation before AUDIT access")
        stopped = bound_json(stop_binding)
        if (stopped.get("protocol") != PROTOCOL or stopped.get("status") != "complete" or stopped["role"] != "STOP"
                or stopped["training_result"] != training_binding or stopped["policy"] != result["policy"]
                or stopped["guidance_or_postcorrection_used"] is not False):
            raise ValueError("AUDIT run does not match the frozen raw STOP method")
        for name, digest in stopped["evaluation_code_sha256"].items(): verify_binding(dict(path=name, sha256=digest))
    elif stop_binding is not None:
        raise ValueError("STOP cannot consume a later evaluation result")
    data = bound_json(result["data"])
    if data["dataset_manifest"] != policy["data_manifest"] or result["data"] != policy["generator_data"]:
        raise ValueError("wrong natural coefficient data binding")
    torch.set_num_threads(policy["runtime"]["threads"]); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    device = torch.device(policy["runtime"]["device"])
    checkpoint = torch.load(verify_binding(result["checkpoint"]), map_location="cpu")
    if (checkpoint["policy"] != result["policy"] or checkpoint["data"] != result["data"]
            or checkpoint["architecture"] != result["architecture"] or checkpoint["prediction_type"] != "v"
            or checkpoint["code_sha256"] != result["code_sha256"]
            or checkpoint["labels_manifest"] != result["labels_manifest"]
            or checkpoint["p_is_actual_denoiser_condition"] is not True
            or checkpoint["primary_sampling_guidance"] is not False
            or checkpoint["primary_sampling_post_correction"] is not False
            or checkpoint["basis"] != data["basis"]):
        raise ValueError("raw direct-P checkpoint differs from selected training result")
    verify_binding(result["labels_manifest"])
    architecture = {key: checkpoint["architecture"][key] for key in
                    ("coefficient_dim", "hidden_dim", "heads", "layers", "feedforward_dim")}
    if (architecture["coefficient_dim"] != 2*data["basis"]["modes"]
            or any(architecture[k] != policy["generator"][k] for k in ("hidden_dim", "heads", "layers", "feedforward_dim"))):
        raise ValueError("wrong direct-P architecture or basis")
    with torch.random.fork_rng(devices=[]): model = JointPercentileDenoiser(**architecture)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.to(device).eval().requires_grad_(False)
    schedule = CosineDiffusionSchedule(policy["generator"]["diffusion_steps"]).to(device)
    if schedule.as_dict() != checkpoint["schedule"]: raise ValueError("sampling schedule differs from training")
    basis = TrajectoryBasis(int(data["basis"]["modes"]))
    cnorm, hnorm = bound_json(data["coefficient_normalizer"]), bound_json(data["history_normalizer"])
    mean, scale = np.asarray(cnorm["mean"]), np.asarray(cnorm["scale"])
    prior = NaturalDiffusionPrior.from_result_binding(policy["history_only_baseline"], device=str(device))
    plugin = FrozenRiskPlugin.from_refinement_binding(policy["risk_plugin_result"], device="cpu")
    output = verify_binding(result["checkpoint"]).parent / (role+"_raw_pilot")
    output.mkdir(parents=True, exist_ok=False)
    code = {name: sha256(ROOT/name) for name in CODE}
    write_json(output/"freeze_before_sampling.json", dict(protocol=PROTOCOL, training_result=training_binding,
        role=role, policy=result["policy"], evaluation_code_sha256=code, STOP_result=stop_binding,
        reference=plugin.describe(), methods=["G0_history_only", "GP_direct_P"], risk_condition_only_in_GP=True,
        guidance_or_postcorrection_used=False, sample_K=1, no_candidate_selection=True))
    cases = select_role_cases(policy["data_manifest"], role=role, per_stratum=policy["evaluation"]["per_N_stratum"],
                             salt=policy["evaluation"]["hash_salt"])
    declarations = [{k: c[k] for k in ("scene_id", "recording_id", "num_agents", "role", "stratum", "selection_hash")} for c in cases]
    write_json(output/"case_selection.json", dict(cases=declarations, selected_by_identity_and_N_only=True))
    rows, sensitivity, started = [], [], time.perf_counter()
    for number, case in enumerate(cases):
        features = model_features(case, hnorm, device)
        seed = int.from_bytes(hashlib.sha256(("natural_diffusion_K1_noise_v1|"+case["scene_id"]).encode()).digest()[:8], "little") % (2**32)
        noise = np.random.default_rng(seed).standard_normal((case["num_agents"], basis.modes, 2)).astype(np.float32)
        z = torch.from_numpy(noise[None]).to(device)
        base = prior.sample(case["history"], case["dimensions"], case["road_boundaries"], case["ego_mask"], initial_noise=noise)["future"]
        arrays = {k: case[k] for k in ("history", "dimensions", "road_boundaries", "ego_mask", "agent_ids")}
        arrays.update(initial_noise=noise, future_observed=case["future"], generated_G0=base)
        # GP sampling receives exactly context, z, and requested p. No PET,
        # risk reference or actual future is reachable through this API.
        generated = {}
        for p in policy["sampling"]["p_grid"]:
            condition = torch.tensor([p], dtype=torch.float32, device=device)
            coefficients = direct_p_sample(model, schedule, features, condition, z,
                steps=policy["sampling"]["steps"])
            physical = coefficients[0].detach().cpu().numpy().astype(np.float64)*scale+mean
            future = basis.decode(physical, case["history"][-1])
            if not np.array_equal(future[0], case["history"][-1]): raise ValueError("direct-P changed observed t0")
            key = "generated_GP_p"+format(p,"g").replace(".","_")
            arrays[key], generated[p] = future, future
        # Reference queries and targets below are evaluation-only and occur
        # AFTER every requested GP future for this context has been sampled.
        reference = plugin.condition(case["history"], case["dimensions"], case["road_boundaries"], case["ego_mask"], case["agent_mask"])
        observed_score, observed_quality = reference.score_future(case["future"]), quality_metrics(case["future"], case)
        for method in ("G0_history_only", "GP_direct_P"):
            for p in policy["sampling"]["p_grid"]:
                future = base if method == "G0_history_only" else generated[p]
                key = "generated_G0" if method == "G0_history_only" else "generated_GP_p"+format(p,"g").replace(".","_")
                scored, quality = reference.score_future(future), quality_metrics(future, case)
                rank, target = scored["estimated_rank"], reference.target_spec(p)
                row = dict(scene_id=case["scene_id"], recording_id=case["recording_id"], num_agents=case["num_agents"],
                    role=role, stratum=case["stratum"], method=method, requested_p=p, noise_seed=seed,
                    pet_seconds=scored["pet_seconds"], pet_raw_seconds=scored["pet_raw_seconds"], estimated_rank=rank,
                    target_spec=target, p_mid_absolute_error=abs(rank["p_mid"]-p), p_interval_error=interval_error(p,rank),
                    PET_target_absolute_error_seconds=abs(scored["pet_seconds"]-target["target_pet_seconds"]),
                    critical_container_index=scored["critical_container_index"], witness=scored["metric"]["witness"],
                    observed_PET=observed_score["pet_seconds"], observed_quality=observed_quality, quality=quality,
                    array_key=key, candidate_origin="raw_model_generated_not_natural_observation", K=1,
                    p_supplied_to_denoiser=(method=="GP_direct_P"), physical_target_supplied_to_sampler=False,
                    guidance_or_postcorrection_used=False)
                rows.append(row)
                print(json.dumps(clean_json(dict(stage="raw_direct_P_scored", role=role, case=number+1, N=case["num_agents"],
                    method=method, p=p, PET=row["pet_seconds"], point_error=row["p_mid_absolute_error"],
                    overlap=quality["all_pair_overlap_scene"], road=quality["road_outside_scene"], wall_seconds=time.perf_counter()-started))), flush=True)
        delta = generated[max(generated)]-generated[min(generated)]
        sensitivity.append(dict(scene_id=case["scene_id"], num_agents=case["num_agents"],
            p90_p10_position_RMSE_m=float(np.sqrt(np.mean(delta[...,:2]**2))),
            position_RMSE_convention="scalar_xy_coordinates; vector version reported separately",
            p90_p10_position_vector_RMSE_m=float(np.sqrt(np.mean(np.sum(delta[...,:2]**2,axis=-1)))),
            p90_p10_x_RMSE_m=float(np.sqrt(np.mean(delta[...,0]**2))),
            p90_p10_y_RMSE_m=float(np.sqrt(np.mean(delta[...,1]**2))),
            p90_p10_max_state_difference=float(np.max(np.abs(delta))),
            baseline_same_across_p=True, GP_lateral_path_not_artificially_frozen=True))
        path = output/("case_%02d.npz" % number)
        with path.open("xb") as handle: np.savez_compressed(handle, **arrays)
        binding = dict(path=str(path), sha256=sha256(path))
        current = [r for r in rows if r["scene_id"]==case["scene_id"]]
        for row in current: row["trajectory_artifact"] = binding
        write_json(output/("case_%02d.json" % number), dict(case=declarations[number], rows=current, p_sensitivity=sensitivity[-1]))
    summaries = {method: summarize([r for r in rows if r["method"]==method]) for method in ("G0_history_only", "GP_direct_P")}
    for name, digest in code.items(): verify_binding(dict(path=name,sha256=digest))
    finished = dict(protocol=PROTOCOL, status="complete", role=role, training_result=training_binding, policy=result["policy"],
        evaluation_code_sha256=code, rows=rows, summaries=summaries, p_sensitivity=sensitivity,
        fixed_p05_condition_diagnostic=fixed_p05_diagnostic(rows),
        guidance_or_postcorrection_used=False, control_error_primary="midrank_point_MAE", sample_K=1,
        true_conditional_percentiles_known=False, CAL_decoded_by_generator_evaluation=False,
        AUDIT_is_internal_development_not_final_blind=True, wall_seconds=time.perf_counter()-started)
    final_binding = write_json(output/"results.json", finished)
    print(json.dumps(dict(raw_direct_P_complete=True, role=role, result=final_binding, summaries=summaries)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-result", required=True); parser.add_argument("--training-sha256", required=True)
    parser.add_argument("--role", choices=("STOP","AUDIT"), required=True)
    parser.add_argument("--stop-result"); parser.add_argument("--stop-sha256")
    args = parser.parse_args()
    stop = None if args.stop_result is None else dict(path=args.stop_result,sha256=args.stop_sha256)
    run(dict(path=args.training_result,sha256=args.training_sha256), args.role, stop)
