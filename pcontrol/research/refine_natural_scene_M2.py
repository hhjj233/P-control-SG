#!/usr/bin/env python3
"""Same-seed, STOP-selected natural M2 continuation; never reads CAL/AUDIT."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pcontrol.research import train_natural_scene_reference as parent

POLICY_SHA = "5d6a85a07d442f37d5f786e865321588cc683c80f42c39cd7bc8455fc7ca2e97"
PROTOCOL = "natural_M2_same_seed_continuation_v1"


def read_inputs(policy_path, policy_sha):
    if policy_sha != POLICY_SHA:
        raise ValueError("only preregistered M2 refinement policy allowed")
    binding = dict(path=str(parent.resolve(policy_path)), sha256=policy_sha)
    policy = parent.load_json(binding)
    prepared, old_policy = parent.read_prepared(policy["parent_prepared"])
    result = parent.load_json(policy["parent_result"])
    checkpoint = torch.load(parent.verify_binding(result["checkpoint"]), map_location="cpu")
    # Parent artifacts may record their prepared path as an absolute path.
    if (checkpoint["variant"] != "M2" or checkpoint["data"] != prepared["data"]
            or checkpoint["normalizer"] != prepared["normalizer"]
            or checkpoint["code_sha256"] != prepared["code_sha256"]
            or checkpoint["best_epoch"] != result["best_epoch"]
            or parent.resolve(checkpoint["prepared"]["path"]) != parent.resolve(policy["parent_prepared"]["path"])
            or checkpoint["prepared"]["sha256"] != policy["parent_prepared"]["sha256"]):
        raise ValueError("parent M2 checkpoint identity differs")
    return policy, binding, prepared, old_policy, result, checkpoint


def stop_scores(model, pack, batch_size):
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(pack["target"]), batch_size):
            features, target = parent._batch(pack, slice(start, start + batch_size))
            chunks.append(parent.crps_from_params(model(features), target).numpy())
    scores = np.concatenate(chunks)
    count = pack["agent_mask"].sum(1)
    if not np.isfinite(scores).all() or not (count >= 9).any():
        raise ValueError("finite STOP scores and actual high-N support required")
    return dict(overall=float(scores.mean()), highN=float(scores[count >= 9].mean()),
                selection=float(.5 * scores.mean() + .5 * scores[count >= 9].mean()))


def selection_eligible(scores, baseline, config):
    return (scores["overall"] <= baseline["overall"] * config["overall_STOP_guard_ratio"]
            and scores["highN"] <= baseline["highN"] * config["highN_STOP_guard_ratio"])


def count_weights(counts, recipe):
    if recipe not in ("continue", "highN"):
        raise ValueError("only registered continuation recipes")
    raw = np.ones(len(counts)) if recipe == "continue" else 1. + (np.asarray(counts) >= 9)
    return raw / raw.mean()


def run(policy_path, policy_sha, recipe):
    policy, policy_binding, prepared, old_policy, old_result, old_checkpoint = read_inputs(policy_path, policy_sha)
    if recipe not in ("continue", "highN"):
        raise ValueError("unregistered recipe")
    parent.set_runtime(old_policy)
    cfg = policy["training"]
    code = dict(parent_training=prepared["code_sha256"], refinement=parent.sha256(__file__))
    output = parent.resolve(policy["output_root"]) / "training" / recipe
    output.mkdir(parents=True, exist_ok=False)
    fit = parent.load_pack(prepared["packs"]["FIT"], "FIT")
    stop = parent.load_pack(prepared["packs"]["STOP"], "STOP")
    model = parent.make_model("M2", old_policy)
    model.load_state_dict(old_checkpoint["state_dict"], strict=True)
    baseline = stop_scores(model, stop, cfg["batch_size"])
    if abs(baseline["overall"] - old_result["best_STOP_CRPS_seconds"]) > 1e-12:
        raise ValueError("parent STOP score does not replay")
    weights = count_weights(fit["agent_mask"].sum(1), recipe)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(cfg["seed"])
    best = dict(baseline)
    best_epoch, stale = 0, 0
    significant = best["selection"]
    best_state = copy.deepcopy(model.state_dict())
    started = time.perf_counter()
    with (output / "epochs.jsonl").open("x", encoding="utf-8") as log:
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            order = rng.permutation(len(weights))
            total = 0.
            for start in range(0, len(order), cfg["batch_size"]):
                rows = order[start:start + cfg["batch_size"]]
                features, target = parent._batch(fit, rows)
                optimizer.zero_grad(set_to_none=True)
                scores = parent.crps_from_params(model(features), target, normalized=True)
                loss = (scores * torch.from_numpy(weights[rows])).mean()
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("nonfinite CRPS refinement loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
                if not bool(torch.isfinite(norm)):
                    raise FloatingPointError("nonfinite refinement gradient")
                optimizer.step()
                total += float(loss.detach()) * len(rows) * 4.
            scores = stop_scores(model, stop, cfg["batch_size"])
            eligible = selection_eligible(scores, baseline, cfg)
            if eligible and scores["selection"] < best["selection"]:
                best, best_epoch, best_state = dict(scores), epoch, copy.deepcopy(model.state_dict())
            if eligible and scores["selection"] < significant - cfg["checkpoint_min_delta_seconds"]:
                significant, stale = scores["selection"], 0
            else:
                stale += 1
            row = dict(recipe=recipe, epoch=epoch, STOP=scores, eligible=eligible, best=best, best_epoch=best_epoch,
                       stale=stale, online_weighted_train_CRPS=total / len(weights), wall_seconds=time.perf_counter() - started)
            log.write(json.dumps(row, sort_keys=True) + "\n")
            log.flush()
            print(json.dumps(row, sort_keys=True), flush=True)
            if stale >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    replay = stop_scores(model, stop, cfg["batch_size"])
    if any(abs(replay[key] - best[key]) > 1e-12 for key in best):
        raise ValueError("selected refinement checkpoint fails replay")
    parent.verify_code(prepared["code_sha256"])
    if code["refinement"] != parent.sha256(__file__):
        raise ValueError("refinement source changed during run")
    checkpoint = dict(protocol=PROTOCOL, recipe=recipe, state_dict=best_state, policy=policy_binding,
                      data=prepared["data"], normalizer=prepared["normalizer"], code_sha256=code,
                      parent_checkpoint=old_result["checkpoint"], best_epoch=best_epoch, STOP=best, seed=cfg["seed"])
    with (output / "best.pt").open("xb") as handle:
        torch.save(checkpoint, handle)
    result = dict(protocol=PROTOCOL, status="complete", recipe=recipe, policy=policy_binding,
                  code_sha256=code, parent_checkpoint=old_result["checkpoint"], baseline_STOP=baseline, best_STOP=best,
                  best_epoch=best_epoch, epochs_completed=epoch, seed=cfg["seed"], multi_seed_replication=False,
                  checkpoint=dict(path=str(output / "best.pt"), sha256=parent.sha256(output / "best.pt")),
                  epochs=dict(path=str(output / "epochs.jsonl"), sha256=parent.sha256(output / "epochs.jsonl")),
                  weight_range=[float(weights.min()), float(weights.max())], mean_weight=float(weights.mean()),
                  wall_seconds=time.perf_counter() - started, CAL_decoded=False, AUDIT_decoded=False,
                  generator_training=False, weights_depend_only_on_H_vehicle_count=True)
    binding = parent.write_json(output / "result.json", result)
    print(json.dumps(dict(training_complete=recipe, binding=binding, best_STOP=best)), flush=True)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy", required=True)
    p.add_argument("--policy-sha256", required=True)
    p.add_argument("--recipe", required=True)
    args = p.parse_args()
    run(args.policy, args.policy_sha256, args.recipe)
