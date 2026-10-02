#!/usr/bin/env python3
"""Create immutable index-only complete-clip view; no raw reads or training."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pcontrol.data.complete_scene_view import (
    PROTOCOL, SOURCE_SHA256, ELIGIBILITY, ESTIMAND, complete_eligibility,
    validate_complete_labels, resolve, sha256, verify_binding,
)


def build(config_path, config_sha256):
    config_path = verify_binding({"path": str(config_path), "sha256": config_sha256})
    config = json.loads(config_path.read_text())
    expected = dict(protocol="natural_complete_ego_scene_view_config_v1", eligibility=ELIGIBILITY,
                    estimand=ESTIMAND, window_seconds=[0., 6.96], cap_seconds=4., loss="CRPS",
                    partial_zero_included=False, missingness_weights=False, model_training=False,
                    calibration_training=False, generator_training=False)
    if any(config.get(k) != v for k, v in expected.items()) or set(config) != set(expected) | {"source_manifest", "output_root"}:
        raise ValueError("complete view protocol/window/loss or execution scope differs")
    if config["source_manifest"]["sha256"] != SOURCE_SHA256:
        raise ValueError("only frozen ego scene v1 source is allowed")
    source_path = verify_binding(config["source_manifest"])
    source = json.loads(source_path.read_text())
    output = resolve(config["output_root"])
    # A distinct output namespace, outside the source and all raw paths.
    if output.parent != source_path.parent.parent or output == source_path.parent:
        raise ValueError("view output must be a distinct sibling of source scene data")
    if output.exists():
        raise FileExistsError("never overwrite an existing complete-scene view")
    roles = {name: Counter() for name in ("FIT", "STOP", "CAL", "AUDIT")}
    records = {}
    by_n = {}
    all_ids = []
    for rec, record in sorted(source["recordings"].items()):
        data = record["artifacts"]["data"]
        data_path = verify_binding(data, source_path.parent / f"{rec}.npz")
        with np.load(data_path, allow_pickle=False) as archive:
            keys = ("future_observed_mask_agents", "offsets", "complete_horizon", "point_identified",
                    "pet_value", "label_status", "label_interval_seconds", "scene_id", "num_agents",
                    "recording_id", "role")
            arrays = {key: archive[key] for key in keys}
        eligible = complete_eligibility(arrays)
        validate_complete_labels(arrays, eligible)
        role = record["role"]
        if not np.all(arrays["recording_id"] == rec) or not np.all(arrays["role"] == role):
            raise ValueError("source array recording/role differs")
        rows = np.flatnonzero(eligible).tolist()
        ids = arrays["scene_id"][eligible].tolist()
        all_ids.extend(ids)
        counts = dict(source_scenes=len(eligible), complete_scenes=len(rows), excluded_incomplete_scenes=int((~eligible).sum()),
                      excluded_partial_zero=int((arrays["label_status"] == "certified_zero_with_missing").sum()))
        roles[role].update(counts)
        for n in np.unique(arrays["num_agents"]):
            mask = arrays["num_agents"] == n
            by_n.setdefault(str(n), Counter()).update(source_scenes=int(mask.sum()), complete_scenes=int((mask & eligible).sum()))
        records[rec] = dict(role=role, data=data, selected_rows=rows, selected_scene_ids=ids, counts=counts)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("duplicate scene in complete view")
    total = Counter()
    for counts in roles.values():
        total.update(counts)
    if total["complete_scenes"] != source["counts"]["complete_horizon"]:
        raise ValueError("complete scene denominator changed")
    code = {name: sha256(ROOT / name) for name in (
        "pcontrol/data/complete_scene_view.py", "pcontrol/research/build_natural_complete_scene_view.py")}
    report = dict(protocol=PROTOCOL, status="complete", source_manifest=config["source_manifest"],
                  config={"path": str(config_path.relative_to(ROOT)), "sha256": config_sha256}, code_sha256=code,
                  eligibility=ELIGIBILITY, estimand=ESTIMAND, loss="CRPS", window_seconds=[0., 6.96], cap_seconds=4.,
                  counts=dict(total), by_role={k: dict(v) for k, v in roles.items()},
                  by_N={k: dict(v) for k, v in by_n.items()}, recordings=records,
                  future_availability_used_for_eligibility=True, future_availability_is_model_input=False,
                  partial_zero_included=False, missingness_weights=False, target_cohort_is_original_all_scenes=False,
                  identifiable_complete_clip_target_not_MAR_recovery=True, no_scene_replacement=True,
                  raw_CSV_opened=False, model_training=False, calibration_training=False, generator_training=False,
                  prior_pair_results_reused=False, AUDIT_is_previously_inspected_development_recordings=True)
    output.mkdir()
    with (output / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(dict(path=str(output / "manifest.json"), sha256=sha256(output / "manifest.json"), counts=dict(total)), sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--config-sha256", required=True)
    args = parser.parse_args()
    build(args.config, args.config_sha256)
