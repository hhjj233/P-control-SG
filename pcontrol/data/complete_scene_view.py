"""Explicit E_full-conditioned natural scene view, not missing-data recovery.

E_full uses future availability for *dataset eligibility*, never a model input.
Only full fixed-roster observations enter this view. Partial-zero labels stay
in the original cohort. Prediction features expose H/size/road/ego role only.
"""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PROTOCOL = "natural_complete_ego_scene_reference_view_v1"
SOURCE_SHA256 = "86dbb5356841f4ca44e3e3da5a65612abd3594ae30721ebeea370cc220c79953"
ELIGIBILITY = "all_history_selected_agents_all_175_native_future_frames_observed"
ESTIMAND = "P(Y_scene_6.96s <= y | H, E_full=1)"
ROLE_PURPOSE = {"fit": "FIT", "stop": "STOP", "calibrate": "CAL", "audit": "AUDIT"}
COMPLETE_STATUSES = {"exact_zero", "complete_finite", "complete_capped", "complete_no_shared_occupancy"}


def resolve(path):
    path = Path(path)
    return (path if path.is_absolute() else ROOT / path).resolve()


def sha256(path):
    digest = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_binding(binding, expected_path=None):
    path = resolve(binding["path"])
    if expected_path is not None and path != resolve(expected_path):
        raise ValueError("bound artifact is at the wrong path")
    if sha256(path) != binding["sha256"]:
        raise ValueError("bound artifact hash changed")
    return path


def complete_eligibility(arrays):
    """Eligibility depends exclusively on all-actor native observation masks."""
    mask = arrays["future_observed_mask_agents"]
    offsets = arrays["offsets"]
    if (mask.dtype != np.bool_ or mask.ndim != 2 or mask.shape[1] != 175
            or offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer)
            or len(offsets) < 1 or offsets[0] != 0 or offsets[-1] != len(mask)
            or np.any(np.diff(offsets) < 3)):
        raise ValueError("full native175 masks and valid all-agent ragged offsets required")
    return np.array([mask[lo:hi].all() for lo, hi in zip(offsets[:-1], offsets[1:])], dtype=bool)


def validate_complete_labels(arrays, eligible):
    """Validate point targets after eligibility; never use PET to pick rows."""
    if not np.array_equal(eligible, arrays["complete_horizon"]):
        raise ValueError("stored completeness disagrees with native masks")
    if not arrays["point_identified"][eligible].all():
        raise ValueError("a complete scene lacks a point-identified target")
    values = arrays["pet_value"][eligible]
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 4)):
        raise ValueError("complete target is not finite capped PET")
    if not set(arrays["label_status"][eligible].tolist()) <= COMPLETE_STATUSES:
        raise ValueError("noncomplete label status entered the complete view")
    if not np.array_equal(arrays["label_interval_seconds"][eligible], np.column_stack((values, values))):
        raise ValueError("complete target interval is not a singleton")


def reference_example(arrays, row):
    """Separate allowed features, supervised target, and identity metadata.

    This projection never returns native futures, future masks, label status,
    actor IDs, recording IDs, or future-critical-actor diagnostics as features.
    The caller validates complete-view membership before this projection.
    """
    row = int(row)
    if row < 0 or row >= len(arrays["scene_id"]):
        raise IndexError(row)
    lo, hi = map(int, arrays["offsets"][row:row + 2])
    count = hi - lo
    ego = np.zeros(count, dtype=bool)
    ego[0] = True
    boundary_mask = arrays["carriageway_boundary_mask"][row]
    return {
        "features": {
            "history": arrays["history_agents"][lo:hi].transpose(1, 0, 2).copy(),
            "dimensions": arrays["dimensions_agents"][lo:hi].copy(),
            "road_boundaries": arrays["carriageway_boundaries"][row, boundary_mask].copy(),
            "ego_mask": ego,
        },
        "target": float(arrays["pet_value"][row]),
        "metadata": {"scene_id": str(arrays["scene_id"][row]), "source_row": row,
                     "recording_id": str(arrays["recording_id"][row]), "role": str(arrays["role"][row])},
    }


def collate_reference_examples(examples):
    """Pad for batching with explicit masks; padding never denotes real cars."""
    if not examples:
        raise ValueError("cannot collate an empty batch")
    sizes = [x["features"]["history"].shape[1] for x in examples]
    roads = [len(x["features"]["road_boundaries"]) for x in examples]
    batch, count, boundaries = len(examples), max(sizes), max(roads)
    features = dict(history=np.zeros((batch, 13, count, 4), np.float64),
                    dimensions=np.zeros((batch, count, 2), np.float64),
                    road_boundaries=np.zeros((batch, boundaries), np.float64),
                    road_boundary_mask=np.zeros((batch, boundaries), bool),
                    ego_mask=np.zeros((batch, count), bool),
                    agent_mask=np.zeros((batch, count), bool))
    for i, example in enumerate(examples):
        source = example["features"]
        n, r = sizes[i], roads[i]
        features["history"][i, :, :n] = source["history"]
        features["dimensions"][i, :n] = source["dimensions"]
        features["road_boundaries"][i, :r] = source["road_boundaries"]
        features["road_boundary_mask"][i, :r] = True
        features["ego_mask"][i, :n] = source["ego_mask"]
        features["agent_mask"][i, :n] = True
    return dict(features=features, target=np.array([x["target"] for x in examples], np.float64),
                metadata=[x["metadata"] for x in examples])


class CompleteSceneReferenceView:
    """Hash-bound lazy reader; each instance has exactly one recording role."""

    def __init__(self, manifest_path, manifest_sha256, *, purpose):
        if purpose not in ROLE_PURPOSE:
            raise ValueError("explicit fit/stop/calibrate/audit purpose required")
        path = verify_binding({"path": str(manifest_path), "sha256": manifest_sha256})
        self.manifest = json.loads(path.read_text())
        self.role = ROLE_PURPOSE[purpose]
        if (self.manifest.get("protocol") != PROTOCOL
                or self.manifest.get("estimand") != ESTIMAND
                or self.manifest.get("eligibility") != ELIGIBILITY
                or self.manifest.get("source_manifest", {}).get("sha256") != SOURCE_SHA256
                or self.manifest.get("partial_zero_included") is not False):
            raise ValueError("not the frozen complete-scene estimand")
        source_path = verify_binding(self.manifest["source_manifest"])
        self.source = json.loads(source_path.read_text())
        if set(self.manifest["recordings"]) != set(self.source["recordings"]):
            raise ValueError("view must preserve the full source recording roster")
        if set(self.manifest["by_role"]) != set(ROLE_PURPOSE.values()):
            raise ValueError("view must preserve all four source roles")
        for role in ROLE_PURPOSE.values():
            if self.manifest["by_role"][role]["complete_scenes"] != self.source["by_role"][role]["complete_horizon"]:
                raise ValueError("view role total differs from all source complete clips")
        for rec, entry in self.manifest["recordings"].items():
            original = self.source["recordings"][rec]
            if (entry["role"] != original["role"] or entry["data"] != original["artifacts"]["data"]
                    or len(entry["selected_rows"]) != original["counts"]["complete_horizon"]):
                raise ValueError("view changed a source role, binding, or complete denominator")
        self.rows = []
        self.paths = {}
        for rec, entry in self.manifest["recordings"].items():
            if entry["role"] != self.role:
                continue
            original = self.source["recordings"][rec]
            if original["role"] != self.role or entry["data"] != original["artifacts"]["data"]:
                raise ValueError("view changed source binding or recording role")
            if len(entry["selected_rows"]) != len(entry["selected_scene_ids"]):
                raise ValueError("view row/identity lengths differ")
            self.paths[rec] = verify_binding(entry["data"], source_path.parent / f"{rec}.npz")
            self.rows.extend((rec, row, sid) for row, sid in zip(entry["selected_rows"], entry["selected_scene_ids"]))
        if len(self.rows) != self.manifest["by_role"][self.role]["complete_scenes"]:
            raise ValueError("view role denominator changed")
        if len({sid for _, _, sid in self.rows}) != len(self.rows):
            raise ValueError("duplicate complete scene identities")

    def __len__(self):
        return len(self.rows)

    @lru_cache(maxsize=2)
    def _record(self, recording):
        # Do not load future trajectory arrays or pair-diagnostic ledger here.
        entry = self.manifest["recordings"][recording]
        verify_binding(entry["data"], self.paths[recording])
        keys = ("offsets", "future_observed_mask_agents", "complete_horizon", "point_identified",
                "pet_value", "label_status", "label_interval_seconds", "scene_id", "recording_id",
                "role", "history_agents", "dimensions_agents", "carriageway_boundaries",
                "carriageway_boundary_mask")
        with np.load(self.paths[recording], allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in keys}
        verify_binding(entry["data"], self.paths[recording])
        eligible = complete_eligibility(arrays)
        validate_complete_labels(arrays, eligible)
        if (np.flatnonzero(eligible).tolist() != entry["selected_rows"]
                or arrays["scene_id"][eligible].tolist() != entry["selected_scene_ids"]
                or not np.all(arrays["role"] == self.role)):
            raise ValueError("view membership/role does not equal all and only complete clips")
        del arrays["future_observed_mask_agents"]
        return arrays

    def __getitem__(self, index):
        rec, row, scene = self.rows[index]
        example = reference_example(self._record(rec), row)
        if example["metadata"]["scene_id"] != scene:
            raise ValueError("view identity differs from source row")
        return example
