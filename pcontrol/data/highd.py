"""Train-only, strict-N9 extraction from actual highD CSV observations.

This module deliberately has no dependency on the legacy scenario builders,
simulation caches, normalizers, or checkpoints.  It does not choose events,
compute risk, fill missing observations, or alter an observed trajectory.

highD's CSV names ``width`` and ``height`` mean longitudinal vehicle length
and lateral vehicle width.  Raw ``x,y`` are the upper-left box corner in a
coordinate system whose y axis points down.  Our state stores box centres,
with x forward and y left, anchored at the ego centre at t0.  Consequently the
transform is diag(sign(ego_vx), -sign(ego_vx)), NOT a raw x/y sign flip.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


PROTOCOL = "natural_highd_strict_n9_observed_centres_v1"
SLOT_NAMES = (
    "ego", "front", "front_left", "left", "back_left",
    "back", "back_right", "right", "front_right",
)
SLOT_SOURCE_COLUMNS = (
    None, "precedingId", "leftPrecedingId", "leftAlongsideId",
    "leftFollowingId", "followingId", "rightFollowingId",
    "rightAlongsideId", "rightPrecedingId",
)
INTEGER_COLUMNS = ("frame", "id", "laneId") + SLOT_SOURCE_COLUMNS[1:]
FLOAT_COLUMNS = ("x", "y", "width", "height", "xVelocity", "yVelocity")
TRACK_COLUMNS = INTEGER_COLUMNS + FLOAT_COLUMNS
META_COLUMNS = (
    "id", "width", "height", "initialFrame", "finalFrame", "class",
    "drivingDirection",
)


class HighDContractError(ValueError):
    """An input contract is invalid; not an ordinary rejected candidate."""


def recording_id(value: Any) -> str:
    token = str(value)
    if not re.fullmatch(r"\d{1,2}", token) or not 1 <= int(token) <= 60:
        raise HighDContractError("recording must be a highD integer ID in 01..60")
    return f"{int(token):02d}"


def load_split_assignment(path: Path) -> Dict[str, Tuple[str, ...]]:
    """Read the existing roster only; ignore its legacy cache path/counts."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    raw = payload.get("splits")
    if not isinstance(raw, dict) or set(raw) != {"train", "val", "test"}:
        raise HighDContractError("split file requires train/val/test recording lists")
    result: Dict[str, Tuple[str, ...]] = {}
    seen = set()
    for split in ("train", "val", "test"):
        if not isinstance(raw[split], list) or not raw[split]:
            raise HighDContractError(f"empty or invalid {split} recording list")
        ids = tuple(recording_id(value) for value in raw[split])
        if len(set(ids)) != len(ids) or seen.intersection(ids):
            raise HighDContractError("duplicate or overlapping recording assignments")
        seen.update(ids)
        result[split] = ids
    return result


def inventory_files(data_root: Path, split_path: Path) -> Dict[str, Any]:
    """Filename/stat inventory only, including locked splits; opens no CSV."""
    splits = load_split_assignment(split_path)
    root = Path(data_root).resolve()
    return {
        "data_root": str(root), "trajectory_contents_opened": False,
        "splits": {
            split: {
                rec: {
                    suffix: (root / f"{rec}_{suffix}.csv").is_file()
                    for suffix in ("tracks", "tracksMeta", "recordingMeta")
                }
                for rec in ids
            }
            for split, ids in splits.items()
        },
    }


def _train_id(value: Any, split_path: Path) -> str:
    rec = recording_id(value)
    if rec not in load_split_assignment(split_path)["train"]:
        # This guard runs before any CSV existence check or file open.
        raise PermissionError(f"recording {rec} is not in the authorized train roster")
    return rec


def _lane_markings(value: Any) -> np.ndarray:
    try:
        points = np.asarray([float(v) for v in str(value).split(";")], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise HighDContractError("invalid raw lane markings") from exc
    if len(points) < 2 or not np.isfinite(points).all() or not np.all(np.diff(points) > 0):
        raise HighDContractError("lane markings must be finite, strictly increasing boundaries")
    return points


def read_train_recording_metadata(
    data_root: Path, recording: Any, *, split_path: Path,
) -> Dict[str, Any]:
    """Metadata-only train read; does not decode trajectory or tracksMeta CSVs."""
    rec = _train_id(recording, split_path)
    path = Path(data_root).resolve() / f"{rec}_recordingMeta.csv"
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise HighDContractError("recording metadata must contain exactly one row")
    row = dict(rows[0])
    if recording_id(row["id"]) != rec:
        raise HighDContractError("recording metadata ID disagrees with filename")
    row["frameRate"] = float(row["frameRate"])
    if row["frameRate"] != 25.0:
        raise HighDContractError("this protocol requires raw frameRate=25 Hz")
    _lane_markings(row["upperLaneMarkings"])
    _lane_markings(row["lowerLaneMarkings"])
    return row


def _typed_tracks(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(TRACK_COLUMNS).difference(frame.columns)
    if missing:
        raise HighDContractError(f"missing raw track columns: {sorted(missing)}")
    result = frame.loc[:, list(TRACK_COLUMNS)].copy()
    for column in INTEGER_COLUMNS:
        values = pd.to_numeric(result[column], errors="raise").to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.equal(values, np.floor(values)).all():
            raise HighDContractError(f"raw {column} must be finite integer values")
        if np.any(np.abs(values) > 2 ** 53):
            raise HighDContractError(f"raw {column} exceeds exactly representable ID range")
        result[column] = values.astype(np.int64)
    for column in FLOAT_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="raise").astype(np.float64)
    if (result["frame"] <= 0).any() or (result["id"] <= 0).any():
        raise HighDContractError("raw frame and vehicle IDs must be positive")
    if (result.loc[:, list(SLOT_SOURCE_COLUMNS[1:])] < 0).any().any():
        raise HighDContractError("raw missing-neighbor IDs must use zero, not negative IDs")
    if result.duplicated(["id", "frame"]).any():
        raise HighDContractError("duplicate (vehicle ID, frame) observations")
    return result


@dataclass(frozen=True)
class TrackMetadata:
    initial_frame: int
    final_frame: int
    length_m: float
    width_m: float
    vehicle_class: str
    driving_direction: int


@dataclass
class RawRecording:
    """Validated raw observations; create production instances with the guarded loader."""

    recording_id: str
    tracks: pd.DataFrame
    track_metadata: Mapping[int, TrackMetadata]
    frame_rate: float
    upper_lane_markings_raw_m: np.ndarray
    lower_lane_markings_raw_m: np.ndarray
    source_paths: Mapping[str, str]
    provenance: str

    def __post_init__(self) -> None:
        self.recording_id = recording_id(self.recording_id)
        if self.frame_rate != 25.0:
            raise HighDContractError("this protocol requires raw frameRate=25 Hz")
        self.tracks = _typed_tracks(self.tracks)
        for boundaries in (self.upper_lane_markings_raw_m, self.lower_lane_markings_raw_m):
            array = np.asarray(boundaries, dtype=np.float64)
            if array.ndim != 1 or len(array) < 2 or not np.isfinite(array).all() or not np.all(np.diff(array) > 0):
                raise HighDContractError("invalid recording lane boundary arrays")
        self._by_id_frame = self.tracks.set_index(["id", "frame"]).sort_index()

    def rows(self, actor_id: int, frames: Sequence[int]) -> Optional[pd.DataFrame]:
        index = pd.MultiIndex.from_product([[int(actor_id)], np.asarray(frames, dtype=np.int64)])
        positions = self._by_id_frame.index.get_indexer(index)
        if np.any(positions < 0):
            return None
        return self._by_id_frame.iloc[positions]


def load_train_recording(
    data_root: Path, recording: Any, *, split_path: Path,
) -> RawRecording:
    """Read only an explicitly train-authorized raw recording; no cache fallback."""
    rec = _train_id(recording, split_path)
    root = Path(data_root).resolve()
    paths = {key: root / f"{rec}_{key}.csv" for key in ("tracks", "tracksMeta", "recordingMeta")}
    record = read_train_recording_metadata(root, rec, split_path=split_path)
    tracks = pd.read_csv(paths["tracks"], usecols=list(TRACK_COLUMNS))
    meta_table = pd.read_csv(paths["tracksMeta"], usecols=list(META_COLUMNS))
    if meta_table["id"].duplicated().any():
        raise HighDContractError("duplicate vehicle IDs in tracksMeta")
    metadata: Dict[int, TrackMetadata] = {}
    for row in meta_table.to_dict("records"):
        integers = np.asarray([row[k] for k in ("id", "initialFrame", "finalFrame", "drivingDirection")], dtype=np.float64)
        if not np.isfinite(integers).all() or not np.equal(integers, np.floor(integers)).all():
            raise HighDContractError("non-integer tracksMeta identifiers/frame/direction")
        actor_id, first, last, direction = map(int, integers)
        length, width = float(row["width"]), float(row["height"])
        if actor_id <= 0 or first <= 0 or last < first or direction not in (1, 2):
            raise HighDContractError("invalid tracksMeta frame extent or direction")
        if not np.isfinite([length, width]).all() or min(length, width) <= 0:
            raise HighDContractError("invalid tracksMeta dimensions")
        metadata[actor_id] = TrackMetadata(first, last, length, width, str(row["class"]), direction)
    return RawRecording(
        rec, tracks, metadata, float(record["frameRate"]),
        _lane_markings(record["upperLaneMarkings"]), _lane_markings(record["lowerLaneMarkings"]),
        {key: str(path) for key, path in paths.items()}, "raw_highd_train_csv",
    )


@dataclass(frozen=True)
class ExtractionPolicy:
    history_steps: int = 13
    future_steps_including_t0: int = 88
    stride_frames: int = 2
    max_neighbor_distance_m: Optional[float] = None
    only_cars: bool = False
    # Numerical tolerance only.  Any wider annotation tolerance changes E and
    # must be recorded as an explicit, separately frozen protocol choice.
    geometry_tolerance_m: float = 1e-6

    def __post_init__(self) -> None:
        if (self.history_steps, self.future_steps_including_t0, self.stride_frames) != (13, 88, 2):
            raise HighDContractError("this protocol fixes H13/F88/stride2")
        if not np.isfinite(self.geometry_tolerance_m) or self.geometry_tolerance_m < 0:
            raise HighDContractError("invalid geometry tolerance")
        if self.max_neighbor_distance_m is not None and (
            not np.isfinite(self.max_neighbor_distance_m) or self.max_neighbor_distance_m <= 0
        ):
            raise HighDContractError("max neighbor distance must be positive or None")


@dataclass(frozen=True)
class NaturalN9Sample:
    recording_id: str
    t0_frame: int
    source_agent_ids: np.ndarray
    history_frame_ids: np.ndarray
    future_frame_ids: np.ndarray
    history: np.ndarray
    future: np.ndarray
    history_lane_ids: np.ndarray
    future_lane_ids: np.ndarray
    sizes_length_width_m: np.ndarray
    vehicle_classes: Tuple[str, ...]
    road_geometry: Mapping[str, Any]
    metadata: Mapping[str, Any]


@dataclass(frozen=True)
class ExtractionResult:
    sample: Optional[NaturalN9Sample]
    failure_code: Optional[str]
    details: Mapping[str, Any]

    @property
    def ok(self) -> bool:
        return self.sample is not None


def _readonly(array: Any, dtype: Any = None) -> np.ndarray:
    out = np.array(array, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _reject(code: str, **details: Any) -> ExtractionResult:
    return ExtractionResult(None, code, details)


def extract_one(
    recording: RawRecording, t0_frame: int, ego_id: int, *,
    policy: ExtractionPolicy = ExtractionPolicy(),
) -> ExtractionResult:
    """Extract one strict9 observed window, or return an explicit rejection.

    Actor selection and geometric filters use t0 only.  Future reads are only
    exact-row coverage/data-integrity checks and extraction, not risk-based
    selection.  IDs are never refreshed after t0.  No semantic event filter
    is imposed here: downstream event selection must preserve its own ledger.
    """
    if isinstance(t0_frame, bool) or int(t0_frame) != t0_frame or t0_frame <= 0:
        raise HighDContractError("t0_frame must be a positive integer")
    if isinstance(ego_id, bool) or int(ego_id) != ego_id or ego_id <= 0:
        raise HighDContractError("ego_id must be a positive integer")
    t0_frame, ego_id = int(t0_frame), int(ego_id)
    ego_rows = recording.rows(ego_id, [t0_frame])
    if ego_rows is None:
        return _reject("ego_missing_t0", actor_id=ego_id)
    ego = ego_rows.iloc[0]
    if not np.isfinite(ego.loc[list(FLOAT_COLUMNS)].to_numpy(dtype=np.float64)).all():
        return _reject("nonfinite_t0", slot=0, actor_id=ego_id)
    ego_meta = recording.track_metadata.get(ego_id)
    if ego_meta is None:
        return _reject("missing_track_metadata", slot=0, actor_id=ego_id)
    if ego_meta.driving_direction not in (1, 2):
        return _reject("ego_direction_undefined")
    # Static road direction handles stopped egos.  Actual vx corroborates it
    # whenever a longitudinal motion sign is defined.
    sign = 1 if ego_meta.driving_direction == 2 else -1
    if float(ego["xVelocity"]) * sign < -1e-6:
        return _reject("ego_direction_velocity_disagreement")
    ids = [ego_id] + [int(ego[column]) for column in SLOT_SOURCE_COLUMNS[1:]]
    if any(value <= 0 for value in ids):
        return _reject("missing_directional_slot", slots=[i for i, value in enumerate(ids) if value <= 0])
    if len(set(ids)) != 9:
        return _reject("duplicate_slot_actor", source_agent_ids=ids)
    history_frames = t0_frame - np.arange(12, -1, -1, dtype=np.int64) * 2
    future_frames = t0_frame + np.arange(88, dtype=np.int64) * 2
    requested_frames = np.concatenate([history_frames[:-1], future_frames])
    tol = policy.geometry_tolerance_m
    ego_center = np.array([ego.x + ego.width / 2, ego.y + ego.height / 2], dtype=np.float64)
    rotation = np.array([sign, -sign], dtype=np.float64)
    ego_lane = int(ego["laneId"])
    sizes, classes, values, lanes, geometry_diagnostics = [], [], [], [], []
    expected_direction = 2 if sign > 0 else 1
    for slot, actor_id in enumerate(ids):
        meta = recording.track_metadata.get(actor_id)
        if meta is None:
            return _reject("missing_track_metadata", slot=slot, actor_id=actor_id)
        if meta.driving_direction != expected_direction:
            return _reject("opposite_driving_direction", slot=slot, actor_id=actor_id)
        if policy.only_cars and meta.vehicle_class.lower() != "car":
            return _reject("vehicle_class_excluded", slot=slot, actor_id=actor_id)
        anchor_rows = recording.rows(actor_id, [t0_frame])
        if anchor_rows is None:
            return _reject("actor_missing_t0", slot=slot, actor_id=actor_id)
        anchor = anchor_rows.iloc[0]
        av = anchor.loc[list(FLOAT_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(av).all():
            return _reject("nonfinite_t0", slot=slot, actor_id=actor_id)
        if float(anchor["xVelocity"]) * sign < -1e-6:
            return _reject("opposite_t0_velocity", slot=slot, actor_id=actor_id)
        center = np.array([anchor.x + anchor.width / 2, anchor.y + anchor.height / 2])
        dx, dy = (center - ego_center) * rotation
        if slot and policy.max_neighbor_distance_m is not None and np.hypot(dx, dy) > policy.max_neighbor_distance_m:
            return _reject("neighbor_distance_exceeded", slot=slot, actor_id=actor_id)
        lane = int(anchor["laneId"])
        if slot in (1, 5) and lane != ego_lane:
            return _reject("slot_lane_mismatch", slot=slot, actor_id=actor_id)
        if slot in (2, 3, 4) and (lane != ego_lane - sign or dy <= tol):
            return _reject("slot_left_geometry_mismatch", slot=slot, actor_id=actor_id)
        if slot in (6, 7, 8) and (lane != ego_lane + sign or dy >= -tol):
            return _reject("slot_right_geometry_mismatch", slot=slot, actor_id=actor_id)
        half_sum_length = 0.5 * (float(anchor.width) + float(ego.width))
        if slot in (1, 2, 8) and dx <= tol:
            return _reject("slot_front_geometry_mismatch", slot=slot, actor_id=actor_id)
        if slot in (4, 5, 6) and dx >= -tol:
            return _reject("slot_back_geometry_mismatch", slot=slot, actor_id=actor_id)
        if slot in (3, 7) and abs(dx) > half_sum_length + tol:
            # Official alongside IDs are authoritative.  CSV annotations can
            # differ slightly from rounded box-edge overlap; do not invent an
            # additional population-selection criterion or reassign the slot.
            geometry_diagnostics.append({
                "code": "alongside_longitudinal_boxes_do_not_overlap",
                "slot": slot, "actor_id": actor_id,
                "longitudinal_box_gap_m": float(abs(dx) - half_sum_length),
            })
        if meta.initial_frame > int(history_frames[0]) or meta.final_frame < int(future_frames[-1]):
            return _reject("insufficient_track_extent", slot=slot, actor_id=actor_id)
        rows = recording.rows(actor_id, requested_frames)
        if rows is None:
            return _reject("missing_observed_frame", slot=slot, actor_id=actor_id)
        raw = rows.loc[:, list(FLOAT_COLUMNS)].to_numpy(dtype=np.float64)
        if not np.isfinite(raw).all():
            return _reject("nonfinite_observed_state", slot=slot, actor_id=actor_id)
        if not np.all(raw[:, 2:4] > 0):
            return _reject("invalid_observed_dimensions", slot=slot, actor_id=actor_id)
        if not np.allclose(raw[:, 2:4], [meta.length_m, meta.width_m], rtol=0, atol=1e-9):
            return _reject("dimensions_disagree_with_metadata", slot=slot, actor_id=actor_id)
        centres = raw[:, :2] + raw[:, 2:4] / 2
        state = np.concatenate([(centres - ego_center) * rotation, raw[:, 4:6] * rotation], axis=-1)
        values.append(state)
        lanes.append(rows["laneId"].to_numpy(dtype=np.int64))
        sizes.append([meta.length_m, meta.width_m])
        classes.append(meta.vehicle_class)
    states = np.stack(values, axis=1)
    lane_array = np.stack(lanes, axis=1)
    upper = np.asarray(recording.upper_lane_markings_raw_m, dtype=np.float64)
    lower = np.asarray(recording.lower_lane_markings_raw_m, dtype=np.float64)
    road = {
        "upper_lane_markings_raw_m": _readonly(upper),
        "lower_lane_markings_raw_m": _readonly(lower),
        "upper_lane_markings_canonical_y_m": _readonly(np.sort(-sign * (upper - ego_center[1]))),
        "lower_lane_markings_canonical_y_m": _readonly(np.sort(-sign * (lower - ego_center[1]))),
        "ego_t0_raw_lane_id": ego_lane,
        "ego_carriageway": "lower" if sign > 0 else "upper",
        "lane_markings_snapped": False,
    }
    sample = NaturalN9Sample(
        recording.recording_id, t0_frame, _readonly(ids, np.int64),
        _readonly(history_frames), _readonly(future_frames),
        _readonly(states[:13]), _readonly(states[12:]),
        _readonly(lane_array[:13]), _readonly(lane_array[12:]),
        _readonly(sizes, np.float64), tuple(classes), road,
        {
            "protocol": PROTOCOL, "source": recording.provenance,
            "source_paths": dict(recording.source_paths), "dt_s": 2 / recording.frame_rate,
            "raw_frame_rate_hz": recording.frame_rate, "raw_stride_frames": 2,
            "history_duration_s": 24 / recording.frame_rate,
            "future_duration_s": 174 / recording.frame_rate,
            "raw_t0_ego_center_xy_m": ego_center.tolist(), "forward_sign": sign,
            "state_position_convention": "box_center_m_ego_t0_origin_x_forward_y_left",
            "size_columns": ["longitudinal_length_m", "lateral_width_m"],
            "slot_names": list(SLOT_NAMES), "slot_selection_view": "raw_t0_only",
            "geometry_tolerance_m": tol, "max_neighbor_distance_m": policy.max_neighbor_distance_m,
            "only_cars": policy.only_cars, "same_ids_for_all_frames": True,
            "geometry_diagnostics": geometry_diagnostics,
            "driving_direction_source": "tracksMeta_corroborated_by_moving_t0_velocity",
            "interpolation": False, "extrapolation": False, "smoothing": False,
            "terminal_projection": False, "future_risk_used_for_selection": False,
        },
    )
    return ExtractionResult(sample, None, {})
