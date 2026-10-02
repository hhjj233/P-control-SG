"""Observed variable-N history clips with ego--focal future PET supervision.

The raw recording loader decodes an authorized train CSV.  Sample membership
uses only t0 and actual history; background future observations are never
indexed for context membership, exported, or supplied to the estimator.
Only the observed ego/focal future is extracted and assessed.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pcontrol.data.road_semantics import infer_lane_ids_from_boundaries, semantic_validity
from pcontrol.data.highd import RawRecording, load_split_assignment, load_train_recording, recording_id
from pcontrol.data.pair_pet import PROTOCOL as PET_PROTOCOL, observed_pair_cut_in_pet


PROTOCOL = "natural_highd_variable_context_event_clips_v2"
EXTRACTION_PROTOCOL = "natural_highd_observed_history_context_pair_future_v2"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_COLUMNS = ["x", "y", "width", "height", "xVelocity", "yVelocity"]
POPULATION = (
    "original-train highD natural pair events; t0 positive frame%25==0; "
    "context is every same-carriageway car/truck within longitudinal 120m at t0 "
    "with actual H13 stride2, minimum3 actors and no maximum/truncation; "
    "all t0 ahead-adjacent context focal candidates; observed ego/focal H13/F88; "
    "pair-only native cut-in semantics; one closest-entry-minus3s window per "
    "recording/ego/focal/raw-geometric-entry event, earlier t0 breaks ties; "
    "reference eligibility requires complete observed pair conflict occupancy"
)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _readonly(value, dtype=None):
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _raw_boundaries(recording, direction):
    return np.asarray(recording.lower_lane_markings_raw_m if direction == 2 else recording.upper_lane_markings_raw_m, dtype=np.float64)


def _raw_lane(center_y, boundaries):
    values = np.asarray(center_y)
    lane = np.searchsorted(boundaries, values, side="right") - 1
    return np.where((values >= boundaries[0]) & (values < boundaries[-1]), lane, -1)


def _centres_and_states(raw, anchor, sign):
    return np.concatenate([
        (raw[..., :2] + .5 * raw[..., 2:4] - anchor) * [sign, -sign],
        raw[..., 4:6] * [sign, -sign],
    ], axis=-1)


def _valid_raw_rows(rows, meta):
    if rows is None:
        return False
    raw = rows.loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64)
    return bool(np.isfinite(raw).all() and np.all(raw[:, 2:4] > 0)
                and np.allclose(raw[:, 2:4], [meta.length_m, meta.width_m], rtol=0, atol=1e-9))


@dataclass(frozen=True)
class ObservedContext:
    recording_id: str
    t0_frame: int
    ego_id: int
    source_agent_ids: np.ndarray
    history: np.ndarray  # [13,N,4], ego first, remaining raw IDs ascending
    dimensions: np.ndarray
    history_lane_ids: np.ndarray
    history_frame_ids: np.ndarray
    canonical_boundaries: np.ndarray
    raw_anchor_center: np.ndarray
    forward_sign: int
    t0_geometric_lanes: np.ndarray
    source_vehicle_classes: Tuple[str, ...]

    @property
    def num_agents(self):
        return len(self.source_agent_ids)

    def focal_candidates(self):
        ego_lane = self.t0_geometric_lanes[0]
        return tuple(int(actor) for i, actor in enumerate(self.source_agent_ids)
                     if i and self.history[-1, i, 0] > 0
                     and abs(int(self.t0_geometric_lanes[i]) - int(ego_lane)) == 1)


def extract_context(recording, t0_frame, ego_id, *, frame_rows=None, audit_counts=None):
    """Return all eligible t0/history actors; never examine their future rows."""
    t0, ego_id = int(t0_frame), int(ego_id)
    diagnostics = audit_counts if audit_counts is not None else Counter()
    if t0 <= 0 or t0 % 25:
        raise ValueError("context t0 must be a positive raw frame divisible by25")
    if frame_rows is None:
        frame_rows = recording.tracks.loc[recording.tracks.frame == t0].set_index("id")
    if ego_id not in frame_rows.index:
        return None, "ego_missing_t0"
    meta = recording.track_metadata.get(ego_id)
    if meta is None or meta.driving_direction not in (1, 2) or meta.vehicle_class.lower() not in ("car", "truck"):
        return None, "ego_static_metadata_invalid"
    sign = 1 if meta.driving_direction == 2 else -1
    ego = frame_rows.loc[ego_id]
    ego_raw = ego.loc[RAW_COLUMNS].to_numpy(dtype=np.float64)
    if not np.isfinite(ego_raw).all():
        return None, "ego_nonfinite_t0"
    anchor = ego_raw[:2] + .5 * ego_raw[2:4]
    boundaries_raw = _raw_boundaries(recording, meta.driving_direction)
    boundaries = np.sort(-sign * (boundaries_raw - anchor[1]))
    if int(_raw_lane(0., boundaries)) < 0:
        return None, "ego_t0_outside_carriageway"
    frames = np.arange(t0-24, t0+1, 2, dtype=np.int64)
    actor_rows, actor_metadata = [], []
    diagnostics["t0_actor_rows_considered_across_materialized_contexts"] += len(frame_rows)
    for actor, row in frame_rows.iterrows():
        actor = int(actor)
        actor_meta = recording.track_metadata.get(actor)
        if actor_meta is None or actor_meta.driving_direction != meta.driving_direction or actor_meta.vehicle_class.lower() not in ("car", "truck"):
            diagnostics["static_metadata_direction_or_class_excluded"] += 1
            continue
        values = row.loc[RAW_COLUMNS].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            diagnostics["nonfinite_t0_actor"] += 1
            continue
        centre = values[:2] + .5 * values[2:4]
        if abs(float(centre[0] - anchor[0])) > 120.:
            diagnostics["outside_t0_longitudinal_roi"] += 1
            continue
        if int(_raw_lane(-sign*(centre[1]-anchor[1]), boundaries)) < 0:
            diagnostics["outside_t0_carriageway"] += 1
            continue
        if actor_meta.initial_frame > frames[0] or actor_meta.final_frame < t0:
            diagnostics["history_metadata_extent_incomplete"] += 1
            continue
        rows = recording.rows(actor, frames)
        if not _valid_raw_rows(rows, actor_meta):
            diagnostics["history_observations_missing_or_invalid"] += 1
            continue
        raw_history = rows.loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64)
        centers_y = -sign*(raw_history[:, 1]+.5*raw_history[:, 3]-anchor[1])
        if not np.all(_raw_lane(centers_y, boundaries) >= 0):
            diagnostics["history_centers_outside_actual_carriageway"] += 1
            continue
        actor_rows.append((actor, rows))
        actor_metadata.append((actor, actor_meta))
    row_map, metadata_map = dict(actor_rows), dict(actor_metadata)
    if ego_id not in row_map:
        return None, "ego_observed_history_incomplete"
    ids = [ego_id] + sorted(actor for actor in row_map if actor != ego_id)
    if len(ids) < 3:
        return None, "fewer_than_three_actual_context_actors"
    raw = np.stack([row_map[actor].loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64) for actor in ids], axis=1)
    states = _centres_and_states(raw, anchor, sign)
    t0_lanes = _raw_lane(states[-1, :, 1], boundaries)
    return ObservedContext(
        recording.recording_id, t0, ego_id, _readonly(ids, np.int64), _readonly(states),
        _readonly([[metadata_map[actor].length_m, metadata_map[actor].width_m] for actor in ids]),
        _readonly(np.stack([row_map[actor]["laneId"].to_numpy(dtype=np.int64) for actor in ids], axis=1)),
        _readonly(frames), _readonly(boundaries), _readonly(anchor), sign, _readonly(t0_lanes, np.int64),
        tuple(metadata_map[actor].vehicle_class for actor in ids),
    ), None


def extract_pair_future(recording, context, focal_id):
    """Only ego and requested, already-in-context focal future rows are indexed."""
    focal_id = int(focal_id)
    if focal_id not in context.focal_candidates():
        return None, None, "focal_not_t0_ahead_adjacent_context_actor"
    frames = np.arange(context.t0_frame, context.t0_frame+175, 2, dtype=np.int64)
    ids = [context.ego_id, focal_id]
    arrays = []
    for actor in ids:
        meta = recording.track_metadata[actor]
        if meta.final_frame < frames[-1]:
            return None, None, "pair_future_metadata_extent_incomplete"
        rows = recording.rows(actor, frames)
        if not _valid_raw_rows(rows, meta):
            return None, None, "pair_future_observations_incomplete_or_invalid"
        arrays.append(rows.loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64))
    raw = np.stack(arrays, axis=1)
    future = _centres_and_states(raw, context.raw_anchor_center, context.forward_sign)
    sizes = np.asarray([[recording.track_metadata[actor].length_m, recording.track_metadata[actor].width_m] for actor in ids])
    positions = [0, int(np.flatnonzero(context.source_agent_ids == focal_id)[0])]
    if not np.array_equal(context.history[-1, positions], future[0]):
        raise RuntimeError("observed pair history t0 differs from observed future t0")
    return _readonly(future), _readonly(sizes), None


def _pair_semantic(future, sizes, boundaries, mode):
    origin = future.copy()
    origin[..., :2] -= .5 * sizes[None]
    lanes = infer_lane_ids_from_boundaries(origin, boundaries, widths=sizes[:, 1])
    semantic = semantic_validity(origin, lanes, lengths=sizes[:, 0], widths=sizes[:, 1], semantic_mode=mode, ego_index=0, actor_index=1)
    checks = dict(semantic["checks"])
    checks["actor_starts_in_immediately_adjacent_lane"] = bool(abs(int(lanes[0, 1])-int(lanes[0, 0])) == 1)
    checks["actor_center_ahead_at_t0"] = bool(future[0, 1, 0] > future[0, 0, 0])
    semantic["checks"] = checks
    semantic["failure_reasons"] = [key for key, passed in checks.items() if not passed]
    semantic["valid"] = not semantic["failure_reasons"]
    return semantic, lanes


def _resolve_entry(recording, context, focal_id, future_lanes):
    indices = np.flatnonzero(future_lanes[:, 1] == future_lanes[0, 0])
    if not len(indices) or indices[0] == 0:
        return None, {"reason": "no_future_geometric_entry"}
    k = int(indices[0])
    start = context.t0_frame + (k-1)*2
    frames = np.arange(start, start+3)
    rows = recording.rows(focal_id, frames)
    meta = recording.track_metadata[focal_id]
    if not _valid_raw_rows(rows, meta):
        return None, {"reason": "raw_entry_bracket_observations_missing_or_invalid"}
    raw = rows.loc[:, RAW_COLUMNS].to_numpy(dtype=np.float64)
    centres = _centres_and_states(raw, context.raw_anchor_center, context.forward_sign)
    # Use the native origin->centre operation order at a boundary.
    centres_y = (centres[:, 1] - .5*meta.width_m) + .5*meta.width_m
    lanes = _raw_lane(centres_y, context.canonical_boundaries)
    same = lanes == future_lanes[0, 0]
    changes = np.flatnonzero((~same[:-1]) & same[1:]) + 1
    if not len(changes):
        return None, {"reason": "raw_entry_geometry_disagrees_with_sampled_entry"}
    j = int(changes[0])
    official = rows.laneId.to_numpy(dtype=np.int64)
    ego_lane = int(context.history_lane_ids[-1, 0])
    return int(frames[j]), {
        "raw_geometric_entry_frame": int(frames[j]), "sampled_entry_frame": context.t0_frame+k*2,
        "official_lane_before": int(official[j-1]), "official_lane_after": int(official[j]),
        "official_ego_t0_lane": ego_lane,
        "official_lane_transition_agrees": bool(official[j-1] != ego_lane and official[j] == ego_lane),
    }


@dataclass(frozen=True)
class ClipEvent:
    context: ObservedContext
    focal_id: int
    mode: str
    raw_entry_frame: int
    pair_future: np.ndarray
    pair_sizes: np.ndarray
    semantic: Mapping[str, Any]
    entry_diagnostic: Mapping[str, Any]

    @property
    def key(self):
        return self.context.recording_id, self.context.ego_id, self.focal_id, self.raw_entry_frame

    @property
    def event_id(self):
        rec, ego, focal, entry = self.key
        return f"highd{rec}_ego{ego}_focal{focal}_entry{entry}"

    @property
    def physical_event_id(self):
        return f"highd{self.context.recording_id}_focal{self.focal_id}_entry{self.raw_entry_frame}"

    @property
    def clip_id(self):
        return f"highd{self.context.recording_id}_ego{self.context.ego_id}_t0{self.context.t0_frame}"

    @property
    def selection_order(self):
        t0 = self.context.t0_frame
        return abs(t0-(self.raw_entry_frame-75)), t0

    @property
    def actor_order(self):
        ids = self.context.source_agent_ids
        focal = int(np.flatnonzero(ids == self.focal_id)[0])
        return [0, focal] + [i for i in range(1, len(ids)) if i != focal]


def _grid_frames(recording):
    grid = recording.tracks.loc[recording.tracks.frame % 25 == 0]
    return {int(frame): rows.set_index("id") for frame, rows in grid.groupby("frame", sort=True)}, len(grid)


def optimized_pair_queries(recording, frame_map):
    """Equivalent query pushdown: future lane changes, never risk, bound t0s.

    An accepted cut-in must contain an actual geometric transition into the
    unchanged ego lane.  Enumerating those transitions avoids materializing
    histories for grid rows that cannot possibly satisfy that semantic event.
    Final semantics and context eligibility are still checked from observations.
    """
    queries = set()
    counts = Counter()
    for actor, track in recording.tracks.groupby("id", sort=False):
        actor = int(actor)
        meta = recording.track_metadata.get(actor)
        if meta is None or meta.driving_direction not in (1, 2):
            continue
        rows = track.sort_values("frame")
        center_y = rows.y.to_numpy(dtype=np.float64) + .5*rows.height.to_numpy(dtype=np.float64)
        boundaries = _raw_boundaries(recording, meta.driving_direction)
        frames = rows.frame.to_numpy(dtype=np.int64)
        # Proposal-only superset: both boundary tie conventions plus a tiny
        # floating-point guard.  Canonical reflection reverses half-open ties;
        # ego-relative subtraction can also move an exactly-on-line centre by
        # an ulp.  Never prune a native-valid event on this coarse raw pass.
        # These variants do NOT alter final context, semantic or PET geometry.
        scale = max(1., float(np.max(np.abs(boundaries))), float(np.nanmax(np.abs(center_y))))
        guard = 64*np.finfo(np.float64).eps*scale
        variants = [center_y, center_y-guard, center_y+guard]
        change_set = set()
        for values in variants:
            for side in ("left", "right"):
                lanes = np.searchsorted(boundaries, values, side=side)-1
                inside = (values >= boundaries[0]) & (values <= boundaries[-1])
                lanes = np.where(inside & (lanes < len(boundaries)-1), lanes, -1)
                changes = np.flatnonzero((lanes[1:] != lanes[:-1]) & (lanes[1:] >= 0) & (lanes[:-1] >= 0) & (np.diff(frames) == 1)) + 1
                change_set.update(int(j) for j in changes)
        changes = sorted(change_set)
        counts["raw_geometry_lane_transition_proposal_frames"] += len(changes)
        sign = 1 if meta.driving_direction == 2 else -1
        for j in changes:
            entry = int(frames[j])
            first_t0 = max(25, ((entry-174+24)//25)*25)
            for t0 in range(first_t0, entry, 25):
                counts["optimized_focal_transition_grid_proposals"] += 1
                frame = frame_map.get(t0)
                if frame is None or actor not in frame.index:
                    continue
                focal = frame.loc[actor]
                focal_centre = np.asarray([focal.x+.5*focal.width, focal.y+.5*focal.height])
                for ego, row in frame.iterrows():
                    ego = int(ego)
                    ego_meta = recording.track_metadata.get(ego)
                    if ego == actor or ego_meta is None or ego_meta.driving_direction != meta.driving_direction:
                        continue
                    ego_centre = np.asarray([row.x+.5*row.width, row.y+.5*row.height])
                    dx = sign*(focal_centre[0]-ego_centre[0])
                    canonical_bounds = np.sort(-sign*(boundaries-ego_centre[1]))
                    ego_lane = int(_raw_lane(0., canonical_bounds))
                    focal_lane = int(_raw_lane(-sign*(focal_centre[1]-ego_centre[1]), canonical_bounds))
                    if 0 < dx <= 120. and ego_lane >= 0 and focal_lane >= 0 and abs(focal_lane-ego_lane) == 1:
                        queries.add((t0, ego, actor))
    counts["optimized_unique_pair_t0_proposals"] = len(queries)
    return sorted(queries), counts


def brute_force_pair_queries(recording, frame_map):
    """Small-fixture oracle: all t0 contexts and all ahead-adjacent focal IDs."""
    queries = []
    for t0, frame in frame_map.items():
        for ego in sorted(frame.index):
            context, _ = extract_context(recording, t0, int(ego), frame_rows=frame)
            if context is not None:
                queries.extend((t0, int(ego), focal) for focal in context.focal_candidates())
    return queries, Counter(bruteforce_pair_t0_proposals=len(queries))


@dataclass
class ClipScan:
    recording_id: str
    counts: Counter
    rejection_counts: Counter
    selected_events: Tuple[ClipEvent, ...]
    event_window_counts: Mapping[Tuple[str, int, int, int], int]
    selected_context_n_histogram: Mapping[int, int]
    actor_exclusion_counts: Mapping[str, int]


def scan_recording_clips(recording, *, query_mode="optimized"):
    frame_map, grid_actor_rows = _grid_frames(recording)
    if query_mode == "optimized":
        queries, counts = optimized_pair_queries(recording, frame_map)
    elif query_mode == "bruteforce":
        queries, counts = brute_force_pair_queries(recording, frame_map)
    else:
        raise ValueError("query_mode must be optimized or brute force test oracle")
    counts["grid_t0_actor_rows"] = grid_actor_rows
    counts["grid_unique_frames"] = len(frame_map)
    rejected, windows, best, actor_exclusions = Counter(), Counter(), {}, Counter()
    context_key, context, context_reason = None, None, None
    for t0, ego, focal in queries:
        if context_key != (t0, ego):
            context_key = t0, ego
            context, context_reason = extract_context(recording, t0, ego, frame_rows=frame_map[t0], audit_counts=actor_exclusions)
            counts["materialized_context_attempts"] += 1
            if context is not None:
                counts["valid_actual_history_contexts"] += 1
                counts["all_t0_ahead_adjacent_focal_ids_in_materialized_contexts"] += len(context.focal_candidates())
        if context is None:
            rejected[str(context_reason)] += 1
            continue
        if focal not in context.focal_candidates():
            rejected["focal_not_t0_ahead_adjacent_context_actor"] += 1
            continue
        counts["t0_context_qualified_pair_windows"] += 1
        future, sizes, reason = extract_pair_future(recording, context, focal)
        if reason:
            rejected[reason] += 1
            continue
        counts["complete_observed_pair_future_windows"] += 1
        focal_slot = int(np.flatnonzero(context.source_agent_ids == focal)[0])
        mode = "left_cut_in" if context.history[-1, focal_slot, 1] > 0 else "right_cut_in"
        semantic, lanes = _pair_semantic(future, sizes, context.canonical_boundaries, mode)
        if not semantic["valid"]:
            for failure in semantic["failure_reasons"]:
                rejected["pair_semantic_" + failure] += 1
            continue
        counts["semantic_valid_pair_windows"] += 1
        entry, diagnostic = _resolve_entry(recording, context, focal, lanes)
        if entry is None:
            rejected[diagnostic["reason"]] += 1
            continue
        event = ClipEvent(context, focal, mode, entry, future, sizes, semantic, diagnostic)
        windows[event.key] += 1
        counts["entry_resolved_pair_windows"] += 1
        if event.key not in best or event.selection_order < best[event.key].selection_order:
            best[event.key] = event
    selected = tuple(best[key] for key in sorted(best))
    counts["deduplicated_pair_events"] = len(selected)
    counts["distinct_physical_focal_lane_change_events"] = len({event.physical_event_id for event in selected})
    counts["pet_calls_during_selection"] = 0
    counts["background_future_observations_extracted"] = 0
    histogram = Counter(event.context.num_agents for event in selected)
    return ClipScan(recording.recording_id, counts, rejected, selected, dict(windows), dict(histogram), dict(actor_exclusions))


def _road_features(context):
    bounds = context.canonical_boundaries
    lane = int(context.t0_geometric_lanes[0])
    return np.asarray([bounds[0], bounds[lane], bounds[lane+1], bounds[-1]], dtype=np.float64)


def score_selected_clip_events(recording, scan):
    """Measure fixed selected pair events; background safety remains unknown."""
    counts, records, eligible = Counter(scan.counts), [], []
    invalid_reasons = Counter()
    for event in scan.selected_events:
        context = event.context
        measurement = observed_pair_cut_in_pet(
            event.pair_future, sizes_length_width_m=event.pair_sizes,
            lane_boundaries_y=context.canonical_boundaries, dt=.08, semantic_mode=event.mode,
        )
        if not measurement["semantic_valid"]:
            raise RuntimeError("selected pair population semantics differ from PET measurement semantics")
        counts["pet_calls_after_event_selection"] += 1
        order = event.actor_order
        ordered_ids = context.source_agent_ids[order]
        # This is a tracksMeta-only observability diagnostic, not a member
        # filter; no background future state rows are fetched here.
        background_without_future_extent = [int(actor) for actor in ordered_ids[2:]
                                            if recording.track_metadata[int(actor)].final_frame < context.t0_frame+174]
        counts["selected_events_with_background_metadata_future_extent_missing"] += bool(background_without_future_extent)
        counts["background_metadata_future_extent_missing_occurrences"] += len(background_without_future_extent)
        record = {
            "recording_id": scan.recording_id, "event_id": event.event_id,
            "physical_event_id": event.physical_event_id, "clip_id": event.clip_id,
            "ego_id": context.ego_id, "focal_id": event.focal_id,
            "source_agent_ids": ordered_ids.tolist(), "actual_context_num_agents": context.num_agents,
            "ego_index": 0, "focal_index": 1, "semantic_mode": event.mode,
            "selected_t0_frame": context.t0_frame, "raw_entry_frame": event.raw_entry_frame,
            "t0_to_entry_s": (event.raw_entry_frame-context.t0_frame)/25.,
            "distance_to_preferred_t0_frames": event.selection_order[0],
            "semantic_window_count_for_pair_event": scan.event_window_counts[event.key],
            "history_frame_ids": context.history_frame_ids.tolist(),
            "future_frame_ids": list(range(context.t0_frame, context.t0_frame+175, 2)),
            "raw_t0_ego_center_xy_m": context.raw_anchor_center.tolist(),
            "forward_sign": context.forward_sign, "canonical_carriageway_boundaries": context.canonical_boundaries.tolist(),
            "entry_diagnostic": dict(event.entry_diagnostic),
            "selection_uses_pet": False, "selection_uses_collision": False,
            "invalid_pet_triggers_reselection": False, "pair_future_only": True,
            "context_membership_uses_background_future": False,
            "background_metadata_future_extent_missing_ids": background_without_future_extent,
            "background_future_observations_checked": False, "background_collision_evaluated": False,
            "background_collision_count": None, "measurement": measurement,
        }
        records.append(record)
        if measurement["reference_eligible"]:
            eligible.append((event, measurement))
            counts["reference_eligible_events"] += 1
            counts["cap_events"] += bool(measurement["cap_flag"])
            counts["zero_events"] += bool(measurement["zero_atom_flag"])
            counts["positive_uncapped_events"] += 0. < measurement["pet_value_seconds"] < 4.
        else:
            counts["pet_invalid_events"] += 1
            invalid_reasons[measurement["reason"]] += 1
        counts["focal_collision_events"] += bool(measurement["focal_collision"])
        counts["pair_nonmonotone_longitudinal_events"] += not all(measurement["longitudinal_nondecreasing_pair"])
    counts["reference_eligible_physical_focal_lane_change_events"] = len({event.physical_event_id for event, _ in eligible})
    counts["reference_eligible_ego_focal_pairs"] = len({(event.context.ego_id, event.focal_id) for event, _ in eligible})
    counts["reference_eligible_unique_history_clips"] = len({event.clip_id for event, _ in eligible})
    for key in (
        "grid_t0_actor_rows", "materialized_context_attempts", "valid_actual_history_contexts",
        "t0_context_qualified_pair_windows", "complete_observed_pair_future_windows",
        "semantic_valid_pair_windows", "deduplicated_pair_events", "reference_eligible_events",
        "pet_invalid_events", "cap_events", "zero_events", "positive_uncapped_events",
        "pet_calls_after_event_selection", "focal_collision_events",
        "selected_events_with_background_metadata_future_extent_missing",
    ):
        counts.setdefault(key, 0)
    if counts["reference_eligible_events"] + counts["pet_invalid_events"] != len(scan.selected_events):
        raise RuntimeError("pair-event denominator not preserved")
    return records, eligible, counts, invalid_reasons


def eligible_clip_arrays(eligible, *, metadata):
    histories, dimensions, actor_ids, history_lanes = [], [], [], []
    offsets = [0]
    for event, _ in eligible:
        order = event.actor_order
        histories.append(event.context.history[:, order].transpose(1, 0, 2))
        dimensions.append(event.context.dimensions[order])
        actor_ids.append(event.context.source_agent_ids[order])
        history_lanes.append(event.context.history_lane_ids[:, order].T)
        offsets.append(offsets[-1]+len(order))
    n = len(eligible)
    max_boundaries = max((len(event.context.canonical_boundaries) for event, _ in eligible), default=0)
    all_boundaries = np.zeros((n, max_boundaries), dtype=np.float64)
    boundary_mask = np.zeros((n, max_boundaries), dtype=bool)
    for i, (event, _) in enumerate(eligible):
        bounds = event.context.canonical_boundaries
        all_boundaries[i, :len(bounds)] = bounds
        boundary_mask[i, :len(bounds)] = True
    return {
        "history_agents": np.concatenate(histories) if n else np.empty((0, 13, 4), np.float64),
        "dimensions_agents": np.concatenate(dimensions) if n else np.empty((0, 2), np.float64),
        "agent_ids": np.concatenate(actor_ids) if n else np.empty((0,), np.int64),
        "history_lane_ids_agents": np.concatenate(history_lanes) if n else np.empty((0, 13), np.int64),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "num_agents": np.diff(np.asarray(offsets, dtype=np.int64)),
        "road_geometry": np.stack([_road_features(event.context) for event, _ in eligible]) if n else np.empty((0, 4), np.float64),
        "carriageway_boundaries": all_boundaries, "carriageway_boundary_mask": boundary_mask,
        "semantics": np.asarray([0 if event.mode == "left_cut_in" else 1 for event, _ in eligible], dtype=np.int64),
        "pet": np.asarray([measurement["pet_value_seconds"] for _, measurement in eligible], dtype=np.float64),
        "pet_raw": np.asarray([measurement["pet_raw_seconds"] for _, measurement in eligible], dtype=np.float64),
        "pair_future": np.stack([event.pair_future for event, _ in eligible]) if n else np.empty((0, 88, 2, 4), np.float64),
        "recording_id": np.asarray([event.context.recording_id for event, _ in eligible], dtype="U2"),
        "event_id": np.asarray([event.event_id for event, _ in eligible], dtype="U96"),
        "physical_event_id": np.asarray([event.physical_event_id for event, _ in eligible], dtype="U96"),
        "clip_id": np.asarray([event.clip_id for event, _ in eligible], dtype="U96"),
        "t0_frame": np.asarray([event.context.t0_frame for event, _ in eligible], dtype=np.int64),
        "raw_entry_frame": np.asarray([event.raw_entry_frame for event, _ in eligible], dtype=np.int64),
        "history_frame_ids": np.stack([event.context.history_frame_ids for event, _ in eligible]) if n else np.empty((0, 13), np.int64),
        "future_frame_ids": np.stack([np.arange(event.context.t0_frame, event.context.t0_frame+175, 2) for event, _ in eligible]) if n else np.empty((0, 88), np.int64),
        "metadata_json": np.asarray(json.dumps(metadata, sort_keys=True, allow_nan=False)),
    }


def _json_write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")


def _path(value):
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT/path).resolve()


@dataclass(frozen=True)
class ClipBuildConfig:
    raw_root: Path
    split_path: Path
    output_root: Path
    recordings: Tuple[str, ...]
    data_protocol_path: Path
    data_protocol_sha256: str
    development_split_path: Path
    development_split_sha256: str
    grid_stride_raw_frames: int = 25
    preferred_pre_entry_s: float = 3.0
    context_longitudinal_radius_m: float = 120.0
    min_context_agents: int = 3
    max_context_agents: Optional[int] = None

    def __post_init__(self):
        for key in ("raw_root", "split_path", "output_root", "data_protocol_path", "development_split_path"):
            object.__setattr__(self, key, _path(getattr(self, key)))
        ids = tuple(recording_id(value) for value in self.recordings)
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("recordings must be unique and nonempty")
        object.__setattr__(self, "recordings", ids)
        self.validate()

    def validate(self):
        parent = load_split_assignment(self.split_path)
        if not set(self.recordings).issubset(set(parent["train"])):
            raise PermissionError("v2 clips may only read original train recordings")
        for prefix in ("data_protocol", "development_split"):
            if sha256_file(getattr(self, prefix+"_path")) != getattr(self, prefix+"_sha256"):
                raise ValueError(f"{prefix} SHA256 differs from authorization")
        if self.output_root == self.raw_root or self.raw_root in self.output_root.parents:
            raise ValueError("outputs may not be inside the immutable raw directory")
        if (self.grid_stride_raw_frames, self.preferred_pre_entry_s, self.context_longitudinal_radius_m,
            self.min_context_agents, self.max_context_agents) != (25, 3., 120., 3, None):
            raise ValueError("v2 executable grid/alignment/ROI/minN/maxN differ from frozen policy")
        _validate_declared_protocol(self, parent)

    @classmethod
    def from_mapping(cls, value):
        expected = set(cls.__dataclass_fields__) | {"protocol", "source_origin", "natural_only", "allow_simulated_futures"}
        if set(value) != expected:
            raise ValueError(f"v2 config keys differ: {sorted(set(value).symmetric_difference(expected))}")
        if value["protocol"] != PROTOCOL or value["source_origin"] != "natural_observation" or value["natural_only"] is not True or value["allow_simulated_futures"] is not False:
            raise ValueError("config must explicitly authorize this natural-only v2 protocol")
        return cls(**{key: value[key] for key in cls.__dataclass_fields__})

    def as_json(self):
        fields = {key: str(getattr(self, key)) if key.endswith("_path") or key.endswith("_root") else getattr(self, key) for key in self.__dataclass_fields__}
        fields.update(protocol=PROTOCOL, source_origin="natural_observation", natural_only=True, allow_simulated_futures=False)
        return fields


def _validate_declared_protocol(config, parent):
    """The root-owned v2 protocol is parsed, not merely hash-listed."""
    protocol = json.loads(config.data_protocol_path.read_text())
    split = json.loads(config.development_split_path.read_text())
    parent_sha = sha256_file(config.split_path)
    if _path(protocol["raw_data_root"]) != config.raw_root:
        raise ValueError("raw directory differs from bound natural v2 protocol")
    if _path(protocol["authorized_recording_roster"]) != config.split_path or protocol["authorized_roster_sha256"] != parent_sha:
        raise ValueError("v2 data protocol original roster differs")
    if _path(split["authorized_parent_roster"]) != config.split_path or split["authorized_parent_roster_sha256"] != parent_sha:
        raise ValueError("v2 development original roster differs")
    assignments = []
    for key in ("T", "V_ref", "audit_only"):
        if not isinstance(split.get(key), list):
            raise ValueError(f"development {key} must be a recording list")
        assignments.extend(recording_id(value) for value in split[key])
    if len(assignments) != len(set(assignments)) or set(assignments) != set(parent["train"]):
        raise ValueError("T/V_ref/audit_only must partition the original train roster exactly")
    for key, parent_key in (("locked_parent_val", "val"), ("locked_parent_test", "test")):
        if set(split.get(key, ())) != set(parent[parent_key]):
            raise ValueError("original protected recording sets changed")
    if split.get("V_gen") != [] or split.get("Test") != []:
        raise ValueError("v2 reference pilot does not authorize final or generator testing")
    # Detailed v2 executable-field binding is completed against the root-owned
    # protocol before production execution; no permissive schema fallback.
    _validate_v2_execution_fields(protocol, split)


def _validate_v2_execution_fields(protocol, split):
    required = {
        "protocol": "natural_highd_variable_context_reference_pilot_v2",
        "source": "raw_highd_observed_trajectories_only", "authorized_content_split": "train",
        "locked_content_splits": ["val", "test"], "build_protocol": PROTOCOL,
        "extraction_protocol": EXTRACTION_PROTOCOL, "raw_hz": 25, "raw_stride_frames": 2,
        "history_steps": 13, "pair_future_steps_including_t0": 88,
        "history_duration_s": .96, "future_duration_s": 6.96,
        "t0_grid": "positive_raw_frame_divisible_by_25", "grid_stride_raw_frames": 25,
        "preferred_pre_entry_s": 3.0, "roi_longitudinal_half_extent_m": 120.0,
        "minimum_context_agents": 3, "maximum_context_agents": None,
        "context_membership": "all_t0_observed_same_carriageway_actual_vehicles_with_absolute_center_dx_at_most_120m_and_complete_finite_actual_H13_inside_road",
        "context_selection_uses_background_future": False, "background_future_required": False,
        "background_future_exported": False, "background_collision_evaluated": False,
        "background_missing_future_interpreted_as_safe": False,
        "pair_roles": {"ego_index": 0, "focal_index": 1},
        "bystander_container_order": "ascending_raw_vehicle_ID_after_ego_and_focal_not_a_learned_role",
        "focal_candidate_rule": "every_t0_context_vehicle_with_center_ahead_of_ego_and_in_immediately_adjacent_geometric_lane",
        "event_modes": ["left_cut_in", "right_cut_in"], "focal_candidates_ranked_by_future_risk": False,
        "pair_requires_complete_actual_H13_F88": True,
        "pair_semantics": "ego_stays_in_lane_focal_enters_from_immediately_adjacent_lane_and_stays_both_within_full_actual_carriageway",
        "event_key": ["recording_id", "ego_raw_ID", "focal_raw_ID", "raw_geometric_lane_entry_frame"],
        "physical_event_key": ["recording_id", "focal_raw_ID", "raw_geometric_lane_entry_frame"],
        "context_clip_key": ["recording_id", "ego_raw_ID", "t0_frame"],
        "event_deduplication": "one_semantically_valid_pair_and_history_context_window_per_event_closest_to_entry_minus_3s_tie_earlier_t0",
        "event_time_definition": "raw_25Hz_center_crosses_actual_ego_lane_boundary_consistent_with_native_PET_semantics",
        "window_selection_uses_pet": False, "window_selection_uses_collision": False,
        "replace_window_when_pet_unobserved": False,
        "coordinate_transform": "box_centers_relative_to_ego_t0_x_forward_y_left_rigid_reflection_and_translation_only",
        "driving_direction_source": "tracksMeta_static_road_direction",
        "t0_negative_longitudinal_velocity_rejected": False, "finite_observed_velocities_modified": False,
        "road_geometry_features": ["carriageway_outer_right_y_m", "ego_lane_right_y_m", "ego_lane_left_y_m", "carriageway_outer_left_y_m"],
        "semantic_road_boundaries": "all_actual_carriageway_boundaries_no_cropping",
        "vehicle_classes": "cars_and_trucks_as_observed", "only_cars": False,
        "interpolation_of_missing_states": False, "fabricated_or_duplicated_actors": False,
        "actor_reassignment_over_time": False, "lane_center_snapping": False,
        "trajectory_smoothing": False, "terminal_projection": False,
        "simulation_or_generated_futures": False, "clustering_or_cross_history_neighbor_reference": False,
        "pet_metric": PET_PROTOCOL, "pet_cap_s": 4.0, "unobserved_occupancy_is_cap": False,
        "collision_overrides_pet": False, "legacy_checkpoint_initialization": False,
        "generator_training_authorized_in_this_stage": False,
        "complete_variable_N_future_training_targets_available": False, "old_v1_data_merged_into_v2": False,
    }
    for key, expected in required.items():
        if key not in protocol or protocol[key] != expected or (isinstance(expected, bool) and protocol[key] is not expected):
            raise ValueError(f"bound v2 protocol {key!r} differs from executable behavior")
    if split.get("protocol") != "natural_clip_reference_recording_split_v2":
        raise ValueError("bound development split is not v2")
    if split.get("frozen_before_v2_event_counts_or_model_outcomes") is not True:
        raise ValueError("v2 development split must be frozen before v2 outcomes")
    if split.get("split_unit") != "recording" or split.get("salt") != "natural_reference_development_v1":
        raise ValueError("v2 split must use the declared recording-unit hash assignment")
    parent = load_split_assignment(_path(split["authorized_parent_roster"]))
    order = sorted(parent["train"], key=lambda rec: hashlib.sha256(f"natural_reference_development_v1|{rec}".encode()).hexdigest())
    if set(split["V_ref"]) != set(order[:6]) or set(split["T"]) != set(order[6:]) or split["audit_only"]:
        raise ValueError("v2 T/V_ref assignment differs from outcome-blind frozen recording hash split")


def build_clip_dataset(config, *, config_sha256):
    config.validate()
    config.output_root.mkdir(parents=True, exist_ok=False)
    effective = config.as_json()
    effective_sha = hashlib.sha256(json.dumps(effective, sort_keys=True, indent=2, allow_nan=False).encode()).hexdigest()
    _json_write_new(config.output_root/"effective_config.json", effective)
    guards = dict(source_origin="natural_observation", natural_only=True, allow_simulated_futures=False,
        source_trajectories_modified=False, history_context_complete_observed=True,
        context_membership_uses_background_future=False, background_future_exported=False,
        pair_future_only=True, ego_index=0, focal_index=1, context_min_actual_agents=3, context_max_agents=None)
    bindings = {
        "data_protocol_binding": {"path": str(config.data_protocol_path), "sha256": config.data_protocol_sha256},
        "development_split_binding": {"path": str(config.development_split_path), "sha256": config.development_split_sha256},
        "split_binding": {"path": str(config.split_path), "sha256": sha256_file(config.split_path)},
    }
    paths = [Path(__file__), Path(__file__).with_name("highd.py"), Path(__file__).with_name("pair_pet.py"),
        Path(__file__).with_name("pet.py"), PROJECT_ROOT/"pcontrol/data/road_semantics.py",
        PROJECT_ROOT/"pcontrol/data/cut_in_pet.py", PROJECT_ROOT/"pcontrol/research/build_natural_highd_clips.py"]
    manifest = dict(protocol=PROTOCOL, extraction_protocol=EXTRACTION_PROTOCOL, metric_version=PET_PROTOCOL,
        population_E=POPULATION, config_sha256=config_sha256, effective_config_sha256=effective_sha,
        code_sha256={str(path.resolve()): sha256_file(path) for path in paths},
        recordings=list(config.recordings), input_bindings={}, recording_artifacts={},
        query_optimization="raw_geometric_lane_transition_semantic_query_pushdown_not_risk_filter",
        raw_csv_loader_decodes_authorized_recording=True,
        heldout_trajectory_contents_opened=False, legacy_cache_or_weights_used=False,
        reference_uses_clustering=False,
        background_collision_evaluated=False, background_collision_count=None,
        **guards, **bindings)
    totals, all_n_histogram = Counter(), Counter()
    for rec in config.recordings:
        source_paths = {key: config.raw_root/f"{rec}_{key}.csv" for key in ("tracks", "tracksMeta", "recordingMeta")}
        snapshots = {key: (path.stat().st_size, path.stat().st_mtime_ns) for key, path in source_paths.items()}
        sources = {key: {"path": str(path), "sha256": sha256_file(path)} for key, path in source_paths.items()}
        recording = load_train_recording(config.raw_root, rec, split_path=config.split_path)
        scan = scan_recording_clips(recording)
        records, eligible, counts, invalid_reasons = score_selected_clip_events(recording, scan)
        for key, path in source_paths.items():
            if (path.stat().st_size, path.stat().st_mtime_ns) != snapshots[key]:
                raise RuntimeError("raw source changed during clip extraction")
        n_histogram = Counter(event.context.num_agents for event, _ in eligible)
        metadata = dict(protocol=PROTOCOL, extraction_protocol=EXTRACTION_PROTOCOL, metric_version=PET_PROTOCOL,
            recording_id=rec, source_bindings=sources, config_sha256=config_sha256,
            effective_config_sha256=effective_sha, reference_eligible_events=len(eligible),
            context_n_histogram=dict(n_histogram), population_E=POPULATION,
            raw_carriageway_boundaries_m={"upper": recording.upper_lane_markings_raw_m.tolist(), "lower": recording.lower_lane_markings_raw_m.tolist()},
            **guards, **bindings)
        arrays = eligible_clip_arrays(eligible, metadata=metadata)
        array_path = config.output_root/f"{rec}.npz"
        with array_path.open("xb") as handle:
            np.savez_compressed(handle, **arrays)
        ledger_path = config.output_root/f"{rec}.events.audit.jsonl"
        with ledger_path.open("x", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row, sort_keys=True, allow_nan=False)+"\n")
        audit_path = config.output_root/f"{rec}.audit.json"
        _json_write_new(audit_path, dict(recording_id=rec, counts=dict(counts), rejection_counts=dict(scan.rejection_counts),
            context_actor_exclusion_counts=dict(scan.actor_exclusion_counts),
            pet_invalid_reasons=dict(invalid_reasons), selected_context_n_histogram=scan.selected_context_n_histogram,
            reference_eligible_context_n_histogram=dict(n_histogram), event_selection_precedes_pet=True,
            invalid_pet_triggers_reselection=False, background_future_used_for_context_selection=False,
            physical_focal_events_are_not_assumed_independent_ego_focal_pair_events=True))
        manifest["input_bindings"][rec] = sources
        manifest["recording_artifacts"][rec] = dict(recording_id=rec, path=str(array_path), sha256=sha256_file(array_path),
            rows=len(eligible), num_rows=len(eligible), counts=dict(counts), context_n_histogram=dict(n_histogram),
            audit={"path": str(audit_path), "sha256": sha256_file(audit_path)}, events={"path": str(ledger_path), "sha256": sha256_file(ledger_path)})
        totals.update(counts)
        all_n_histogram.update(n_histogram)
        print(json.dumps({"recording_complete": rec, "counts": dict(counts), "context_n_histogram": dict(n_histogram)}, sort_keys=True), flush=True)
        del recording, scan, records, eligible, arrays
    manifest.update(status="complete", counts=dict(totals), reference_eligible_context_n_histogram=dict(all_n_histogram))
    _json_write_new(config.output_root/"manifest.json", manifest)
    _json_write_new(config.output_root/"summary.json", dict(protocol=PROTOCOL, status="complete", counts=dict(totals),
        reference_eligible_context_n_histogram=dict(all_n_histogram), population_E=POPULATION,
        manifest={"path": str(config.output_root/"manifest.json"), "sha256": sha256_file(config.output_root/"manifest.json")},
        heldout_trajectory_contents_opened=False, source_trajectories_modified=False))
    return manifest
