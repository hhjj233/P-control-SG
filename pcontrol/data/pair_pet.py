"""Pair-only natural PET measurement for variable-N history reference data.

This does not construct a two-vehicle research scene: the reference history
contains all eligible actual context vehicles. Only ego/focal futures are
needed to measure this scalar label. Background future safety is unknown,
not reported as collision-free. Legacy N9 code and datasets remain unchanged.
"""
from __future__ import annotations

import numpy as np

from pcontrol.data.road_semantics import (
    infer_lane_ids_from_boundaries, semantic_validity,
    multiagent_collision_diagnostics,
)
from pcontrol.data.cut_in_pet import PET_ADAPTER_VERSION, cut_in_pet_from_lane_transition
from pcontrol.data.pet import _crossing_time

PROTOCOL = "natural_observed_pair_pet_complete_occupancy_v2"


def observed_pair_cut_in_pet(
    future_centers, *, sizes_length_width_m, lane_boundaries_y, dt,
    semantic_mode,
):
    future = np.asarray(future_centers, dtype=np.float64)
    sizes = np.asarray(sizes_length_width_m, dtype=np.float64)
    if future.shape != (88, 2, 4) or sizes.shape != (2, 2):
        raise ValueError("pair future must be actual F88 [88,2,4], ego=0/focal=1")
    if semantic_mode not in ("left_cut_in", "right_cut_in"):
        raise ValueError("explicit left/right cut-in mode required")
    if not np.isfinite(future).all() or not np.isfinite(sizes).all() or np.any(sizes <= 0):
        raise ValueError("actual pair observations/dimensions must be finite and positive-sized")
    if not np.isfinite(dt) or abs(float(dt) - .08) > 1e-12:
        raise ValueError("observed pair v2 freezes 0.08s sampling")
    origin = future.copy()
    origin[..., :2] -= .5 * sizes[None]
    lengths, widths = sizes[:, 0], sizes[:, 1]
    lanes = infer_lane_ids_from_boundaries(origin, lane_boundaries_y, widths=widths)
    semantic = semantic_validity(origin, lanes, lengths=lengths, widths=widths,
                                 semantic_mode=semantic_mode, ego_index=0, actor_index=1)
    # The legacy generic semantic function calls any distinct lane 'adjacent'.
    # N9 selection had already enforced actual adjacency and actor-ahead. Make
    # that source-population condition explicit for the new free role indices.
    checks = dict(semantic['checks'])
    checks['actor_starts_in_immediately_adjacent_lane'] = bool(abs(int(lanes[0, 1]) - int(lanes[0, 0])) == 1)
    checks['actor_center_ahead_at_t0'] = bool(future[0, 1, 0] > future[0, 0, 0])
    failed = [key for key, passed in checks.items() if not passed]
    collision = multiagent_collision_diagnostics(origin, lengths=lengths, widths=widths,
                                                 ego_index=0, target_actor_index=1)
    record = dict(
        metric_version=PROTOCOL, base_native_adapter=PET_ADAPTER_VERSION,
        semantic_mode=semantic_mode, semantic_valid=not failed,
        semantic_failure_reasons=failed, semantic_checks=checks,
        ego_index=0, target_actor_index=1, pair_future_only=True,
        pet_raw_seconds=None, pet_value_seconds=None, native_legacy_cap_value=None,
        pet_status="invalid", reference_eligible=False, cap_flag=False,
        zero_atom_flag=False, focal_collision=collision['ego_target_collision'],
        background_collision_evaluated=False, secondary_collision_count=None,
        collision_overrides_pet=False, trajectory_modified=False,
        occupation_times=None, transition_index=None,
        longitudinal_nondecreasing_pair=np.all(np.diff(future[..., 0], axis=0) >= -1e-8, axis=0).tolist(),
    )
    if failed:
        record['reason'] = 'outside_declared_observed_pair_cut_in_population'
        return record
    transition = int(np.flatnonzero(lanes[:, 1] == lanes[0, 0])[0])
    native = cut_in_pet_from_lane_transition(origin, lanes, lengths=lengths, dt=dt,
                                            ego_index=0, actor_index=1, cap_s=4.)
    record.update(transition_index=transition, native_legacy_cap_value=float(native))
    centers_x = origin[..., 0] + .5 * lengths[None]
    point = float(centers_x[transition, 1])
    record['conflict_point_x_m'] = point
    intervals = []
    missing = []
    for actor in (0, 1):
        front = _crossing_time(centers_x[:, actor] + .5 * lengths[actor], point, dt)
        rear = _crossing_time(centers_x[:, actor] - .5 * lengths[actor], point, dt)
        if front is None or rear is None:
            missing.append(dict(actor_index=actor, front_unobserved=front is None, rear_unobserved=rear is None))
        else:
            intervals.append((min(front, rear), max(front, rear)))
    if missing:
        record.update(reason='conflict_point_occupancy_not_fully_observed', missing_crossings=missing)
        return record
    a, b = intervals
    raw = b[0] - a[1] if a[1] < b[0] else a[0] - b[1] if b[1] < a[0] else 0.
    value = float(np.clip(raw, 0., 4.))
    if value != native:
        raise RuntimeError("pair extraction differs from the frozen native PET formula")
    record.update(
        pet_raw_seconds=float(raw), pet_value_seconds=value,
        pet_status='capped' if raw >= 4. else 'finite', reference_eligible=True,
        cap_flag=raw >= 4., zero_atom_flag=value == 0.,
        reason='complete_observed_occupancy', occupation_times=[list(v) for v in intervals],
    )
    return record
