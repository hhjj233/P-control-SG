"""Post-encroachment time of a lane transition.

Copied verbatim from the authors' earlier evaluation code.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


PET_ADAPTER_VERSION = "lane_transition_conflict_point_v1"


def cut_in_pet_from_lane_transition(
    trajectory: np.ndarray,
    lane_ids: np.ndarray,
    *,
    lengths: np.ndarray,
    dt: float,
    ego_index: int = 0,
    actor_index: int = 1,
    cap_s: float = 30.0,
) -> float:
    """PET at the longitudinal conflict point induced by a cut-in transition.

    The adversarial vehicle must start in an adjacent lane and enter the ego's
    unchanged lane. The conflict point is its longitudinal center at the first
    in-lane frame. PET is the temporal gap between the two vehicles' box
    occupancy intervals at that point. This is a documented project adapter;
    the paper does not disclose its conflict-point implementation.
    """

    traj = np.asarray(trajectory, dtype=np.float64)
    lanes = np.asarray(lane_ids)
    sizes = np.asarray(lengths, dtype=np.float64).reshape(-1)
    if traj.ndim != 3 or traj.shape[-1] < 2 or lanes.shape != traj.shape[:2]:
        raise ValueError("trajectory/lane_ids must be [T,N,D] and [T,N]")
    if sizes.shape[0] != traj.shape[1] or not bool(
        np.isfinite(sizes).all() and (sizes > 0.0).all()
    ):
        raise ValueError("lengths must contain one positive finite value per agent")
    if not np.isfinite(float(dt)) or float(dt) <= 0.0:
        raise ValueError("dt must be finite and positive")
    if not np.isfinite(float(cap_s)) or float(cap_s) <= 0.0:
        raise ValueError("cap_s must be finite and positive")
    ego, actor = int(ego_index), int(actor_index)
    if ego == actor or min(ego, actor) < 0 or max(ego, actor) >= traj.shape[1]:
        raise ValueError("invalid ego/actor indices")
    ego_lane = lanes[0, ego]
    if bool(np.any(lanes[:, ego] != ego_lane)):
        raise ValueError("ego changes lane in a cut-in protocol sample")
    if lanes[0, actor] == ego_lane:
        raise ValueError("cut-in actor must start in an adjacent lane")
    entered = np.flatnonzero(lanes[:, actor] == ego_lane)
    if entered.size == 0:
        return float(cap_s)
    transition = int(entered[0])
    conflict_x = float(traj[transition, actor, 0] + 0.5 * sizes[actor])

    def _crossing_time(values: np.ndarray, target: float):
        difference = values - float(target)
        exact = np.flatnonzero(np.isclose(difference, 0.0, atol=1e-8))
        if exact.size:
            return float(exact[0]) * float(dt)
        crossings = np.flatnonzero(difference[:-1] * difference[1:] < 0.0)
        if crossings.size == 0:
            return None
        index = int(crossings[0])
        denominator = values[index + 1] - values[index]
        if abs(float(denominator)) < 1e-12:
            return None
        fraction = (float(target) - values[index]) / denominator
        return (float(index) + float(fraction)) * float(dt)

    def _occupancy_interval(vehicle: int):
        center = traj[:, vehicle, 0] + 0.5 * sizes[vehicle]
        front_time = _crossing_time(center + 0.5 * sizes[vehicle], conflict_x)
        rear_time = _crossing_time(center - 0.5 * sizes[vehicle], conflict_x)
        if front_time is None or rear_time is None:
            return None
        return min(front_time, rear_time), max(front_time, rear_time)

    ego_interval = _occupancy_interval(ego)
    actor_interval = _occupancy_interval(actor)
    if ego_interval is None or actor_interval is None:
        return float(cap_s)
    if ego_interval[1] < actor_interval[0]:
        pet = actor_interval[0] - ego_interval[1]
    elif actor_interval[1] < ego_interval[0]:
        pet = ego_interval[0] - actor_interval[1]
    else:
        pet = 0.0
    return float(np.clip(pet, 0.0, float(cap_s)))
