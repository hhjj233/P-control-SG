"""Lane inference, semantic validity and collision diagnostics for highD scenes.

Copied verbatim from the authors' earlier evaluation code.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


def _normalize_mode(value: Any) -> str:
    mode = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "carfollowing": "car_following",
        "following": "car_following",
        "cutin": "cut_in",
        "left_cutin": "left_cut_in",
        "right_cutin": "right_cut_in",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"car_following", "cut_in", "left_cut_in", "right_cut_in"}:
        raise ValueError(f"unsupported semantic_mode {value!r}")
    return mode


def infer_lane_ids_from_boundaries(
    trajectory: np.ndarray,
    lane_boundaries_y: np.ndarray,
    *,
    widths: np.ndarray,
    agent_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Infer discrete lanes from vehicle-centre y and ordered boundaries.

    ``trajectory[..., 1]`` follows the highD cache convention (box origin), so
    half a vehicle width is added before binning.  Centres outside the road
    corridor receive lane id ``-1`` and consequently fail semantic validity.
    """

    traj = np.asarray(trajectory, dtype=np.float64)
    boundaries = np.asarray(lane_boundaries_y, dtype=np.float64).reshape(-1)
    width = np.asarray(widths, dtype=np.float64).reshape(-1)
    if traj.ndim != 3 or traj.shape[-1] < 2:
        raise ValueError("trajectory must be [T,N,D>=2]")
    if width.shape != (traj.shape[1],) or not bool(np.isfinite(width).all()):
        raise ValueError("widths must contain one finite value per agent")
    if agent_mask is None:
        valid = np.ones((traj.shape[1],), dtype=bool)
    else:
        valid = np.asarray(agent_mask, dtype=bool).reshape(-1)
        if valid.shape != (traj.shape[1],):
            raise ValueError("agent_mask must contain one value per agent")
    if not bool((width[valid] > 0.0).all()):
        raise ValueError("widths must be positive for every valid agent")
    if boundaries.size < 2 or not bool(
        np.isfinite(boundaries).all() and (np.diff(boundaries) > 0.0).all()
    ):
        raise ValueError("lane_boundaries_y must be finite and increasing")
    centers = traj[..., 1] + 0.5 * width[None, :]
    lane = np.searchsorted(boundaries, centers, side="right") - 1
    inside = (centers >= boundaries[0]) & (centers < boundaries[-1])
    lane = np.where(inside & valid[None, :], lane, -1)
    return lane.astype(np.int64)


def semantic_validity(
    trajectory: np.ndarray,
    lane_ids: np.ndarray,
    *,
    lengths: np.ndarray,
    widths: np.ndarray,
    semantic_mode: Any,
    ego_index: int = 0,
    actor_index: int = 1,
    agent_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Evaluate the requested car-following/cut-in trajectory semantics."""

    traj = np.asarray(trajectory, dtype=np.float64)
    lanes = np.asarray(lane_ids, dtype=np.int64)
    length = np.asarray(lengths, dtype=np.float64).reshape(-1)
    width = np.asarray(widths, dtype=np.float64).reshape(-1)
    mode = _normalize_mode(semantic_mode)
    if traj.ndim != 3 or traj.shape[1] < 2 or traj.shape[-1] < 2:
        raise ValueError("semantic evaluation requires [T,N>=2,D>=2]")
    if lanes.shape != traj.shape[:2]:
        raise ValueError("lane_ids must align with trajectory [T,N]")
    agents = int(traj.shape[1])
    if length.shape != (agents,) or width.shape != (agents,):
        raise ValueError("lengths/widths must contain one value per vehicle")
    ego = int(ego_index)
    actor = int(actor_index)
    if ego < 0 or ego >= agents or actor < 0 or actor >= agents or ego == actor:
        raise ValueError("ego_index/actor_index must identify two different agents")
    if agent_mask is None:
        valid_agents = np.ones((agents,), dtype=bool)
    else:
        valid_agents = np.asarray(agent_mask, dtype=bool).reshape(-1)
        if valid_agents.shape != (agents,):
            raise ValueError("agent_mask must contain one value per vehicle")
    if not bool(valid_agents[ego] and valid_agents[actor]):
        raise ValueError("ego and target actor must both be valid")

    center_x = traj[..., 0] + 0.5 * length[None, :]
    center_y = traj[..., 1] + 0.5 * width[None, :]
    checks: Dict[str, bool] = {
        "all_agents_in_lane_corridor": bool((lanes[:, valid_agents] >= 0).all()),
        "ego_lane_constant": bool(np.all(lanes[:, ego] == lanes[0, ego])),
    }
    if mode == "car_following":
        checks.update(
            {
                "same_lane_throughout": bool(
                    np.all(lanes[:, actor] == lanes[:, ego])
                ),
                "leader_ahead_throughout": bool(
                    np.all(center_x[:, actor] > center_x[:, ego])
                ),
            }
        )
        realized_mode = "car_following" if all(checks.values()) else "invalid"
    else:
        same = lanes[:, actor] == lanes[:, ego]
        entered = np.flatnonzero(same)
        stays_after_entry = bool(entered.size and same[int(entered[0]) :].all())
        initial_delta_y = float(center_y[0, actor] - center_y[0, ego])
        realized_direction = (
            "left_cut_in"
            if initial_delta_y > 0.0
            else "right_cut_in" if initial_delta_y < 0.0 else "ambiguous_cut_in"
        )
        direction_matches = mode == "cut_in" or realized_direction == mode
        checks.update(
            {
                "actor_starts_adjacent": bool(
                    lanes[0, actor] != lanes[0, ego]
                ),
                "actor_enters_ego_lane": bool(entered.size),
                "actor_stays_after_entry": stays_after_entry,
                "actor_ends_in_ego_lane": bool(
                    lanes[-1, actor] == lanes[-1, ego]
                ),
                "direction_matches_request": bool(direction_matches),
            }
        )
        realized_mode = realized_direction if all(checks.values()) else "invalid"
    failed = [name for name, passed in checks.items() if not passed]
    return {
        "valid": not failed,
        "requested_mode": mode,
        "realized_mode": realized_mode,
        "ego_index": ego,
        "target_actor_index": actor,
        "checks": checks,
        "failure_reasons": failed,
    }


def multiagent_collision_diagnostics(
    trajectory: np.ndarray,
    *,
    lengths: np.ndarray,
    widths: np.ndarray,
    ego_index: int = 0,
    target_actor_index: int = 1,
    agent_mask: Optional[np.ndarray] = None,
    margin_m: float = 0.0,
) -> Dict[str, Any]:
    """Report focal and secondary box-overlap collisions for arbitrary N."""

    traj = np.asarray(trajectory, dtype=np.float64)
    if traj.ndim != 3 or traj.shape[-1] < 2:
        raise ValueError("trajectory must be [T,N,D>=2]")
    _, agents, _ = traj.shape
    length = np.asarray(lengths, dtype=np.float64).reshape(-1)
    width = np.asarray(widths, dtype=np.float64).reshape(-1)
    if length.shape != (agents,) or width.shape != (agents,):
        raise ValueError("lengths/widths must contain one value per agent")
    if agent_mask is None:
        valid = np.ones((agents,), dtype=bool)
    else:
        valid = np.asarray(agent_mask, dtype=bool).reshape(-1)
        if valid.shape != (agents,):
            raise ValueError("agent_mask must contain one value per agent")
    ego = int(ego_index)
    target = int(target_actor_index)
    if (
        ego < 0
        or ego >= agents
        or target < 0
        or target >= agents
        or ego == target
        or not valid[ego]
        or not valid[target]
    ):
        raise ValueError("ego/target indices must identify valid different agents")
    if not bool(
        np.isfinite(traj[:, valid, :2]).all()
        and np.isfinite(length[valid]).all()
        and np.isfinite(width[valid]).all()
        and (length[valid] > 0.0).all()
        and (width[valid] > 0.0).all()
    ):
        raise ValueError("valid trajectories and dimensions must be finite/positive")

    center_x = traj[..., 0] + 0.5 * length[None, :]
    center_y = traj[..., 1] + 0.5 * width[None, :]
    half_length = 0.5 * length + float(margin_m)
    half_width = 0.5 * width + float(margin_m)
    pair_events: Dict[Tuple[int, int], np.ndarray] = {}
    valid_indices = np.flatnonzero(valid)
    for left_position, left in enumerate(valid_indices):
        for right in valid_indices[left_position + 1 :]:
            overlap_x = np.abs(center_x[:, left] - center_x[:, right]) < (
                half_length[left] + half_length[right]
            )
            overlap_y = np.abs(center_y[:, left] - center_y[:, right]) < (
                half_width[left] + half_width[right]
            )
            pair_events[(int(left), int(right))] = overlap_x & overlap_y

    collided_pairs = [pair for pair, event in pair_events.items() if bool(event.any())]
    focal = tuple(sorted((ego, target)))
    ego_non_target = [
        pair
        for pair in collided_pairs
        if ego in pair and pair != focal
    ]
    target_non_ego = [
        pair
        for pair in collided_pairs
        if target in pair and ego not in pair
    ]
    bystander_bystander = [
        pair
        for pair in collided_pairs
        if ego not in pair and target not in pair
    ]
    # Backward-compatible aggregate: historically every pair not involving
    # ego was called background-background, including target-bystander pairs.
    background = [pair for pair in collided_pairs if ego not in pair]
    first_steps = [
        int(np.flatnonzero(pair_events[pair])[0]) for pair in collided_pairs
    ]
    return {
        "any_collision": bool(collided_pairs),
        "ego_target_collision": bool(focal in collided_pairs),
        "ego_non_target_collision": bool(ego_non_target),
        "target_non_ego_collision": bool(target_non_ego),
        "bystander_bystander_collision": bool(bystander_bystander),
        "background_background_collision": bool(background),
        "background_background_collision_semantics": (
            "legacy_any_pair_excluding_ego_includes_target_bystander"
        ),
        "collided_pair_count": int(len(collided_pairs)),
        "collided_pairs": [list(pair) for pair in collided_pairs],
        "first_collision_step": min(first_steps) if first_steps else None,
        "valid_agent_count": int(valid.sum()),
        "margin_m": float(margin_m),
    }
