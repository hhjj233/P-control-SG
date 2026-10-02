"""Authenticated complete-scene observations for display, never model inputs.

Only existing AUDIT NPZ members are decoded; no raw CSV or generator access.
Case selection deliberately receives identities and counts, not model scores.
"""
import hashlib
import json

import numpy as np

from pcontrol.data.complete_scene_view import ELIGIBILITY, ESTIMAND, verify_binding
from pcontrol.data.expanded_scene_view import validate_manifest


DATA_SHA256 = "e05bd02e52a81ece015d86be8e106d7c353db42cc29377df364d9306d88c9012"
SALT = "natural_scene_refinement_visualization_v1"
STRATA = (("N3_5", 3, 5), ("N6_8", 6, 8), ("N9_plus", 9, 1000000))
AUDIT_RECORDINGS = frozenset(("01", "03", "06", "41", "43", "51"))


def select_cases(identity):
    ids = np.asarray(identity["scene_id"]).astype(str)
    records = np.asarray(identity["recording_id"]).astype(str)
    counts = np.asarray(identity["num_agents"])
    if (ids.ndim != 1 or records.shape != ids.shape or counts.shape != ids.shape
            or len(set(ids)) != len(ids) or not np.issubdtype(counts.dtype, np.integer)
            or np.any(counts < 3)):
        raise ValueError("unique scene identities and actual integer N>=3 required")
    hashes = [hashlib.sha256((SALT + "|" + sid).encode()).hexdigest() for sid in ids]
    selections, first = [], {}
    groups = [(name, (counts >= lo) & (counts <= hi)) for name, lo, hi in STRATA]
    groups.append(("highest_N", counts == counts.max()))
    for name, mask in groups:
        eligible = np.flatnonzero(mask)
        if not len(eligible):
            selections.append(dict(selection_slot=name, status="empty", eligible_rows=0))
            continue
        row = min(eligible, key=lambda i: (hashes[i], ids[i]))
        sid = str(ids[row])
        selections.append(dict(selection_slot=name, status="selected", scene_id=sid,
            recording_id=str(records[row]), num_agents=int(counts[row]), selection_hash=hashes[row],
            eligible_rows=int(len(eligible)), duplicate_of=first.get(sid),
            score_or_PET_used_for_selection=False))
        first.setdefault(sid, name)
    return selections


def selection_manifest(identity):
    selections = select_cases(identity)
    return dict(salt=SALT, rule="within each N stratum take minimum SHA256(salt|scene_id); highest-N ties use same hash",
        slots=selections, unique_scenes=len({r["scene_id"] for r in selections if r["status"] == "selected"}),
        duplicate_policy="preserve selection slots; render duplicate scene only once; never replace it",
        purposeful_N_stratification=True, model_scores_or_PET_used=False, random_population_sample=False)


def load_scene_case(dataset_binding, selected):
    if dataset_binding["sha256"] != DATA_SHA256:
        raise ValueError("only the frozen expanded complete-scene manifest may supply cases")
    path = verify_binding(dataset_binding)
    manifest = json.loads(path.read_text())
    _base, source = validate_manifest(manifest)
    rec, sid = str(selected["recording_id"]), str(selected["scene_id"])
    if rec not in AUDIT_RECORDINGS:
        raise PermissionError("visual cases are restricted to the six internal-development AUDIT recordings")
    entry = manifest["recordings"][rec]
    if (entry["role"] != "AUDIT" or entry["source_kind"] != "unchanged_original_complete_view"
            or sid not in entry["selected_scene_ids"]
            or entry["data"] != source["recordings"][rec]["artifacts"]["data"]):
        raise ValueError("selected case is not an unchanged E_full AUDIT scene")
    source_path = verify_binding(manifest["source_scene_manifest"])
    shard = verify_binding(entry["data"], source_path.parent / (rec + ".npz"))
    keys = ("scene_id", "recording_id", "role", "offsets", "num_agents", "ego_id", "t0_frame", "agent_ids",
            "history_agents", "dimensions_agents", "future_native_agents", "future_observed_mask_agents",
            "history_frame_ids", "future_frame_ids", "carriageway_boundaries", "carriageway_boundary_mask",
            "pet_value", "complete_horizon", "point_identified", "label_interval_seconds", "label_status")
    # NPZ decodes the selected recording's members, not unrelated record futures.
    with np.load(shard, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in keys}
    verify_binding(entry["data"], shard)
    rows = np.flatnonzero(arrays["scene_id"] == sid)
    if len(rows) != 1:
        raise ValueError("scene must be resolved by identity, never a global prediction row index")
    row = int(rows[0])
    if row not in entry["selected_rows"] or str(arrays["role"][row]) != "AUDIT" or str(arrays["recording_id"][row]) != rec:
        raise ValueError("cached row has a different algorithm role or recording")
    lo, hi = map(int, arrays["offsets"][row:row + 2])
    n = hi - lo
    ids = arrays["agent_ids"][lo:hi].copy()
    h = arrays["history_agents"][lo:hi].transpose(1, 0, 2).copy()
    f = arrays["future_native_agents"][lo:hi].transpose(1, 0, 2).copy()
    mask = arrays["future_observed_mask_agents"][lo:hi].T.copy()
    dims = arrays["dimensions_agents"][lo:hi].copy()
    boundary_mask = arrays["carriageway_boundary_mask"][row]
    bounds = arrays["carriageway_boundaries"][row, boundary_mask].copy()
    hframes, fframes = arrays["history_frame_ids"][row].copy(), arrays["future_frame_ids"][row].copy()
    pet = float(arrays["pet_value"][row])
    if (n != int(selected["num_agents"]) or n != int(arrays["num_agents"][row]) or n < 3
            or len(set(ids.tolist())) != n or int(ids[0]) != int(arrays["ego_id"][row])
            or h.shape != (13, n, 4) or f.shape != (175, n, 4) or mask.shape != (175, n)
            or mask.dtype != np.bool_ or not mask.all() or dims.shape != (n, 2)):
        raise ValueError("all original N vehicles require actual complete H13/F175 and fixed unique identities")
    if (not all(np.isfinite(a).all() for a in (h, f, dims, bounds)) or np.any(dims <= 0)
            or not np.array_equal(h[-1], f[0]) or not np.all(np.diff(bounds) > 0)
            or hframes.shape != (13,) or fframes.shape != (175,)
            or not np.all(np.diff(hframes) == 2) or not np.all(np.diff(fframes) == 1)
            or hframes[-1] != fframes[0] or fframes[0] != arrays["t0_frame"][row]):
        raise ValueError("cached actual coordinates, sizes or native frame axes are inconsistent")
    if (not bool(arrays["complete_horizon"][row]) or not bool(arrays["point_identified"][row])
            or not np.isfinite(pet) or not 0 <= pet <= 4
            or not np.array_equal(arrays["label_interval_seconds"][row], [pet, pet])):
        raise ValueError("case does not have a complete-scene point PET")
    return dict(scene_id=sid, recording_id=rec, source_row=row, num_agents=n, agent_ids=ids,
        history=h, future=f, observed_mask=mask, dimensions=dims, road_boundaries=bounds,
        history_frame_ids=hframes, future_frame_ids=fframes, t0_frame=int(fframes[0]), pet=pet,
        label_status=str(arrays["label_status"][row]), history_dt=.08, future_dt=.04,
        source_bindings=dict(dataset_manifest=dataset_binding, source_manifest=manifest["source_scene_manifest"], shard=entry["data"]),
        provenance=dict(actual_observed_trajectories=True, generated_trajectories=False, raw_CSV_opened=False,
            fixed_t0_context=True, future_selected_critical_actor_input=False, all_N_vehicles_full_horizon_observed=True,
            NPZ_member_decode_scope="selected recording fields; unrelated recording future arrays not decoded",
            source_row_resolved_by_scene_id=True, modified_source_arrays=False, interpolated_or_extrapolated_frames=0,
            eligibility=ELIGIBILITY, estimand=ESTIMAND, evaluation_scope="previously inspected internal development; not final blind test"))


def display_frames(case):
    h, f = np.asarray(case["history"]), np.asarray(case["future"])
    n = int(case["num_agents"])
    if (h.shape != (13, n, 4) or f.shape != (175, n, 4)
            or not np.array_equal(h[-1], f[0]) or not np.asarray(case["observed_mask"]).all()
            or not np.isfinite(h).all() or not np.isfinite(f).all()):
        raise ValueError("display requires the authenticated full-context natural arrays")
    states = np.concatenate((h[:-1], f), axis=0)
    times = np.concatenate((np.arange(-12, 0) * .08, np.arange(175) * .04))
    xy = states[..., :2].copy()
    xy[..., 0] -= states[:, 0:1, 0]
    dims = np.asarray(case["dimensions"])
    bounds = np.asarray(case["road_boundaries"])
    xlim = (float(np.min(xy[..., 0] - dims[None, :, 0] / 2)) - 5,
            float(np.max(xy[..., 0] + dims[None, :, 0] / 2)) + 5)
    ylim = (min(float(bounds[0]), float(np.min(xy[..., 1] - dims[None, :, 1] / 2))) - 1,
            max(float(bounds[-1]), float(np.max(xy[..., 1] + dims[None, :, 1] / 2))) + 1)
    return dict(xy=xy, times=times, duration_ms=[80] * 12 + [40] * 175,
        xlim=xlim, ylim=ylim, dimensions=dims, road_boundaries=bounds, agent_ids=np.asarray(case["agent_ids"]))
