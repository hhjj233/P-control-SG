"""Observed-only, reason-resolved wrapper of the existing focal PET adapter.

The native longitudinal-conflict-point formula is retained. Its cap-valued
sentinels for unobserved crossings are NOT accepted as natural PET labels.
No trajectory is projected, smoothed, simulated, or extrapolated here.
"""
from __future__ import annotations

import numpy as np
from pcontrol.data.road_semantics import infer_lane_ids_from_boundaries, semantic_validity, multiagent_collision_diagnostics
from pcontrol.data.cut_in_pet import cut_in_pet_from_lane_transition, PET_ADAPTER_VERSION

PROTOCOL="natural_observed_focal_pet_complete_occupancy_v1"


def _crossing_time(values,target,dt):
    difference=values-float(target)
    exact=np.flatnonzero(np.isclose(difference,0.,atol=1e-8))
    if exact.size:return float(exact[0])*float(dt)
    crossings=np.flatnonzero(difference[:-1]*difference[1:]<0.)
    if not crossings.size:return None
    i=int(crossings[0]);denominator=values[i+1]-values[i]
    if abs(float(denominator))<1e-12:return None
    return (i+float((target-values[i])/denominator))*float(dt)


def observed_cut_in_pet(future_centers,*,sizes_length_width_m,lane_boundaries_y,dt,target_actor_slot):
    future=np.asarray(future_centers,dtype=np.float64)
    sizes=np.asarray(sizes_length_width_m,dtype=np.float64)
    target=int(target_actor_slot)
    if future.ndim!=3 or future.shape[1:]!=(9,4) or sizes.shape!=(9,2) or target not in (2,8):
        raise ValueError("strict N9 centre-state PET input contract differs")
    if not np.isfinite(future).all() or not np.isfinite(sizes).all() or np.any(sizes<=0) or not np.isfinite(dt) or dt<=0:
        raise ValueError("nonfinite/missing observations or invalid geometry/dt")
    origin=future.copy();origin[...,:2]-=.5*sizes[None]
    lengths,widths=sizes[:,0],sizes[:,1]
    lanes=infer_lane_ids_from_boundaries(origin,np.asarray(lane_boundaries_y),widths=widths)
    mode="left_cut_in" if target==2 else "right_cut_in"
    semantic=semantic_validity(origin,lanes,lengths=lengths,widths=widths,semantic_mode=mode,ego_index=0,actor_index=target)
    collisions=multiagent_collision_diagnostics(origin,lengths=lengths,widths=widths,ego_index=0,target_actor_index=target)
    record=dict(metric_version=PROTOCOL,base_native_adapter=PET_ADAPTER_VERSION,semantic_mode=mode,
        semantic_valid=semantic["valid"],semantic_failure_reasons=semantic["failure_reasons"],
        target_actor_slot=target,pet_raw_seconds=None,pet_value_seconds=None,native_legacy_cap_value=None,
        pet_status="invalid",reference_eligible=False,cap_flag=False,zero_atom_flag=False,
        focal_collision=collisions["ego_target_collision"],
        secondary_collision_count=sum(tuple(pair)!=(0,target) for pair in collisions["collided_pairs"]),
        collision_diagnostics=collisions,collision_overrides_pet=False,
        trajectory_modified=False,occupation_times=None,transition_index=None)
    if not semantic["valid"]:
        record["reason"]="outside_declared_natural_cut_in_population"
        return record
    ego_lane=lanes[0,0];transition=int(np.flatnonzero(lanes[:,target]==ego_lane)[0])
    record["transition_index"]=transition
    native=cut_in_pet_from_lane_transition(origin,lanes,lengths=lengths,dt=dt,ego_index=0,actor_index=target,cap_s=4.)
    record["native_legacy_cap_value"]=float(native)
    # Use the adapter's exact origin->centre operation order, including the
    # box-centre conflict point at first in-lane sample.
    center=origin[...,0]+.5*lengths[None]
    point=float(center[transition,target]);intervals=[]
    for actor in (0,target):
        front=_crossing_time(center[:,actor]+.5*lengths[actor],point,dt)
        rear=_crossing_time(center[:,actor]-.5*lengths[actor],point,dt)
        if front is None or rear is None:
            record.update(reason="conflict_point_occupancy_not_fully_observed",conflict_point_x_m=point)
            return record
        intervals.append((min(front,rear),max(front,rear)))
    a,b=intervals
    raw=b[0]-a[1] if a[1]<b[0] else a[0]-b[1] if b[1]<a[0] else 0.
    value=float(np.clip(raw,0.,4.))
    if value!=native:raise RuntimeError("reason-resolved natural PET differs from frozen native formula")
    record.update(pet_raw_seconds=float(raw),pet_value_seconds=value,
        pet_status="capped" if raw>=4. else "finite",reference_eligible=True,
        cap_flag=raw>=4.,zero_atom_flag=value==0.,reason="complete_observed_occupancy",
        occupation_times=[list(v) for v in intervals],conflict_point_x_m=point)
    return record
