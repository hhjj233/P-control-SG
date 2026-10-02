"""Exact scene-PET broadphase accelerator; canonical definition is unchanged.

Every actor remains in pair_results. Only provably disjoint swept AABBs skip
the original segment search. Extents use the canonical segment arithmetic,
including its floating-point endpoint reconstruction and closed-box contact.
The old scorer stays untouched and remains the reporting oracle.
"""
import numpy as np
from . import scene_pet as canonical
PROTOCOL=canonical.PROTOCOL
_grid=canonical._grid
_result=canonical._result


def pair_occupancy_pet(ego,other,*,times,ego_mask,other_mask,ego_dimensions,other_dimensions,
                       sample_period=.04,window=(0.,6.96),cap_seconds=4.):
    times=_grid(times,sample_period,window,cap_seconds)
    ego,ego_mask,ed=canonical._actor(ego,ego_mask,ego_dimensions,times)
    other,other_mask,od=canonical._actor(other,other_mask,other_dimensions,times)
    left=canonical._segments(ego,ego_mask,times);right=canonical._segments(other,other_mask,times)
    if len(left['time']) and len(right['time']):
        half=.5*(ed+od)
        # Keep the exact operand order used in the canonical per-segment test.
        low=np.minimum(left['p'],left['p']+left['delta']).min(0)
        high=np.maximum(left['p'],left['p']+left['delta']).max(0)
        rlow=np.minimum(right['p'],right['p']+right['delta']).min(0)
        rhigh=np.maximum(right['p'],right['p']+right['delta']).max(0)
        if np.any((low>rhigh+half)|(high<rlow-half)):
            return _result(float('inf'),bool(ego_mask.all() and other_mask.all()),None,cap_seconds,
                dict(ego_pieces=int(len(left['time'])),other_pieces=int(len(right['time'])),
                     broadphase_candidates=0,solved_segment_pairs=0))
    return canonical.pair_occupancy_pet(ego,other,times=times,ego_mask=ego_mask,other_mask=other_mask,
        ego_dimensions=ed,other_dimensions=od,sample_period=sample_period,window=window,cap_seconds=cap_seconds)


def scene_occupancy_pet(future, observed_mask, dimensions, *, times,
                        ego_index=0, sample_period=.04, window=(0., 6.96),
                        cap_seconds=4.):
    """Minimum over every supplied non-ego actor; never select a focal input.

    The caller freezes the full actor roster from observed history. This
    function does not silently drop missing actors or add future-selected IDs.
    All pair results are retained, including intervals and observed witnesses.
    """
    future = np.asarray(future, dtype=np.float64)
    observed_mask = np.asarray(observed_mask)
    dimensions = np.asarray(dimensions, dtype=np.float64)
    times = _grid(times, sample_period, window, cap_seconds)
    if (future.ndim != 3 or future.shape[0] != times.size or future.shape[2] != 4
            or future.shape[1] < 2 or observed_mask.shape != future.shape[:2]
            or dimensions.shape != (future.shape[1], 2) or observed_mask.dtype != np.bool_
            or not isinstance(ego_index, (int, np.integer)) or not 0 <= ego_index < future.shape[1]):
        raise ValueError("scene needs [T,N>=2,4], bool[T,N], dimensions[N,2], valid ego index")
    pairs, best, witness = [], float('inf'), None
    for actor in range(future.shape[1]):
        if actor == ego_index:
            continue
        result = pair_occupancy_pet(future[:, ego_index], future[:, actor], times=times,
            ego_mask=observed_mask[:, ego_index], other_mask=observed_mask[:, actor],
            ego_dimensions=dimensions[ego_index], other_dimensions=dimensions[actor],
            sample_period=sample_period, window=window, cap_seconds=cap_seconds)
        result['other_index'] = actor
        pairs.append(result)
        if result['observed_min_seconds'] < best:
            best = result['observed_min_seconds']
            witness = dict(result['witness'], other_index=actor, ego_index=int(ego_index))
    result = _result(best, bool(observed_mask.all()), witness, cap_seconds,
                     dict(num_agents=int(future.shape[1]), num_pairs=len(pairs)))
    observed_critical = None if witness is None else witness['other_index']
    result.update(ego_index=int(ego_index), pair_results=pairs,
                  observed_critical_other_index=observed_critical,
                  critical_other_index=observed_critical if result['point_identified'] else None)
    return result
