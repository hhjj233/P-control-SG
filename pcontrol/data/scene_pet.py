"""Window-restricted minimum pointwise AABB occupancy PET.

For each ego--other pair, minimize |s-t| over observed piecewise-linear
segments whose axis-aligned vehicle boxes share a spatial point. This is a
new measurement, NOT the legacy cut-in entry-x PET or an SSAM implementation.
Closed boxes include boundary contact. No interpolation crosses a missing
frame, no trajectory is exported, and no observation is extrapolated.

Completeness is relative to the supplied *full*, uniformly sampled window.
The default contract is original 25 Hz observations on [0, 6.96] seconds.
With missing observations the observed minimum is an upper bound, not a
point label; only an observed zero certifies the capped label despite gaps.
"""
from __future__ import annotations

import numpy as np


PROTOCOL = "natural_window_min_pointwise_aabb_occupancy_pet_v1"


def _grid(times, sample_period, window, cap_seconds):
    times = np.asarray(times, dtype=np.float64)
    window = np.asarray(window, dtype=np.float64)
    if (times.ndim != 1 or times.size < 2 or window.shape != (2,)
            or not np.isfinite(times).all() or not np.isfinite(window).all()
            or not np.isfinite(sample_period) or sample_period <= 0
            or not np.isfinite(cap_seconds) or cap_seconds <= 0
            or window[1] <= window[0]):
        raise ValueError("finite positive timing/cap and a complete time grid required")
    expected = window[0] + np.arange(times.size) * sample_period
    if (not np.allclose(times, expected, rtol=0, atol=1e-10)
            or abs(times[-1] - window[1]) > 1e-10):
        raise ValueError("times must retain every original sample across the full window")
    return times


def _actor(observations, mask, dimensions, times):
    values = np.asarray(observations, dtype=np.float64)
    mask = np.asarray(mask)
    dims = np.asarray(dimensions, dtype=np.float64)
    if (values.shape != (times.size, 4) or mask.shape != (times.size,)
            or mask.dtype != np.bool_ or dims.shape != (2,)
            or not np.isfinite(dims).all() or np.any(dims <= 0)):
        raise ValueError("actor needs [T,4], boolean mask[T], positive [length,width]")
    if not np.isfinite(values[mask]).all():
        raise ValueError("observed actor rows must be finite; mark missing rows invalid")
    return values, mask, dims


def _segments(values, mask, times):
    """Adjacent real frames, plus otherwise-unrepresented valid singletons."""
    starts = np.flatnonzero(mask[:-1] & mask[1:])
    covered = np.zeros(mask.size, dtype=bool)
    covered[starts] = True
    covered[starts + 1] = True
    isolated = np.flatnonzero(mask & ~covered)
    first = np.concatenate((starts, isolated))
    last = np.concatenate((starts + 1, isolated))
    order = np.argsort(first, kind="stable")
    first, last = first[order], last[order]
    return dict(p=values[first, :2], delta=values[last, :2] - values[first, :2],
                time=times[first], duration=times[last] - times[first],
                first=first, last=last)


def _clip_polygon(polygon, a, b, c):
    """Clip a convex polygon by a*u+b*v <= c, without a geometry epsilon."""
    if not polygon:
        return []
    output = []
    previous = polygon[-1]
    previous_value = a * previous[0] + b * previous[1] - c
    for current in polygon:
        current_value = a * current[0] + b * current[1] - c
        if (current_value <= 0) != (previous_value <= 0):
            ratio = previous_value / (previous_value - current_value)
            output.append(previous + ratio * (current - previous))
        if current_value <= 0:
            output.append(current)
        previous, previous_value = current, current_value
    return output


def _segment_minimum(left, right, i, j, half_sizes):
    polygon = [np.array([0., 0.]), np.array([1., 0.]),
               np.array([1., 1.]), np.array([0., 1.])]
    displacement = left['p'][i] - right['p'][j]
    for axis in (0, 1):
        a, b = left['delta'][i, axis], -right['delta'][j, axis]
        polygon = _clip_polygon(polygon, a, b, half_sizes[axis] - displacement[axis])
        polygon = _clip_polygon(polygon, -a, -b, half_sizes[axis] + displacement[axis])
        if not polygon:
            return None
    vertices = np.asarray(polygon)
    differences = (left['time'][i] - right['time'][j]
                   + vertices[:, 0] * left['duration'][i]
                   - vertices[:, 1] * right['duration'][j])
    lo, hi = int(np.argmin(differences)), int(np.argmax(differences))
    if differences[lo] <= 0 <= differences[hi]:
        distance = 0.
        if differences[hi] == differences[lo]:
            uv = vertices[lo]
        else:
            fraction = -differences[lo] / (differences[hi] - differences[lo])
            uv = vertices[lo] + fraction * (vertices[hi] - vertices[lo])
    else:
        nearest = lo if differences[lo] > 0 else hi
        distance, uv = float(abs(differences[nearest])), vertices[nearest]
    return distance, uv


def _result(observed_min, complete, witness, cap_seconds, diagnostics):
    upper = float(min(observed_min, cap_seconds))
    certified = complete or observed_min == 0.
    if observed_min == 0:
        status = 'exact_zero' if complete else 'certified_zero_with_missing'
    elif not complete:
        status = 'partial_observation_interval'
    elif np.isinf(observed_min):
        status = 'complete_no_shared_occupancy'
    elif observed_min >= cap_seconds:
        status = 'complete_capped'
    else:
        status = 'complete_finite'
    return dict(metric_version=PROTOCOL, observed_min_seconds=float(observed_min),
                observed_min_capped_seconds=upper, complete=bool(complete),
                point_identified=bool(certified), status=status,
                pet_value_seconds=upper if certified else None,
                label_interval_seconds=[upper, upper] if certified else [0., upper],
                cap_seconds=float(cap_seconds), witness=witness,
                interpolation='linear_between_adjacent_observed_original_frames_only',
                diagnostics=diagnostics)


def pair_occupancy_pet(ego, other, *, times, ego_mask, other_mask,
                       ego_dimensions, other_dimensions, sample_period=.04,
                       window=(0., 6.96), cap_seconds=4.):
    """Measure a pair's observed minimum and conservative missing-data interval.

    ``observed_min_seconds=inf`` means no shared occupancy in the observed
    pieces. It identifies a capped point label only when both full masks are true.
    Positions are box centres; dimensions are static longitudinal length and
    lateral width. Velocities are validated but not integrated or extrapolated.
    """
    times = _grid(times, sample_period, window, cap_seconds)
    ego, ego_mask, ego_dims = _actor(ego, ego_mask, ego_dimensions, times)
    other, other_mask, other_dims = _actor(other, other_mask, other_dimensions, times)
    left, right = _segments(ego, ego_mask, times), _segments(other, other_mask, times)
    half_sizes = .5 * (ego_dims + other_dims)
    best, witness, tested, candidates = float('inf'), None, 0, 0
    right_low = np.minimum(right['p'], right['p'] + right['delta'])
    right_high = np.maximum(right['p'], right['p'] + right['delta'])
    for i in range(left['time'].size):
        low = np.minimum(left['p'][i], left['p'][i] + left['delta'][i])
        high = np.maximum(left['p'][i], left['p'][i] + left['delta'][i])
        spatial = np.all((low <= right_high + half_sizes)
                         & (high >= right_low - half_sizes), axis=1)
        time_bound = np.maximum.reduce((
            np.zeros(right['time'].size),
            left['time'][i] - right['time'] - right['duration'],
            right['time'] - left['time'][i] - left['duration'][i]))
        possible = np.flatnonzero(spatial & (time_bound < best))
        candidates += int(possible.size)
        for j in possible[np.argsort(time_bound[possible], kind='stable')]:
            if time_bound[j] >= best:
                continue
            tested += 1
            answer = _segment_minimum(left, right, i, j, half_sizes)
            if answer is None or answer[0] >= best:
                continue
            best, uv = answer
            s = left['time'][i] + uv[0] * left['duration'][i]
            t = right['time'][j] + uv[1] * right['duration'][j]
            ego_point = left['p'][i] + uv[0] * left['delta'][i]
            other_point = right['p'][j] + uv[1] * right['delta'][j]
            overlap_low = np.maximum(ego_point - ego_dims / 2, other_point - other_dims / 2)
            overlap_high = np.minimum(ego_point + ego_dims / 2, other_point + other_dims / 2)
            witness = dict(ego_time_seconds=float(s), other_time_seconds=float(t),
                           shared_point_xy_m=((overlap_low + overlap_high) / 2).tolist(),
                           ego_segment_frames=[int(left['first'][i]), int(left['last'][i])],
                           other_segment_frames=[int(right['first'][j]), int(right['last'][j])],
                           segment_coordinates_uv=uv.tolist())
            if best == 0:
                break
        if best == 0:
            break
    return _result(best, bool(ego_mask.all() and other_mask.all()), witness,
                   cap_seconds, dict(ego_pieces=int(left['time'].size),
                       other_pieces=int(right['time'].size),
                       broadphase_candidates=candidates, solved_segment_pairs=tested))


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
