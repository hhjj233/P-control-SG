"""Descriptive native-frame scene quality; no risk optimization or data access."""
import numpy as np


def longest_sample_span(mask, dt):
    """Elapsed span of longest run of marked samples, not continuous certification."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or not np.isfinite(dt) or dt <= 0:
        raise ValueError("one temporal mask and positive dt required")
    edges = np.diff(np.r_[False, mask, False].astype(int))
    lengths = np.flatnonzero(edges == -1)-np.flatnonzero(edges == 1)
    return float(max(int(lengths.max())-1, 0)*dt) if len(lengths) else 0.


def scene_quality(future, dimensions, ego_mask, dt=.04):
    f = np.asarray(future, dtype=np.float64); d = np.asarray(dimensions, dtype=np.float64); ego = np.asarray(ego_mask)
    if (f.ndim != 3 or f.shape[0] < 3 or f.shape[1] < 2 or f.shape[2] != 4
            or d.shape != (f.shape[1], 2) or ego.shape != (f.shape[1],) or ego.dtype != bool or ego.sum() != 1
            or not np.isfinite(f).all() or not np.isfinite(d).all() or np.any(d <= 0)
            or not np.isfinite(dt) or dt <= 0):
        raise ValueError("finite multi-agent trajectory, sizes, one ego, and positive dt required")
    e = int(np.flatnonzero(ego)[0]); result = dict(num_agents=f.shape[1], frames=f.shape[0], dt_seconds=float(dt))
    speed = np.linalg.norm(f[..., 2:4], axis=-1)
    acceleration = np.linalg.norm(np.diff(f[..., 2:4], axis=0)/dt, axis=-1)
    jerk = np.linalg.norm(np.diff(f[..., 2:4], n=2, axis=0)/dt**2, axis=-1)
    for name, mask in (("all", np.ones(len(d), bool)), ("ego", ego), ("background", ~ego)):
        for variable, array in (("speed_mps", speed), ("acceleration_mps2", acceleration), ("jerk_mps3", jerk)):
            values = array[:, mask]
            result[f"{name}_{variable}_mean"] = float(values.mean())
            result[f"{name}_{variable}_p95"] = float(np.quantile(values, .95))
            result[f"{name}_{variable}_rms"] = float(np.sqrt(np.mean(values**2)))
            result[f"{name}_{variable}_max"] = float(values.max())
    i, j = np.triu_indices(len(d), 1)
    delta = np.abs(f[:, i, :2]-f[:, j, :2])-.5*(d[i]+d[j])[None]
    surface = np.linalg.norm(np.maximum(delta, 0.), axis=-1)
    lateral_overlap = delta[..., 1] <= 0.
    long_gap = np.maximum(delta[..., 0], 0.)
    for name, pair_mask in (("ego_environment", (i == e) | (j == e)), ("background_pairs", (i != e) & (j != e))):
        n = int(pair_mask.sum()); prefix = name+"_"
        result[prefix+"pair_count"] = n
        if not n:
            for key in ("nearest_surface_min_m", "nearest_surface_p05_m", "longitudinal_gap_min_m"):
                result[prefix+key] = None
        else:
            nearest = surface[:, pair_mask].min(1)
            result[prefix+"nearest_surface_min_m"] = float(nearest.min())
            result[prefix+"nearest_surface_p05_m"] = float(np.quantile(nearest, .05))
            valid = lateral_overlap[:, pair_mask]
            result[prefix+"longitudinal_gap_min_m"] = float(long_gap[:, pair_mask][valid].min()) if valid.any() else None
        for threshold in (1., 2.):
            close = lateral_overlap[:, pair_mask] & (long_gap[:, pair_mask] < threshold)
            any_close = close.any(1)
            key = prefix+f"long_gap_under{threshold:g}m_"
            result[key+"frame_fraction"] = float(any_close.mean())
            result[key+"longest_sample_span_seconds"] = longest_sample_span(any_close, dt)
            result[key+"pair_count"] = int(close.any(0).sum())
    return result
