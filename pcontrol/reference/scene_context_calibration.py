"""Small history-conditioned monotone warps and mixed-CDF diagnostics.

No file access. Reuses frozen exact physical-piece kernels without changing
their source. Context is history-only; deterministic convex combinations of
monotone maps are not empirical-neighborhood reference distributions.
"""
from dataclasses import dataclass

import numpy as np

from .scene_calibration import (
    SceneCDFWarp, U_KNOTS, _base, _query, _physical_pieces, _piece_basis,
    precompute_crps_quadratic,
)

PROTOCOL = "natural_scene_context_monotone_CRPS_warp_v2"
CONTEXT_NAMES = ("num_agents", "ego_speed_mps", "longitudinal_gap_m", "ego_lateral_speed_mps")
FAMILIES = {"global": 1, "count": 2, "speed": 2, "count_speed": 4, "count_gap": 4}
GATE_SPEC = "count=clip((N-5)/8);speed=clip((v-5)/25);gap=clip(g/40);all_clip_0_1"


def history_context(features):
    """Accept only one unpadded example's predictor features, in SI units."""
    if set(features) != {"history", "dimensions", "road_boundaries", "ego_mask"}:
        raise ValueError("literal history-only feature allowlist required")
    h = np.asarray(features["history"], dtype=np.float64)
    d = np.asarray(features["dimensions"], dtype=np.float64)
    ego = np.asarray(features["ego_mask"])
    if (h.ndim != 3 or h.shape[0] != 13 or h.shape[2] != 4 or h.shape[1] < 2
            or d.shape != (h.shape[1], 2) or ego.dtype != bool
            or ego.shape != (h.shape[1],) or ego.sum() != 1
            or not np.isfinite(h).all() or not np.isfinite(d).all() or np.any(d <= 0)):
        raise ValueError("finite complete H13, dimensions and one ego required")
    e = int(np.flatnonzero(ego)[0]); last = h[-1]
    # Dimension order is longitudinal length, lateral width. A vehicle in a
    # disjoint lateral footprint is not a longitudinal leader/follower here.
    lateral = np.abs(last[:, 1]-last[e, 1]) <= .5*(d[:, 1]+d[e, 1])
    eligible = lateral & ~ego
    gaps = np.maximum(np.abs(last[:, 0]-last[e, 0])-.5*(d[:, 0]+d[e, 0]), 0.)
    gap = float(np.min(gaps[eligible])) if eligible.any() else 120.
    return np.array([h.shape[1], np.linalg.norm(last[e, 2:4]), gap,
                     abs(last[e, 3])], dtype=np.float64)


def validate_context(context):
    c = np.asarray(context, dtype=np.float64)
    if (c.ndim != 2 or c.shape[1] != len(CONTEXT_NAMES) or not len(c)
            or not np.isfinite(c).all() or np.any(c < 0)
            or np.any(c[:, 0] < 2) or np.any(c[:, 0] != np.floor(c[:, 0]))):
        raise ValueError("context must be finite nonnegative [rows, N/speed/gap/lateral_speed]")
    return c


def context_weights(context, family):
    c = validate_context(context)
    if family not in FAMILIES:
        raise ValueError("unknown context warp family")
    n = np.clip((c[:, 0]-5.)/8., 0., 1.)
    v = np.clip((c[:, 1]-5.)/25., 0., 1.)
    g = np.clip(c[:, 2]/40., 0., 1.)
    if family == "global":
        return np.ones((len(c), 1))
    if family in ("count", "speed"):
        t = n if family == "count" else v
        return np.column_stack((1.-t, t))
    t = v if family == "count_speed" else g
    return np.column_stack(((1.-n)*(1.-t), (1.-n)*t, n*(1.-t), n*t))


def interval_quadratic(masses, counts, observed_pet, upper, *, base_knots=None):
    """Integral over [0,upper] for ALL outcomes, including those above upper.

    Returns row quadratics for the seven internal nodes of one effective warp.
    Exact Gauss3 on all base/warp/observation/threshold breakpoints.
    """
    masses, counts, knots = _base(masses, counts, base_knots)
    y = np.broadcast_to(np.asarray(observed_pet, dtype=np.float64), (len(masses),))
    if (not np.isfinite(upper) or not 0 < upper <= knots[-1]
            or not np.isfinite(y).all() or np.any((y < 0) | (y > knots[-1]))):
        raise ValueError("valid fixed integration limit and observed support required")
    A = np.zeros((len(y), 7, 7)); b = np.zeros((len(y), 7)); const = np.zeros(len(y))
    gx, gw = np.polynomial.legendre.leggauss(3)
    for i, (mass, target) in enumerate(zip(masses, y)):
        physical, left, right = _physical_pieces(mass, knots, U_KNOTS)
        points = np.r_[physical, target, upper]
        points = np.unique(points[(points >= 0) & (points <= upper)])
        lengths = np.diff(points)
        at = (.5*(points[1:]+points[:-1])[:, None]+.5*lengths[:, None]*gx).ravel()
        w = (.5*lengths[:, None]*gw).ravel()
        basis = _piece_basis(physical, left, right, at)
        X = basis[:, 1:-1]; offset = basis[:, -1]-(target <= at)
        A[i] = (X.T*w)@X; b[i] = 2*X.T@(w*offset); const[i] = np.sum(w*offset**2)
    return dict(A=A, b=b, c=const, upper_seconds=float(upper))


def evaluate_quadratic(quadratic, effective_nodes):
    theta = np.asarray(effective_nodes, dtype=np.float64)[:, 1:-1]
    if theta.shape != quadratic["b"].shape:
        raise ValueError("one effective warp per observation required")
    return np.einsum("ni,nij,nj->n", theta, quadratic["A"], theta) + np.sum(quadratic["b"]*theta, axis=1)+quadratic["c"]


@dataclass(frozen=True)
class ContextCDFWarp:
    family: str
    node_values: np.ndarray

    def __post_init__(self):
        nodes = np.array(self.node_values, dtype=np.float64, copy=True)
        if (self.family not in FAMILIES or nodes.shape != (FAMILIES[self.family], 9)
                or not np.isfinite(nodes).all() or np.any(nodes[:, 0] != 0)
                or np.any(nodes[:, -1] != 1) or np.any(np.diff(nodes, axis=1) < 0)):
            raise ValueError("monotone component warps with exact endpoints required")
        nodes.setflags(write=False); object.__setattr__(self, "node_values", nodes)

    @classmethod
    def identity(cls, family="global"):
        if family not in FAMILIES:
            raise ValueError("unknown family")
        return cls(family, np.tile(U_KNOTS, (FAMILIES[family], 1)))

    @classmethod
    def from_legacy(cls, warp):
        if not isinstance(warp, SceneCDFWarp):
            raise TypeError("frozen SceneCDFWarp required")
        return cls(warp.family, warp.node_values)

    def as_dict(self):
        return dict(protocol=PROTOCOL, family=self.family, node_values=self.node_values.tolist(),
                    u_knots=U_KNOTS.tolist(), context_names=list(CONTEXT_NAMES), gate_spec=GATE_SPEC)

    @classmethod
    def from_dict(cls, value):
        if (set(value) != {"protocol", "family", "node_values", "u_knots", "context_names", "gate_spec"}
                or value["protocol"] != PROTOCOL or value["u_knots"] != U_KNOTS.tolist()
                or value["context_names"] != list(CONTEXT_NAMES) or value["gate_spec"] != GATE_SPEC):
            raise ValueError("wrong context warp schema or gate specification")
        return cls(value["family"], value["node_values"])

    def row_nodes(self, context):
        nodes = context_weights(context, self.family)@self.node_values
        # Constant endpoints are a model invariant, not fitted probabilities.
        nodes[:, 0] = 0.; nodes[:, -1] = 1.
        return nodes

    def _rows(self, method, masses, context, query, *, base_knots=None, **kwargs):
        c = validate_context(context); m, _, knots = _base(masses, c[:, 0], base_knots)
        if len(c) != len(m):
            raise ValueError("one context per distribution row required")
        q = _query(query, len(m)); nodes = self.row_nodes(c)
        if method == "quantile" and (not np.isfinite(q).all() or np.any((q < 0) | (q > 1))):
            raise ValueError("quantile levels must be finite in [0,1]")
        result = np.empty_like(q)
        for i, node in enumerate(nodes):
            physical, left_basis, right_basis = _physical_pieces(m[i], knots, U_KNOTS)
            if method == "cdf":
                basis = _piece_basis(physical, left_basis, right_basis, q[i], kwargs.get("side", "right"))
                result[i] = np.sum(basis*node, axis=-1)
            else:
                # Use the SAME explicit reduction as CDF evaluation. Mixing a
                # vector dot and a matrix GEMV can differ by one ulp at an atom
                # boundary, wrongly sending an exact cap-left query to the cap.
                left = np.sum(left_basis*node, axis=-1)
                right = np.sum(right_basis*node, axis=-1)
                levels = q[i].ravel()
                index = np.argmax(right[None] >= levels[:, None], axis=-1)
                previous = np.maximum(index-1, 0)
                lo, hi = right[previous], left[index]
                values = physical[previous]+(levels-lo)/np.where(hi > lo, hi-lo, 1.)*(physical[index]-physical[previous])
                values = np.where((index == 0) | (levels > left[index]), physical[index], values)
                values = np.where(levels <= right[0], 0., values)
                values = np.where(levels == 0, 0., np.where(levels == 1, knots[-1], values))
                result[i] = values.reshape(q[i].shape)
        return result

    def cdf(self, masses, context, y, *, side="right", base_knots=None):
        return self._rows("cdf", masses, context, y, side=side, base_knots=base_knots)

    def quantile(self, masses, context, u, *, base_knots=None):
        return self._rows("quantile", masses, context, u, base_knots=base_knots)

    def rank(self, masses, context, y, *, base_knots=None):
        c = validate_context(context)
        m, _, knots = _base(masses, c[:, 0], base_knots); y = _query(y, len(m))
        if not np.isfinite(y).all() or np.any((y < 0) | (y > knots[-1])):
            raise ValueError("rank requires finite observed support without clipping")
        left = self.cdf(m, c, y, side="left", base_knots=knots)
        right = self.cdf(m, c, y, base_knots=knots)
        return dict(p_low=1.-right, p_up=1.-left, p_mid=1.-.5*(left+right))

    def crps(self, masses, context, y, *, base_knots=None):
        c = validate_context(context)
        q = precompute_crps_quadratic(masses, c[:, 0], y, family="global", base_knots=base_knots)
        return evaluate_quadratic(q, self.row_nodes(c))


def fit_context_warp(quadratic, context, *, family, ridge, maxiter=1000):
    """Convex CRPS fit; caller supplies calibration rows, no hidden data access."""
    from scipy.optimize import minimize
    weights = context_weights(context, family); k = weights.shape[1]
    if (len(weights) != len(quadratic["c"]) or not np.isfinite(ridge) or ridge < 0
            or type(maxiter) is not int or maxiter < 1):
        raise ValueError("paired rows and finite nonnegative regularization required")
    cap = float(quadratic["cap_seconds"])
    A = np.einsum("nk,nl,nij->kilj", weights, weights, quadratic["A"]).reshape(7*k, 7*k)/(len(weights)*cap)
    b = np.einsum("nk,ni->ki", weights, quadratic["b"]).ravel()/(len(weights)*cap)
    const = float(np.mean(quadratic["c"])/cap)
    identity = np.tile(U_KNOTS[1:-1], k)
    constraint = np.zeros((6*k, 7*k))
    for block in range(k):
        for j in range(6):
            constraint[block*6+j, block*7+j] = -1.
            constraint[block*6+j, block*7+j+1] = 1.

    def objective(theta):
        delta = theta-identity
        return float(theta@A@theta+b@theta+const+.5*ridge*(delta@delta))

    def gradient(theta):
        return (A+A.T)@theta+b+ridge*(theta-identity)

    result = minimize(objective, identity, jac=gradient, method="SLSQP",
                      bounds=[(0., 1.)]*(7*k), constraints=dict(type="ineq", fun=lambda t: constraint@t,
                                                             jac=lambda t: constraint),
                      options=dict(maxiter=maxiter, ftol=1e-12, disp=False))
    raw = np.asarray(result.x)
    if not np.isfinite(raw).all() or not np.isfinite(result.fun):
        raise FloatingPointError("nonfinite context warp fit")
    violation = max(0., float(-raw.min()), float(raw.max()-1.), float(-(constraint@raw).min()))
    if violation > 1e-10:
        raise FloatingPointError("materially infeasible monotone calibration")
    theta = np.maximum.accumulate(np.clip(raw.reshape(k, 7), 0., 1.), axis=1)
    nodes = np.column_stack((np.zeros(k), theta, np.ones(k)))
    report = dict(success=bool(result.success), status=int(result.status), message=str(result.message),
                  rows=len(weights), family=family, ridge=float(ridge), parameter_count=7*k,
                  iterations=int(result.nit), objective="mean_CRPS_over_cap_plus_identity_ridge",
                  initial_objective=objective(identity), final_objective=objective(theta.ravel()),
                  parameter_polish_max=float(np.max(np.abs(theta.ravel()-raw))),
                  effective_component_weight_sums=weights.sum(0).tolist(),
                  labels_modified=False, future_or_recording_features=False)
    return ContextCDFWarp(family, nodes), report


def expected_pit_contributions(left, right, levels):
    left = np.asarray(left); right = np.asarray(right); levels = np.asarray(levels)
    if (left.ndim != 1 or right.shape != left.shape or not np.isfinite(left).all()
            or not np.isfinite(right).all() or np.any(left > right) or np.any(left < 0) or np.any(right > 1)):
        raise ValueError("valid paired CDF sides required")
    width = right-left; atom = width > 0
    return np.where(atom[:, None], np.clip((levels[None]-left[:, None])/np.where(atom, width, 1.)[:, None], 0., 1.),
                    right[:, None] <= levels[None])


def diagnostic_masks(context):
    c = validate_context(context)
    return {
        "N": {"N3_5": c[:, 0] < 6, "N6_8": (c[:, 0] >= 6) & (c[:, 0] < 9), "N9_plus": c[:, 0] >= 9},
        "speed": {"v_under10": c[:, 1] < 10, "v10_25": (c[:, 1] >= 10) & (c[:, 1] < 25), "v25_plus": c[:, 1] >= 25},
        "gap": {"gap_under5": c[:, 2] < 5, "gap5_20": (c[:, 2] >= 5) & (c[:, 2] < 20), "gap20_plus": c[:, 2] >= 20},
    }


def diagnostic_arrays(masses, context, y, nodes, policy, quadratics):
    """Score one effective mixed CDF per row; no future-dependent selection."""
    c = validate_context(context); m, _, _ = _base(masses, c[:, 0], None)
    if len(c) != len(m) or np.asarray(y).shape != (len(m),) or nodes.shape != (len(m), 9):
        raise ValueError("row nodes are misaligned")
    thresholds = np.asarray(policy["thresholds_seconds"])
    levels = np.asarray(policy["quantile_levels"])
    result = {key: np.empty(len(m)) for key in ("cdf_left", "cdf_right", "zero_mass", "cap_mass")}
    result["threshold_cdf"] = np.empty((len(m), len(thresholds)))
    result["quantiles"] = np.empty((len(m), len(levels)))
    for i, node in enumerate(nodes):
        w = ContextCDFWarp("global", node[None]); mass = m[i:i+1]; row_context = c[i:i+1]
        result["cdf_left"][i] = w.cdf(mass, row_context, y[i], side="left")[0]
        result["cdf_right"][i] = w.cdf(mass, row_context, y[i])[0]
        result["zero_mass"][i] = w.cdf(mass, row_context, 0.)[0]
        result["cap_mass"][i] = 1.-w.cdf(mass, row_context, 4., side="left")[0]
        result["threshold_cdf"][i] = w.cdf(mass, row_context, thresholds[None])[0]
        result["quantiles"][i] = w.quantile(mass, row_context, levels[None])[0]
    for name, quadratic in quadratics.items():
        result[name] = evaluate_quadratic(quadratic, nodes)
    residual = y[:, None]-result["quantiles"]
    result["pinball"] = np.maximum(levels*residual, (levels-1.)*residual)
    result["pit_contribution"] = expected_pit_contributions(result["cdf_left"], result["cdf_right"], policy["PIT_grid"])
    return result


def summarize_diagnostics(arrays, y, context, records, policy):
    """All rows remain in proper scores; grouped metrics are descriptive."""
    y = np.asarray(y); c = validate_context(context); records = np.asarray(records)
    thresholds = np.asarray(policy["thresholds_seconds"]); grid = np.asarray(policy["PIT_grid"])
    levels = np.asarray(policy["quantile_levels"])

    def score(mask):
        count = int(mask.sum())
        if not count:
            return dict(scenes=0)
        values = {name: float(a[mask].mean()) for name, a in arrays.items() if name.startswith("CRPS") or name.startswith("twCRPS")}
        pred = arrays["threshold_cdf"][mask].mean(0); obs = (y[mask, None] <= thresholds).mean(0)
        pit = arrays["pit_contribution"][mask].mean(0)
        below = (y[mask, None] < arrays["quantiles"][mask]).mean(0)
        at_or_below = (y[mask, None] <= arrays["quantiles"][mask]).mean(0)
        gap = np.maximum.reduce((below-levels, levels-at_or_below, np.zeros_like(levels)))
        values.update(scenes=count, recordings=len(np.unique(records[mask])),
                      expected_PIT_grid_MAE=float(np.abs(pit-grid).mean()),
                      expected_PIT_grid_KS=float(np.abs(pit-grid).max()),
                      threshold_MAE=float(np.abs(pred-obs).mean()), threshold_predicted=pred.tolist(), threshold_observed=obs.tolist(),
                      expected_PIT_cdf=pit.tolist(), pinball=arrays["pinball"][mask].mean(0).tolist(),
                      quantile_coverage_gap=gap.tolist(), zero_Brier=float(np.mean((arrays["zero_mass"][mask]-(y[mask] == 0))**2)),
                      cap_Brier=float(np.mean((arrays["cap_mass"][mask]-(y[mask] == 4))**2)))
        values["local_component_score"] = values["expected_PIT_grid_MAE"]+2*values["threshold_MAE"]
        return values

    overall = score(np.ones(len(y), bool))
    groups = {axis: {name: score(mask) for name, mask in bins.items()} for axis, bins in diagnostic_masks(c).items()}
    components = [overall["local_component_score"]]
    included = {}
    for axis, bins in groups.items():
        eligible = [name for name, item in bins.items() if item["scenes"] >= policy["diagnostic_groups"]["minimum_rows"]]
        included[axis] = eligible
        if eligible:
            components.append(float(np.mean([bins[name]["local_component_score"] for name in eligible])))
    return dict(overall=overall, groups=groups,
                by_recording={str(r): score(records == r) for r in np.unique(records)},
                context_selection_score=float(np.mean(components)), groups_used_for_selection=included,
                small_groups_reported_but_not_used_for_selection=True,
                per_history_conditional_calibration_guarantee=False)
