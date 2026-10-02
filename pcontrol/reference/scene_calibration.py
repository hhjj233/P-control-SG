"""Convex-CRPS calibration of the WHOLE scene CDF, without data/file access.

G_N(y)=h_N(F(y)). A continuous monotone probability-axis piecewise-linear h
fixes 0/1 but may change the base zero/cap atom probabilities. The count family
blends two warps with clip((N-5)/8,0,1), a frozen history-known count feature.
This is not a relabeling of the old uniform physical bins: exact scoring splits
them at probability-warp crossings. Warp knots inside base atoms create NO
interior physical knot. All CDF sides, quantiles, ranks and CRPS use one object.

Fitting minimizes mean(CRPS/cap) + ridge/2 * sum((internal_nodes-identity)^2).
The data-only quadratic part is precomputed by three-point Gauss integration
on exactly linear physical pieces, then SLSQP enforces monotone node values.
Only the caller controls CAL access/CV; no calibration guarantee is asserted.
"""
from dataclasses import dataclass

import numpy as np


PROTOCOL = 'natural_scene_joint_CDF_monotone_CRPS_warp_v1'
U_KNOTS = np.array([0., .05, .1, .25, .5, .75, .9, .95, 1.])
U_KNOTS.setflags(write=False)


def _base(masses, counts, knots):
    masses = np.asarray(masses, dtype=np.float64)
    if masses.ndim == 1:
        masses = masses[None]
    if masses.ndim != 2 or masses.shape[0] == 0 or masses.shape[1] < 3:
        raise ValueError('base joint masses need [N, zero+bins+cap]')
    counts = np.broadcast_to(np.asarray(counts, dtype=np.float64), (len(masses),)).copy()
    knots = np.linspace(0., 4., masses.shape[1]-1) if knots is None else np.asarray(knots, dtype=np.float64)
    if (knots.shape != (masses.shape[1]-1,) or not np.isfinite(knots).all()
            or knots[0] != 0 or np.any(np.diff(knots) <= 0)
            or not np.isfinite(masses).all() or np.any(masses < 0)
            or not np.allclose(masses.sum(-1), 1., rtol=0, atol=2e-12)
            or not np.isfinite(counts).all() or np.any(counts < 1)
            or np.any(counts != np.floor(counts))):
        raise ValueError('valid fixed knots, normalized probabilities and actual integer counts required')
    return masses, counts, knots


def _query(query, rows):
    query = np.asarray(query, dtype=np.float64)
    if query.ndim == 0:
        query = query[None]
    if query.shape[0] not in (1, rows) or np.isnan(query).any():
        raise ValueError('query leading axis must be 1 or N; use [1,Q] for shared grids')
    return np.broadcast_to(query, (rows,) + query.shape[1:])


def _base_cdf(masses, knots, query, side='right'):
    if side not in ('left', 'right'):
        raise ValueError('CDF side must be left or right')
    y = _query(query, len(masses))
    prefix = (len(masses),) + (1,) * (y.ndim-1)
    zero, cap = masses[:, 0].reshape(prefix), masses[:, -1].reshape(prefix)
    bins = masses[:, 1:-1].reshape(prefix + (masses.shape[1]-2,))
    fraction = np.clip((y[..., None]-knots[:-1]) / np.diff(knots), 0., 1.)
    continuous = np.clip(zero + np.sum(bins*fraction, axis=-1), 0., 1.)
    answer = np.where(y == 0, zero if side == 'right' else 0., continuous)
    answer = np.where(y == knots[-1], 1. if side == 'right' else 1.-cap, answer)
    return np.where(y < 0, 0., np.where(y > knots[-1], 1., answer))


def _base_quantile(masses, knots, query):
    u = _query(query, len(masses))
    if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError('finite quantile levels in [0,1] required')
    prefix = (len(masses),) + (1,) * (u.ndim-1)
    zero = masses[:, 0].reshape(prefix)
    bins = masses[:, 1:-1].reshape(prefix + (masses.shape[1]-2,))
    ends = zero[..., None] + np.cumsum(bins, axis=-1)
    starts = np.concatenate((zero[..., None], ends[..., :-1]), axis=-1)
    index = np.argmax(ends >= u[..., None], axis=-1)
    shape = u.shape + (bins.shape[-1],)
    chosen_start = np.take_along_axis(np.broadcast_to(starts, shape), index[..., None], -1)[..., 0]
    chosen_mass = np.take_along_axis(np.broadcast_to(bins, shape), index[..., None], -1)[..., 0]
    fraction = np.clip((u-chosen_start)/np.where(chosen_mass > 0, chosen_mass, 1.), 0., 1.)
    answer = knots[index] + fraction*np.diff(knots)[index]
    answer = np.where(u > ends[..., -1], knots[-1], answer)
    answer = np.where(u <= zero, 0., answer)
    return np.where(u == 0, 0., np.where(u == 1, knots[-1], answer))


def _hat_basis(u, knots):
    u = np.asarray(u, dtype=np.float64)
    if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
        raise ValueError('CDF probabilities must be finite in [0,1]')
    index = np.clip(np.searchsorted(knots, u, side='right')-1, 0, len(knots)-2)
    fraction = (u-knots[index])/(knots[index+1]-knots[index])
    basis = np.zeros(u.shape+(len(knots),), dtype=np.float64)
    np.put_along_axis(basis, index[..., None], (1.-fraction)[..., None], axis=-1)
    np.put_along_axis(basis, (index+1)[..., None], fraction[..., None], axis=-1)
    return basis


def _physical_pieces(masses, knots, u_knots):
    """Physical knots and probability-hat bases, retaining crossing provenance.

    F^-1(u_k) is a numerical coordinate for the EXACT probability u_k. Never
    recover its probability by round-tripping through F: one ulp below u_k
    would hide the entrance to a flat warp segment from a generalized inverse.
    All nonidentity CDF, inverse and CRPS computations share these same bases.
    """
    levels = u_knots[(u_knots > masses[0]) & (u_knots < 1.-masses[-1])]
    crossings = _base_quantile(masses[None], knots, levels[None])[0] if len(levels) else np.empty(0)
    physical = np.unique(np.r_[knots, crossings])
    left = _base_cdf(masses[None], knots, physical[None], 'left')[0]
    right = _base_cdf(masses[None], knots, physical[None], 'right')[0]
    positive = np.flatnonzero(masses[1:-1] > 0)
    if len(positive):
        trailing = (physical >= knots[positive[-1]+1]) & (physical < knots[-1])
        left[trailing] = right[trailing] = 1.-masses[-1]
    seen = {}
    for coordinate, level in zip(crossings, levels):
        if coordinate <= knots[0] or coordinate >= knots[-1]:
            raise FloatingPointError('an interior warp crossing cannot be represented inside support')
        index = int(np.searchsorted(physical, coordinate))
        if index in seen and seen[index] != level:
            raise FloatingPointError('distinct warp probabilities share one unresolved physical float')
        seen[index] = level
        # These are known probability nodes, not clipped targets or a geometry epsilon.
        left[index] = right[index] = level
    return physical, _hat_basis(left, u_knots), _hat_basis(right, u_knots)


def _piece_basis(physical, left, right, query, side='right'):
    if side not in ('left', 'right'):
        raise ValueError('CDF side must be left or right')
    query = np.asarray(query, dtype=np.float64)
    safe = np.clip(query, physical[0], physical[-1])
    previous = np.clip(np.searchsorted(physical, safe, side='right')-1, 0, len(physical)-2)
    fraction = (safe-physical[previous])/(physical[previous+1]-physical[previous])
    answer = right[previous]+fraction[..., None]*(left[previous+1]-right[previous])
    exact = np.searchsorted(physical, safe, side='left')
    answer = np.where((safe == physical[exact])[..., None],
                       (right if side == 'right' else left)[exact], answer)
    return np.where((query < physical[0])[..., None], left[0],
                     np.where((query > physical[-1])[..., None], right[-1], answer))


def _physical_breaks(masses, knots, u_knots, observed=None):
    physical, _left, _right = _physical_pieces(masses, knots, u_knots)
    return np.unique(np.r_[physical, [] if observed is None else [observed]])


@dataclass(frozen=True)
class SceneCDFWarp:
    family: str
    node_values: np.ndarray
    u_knots: np.ndarray = None

    def __post_init__(self):
        u = np.array(U_KNOTS if self.u_knots is None else self.u_knots, dtype=np.float64, copy=True)
        nodes = np.array(self.node_values, dtype=np.float64, copy=True)
        expected = 1 if self.family == 'global' else 2 if self.family == 'count' else 0
        if (not expected or not np.array_equal(u, U_KNOTS) or nodes.shape != (expected, len(u))
                or not np.isfinite(nodes).all() or np.any(nodes[:, 0] != 0)
                or np.any(nodes[:, -1] != 1) or np.any(np.diff(nodes, axis=-1) < 0)):
            raise ValueError('global/count warp requires frozen u knots and monotone nodes with endpoints 0/1')
        u.setflags(write=False); nodes.setflags(write=False)
        object.__setattr__(self, 'u_knots', u)
        object.__setattr__(self, 'node_values', nodes)

    @classmethod
    def identity(cls, family='global'):
        if family not in ('global', 'count'):
            raise ValueError('family must be global/count')
        return cls(family, np.tile(U_KNOTS, (1 if family == 'global' else 2, 1)))

    def as_dict(self):
        return dict(protocol=PROTOCOL, family=self.family, u_knots=self.u_knots.tolist(),
                    node_values=self.node_values.tolist(),
                    count_blend='clip((N-5)/8,0,1)', changes_endpoint_atoms=True)

    @classmethod
    def from_dict(cls, value):
        if (set(value) != {'protocol','family','u_knots','node_values','count_blend','changes_endpoint_atoms'}
                or value['protocol'] != PROTOCOL or value['count_blend'] != 'clip((N-5)/8,0,1)'
                or value['changes_endpoint_atoms'] is not True):
            raise ValueError('wrong serialized scene calibration schema')
        return cls(value['family'], value['node_values'], value['u_knots'])

    def row_nodes(self, counts):
        counts = np.asarray(counts, dtype=np.float64)
        if not np.isfinite(counts).all() or np.any(counts < 1) or np.any(counts != np.floor(counts)):
            raise ValueError('actual positive integer history counts required')
        if self.family == 'global':
            return np.broadcast_to(self.node_values[0], counts.shape+(len(self.u_knots),))
        blend = np.clip((counts-5.)/8., 0., 1.)[..., None]
        return (1.-blend)*self.node_values[0]+blend*self.node_values[1]

    def cdf(self, masses, counts, y, *, side='right', base_knots=None):
        masses, counts, knots = _base(masses, counts, base_knots)
        y = _query(y, len(masses))
        if np.array_equal(self.node_values, np.tile(self.u_knots, (len(self.node_values), 1))):
            return _base_cdf(masses, knots, y, side)
        answer = np.empty_like(y)
        for row, nodes in enumerate(self.row_nodes(counts)):
            physical, left, right = _physical_pieces(masses[row], knots, self.u_knots)
            answer[row] = _piece_basis(physical, left, right, y[row], side)@nodes
        return answer

    def quantile(self, masses, counts, u, *, base_knots=None):
        masses, counts, knots = _base(masses, counts, base_knots)
        u = _query(u, len(masses))
        if not np.isfinite(u).all() or np.any((u < 0) | (u > 1)):
            raise ValueError('quantile levels must be finite in [0,1]')
        if np.array_equal(self.node_values, np.tile(self.u_knots, (len(self.node_values), 1))):
            return _base_quantile(masses, knots, u)
        # Invert the actual physical linear pieces directly. Chaining h^-1
        # and F^-1 can round a cap-left level onto the cap atom when zero-mass
        # trailing bins exist. No quantile level or label is epsilon-clipped.
        answer = np.empty_like(u)
        for row, mass in enumerate(masses):
            physical, left_basis, right_basis = _physical_pieces(mass, knots, self.u_knots)
            nodes = self.row_nodes(counts[row:row+1])[0]
            left, right = left_basis@nodes, right_basis@nodes
            levels = u[row].ravel()
            index = np.argmax(right[None] >= levels[:, None], axis=-1)
            previous = np.maximum(index-1, 0)
            lo, hi = right[previous], left[index]
            value = physical[previous]+(levels-lo)/np.where(hi > lo, hi-lo, 1.) * (physical[index]-physical[previous])
            value = np.where((index == 0) | (levels > left[index]), physical[index], value)
            value = np.where(levels <= right[0], 0., value)
            value = np.where(levels == 0, 0., np.where(levels == 1, knots[-1], value))
            answer[row] = value.reshape(u[row].shape)
        return answer

    def rank(self, masses, counts, y, *, base_knots=None):
        masses, counts, knots = _base(masses, counts, base_knots)
        y = _query(y, len(masses))
        if not np.isfinite(y).all() or np.any((y < 0) | (y > knots[-1])):
            raise ValueError('rank requires actual finite PET inside the support')
        left = self.cdf(masses, counts, y, side='left', base_knots=knots)
        right = self.cdf(masses, counts, y, side='right', base_knots=knots)
        return dict(p_low=1.-right, p_up=1.-left, p_mid=1.-.5*(right+left))

    def crps(self, masses, counts, observed_pet, *, base_knots=None, normalized=False):
        quadratic = precompute_crps_quadratic(masses, counts, observed_pet,
                                               family=self.family, base_knots=base_knots)
        theta = self.node_values[:, 1:-1].ravel()
        values = np.einsum('i,nij,j->n', theta, quadratic['A'], theta) + quadratic['b']@theta + quadratic['c']
        return values/quadratic['cap_seconds'] if normalized else values


def precompute_crps_quadratic(masses, counts, observed_pet, *, family, base_knots=None):
    """Per-row CRPS seconds = theta.T A theta + b.T theta + c.

    theta flattens one/two sets of seven internal warp ordinates. This routine
    uses only supplied arrays and does not access, select, or identify CAL data.
    Three-point Gauss is exact for degree-two integrands on each constructed
    piece, up to float64 rounding; it is not a quadrature approximation across
    missing warp crossings or endpoint jumps.
    """
    masses, counts, knots = _base(masses, counts, base_knots)
    y = np.broadcast_to(np.asarray(observed_pet, dtype=np.float64), (len(masses),))
    if not np.isfinite(y).all() or np.any((y < 0) | (y > knots[-1])):
        raise ValueError('paired finite actual PET in [0,cap] required; no label clipping')
    if family not in ('global', 'count'):
        raise ValueError('family must be global/count')
    width = 7 if family == 'global' else 14
    A, b, c = np.zeros((len(masses), width, width)), np.zeros((len(masses), width)), np.zeros(len(masses))
    gauss_x, gauss_w = np.polynomial.legendre.leggauss(3)
    piece_counts = []
    for row, (mass, target, count) in enumerate(zip(masses, y, counts)):
        physical, left_basis, right_basis = _physical_pieces(mass, knots, U_KNOTS)
        breaks = np.unique(np.r_[physical, target])
        length = np.diff(breaks)
        positions = .5*(breaks[:-1]+breaks[1:])[:, None]+.5*length[:, None]*gauss_x
        weights = (.5*length[:, None]*gauss_w).ravel()
        positions = positions.ravel()
        basis = _piece_basis(physical, left_basis, right_basis, positions)
        X = basis[:, 1:-1]
        if family == 'count':
            blend = float(np.clip((count-5.)/8., 0., 1.))
            X = np.concatenate(((1.-blend)*X, blend*X), axis=-1)
        offset = basis[:, -1]-(target <= positions)
        A[row] = (X.T*weights)@X
        b[row] = 2.*X.T@(weights*offset)
        c[row] = np.sum(weights*offset*offset)
        piece_counts.append(int(len(length)))
    return dict(A=A, b=b, c=c, cap_seconds=float(knots[-1]),
                physical_piece_counts=piece_counts, family=family,
                integration='Gauss3_exact_quadratic_on_all_physical_and_warp_breakpoints')


@dataclass(frozen=True)
class SceneCalibrationFit:
    warp: SceneCDFWarp
    report: dict


def fit_scene_calibration(masses, counts, observed_pet, *, family, ridge,
                          base_knots=None, maxiter=1000):
    """Fit the convex CRPS objective on caller-supplied calibration arrays only.

    One deterministic identity initialization, analytic gradient, constrained
    SLSQP. No validation/AUDIT input, data-selection rule, or alternate loss.
    Tiny optimizer feasibility error (<=1e-10) is explicitly polished and
    reported; larger violation returns no model and raises. Labels never move.
    A returned unsuccessful optimizer status must not be silently promoted.
    """
    from scipy.optimize import minimize
    if not np.isfinite(ridge) or ridge < 0 or type(maxiter) is not int or maxiter < 1:
        raise ValueError('finite nonnegative ridge and positive maxiter required')
    arrays = precompute_crps_quadratic(masses, counts, observed_pet,
                                        family=family, base_knots=base_knots)
    A = arrays['A'].mean(0)/arrays['cap_seconds']
    b = arrays['b'].mean(0)/arrays['cap_seconds']
    c = float(arrays['c'].mean()/arrays['cap_seconds'])
    identity = np.tile(U_KNOTS[1:-1], 1 if family == 'global' else 2)
    constraint = np.zeros((6*(len(identity)//7), len(identity)))
    for block in range(len(identity)//7):
        for j in range(6):
            constraint[block*6+j, block*7+j] = -1.
            constraint[block*6+j, block*7+j+1] = 1.

    def objective(theta):
        difference = theta-identity
        return float(theta@A@theta+b@theta+c+.5*ridge*(difference@difference))

    def gradient(theta):
        return (A+A.T)@theta+b+ridge*(theta-identity)

    result = minimize(objective, identity, jac=gradient, method='SLSQP',
        bounds=[(0., 1.)]*len(identity),
        constraints=dict(type='ineq', fun=lambda theta: constraint@theta,
                         jac=lambda theta: constraint),
        options=dict(maxiter=maxiter, ftol=1e-12, disp=False))
    raw = np.asarray(result.x, dtype=np.float64)
    if not np.isfinite(raw).all() or not np.isfinite(result.fun):
        raise FloatingPointError('nonfinite scene calibration optimization')
    violation = max(0., float(-raw.min()), float(raw.max()-1.), float(-(constraint@raw).min()))
    if violation > 1e-10:
        raise FloatingPointError('SLSQP returned a materially nonmonotone/infeasible warp')
    theta = np.maximum.accumulate(np.clip(raw.reshape(-1, 7), 0., 1.), axis=-1).ravel()
    nodes = np.column_stack((np.zeros(len(theta)//7), theta.reshape(-1, 7), np.ones(len(theta)//7)))
    warp = SceneCDFWarp(family, nodes)
    plain_final = float(theta@A@theta+b@theta+c)
    plain_initial = float(identity@A@identity+b@identity+c)
    report = dict(protocol=PROTOCOL, family=family, ridge=float(ridge),
        rows=int(len(arrays['c'])), parameter_count=int(len(theta)),
        objective='mean_CRPS_divided_by_cap_plus_ridge_over_2_sum_internal_identity_deviation_squared',
        optimizer='SLSQP_convex_quadratic_analytic_gradient',
        initialization='identity_only', success=bool(result.success),
        optimizer_status=int(result.status), optimizer_message=str(result.message),
        iterations=int(result.nit), maxiter=maxiter, ftol=1e-12,
        initial_mean_normalized_crps=plain_initial, final_mean_normalized_crps=plain_final,
        initial_penalized_objective=objective(identity), final_penalized_objective=objective(theta),
        improvement_normalized_crps=plain_initial-plain_final,
        parameter_feasibility_violation_before_polish=violation,
        maximum_parameter_polish=float(np.max(np.abs(theta-raw))),
        minimum_internal_node_gap=float((constraint@theta).min()),
        labels_modified=False, data_source_access_performed=False,
        integration=arrays['integration'],
        min_physical_pieces=int(min(arrays['physical_piece_counts'])),
        max_physical_pieces=int(max(arrays['physical_piece_counts'])),
        final_model=warp.as_dict())
    return SceneCalibrationFit(warp, report)
