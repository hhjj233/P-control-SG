"""Fixed analytic acceleration-cosine coordinates, not observed trajectories.

Encoding fits actual position residuals by ordinary float64 least squares.
Decoding integrates the acceleration basis analytically, locking actual t0
position and velocity. Reconstructions are explicitly derived approximations.
"""
from dataclasses import dataclass
import numpy as np


PROTOCOL = "natural_anchored_cosine_acceleration_basis_v1"


def basis_matrices(times, modes, horizon=6.96):
    t = np.asarray(times, dtype=np.float64)
    if (t.ndim != 1 or not np.isfinite(t).all() or np.any(t < 0) or np.any(t > horizon)
            or type(modes) is not int or not 1 <= modes <= 64 or not np.isfinite(horizon) or horizon <= 0):
        raise ValueError("finite physical times in a positive horizon and 1..64 modes required")
    position = np.empty((len(t), modes), dtype=np.float64)
    velocity = np.empty_like(position)
    acceleration = np.empty_like(position)
    position[:, 0], velocity[:, 0], acceleration[:, 0] = t*t/2, t, 1.
    if modes > 1:
        omega = np.arange(1, modes, dtype=np.float64)*np.pi/horizon
        phase = t[:, None]*omega[None]
        position[:, 1:] = (1-np.cos(phase))/(omega[None]**2)
        velocity[:, 1:] = np.sin(phase)/omega[None]
        acceleration[:, 1:] = np.cos(phase)
    return position, velocity, acceleration


@dataclass(frozen=True)
class TrajectoryBasis:
    modes: int = 16
    sample_period: float = .04
    frames: int = 175

    def __post_init__(self):
        if self.sample_period != .04 or self.frames != 175:
            raise ValueError("this natural-data protocol preserves all175 native25Hz observations")
        basis_matrices(self.times, self.modes, self.horizon)

    @property
    def horizon(self): return (self.frames-1)*self.sample_period

    @property
    def times(self): return np.arange(self.frames, dtype=np.float64)*self.sample_period

    def as_dict(self):
        bp, _, _ = basis_matrices(self.times, self.modes, self.horizon)
        return dict(protocol=PROTOCOL, modes=self.modes, K=self.modes, mode_indices=list(range(self.modes)),
            sample_period=self.sample_period, frames=self.frames, horizon_seconds=self.horizon,
            time_grid_seconds=self.times.tolist(), basis_has_spline_knots=False,
            coefficient_units="metres_per_second_squared", coefficient_layout="[...,agent,mode,xy]",
            encoding="float64 ordinary least squares of actual xy minus xy0+v0*t; rcond=None; no ridge",
            ridge=0., design_condition_number=float(np.linalg.cond(bp)),
            decoding="analytic double/single cosine integrals for position/velocity",
            initial_xy_and_velocity_hard_locked=True, reconstructed_future_is_original_observation=False,
            acceleration_endpoints_forced_zero=False, terminal_velocity_forced_zero=False)

    def encode(self, future, anchors=None):
        future = np.asarray(future, dtype=np.float64)
        if future.ndim != 3 or future.shape[0] != self.frames or future.shape[2] != 4 or not np.isfinite(future).all():
            raise ValueError("actual finite future[175,N,4] required; no padding is encoded")
        anchors = future[0].copy() if anchors is None else np.asarray(anchors, dtype=np.float64)
        if anchors.shape != future.shape[1:] or not np.array_equal(anchors, future[0]):
            raise ValueError("anchors must exactly equal actual observed t0 states")
        bp, _, _ = basis_matrices(self.times, self.modes, self.horizon)
        residual = future[..., :2]-anchors[None, :, :2]-self.times[:, None, None]*anchors[None, :, 2:]
        solved, _, rank, _ = np.linalg.lstsq(bp, residual.reshape(self.frames, -1), rcond=None)
        if rank != self.modes or not np.isfinite(solved).all():
            raise ValueError("fixed position basis is rank deficient or its fit is nonfinite")
        return solved.reshape(self.modes, future.shape[1], 2).transpose(1, 0, 2).copy()

    def decode(self, coefficients, anchors):
        c = np.asarray(coefficients, dtype=np.float64)
        a = np.asarray(anchors, dtype=np.float64)
        if (c.ndim < 3 or c.shape[-2:] != (self.modes, 2) or a.shape != c.shape[:-2]+(4,)
                or not np.isfinite(c).all() or not np.isfinite(a).all()):
            raise ValueError("finite coefficients[...,N,K,2] and actual anchors[...,N,4] required")
        bp, bv, _ = basis_matrices(self.times, self.modes, self.horizon)
        time = self.times.reshape((1,)*(a.ndim-2)+(self.frames, 1, 1))
        xy = a[..., None, :, :2]+time*a[..., None, :, 2:]+np.einsum("tk,...nkd->...tnd", bp, c)
        velocity = a[..., None, :, 2:]+np.einsum("tk,...nkd->...tnd", bv, c)
        result = np.concatenate((xy, velocity), axis=-1)
        result[..., 0, :, :] = a  # Exact equality, including source floating-point t0 representation.
        return result


def reconstruction_errors(observed, reconstructed):
    x, y = np.asarray(observed), np.asarray(reconstructed)
    if x.shape != y.shape or x.ndim != 3 or x.shape[-1] != 4:
        raise ValueError("matching unpadded [T,N,4] states required")
    squared = (x-y)**2
    return dict(position_rmse_m=float(np.sqrt(squared[..., :2].mean())),
        position_vector_rmse_m=float(np.sqrt(squared[..., :2].sum(-1).mean())),
        velocity_rmse_mps=float(np.sqrt(squared[..., 2:].mean())),
        position_squared_error_sum=float(squared[..., :2].sum()), position_scalar_count=int(squared[..., :2].size),
        velocity_squared_error_sum=float(squared[..., 2:].sum()), velocity_scalar_count=int(squared[..., 2:].size),
        t0_state_exact=bool(np.array_equal(x[0], y[0])))
