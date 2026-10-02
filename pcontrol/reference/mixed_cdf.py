"""One legal mixed distribution reused by every query for a history.

The head assigns joint probability to an optional zero atom, uniform-density
continuous bins, and a cap atom. This module has no encoder, trajectory, data,
retrieval, or random-number dependency. Observations enter only query/score
functions, never the logits-to-distribution factory.

Batch convention: masses have shape ``[*batch, bins]``. Scalar queries apply to
every distribution; ``[*batch]`` queries pair with distributions. Extra trailing
query dimensions are supported, e.g. ``[batch, queries]``. For a shared query
grid with one batch dimension, pass ``grid[None, :]``; a bare ``[batch]`` tensor
always means paired observations, even if its length also equals grid size.

Support starts at zero. Four seconds is a proposed PET cap, not hard-coded data
truth: the final fixed knot specifies the versioned metric's actual cap.
"""
from dataclasses import dataclass
from typing import Any, Tuple

import torch


@dataclass(frozen=True)
class DistributionParams:
    """Cached probability parameters; no renormalization occurs during queries.

    Use :func:`distribution_from_logits` for the jointly normalized learned
    head. Direct construction permits exact zero masses for mathematical tests
    and properly versioned non-neural references; it validates rather than
    silently repairs invalid probabilities. Knots are fixed, shared, 1-D.
    """

    knots: torch.Tensor
    continuous_masses: torch.Tensor
    zero_mass: torch.Tensor
    cap_mass: torch.Tensor
    zero_atom_enabled: bool = True
    model_version: str = "mixed_piecewise_linear_cdf_v1"

    def __post_init__(self) -> None:
        tensors = (self.knots, self.continuous_masses, self.zero_mass, self.cap_mass)
        if not all(isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("All distribution parameters must be torch tensors")
        if self.continuous_masses.dtype not in (torch.float32, torch.float64):
            raise TypeError("Probability parameters require float32 or float64")
        if any(value.dtype != self.continuous_masses.dtype or
               value.device != self.continuous_masses.device for value in tensors):
            raise ValueError("Knots and masses must share dtype and device")
        if self.knots.requires_grad:
            raise ValueError("Knots must be fixed, not learned")
        if self.knots.ndim != 1 or self.knots.numel() < 2:
            raise ValueError("Knots must be a one-dimensional vector of bin edges")
        if self.continuous_masses.ndim < 1:
            raise ValueError("Continuous masses require a final bin dimension")
        if self.continuous_masses.shape[-1] != self.knots.numel() - 1:
            raise ValueError("There must be one continuous mass per knot interval")
        if self.zero_mass.shape != self.batch_shape or self.cap_mass.shape != self.batch_shape:
            raise ValueError("Atom masses must exactly match distribution batch shape")
        if not isinstance(self.zero_atom_enabled, bool):
            raise TypeError("zero_atom_enabled must be explicit boolean metadata")
        if not isinstance(self.model_version, str) or not self.model_version:
            raise ValueError("model_version must be a nonempty string")
        with torch.no_grad():
            if not all(bool(torch.isfinite(value).all()) for value in tensors):
                raise ValueError("Distribution parameters must be finite")
            if self.knots[0].item() != 0.0 or not bool((self.knots[1:] > self.knots[:-1]).all()):
                raise ValueError("Knots must start at zero and increase strictly")
            masses = self.joint_masses
            if bool((masses < 0).any()):
                raise ValueError("Probability masses cannot be negative")
            if not self.zero_atom_enabled and bool((self.zero_mass != 0).any()):
                raise ValueError("Disabled zero atom must have exactly zero mass")
            # Softmax summation is only approximately one in floating point.
            tolerance = 2e-6 if masses.dtype == torch.float32 else 2e-12
            if not bool(torch.allclose(masses.sum(-1), torch.ones_like(self.zero_mass),
                                       atol=tolerance, rtol=0.0)):
                raise ValueError("Joint masses must sum to one; queries never renormalize")

    @property
    def batch_shape(self) -> torch.Size:
        return self.continuous_masses.shape[:-1]

    @property
    def cap_seconds(self) -> torch.Tensor:
        return self.knots[-1]

    @property
    def joint_masses(self) -> torch.Tensor:
        return torch.cat((self.zero_mass.unsqueeze(-1), self.continuous_masses,
                          self.cap_mass.unsqueeze(-1)), dim=-1)


def distribution_from_logits(
    knots: torch.Tensor,
    logits: torch.Tensor,
    *,
    zero_atom_enabled: bool,
    model_version: str = "mixed_piecewise_linear_cdf_v1",
) -> DistributionParams:
    """Normalize all enabled bins/atoms once with a shared softmax.

    Logit order is ``[zero, bins..., cap]`` when zero is enabled and
    ``[bins..., cap]`` otherwise. In the latter case zero is absent from the
    softmax, not merely assigned a low logit. Only history-model logits belong
    here: this factory intentionally accepts no observed PET or requested rank.
    """
    if not isinstance(knots, torch.Tensor) or not isinstance(logits, torch.Tensor):
        raise TypeError("knots and logits must be torch tensors")
    if not isinstance(zero_atom_enabled, bool):
        raise TypeError("zero_atom_enabled must be a boolean")
    if knots.ndim != 1 or logits.ndim < 1:
        raise ValueError("Expected 1-D knots and a final logit dimension")
    n_bins = knots.numel() - 1
    expected = n_bins + 1 + int(zero_atom_enabled)
    if logits.shape[-1] != expected:
        raise ValueError("Logit count must equal continuous bins plus enabled atoms")
    if logits.dtype not in (torch.float32, torch.float64):
        raise TypeError("Logits require float32 or float64")
    if not bool(torch.isfinite(logits).all()):
        raise ValueError("Logits must be finite; use explicit masses for exact degeneracy")
    probabilities = torch.softmax(logits, dim=-1)
    if zero_atom_enabled:
        zero, continuous = probabilities[..., 0], probabilities[..., 1:-1]
    else:
        zero, continuous = torch.zeros_like(probabilities[..., 0]), probabilities[..., :-1]
    return DistributionParams(knots, continuous, zero, probabilities[..., -1],
                              zero_atom_enabled, model_version)


def _broadcast_query(
    params: DistributionParams, value: Any
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Broadcast paired batch axes first and explicit query axes afterward."""
    query = torch.as_tensor(value, dtype=params.continuous_masses.dtype,
                            device=params.continuous_masses.device)
    n_batch = len(params.batch_shape)
    if query.ndim < n_batch:
        query = query.reshape((1,) * (n_batch - query.ndim) + tuple(query.shape))
    query_batch = query.shape[:n_batch]
    query_tail = query.shape[n_batch:]
    try:
        common_batch = torch.broadcast_shapes(params.batch_shape, query_batch)
    except RuntimeError as error:
        raise ValueError("Query batch axes do not broadcast with distribution batch; "
                         "use [1, queries] for a shared grid") from error
    output_shape = tuple(common_batch) + tuple(query_tail)
    query = query.expand(output_shape)
    suffix = (1,) * len(query_tail)
    zero = params.zero_mass.reshape(tuple(params.batch_shape) + suffix).expand(output_shape)
    cap = params.cap_mass.reshape(tuple(params.batch_shape) + suffix).expand(output_shape)
    masses = params.continuous_masses.reshape(
        tuple(params.batch_shape) + suffix + (params.continuous_masses.shape[-1],)
    ).expand(output_shape + (params.continuous_masses.shape[-1],))
    return query, zero, cap, masses


def _validate_observation(params: DistributionParams, observation: torch.Tensor) -> None:
    """Reject missing or uncapped PET; never silently drop invalid score rows."""
    if not bool(torch.isfinite(observation).all()):
        raise ValueError("Observed PET must be finite; invalid rows need an explicit policy")
    if bool(((observation < 0) | (observation > params.cap_seconds)).any()):
        raise ValueError("Observed PET must lie in the frozen [0, cap] support")


def cdf_from_params(params: DistributionParams, y: Any, *, side: str = "right") -> torch.Tensor:
    """Return F(y) or the exact F(y-) without an epsilon approximation.

    Infinite threshold queries are mathematically valid and give 0/1; NaN is
    rejected. Actual observed PET is checked more strictly by rank and scores.
    """
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")
    query, zero, cap, masses = _broadcast_query(params, y)
    if bool(torch.isnan(query).any()):
        raise ValueError("CDF threshold cannot be NaN")
    widths = params.knots[1:] - params.knots[:-1]
    fraction = ((query.unsqueeze(-1) - params.knots[:-1]) / widths).clamp(0.0, 1.0)
    continuous = (zero + (masses * fraction).sum(-1)).clamp(0.0, 1.0)
    at_zero = zero if side == "right" else torch.zeros_like(zero)
    at_cap = torch.ones_like(cap) if side == "right" else 1.0 - cap
    result = torch.where(query == 0, at_zero, continuous)
    result = torch.where(query == params.cap_seconds, at_cap, result)
    result = torch.where(query < 0, torch.zeros_like(result), result)
    return torch.where(query > params.cap_seconds, torch.ones_like(result), result)


def quantile_from_params(params: DistributionParams, quantile_level: Any) -> torch.Tensor:
    """Generalized inverse for 0 < u < 1, with exact atom handling.

    Explicit endpoint extension: Q(0)=0 and Q(1)=cap, the fixed *domain*
    endpoints. When an endpoint has zero mass, this convention does not assert
    it has natural/physical support. No interpolation crosses an atom. Flat
    continuous bins select the leftmost location attaining the requested CDF.
    """
    u, zero, _cap, masses = _broadcast_query(params, quantile_level)
    if not bool(torch.isfinite(u).all()) or bool(((u < 0) | (u > 1)).any()):
        raise ValueError("Quantile levels must be finite in [0, 1]")
    ends = zero.unsqueeze(-1) + masses.cumsum(-1)
    starts = torch.cat((zero.unsqueeze(-1), ends[..., :-1]), dim=-1)
    index = (ends >= u.unsqueeze(-1)).to(torch.int64).argmax(-1, keepdim=True)
    selected_start = starts.gather(-1, index).squeeze(-1)
    selected_mass = masses.gather(-1, index).squeeze(-1)
    safe_mass = torch.where(selected_mass > 0, selected_mass, torch.ones_like(selected_mass))
    fraction = ((u - selected_start) / safe_mass).clamp(0.0, 1.0)
    left = params.knots[:-1][index.squeeze(-1)]
    width = (params.knots[1:] - params.knots[:-1])[index.squeeze(-1)]
    result = left + fraction * width
    result = torch.where(u > ends[..., -1], params.cap_seconds, result)
    result = torch.where(u <= zero, torch.zeros_like(result), result)
    result = torch.where(u == 0, torch.zeros_like(result), result)
    return torch.where(u == 1, params.cap_seconds, result)


@dataclass(frozen=True)
class RankResult:
    """Adversity rank; smaller PET means larger rank, not collision probability."""

    p_low: torch.Tensor
    p_high: torch.Tensor
    p_mid: torch.Tensor
    atom_width: torch.Tensor

    @property
    def p_up(self) -> torch.Tensor:
        """Alias for the upper-rank notation used in some derivations."""
        return self.p_high

    def as_dict(self):
        return {"p_low": self.p_low, "p_high": self.p_high,
                "p_mid": self.p_mid, "atom_width": self.atom_width}


def rank_from_params(params: DistributionParams, y: Any) -> RankResult:
    query, _zero, _cap, _masses = _broadcast_query(params, y)
    _validate_observation(params, query)
    lower = 1.0 - cdf_from_params(params, query, side="right")
    upper = 1.0 - cdf_from_params(params, query, side="left")
    return RankResult(lower, upper, (lower + upper) * 0.5, upper - lower)


def atom_resolution_floor(params: DistributionParams, requested_p: Any) -> torch.Tensor:
    """Infimum midrank error imposed by endpoint atoms alone.

    The rank set is {cap_mass/2}, the continuous CDF's rank interval, and
    {1-zero_mass/2}. Its interval closure gives the same infimum as an open
    interval. If all continuous mass is zero, the interior domain still has a
    constant CDF and thus one mathematical rank value. This diagnostic includes
    every y in the score's domain, *not* only supported/physically reachable y;
    it is not a guarantee of attainable traffic risk.
    """
    p, zero, cap, _masses = _broadcast_query(params, requested_p)
    if not bool(torch.isfinite(p).all()) or bool(((p < 0) | (p > 1)).any()):
        raise ValueError("Requested adversity percentiles must be finite in [0, 1]")
    distance_to_continuous = torch.relu(cap - p) + torch.relu(p - (1.0 - zero))
    return torch.minimum(distance_to_continuous,
                         torch.minimum((p - cap * 0.5).abs(),
                                       (p - (1.0 - zero * 0.5)).abs()))
