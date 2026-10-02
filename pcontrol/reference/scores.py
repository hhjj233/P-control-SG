"""Proper scores and atom diagnostics for the single shared mixed CDF.

Scores are unreduced, preserving the caller's complete observation denominator.
Invalid observations raise explicitly. This module never filters rows, supplies
synthetic PET values, rebalances tails, or mixes score and observation inputs
into a history encoder. CRPS retains gradients through probability masses.
"""
from typing import Any

import torch

from .mixed_cdf import (
    DistributionParams, _broadcast_query, _validate_observation,
    quantile_from_params,
)


def crps_from_params(
    params: DistributionParams, observed_pet: Any, *, normalized: bool = False
) -> torch.Tensor:
    """Analytic CRPS in seconds, or divided by cap with normalized=True.

    Split every bin at the observation and integrate F^2 below it and (F-1)^2
    above it. For a linear function with endpoint values a and b over length d,
    its squared integral is d*(a*a+a*b+b*b)/3. This is equivalent to the
    polynomial antiderivative and avoids subtracting nearly equal primitives.
    Zero/cap atoms affect the interval CDF; endpoints themselves have measure
    zero and receive no artificial integration width.
    """
    y, zero, _cap, masses = _broadcast_query(params, observed_pet)
    _validate_observation(params, y)
    widths = params.knots[1:] - params.knots[:-1]
    end = zero.unsqueeze(-1) + masses.cumsum(-1)
    start = torch.cat((zero.unsqueeze(-1), end[..., :-1]), dim=-1)
    below_length = (y.unsqueeze(-1) - params.knots[:-1]).clamp_min(0.0)
    below_length = torch.minimum(below_length, widths)
    above_length = widths - below_length
    at_split = start + masses * (below_length / widths)
    below = below_length * (start.square() + start * at_split + at_split.square()) / 3.0
    a, b = at_split - 1.0, end - 1.0
    above = above_length * (a.square() + a * b + b.square()) / 3.0
    score = (below + above).sum(-1)
    return score / params.cap_seconds if normalized else score


def pinball_from_params(
    params: DistributionParams, observed_pet: Any, quantile_level: Any, *,
    normalized: bool = False,
) -> torch.Tensor:
    """Quantile loss at probability u (high adversity p corresponds to u=1-p).

    A shared probability grid for a batch is [1, queries]. Paired observations
    [batch] automatically gain trailing singleton query axes when required.
    """
    y, _zero, _cap, _masses = _broadcast_query(params, observed_pet)
    _validate_observation(params, y)
    u, _zero, _cap, _masses = _broadcast_query(params, quantile_level)
    q = quantile_from_params(params, u)
    n_dimensions = max(y.ndim, q.ndim)
    y = y.reshape(tuple(y.shape) + (1,) * (n_dimensions - y.ndim))
    q = q.reshape(tuple(q.shape) + (1,) * (n_dimensions - q.ndim))
    u = u.reshape(tuple(u.shape) + (1,) * (n_dimensions - u.ndim))
    residual = y - q
    score = torch.maximum(u * residual, (u - 1.0) * residual)
    return score / params.cap_seconds if normalized else score


def brier_from_params(params: DistributionParams, observed_pet: Any, *, atom: str) -> torch.Tensor:
    """Unweighted endpoint-event Brier diagnostic; not an added training loss."""
    if atom not in ("zero", "cap"):
        raise ValueError("atom must be 'zero' or 'cap'")
    y, zero, cap, _masses = _broadcast_query(params, observed_pet)
    _validate_observation(params, y)
    probability = zero if atom == "zero" else cap
    outcome = y == (0.0 if atom == "zero" else params.cap_seconds)
    return (probability - outcome.to(probability.dtype)).square()
