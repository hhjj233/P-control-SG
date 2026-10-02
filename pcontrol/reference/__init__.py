"""Shared mathematical reference layer for natural-observation models."""

from .mixed_cdf import (
    DistributionParams,
    RankResult,
    atom_resolution_floor,
    cdf_from_params,
    distribution_from_logits,
    quantile_from_params,
    rank_from_params,
)
from .scores import brier_from_params, crps_from_params, pinball_from_params

__all__ = [
    "DistributionParams", "RankResult", "distribution_from_logits",
    "cdf_from_params", "quantile_from_params", "rank_from_params",
    "atom_resolution_floor", "crps_from_params", "pinball_from_params",
    "brier_from_params",
]
