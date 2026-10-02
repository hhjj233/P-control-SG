"""Training-only natural-sample tangent regularizers for a direct-p denoiser.

The caller supplies frozen g=d(PET)/d(normalized clean coefficients), evaluated
at the supported reconstruction of the natural clean coefficients, and its own
OOF-teacher density f. No geometry,
CDF/model query, label generation, or inference control is performed here.

For v prediction, clean_hat=sqrt(alpha_bar)*x_t-sigma*v_prediction, hence
unit_g·(clean_hat-clean_true)=-sigma*unit_g·(v_prediction-v_target).
Projection loss therefore has the sigma² factor. With h=f*g frozen, estimated
adversity is 1-F, so a unit requested-p response corresponds to
d<h,clean_hat>/dp=-1. Its SmoothL1 residual is 1+that derivative.

This is a fixed-x_t, natural-ground-truth tangent surrogate, NOT the exact PET
gradient at a predicted trajectory or the derivative of a full DDIM path.
Row independence of the denoiser is required when differentiating the sum of
per-scene projections with respect to p[B]. The current set transformer is
independent across batch rows. H/noise/targets must not be functions of p.
"""
import math

import torch
from torch.nn import functional as F

from .diffusion import _clean_coefficients


PROTOCOL = 'natural_GT_risk_tangent_training_regularizers_v1'


def _row_mask(value, batch, device, name):
    if (not isinstance(value, torch.Tensor) or value.shape != (batch,)
            or value.dtype != torch.bool or value.device != device):
        raise ValueError(name + ' must be a device-matched bool[B]')
    return value.detach()


def _expand_rows(value, tensor):
    return value.reshape(value.shape[0], *([1] * (tensor.ndim - 1)))


def natural_tangent_losses(v_prediction, v_target, noisy_coefficients, p, alpha_bar,
                           agent_mask, natural_gradient, density, valid_geometry,
                           valid_density, condition_present, *, projection_weight=1.,
                           jacobian_weight=.1, min_jacobian_sigma=.5):
    """Return weighted EXTRA loss; the caller must retain the complete v-MSE.

    v_prediction/v_target/noisy_coefficients/natural_gradient share floating
    [B,N,K,2] or[B,N,D] shape. p[B] is the SAME tensor used by the denoiser;
    it must require grad only if a positive-weight Jacobian term has eligible
    rows. alpha_bar is[B] or[B,1,...,1]; masks and density are[B]. Agent padding
    is never part of a dot product. All cached directions/densities, targets,
    schedule values and x_t's direct reconstruction term are detached.

    Projection averages only valid-geometry, finite nonzero-g, p-present rows.
    Jacobian additionally requires a valid finite positive density and
    sigma>=min_jacobian_sigma. Upstream valid_density must already incorporate
    density-kink and decoded-vs-original CDF discrepancy eligibility decisions.
    Each term uses its OWN eligible-row denominator, not the full batch size.
    Empty support returns a zero connected to v_prediction's graph. Invalid
    cached directions/densities are sanitized before every multiplication.

    If jacobian_weight=0, no p derivative or higher-order graph is constructed;
    returned derivative/gain placeholders are zero with jacobian_evaluated=False.
    The helper never adds/removes examples from the caller's unweighted v-MSE.
    """
    for name, value in (('projection_weight', projection_weight), ('jacobian_weight', jacobian_weight)):
        if not math.isfinite(float(value)) or float(value) < 0:
            raise ValueError(name + ' must be a finite nonnegative hyperparameter')
    if not math.isfinite(float(min_jacobian_sigma)) or not 0 <= float(min_jacobian_sigma) <= 1:
        raise ValueError('min_jacobian_sigma must lie in [0,1]')
    prediction = _clean_coefficients(v_prediction, agent_mask, name='v prediction')
    target = _clean_coefficients(v_target, agent_mask, name='v target')
    noisy = _clean_coefficients(noisy_coefficients, agent_mask, name='noisy coefficients')
    if any(value.shape != prediction.shape or value.dtype != prediction.dtype for value in (target, noisy)):
        raise ValueError('v prediction, target and noisy coefficients must share shape/dtype')
    batch, device = prediction.shape[0], prediction.device
    geometry = _row_mask(valid_geometry, batch, device, 'valid_geometry')
    density_valid = _row_mask(valid_density, batch, device, 'valid_density')
    present = _row_mask(condition_present, batch, device, 'condition_present')
    if (not isinstance(p, torch.Tensor) or p.shape != (batch,) or p.device != device
            or p.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(p.detach()[present]).all())
            or bool(((p.detach()[present] < 0) | (p.detach()[present] > 1)).any())):
        raise ValueError('p must be the device-matched floating[B] condition with valid present values')
    if (not isinstance(alpha_bar, torch.Tensor) or alpha_bar.ndim < 1 or alpha_bar.shape[0] != batch
            or alpha_bar.numel() != batch or alpha_bar.device != device
            or alpha_bar.dtype not in (torch.float32, torch.float64)):
        raise ValueError('alpha_bar must contain one floating schedule value per scene')
    alpha = alpha_bar.detach().reshape(batch).to(torch.float64)
    if not bool(torch.isfinite(alpha).all()) or bool(((alpha < 0) | (alpha > 1)).any()):
        raise ValueError('alpha_bar values must lie in [0,1]')
    sigma2 = 1. - alpha
    sigma = sigma2.sqrt()
    if (not isinstance(natural_gradient, torch.Tensor) or natural_gradient.shape != prediction.shape
            or natural_gradient.device != device or natural_gradient.dtype not in (torch.float32, torch.float64)
            or not isinstance(density, torch.Tensor) or density.shape != (batch,) or density.device != device
            or density.dtype not in (torch.float32, torch.float64)):
        raise ValueError('frozen natural gradient and density must match coefficient/scene shapes and device')
    # First sanitize padding, then invalidate entire unusable cache rows.
    coefficient_mask = agent_mask.reshape(*agent_mask.shape, *([1] * (prediction.ndim - 2)))
    raw_g = torch.where(coefficient_mask, natural_gradient.detach().to(torch.float64),
                        torch.zeros_like(natural_gradient, dtype=torch.float64))
    finite_g = torch.isfinite(raw_g).flatten(1).all(1)
    clean_g = torch.where(_expand_rows(geometry & finite_g, raw_g), raw_g, torch.zeros_like(raw_g))
    gradient_norm = torch.linalg.vector_norm(clean_g.flatten(1), dim=1)
    nonzero_gradient = torch.isfinite(gradient_norm) & (gradient_norm > 0)
    projection_support = geometry & finite_g & nonzero_gradient & present
    denominator = torch.where(nonzero_gradient, gradient_norm, torch.ones_like(gradient_norm))
    unit_g = clean_g / _expand_rows(denominator, clean_g)
    unit_g = torch.where(_expand_rows(projection_support, unit_g), unit_g, torch.zeros_like(unit_g))
    f = density.detach().to(torch.float64)
    finite_positive_density = torch.isfinite(f) & (f > 0)
    jacobian_support = projection_support & density_valid & finite_positive_density & (sigma >= float(min_jacobian_sigma))
    safe_density = torch.where(jacobian_support, f, torch.zeros_like(f))
    h = clean_g * _expand_rows(safe_density, clean_g)
    # Use FP64 reductions for these scalar diagnostics; the network stays FP32.
    prediction64 = prediction.to(torch.float64)
    target64, noisy64 = target.detach().to(torch.float64), noisy.detach().to(torch.float64)
    projected_error = (unit_g * (prediction64 - target64)).flatten(1).sum(1)
    per_projection = sigma2 * projected_error.square()
    n_projection = int(projection_support.sum().item())
    n_jacobian = int(jacobian_support.sum().item())
    projection_loss = per_projection.sum() / max(n_projection, 1)
    connected_zero = prediction64.sum() * 0.
    derivative = torch.zeros(batch, dtype=torch.float64, device=device) + connected_zero
    residual = torch.zeros_like(derivative) + connected_zero
    per_jacobian = torch.zeros_like(derivative) + connected_zero
    jacobian_loss = connected_zero
    evaluated = float(jacobian_weight) > 0 and n_jacobian > 0
    p_connected = None
    if evaluated:
        if not p.requires_grad:
            raise ValueError('eligible positive-weight Jacobian loss requires the SAME differentiable p used in model forward')
        clean_hat = _expand_rows(alpha.sqrt(), prediction64) * noisy64 - _expand_rows(sigma, prediction64) * prediction64
        cdf_projection = (h * clean_hat).flatten(1).sum(1)
        if not cdf_projection.requires_grad:
            raise ValueError('Jacobian term needs a differentiable model prediction graph')
        raw_derivative = torch.autograd.grad(cdf_projection.sum(), p, create_graph=True,
                                             retain_graph=True, allow_unused=True)[0]
        p_connected = raw_derivative is not None
        if raw_derivative is not None:
            derivative = raw_derivative.to(torch.float64) + connected_zero
        derivative = torch.where(jacobian_support, derivative, torch.zeros_like(derivative))
        if not bool(torch.isfinite(derivative[jacobian_support]).all()):
            raise FloatingPointError('nonfinite natural-tangent condition Jacobian')
        residual = torch.where(jacobian_support, 1. + derivative, torch.zeros_like(derivative))
        per_jacobian = F.smooth_l1_loss(residual, torch.zeros_like(residual), reduction='none', beta=1.)
        jacobian_loss = per_jacobian.sum() / n_jacobian
    gain = -derivative
    counts = dict(batch=batch, projection=n_projection, jacobian=n_jacobian,
                  geometry_valid=int(geometry.sum().item()), p_present=int(present.sum().item()),
                  finite_nonzero_gradient=int((geometry & finite_g & nonzero_gradient).sum().item()),
                  valid_positive_density=int((density_valid & finite_positive_density).sum().item()),
                  sigma_gate=int((sigma >= float(min_jacobian_sigma)).sum().item()))
    diagnostics = dict(protocol=PROTOCOL, jacobian_evaluated=evaluated, p_graph_connected=p_connected,
        min_jacobian_sigma=float(min_jacobian_sigma), sigma=sigma.detach(),
        natural_gradient_norm=gradient_norm.detach(), cdf_tangent_norm=torch.linalg.vector_norm(h.flatten(1), dim=1).detach(),
        mean_adversity_gain=float(gain[jacobian_support].detach().mean()) if evaluated else None,
        mean_absolute_gain_residual=float(residual[jacobian_support].detach().abs().mean()) if evaluated else None,
        projection_weight=float(projection_weight), jacobian_weight=float(jacobian_weight),
        reduction='separate_eligible_present_scene_means', cached_gradient_and_density_detached=True,
        natural_ground_truth_tangent_surrogate=True, predicted_point_exact_PET_gradient=False,
        full_DDIM_path_derivative=False, inference_penalty=False, caller_MSE_must_be_retained=True)
    return dict(loss=float(projection_weight) * projection_loss + float(jacobian_weight) * jacobian_loss,
        projection_loss=projection_loss, jacobian_loss=jacobian_loss,
        per_scene_projection_loss=per_projection, per_scene_jacobian_loss=per_jacobian,
        projection_support_mask=projection_support, jacobian_support_mask=jacobian_support,
        support_counts=counts, cdf_tangent_derivative_wrt_p=derivative,
        adversity_gain=gain, jacobian_residual=residual, diagnostics=diagnostics)
