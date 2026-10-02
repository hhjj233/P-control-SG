"""Training-only terminal percentile VALUE objective, not a slope surrogate.

The caller must compute terminal_pet at the CURRENT generated trajectory and
provide its verified geometry-gradient support. This module neither computes
PET nor modifies/samples trajectories. It must not be fed a stale GT tangent
as though it were the generated trajectory's exact PET. The reference CDF is
estimated and frozen; its numerical control error is not error vs true CDF.
"""
import math

import torch
from torch.nn import functional as F


def terminal_percentile_value_loss(terminal_pet, requested_p, reference, *,
                                  supported_geometry, condition_present=None,
                                  beta=.05, minimum_density=1e-6,
                                  graph_anchor=None):
    """Return masked SmoothL1 plus all-request, detached rank diagnostics.

    PET[B] contains exact finite capped values even for unsupported rows.
    Geometry support only controls gradient use, never metric denominators.
    Endpoint/flat/kink CDF rows are conservatively excluded by reference.density.
    p is a fixed target, not a parameter to optimize. Optional graph_anchor
    connects a zero loss to model output when all PET values are detached.
    The natural v-MSE remains the responsibility of the outer training loop.
    """
    if (not isinstance(terminal_pet, torch.Tensor) or terminal_pet.ndim != 1
            or terminal_pet.numel() == 0 or terminal_pet.dtype not in (torch.float32, torch.float64)
            or not bool(torch.isfinite(terminal_pet).all())):
        raise ValueError('finite floating terminal PET[B] required')
    if (not isinstance(requested_p, torch.Tensor) or requested_p.shape != terminal_pet.shape
            or requested_p.device != terminal_pet.device or not requested_p.is_floating_point()
            or not bool(torch.isfinite(requested_p).all())
            or bool(((requested_p < 0) | (requested_p > 1)).any())):
        raise ValueError('one finite fixed p target in [0,1] per terminal sample required')
    if (not math.isfinite(float(beta)) or beta <= 0
            or not math.isfinite(float(minimum_density)) or minimum_density < 0):
        raise ValueError('positive SmoothL1 beta and nonnegative density threshold required')
    batch = terminal_pet.numel()
    for name, mask in [('supported_geometry', supported_geometry), ('condition_present', condition_present)]:
        if mask is not None and (not isinstance(mask, torch.Tensor) or mask.shape != (batch,)
                                or mask.dtype != torch.bool or mask.device != terminal_pet.device):
            raise ValueError(name+' must be device-matched bool[B]')
    if supported_geometry is None:
        raise ValueError('explicit current-point geometry support is required')
    present = torch.ones_like(supported_geometry) if condition_present is None else condition_present
    if any(parameter.requires_grad for parameter in reference.parameters()):
        raise ValueError('terminal reference must be frozen')
    rank = reference.rank(terminal_pet)
    density_info = reference.density(terminal_pet)
    for name in ('p_low', 'p_mid', 'p_up'):
        if rank[name].shape != (batch,) or not bool(torch.isfinite(rank[name]).all()):
            raise ValueError('reference must return one finite rank per scene')
    rho = density_info['density'].detach()
    smooth = density_info['valid'].detach()
    if rho.shape != (batch,) or smooth.shape != (batch,) or smooth.dtype != torch.bool:
        raise ValueError('CDF density/support must have paired [B] shape')
    target = requested_p.detach().to(torch.float64)
    residual = rank['p_mid'].to(torch.float64) - target
    support = supported_geometry.detach() & present.detach() & smooth & torch.isfinite(rho) & (rho > minimum_density)
    if bool(support.any()) and not terminal_pet.requires_grad:
        raise ValueError('supported terminal loss requires PET connected to generated output')
    per_row = F.smooth_l1_loss(residual, torch.zeros_like(residual), beta=float(beta), reduction='none')
    # Mask scalar loss contributions only. This is NOT a sanitizer for undefined
    # upstream derivatives; the geometry adapter must avoid constructing them.
    masked = torch.where(support, per_row, torch.zeros_like(per_row))
    count = int(support.sum().item())
    loss = masked.sum() / max(count, 1)
    if graph_anchor is not None:
        if (not isinstance(graph_anchor, torch.Tensor) or graph_anchor.device != terminal_pet.device
                or not bool(torch.isfinite(graph_anchor).all())):
            raise ValueError('finite device-matched graph anchor required')
        loss = loss + graph_anchor.sum()*0.
    point = residual.detach().abs()
    interval = torch.maximum(torch.maximum(rank['p_low'].detach()-target,
                                          target-rank['p_up'].detach()), torch.zeros_like(target))
    return dict(loss=loss, per_scene_value_loss=per_row, support_mask=support,
        supported_scenes=count, total_scenes=batch,
        estimated_rank={k:v.detach() for k,v in rank.items()},
        all_request_point_MAE=float(point.mean()), all_request_Fine_at_0_05=float((point<=.05).double().mean()),
        all_request_interval_MAE=float(interval.mean()),
        all_request_point_error=point, all_request_interval_error=interval,
        diagnostics=dict(target_detached=True, current_point_geometry_support_required=True,
            uses_terminal_percentile_value_not_condition_jacobian=True,
            estimated_reference_not_known_true_CDF=True, natural_base_loss_must_be_retained=True,
            unsupported_requests_retained_in_diagnostics=True, trajectory_modified=False,
            endpoint_flat_or_kink_gradients_not_trained=True,
            beta=float(beta), minimum_density=float(minimum_density)))
