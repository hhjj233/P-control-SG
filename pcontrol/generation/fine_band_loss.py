"""Smooth tolerance-hit objective supplement; original rank/PET losses stay.

The fixed .05 tolerance is the existing Fine metric, not a new score. This
term emphasizes errors near the hit boundary, while the retained base losses
still address large errors. No request, label, or trajectory is modified.
"""
import math
import torch


def fine_band_loss(achieved_p,requested_p,support,*,tolerance=.05,temperature=.01,graph_anchor=None):
    if (not isinstance(achieved_p,torch.Tensor) or achieved_p.ndim!=1 or not achieved_p.is_floating_point()
            or not isinstance(requested_p,torch.Tensor) or requested_p.shape!=achieved_p.shape
            or requested_p.device!=achieved_p.device or not requested_p.is_floating_point()
            or support.shape!=achieved_p.shape or support.dtype!=torch.bool or support.device!=achieved_p.device
            or tolerance!=.05 or not math.isfinite(temperature) or temperature<=0):
        raise ValueError('paired probabilities, explicit support, and fixed Fine tolerance required')
    for value in (achieved_p,requested_p):
        if not bool(torch.isfinite(value).all()) or bool(((value<0)|(value>1)).any()):
            raise ValueError('finite probabilities in [0,1] required; no clipping')
    if bool(support.any()) and not achieved_p.requires_grad:raise ValueError('supported achieved rank must remain differentiable')
    error=(achieved_p.double()-requested_p.detach().double()).abs()
    at_zero=torch.sigmoid(error.new_tensor(-tolerance/temperature))
    per_row=torch.sigmoid((error-tolerance)/temperature)-at_zero
    loss=torch.where(support,per_row,torch.zeros_like(per_row)).sum()/support.sum().clamp_min(1)
    if graph_anchor is not None:loss=loss+graph_anchor.sum()*0.
    return dict(loss=loss,per_scene_loss=per_row,support_mask=support.detach(),
        supported_scenes=int(support.sum()),requested_P_unchanged=True,
        tolerance=tolerance,temperature=temperature,not_a_replacement_for_MAE=True)
