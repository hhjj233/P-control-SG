"""Training-only physical PET target loss alongside estimated-rank loss.

The target is F_H^-1(1-P), not an observed/generated-future input. The caller
owns eligibility for mixed-CDF atoms and exact current-geometry support.
Unlike rank loss, this derivative is not multiplied by the current CDF density.
It does NOT create a derivative for capped/no-conflict unsupported geometries.
"""
import math
import torch
from torch.nn import functional as F


def pet_target_value_loss(current_pet,target_pet,supported_geometry,condition_present,*,
                          cap_seconds=4.,beta=.05,graph_anchor=None):
    if (not isinstance(current_pet,torch.Tensor) or current_pet.ndim!=1 or current_pet.numel()==0
            or current_pet.dtype not in (torch.float32,torch.float64)
            or not isinstance(target_pet,torch.Tensor) or target_pet.shape!=current_pet.shape
            or target_pet.device!=current_pet.device or not target_pet.is_floating_point()
            or not math.isfinite(cap_seconds) or cap_seconds!=4. or not math.isfinite(beta) or beta<=0):
        raise ValueError('paired floating PET values/targets and fixed positive scale required')
    for value in (current_pet,target_pet):
        if not bool(torch.isfinite(value).all()) or bool(((value<0)|(value>cap_seconds)).any()):
            raise ValueError('finite canonical PET values in [0,4] required; no clipping')
    for mask in (supported_geometry,condition_present):
        if mask.shape!=current_pet.shape or mask.dtype!=torch.bool or mask.device!=current_pet.device:
            raise ValueError('paired bool geometry/eligibility-presence masks required')
    support=supported_geometry.detach()&condition_present.detach()
    if bool(support.any()) and not current_pet.requires_grad:
        raise ValueError('supported current PET must have an actual geometry gradient')
    residual=(current_pet.double()-target_pet.detach().double())/cap_seconds
    per_row=F.smooth_l1_loss(residual,torch.zeros_like(residual),beta=beta,reduction='none')
    masked=torch.where(support,per_row,torch.zeros_like(per_row));loss=masked.sum()/support.sum().clamp_min(1)
    if graph_anchor is not None:
        if graph_anchor.device!=current_pet.device or not bool(torch.isfinite(graph_anchor).all()):
            raise ValueError('finite device-matched graph anchor required')
        loss=loss+graph_anchor.sum()*0.
    return dict(loss=loss,per_scene_loss=per_row,support_mask=support,
        supported_scenes=int(support.sum()),target_detached=True,
        all_request_PET_target_MAE_seconds=float((current_pet.detach()-target_pet.detach()).abs().mean()),
        current_density_required=False,unsupported_geometry_stays_unsupported=True,
        labels_or_metric_changed=False,estimated_CDF_target_not_known_true_quantile=True)
