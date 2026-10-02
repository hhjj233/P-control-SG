"""Reduce only the irreducible scalar-CDF floor inside the training objective.

Primary errors remain |achieved P - requested P|. For any admissible scalar
outcome this error is bounded below by the supplied frozen-reference infimum;
minimizing excess has the same ordering, without moving the request or CDF.
No physical reachability is inferred from that distribution-only bound.
"""
import torch
from torch.nn import functional as F


def excess_midrank_value_loss(achieved_p,requested_p,scalar_error_infimum,support,*,beta=.05,graph_anchor=None):
    if (achieved_p.ndim!=1 or requested_p.shape!=achieved_p.shape or scalar_error_infimum.shape!=achieved_p.shape
            or support.shape!=achieved_p.shape or support.dtype!=torch.bool
            or not 0<beta<=1 or not achieved_p.is_floating_point() or not requested_p.is_floating_point()):
        raise ValueError('aligned scalar risk values, frozen lower bounds and explicit support required')
    if any(v.device!=achieved_p.device for v in (requested_p,scalar_error_infimum,support)):raise ValueError('one device required')
    if any(not bool(torch.isfinite(v).all()) for v in (achieved_p,requested_p,scalar_error_infimum)):raise ValueError('finite scalar values required')
    if (bool(((achieved_p<0)|(achieved_p>1)|(requested_p<0)|(requested_p>1)).any())
            or bool(((scalar_error_infimum<0)|(scalar_error_infimum>1)).any()) or scalar_error_infimum.requires_grad):
        raise ValueError('valid probabilities and a detached distribution-only floor required')
    if bool(support.any()) and not achieved_p.requires_grad:raise ValueError('supported current rank must remain differentiable')
    point=(achieved_p.double()-requested_p.detach().double()).abs()
    floor=scalar_error_infimum.detach().double()
    if bool((point.detach()+2e-10<floor).any()):raise ValueError('a purported lower bound exceeds an actual rank error')
    excess=(point-floor).clamp_min(0.)
    per_row=F.smooth_l1_loss(excess,torch.zeros_like(excess),beta=beta,reduction='none')
    loss=torch.where(support,per_row,torch.zeros_like(per_row)).sum()/support.sum().clamp_min(1)
    if graph_anchor is not None:loss=loss+graph_anchor.sum()*0.
    return dict(loss=loss,per_scene_value_loss=per_row,support_mask=support.detach(),
        all_request_point_error=point.detach(),all_request_point_MAE=float(point.detach().mean()),
        all_request_Fine_at_0_05=float((point.detach()<=.05).double().mean()),
        all_request_excess_error=excess.detach(),scalar_error_infimum=floor,
        requested_P_unchanged=True,reference_unchanged=True,physical_feasibility_guaranteed=False)
