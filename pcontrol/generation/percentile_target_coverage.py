"""Training target coverage without relabeling any observed natural future.

Eligibility concerns only the frozen scalar CDF image, not physical trajectory
feasibility or known true percentiles. Nominal requested p is never projected,
jittered, or silently replaced. Ineligible requests still count in diagnostics.
"""
import math

import torch


def target_eligibility(reference,requested_p,*,interior_margin=1e-4,atom_tolerance=1e-7):
    if (requested_p.ndim!=1 or not requested_p.is_floating_point()
            or not bool(torch.isfinite(requested_p).all()) or bool(((requested_p<0)|(requested_p>1)).any())
            or not math.isfinite(float(interior_margin)) or not 0<=interior_margin<.5
            or not math.isfinite(float(atom_tolerance)) or not 0<=atom_tolerance<=1e-5):
        raise ValueError('finite p[B] and conservative fixed probability tolerances required')
    p=requested_p.detach().double()
    low=reference.rank(torch.full_like(p,float(reference.cap_seconds)))
    high=reference.rank(torch.zeros_like(p))
    lower=low['p_up'].detach();upper=high['p_low'].detach()
    if bool((lower>upper+1e-12).any()):raise ValueError('invalid continuous midrank image')
    interior=(p>lower+interior_margin)&(p<upper-interior_margin)
    endpoint=(p-low['p_mid'].detach()).abs()<=atom_tolerance
    endpoint=endpoint|((p-high['p_mid'].detach()).abs()<=atom_tolerance)
    distance_interior=torch.maximum(lower-p,torch.maximum(p-upper,torch.zeros_like(p)))
    floor=torch.minimum(distance_interior,torch.minimum((p-low['p_mid'].detach()).abs(),(p-high['p_mid'].detach()).abs()))
    return dict(eligible=interior|endpoint,interior=interior,endpoint_midpoint=endpoint,
        optimistic_endpoint_only_error_floor=floor,continuous_rank_lower=lower,continuous_rank_upper=upper,
        target_unchanged=True,physical_feasibility_guaranteed=False,true_CDF_claimed=False)


def grouped_value_mean(per_request_loss,support,history_presence,slots):
    """Equal present-history weight; within each history average valid slots.

Present histories with no valid target/current derivative contribute zero and
remain in the history denominator. This differs from the old eligible-row mean
and is applied identically to BOTH new paired arms.
"""
    if (type(slots) is not int or slots<1 or history_presence.ndim!=1
            or history_presence.dtype!=torch.bool or per_request_loss.ndim!=1
            or support.shape!=per_request_loss.shape or support.dtype!=torch.bool
            or per_request_loss.numel()!=history_presence.numel()*slots
            or support.device!=per_request_loss.device or history_presence.device!=per_request_loss.device
            or not bool(torch.isfinite(per_request_loss).all())):
        raise ValueError('flat loss/support and equal-slot history presence required')
    loss=per_request_loss.reshape(-1,slots);mask=support.reshape(-1,slots)&history_presence[:,None]
    per_history=torch.where(mask,loss,torch.zeros_like(loss)).sum(1)/mask.sum(1).clamp_min(1)
    return per_history.sum()/history_presence.sum().clamp_min(1)


def make_targets(natural_p,recipe,*,grid=(.1,.5,.9)):
    if natural_p.ndim!=1 or len(grid)!=3:raise ValueError('natural p[B] and three-slot grid required')
    if recipe=='natural_multi_noise':return natural_p.detach().repeat_interleave(3)
    if recipe=='p_grid':
        return torch.tensor(grid,dtype=natural_p.dtype,device=natural_p.device).repeat(natural_p.numel())
    raise ValueError('unregistered target coverage recipe')
