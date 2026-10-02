"""Training-only AABB encounter/escape signals for hard-PET blind regions.

These are spatial constraints in metres, NOT a substitute PET score or its
claimed derivative. At lag q, box intersection is a sufficient condition for
the canonical minimum pointwise occupancy PET to be <= q. Recruitment uses a
finite set of start times, not a complete feasibility solver. All exact PET,
CDF labels, inference and reporting stay outside this module and unchanged.
"""
import math
import torch


def _inputs(future,dimensions,ego_index):
    if (not isinstance(future,torch.Tensor) or future.ndim!=3 or future.shape[0]!=175 or future.shape[2]!=4
            or future.shape[1]<2 or not future.is_floating_point() or not bool(torch.isfinite(future).all())
            or type(ego_index) is not int or not 0<=ego_index<future.shape[1]):
        raise ValueError('complete finite F175/N>=2 and valid ego index required')
    dims=torch.as_tensor(dimensions,dtype=future.dtype,device=future.device).detach()
    if dims.shape!=(future.shape[1],2) or not bool(torch.isfinite(dims).all()) or bool((dims<=0).any()):
        raise ValueError('positive real vehicle length/width required')
    others=torch.arange(future.shape[1],device=future.device)!=ego_index
    return future[:,:,:2],dims,others


def target_lag_gap(future,dimensions,target_pet,*,ego_index=0):
    """Smallest signed separating-axis gap at temporal offset q, both orders."""
    xy,dims,others=_inputs(future,dimensions,ego_index)
    q=torch.as_tensor(target_pet,dtype=future.dtype,device=future.device).detach()
    if q.ndim!=0 or not bool(torch.isfinite(q)) or not 0<=float(q)<=4:
        raise ValueError('fixed target PET in[0,4] required')
    times=torch.arange(175,dtype=future.dtype,device=future.device)*.04
    valid=times+q<=6.96
    shifted=(times[valid]+q)/.04
    left=shifted.floor().long().clamp(0,173);fraction=(shifted-left).clamp(0,1)
    after=xy[left]+fraction[:,None,None]*(xy[left+1]-xy[left])
    before=xy[valid];half=(dims[ego_index]+dims[others])/2
    # Ego before/other after OR other before/ego after, never a fixed focal car.
    a=(before[:,ego_index,None]-after[:,others]).abs()-half[None]
    b=(after[:,ego_index,None]-before[:,others]).abs()-half[None]
    return torch.cat((a.amax(-1).reshape(-1),b.amax(-1).reshape(-1))).min()


def continuous_same_time_gap(future,dimensions,*,ego_index=0):
    """Minimum signed AABB gap over every shared piecewise-linear interval.

The max of four affine separating-axis functions is convex piecewise-linear;
its minimum occurs at an endpoint or a pairwise line intersection. Parallel
lines within 1e-12 metres/interval are treated as parallel for this auxiliary
calculation only. This does not alter the canonical PET implementation.
"""
    xy,dims,others=_inputs(future,dimensions,ego_index)
    relative=xy[:,others]-xy[:,ego_index,None];half=(dims[ego_index]+dims[others])/2
    start=relative[:-1];delta=relative[1:]-start
    a=torch.stack((start[...,0]-half[:,0],-start[...,0]-half[:,0],
                   start[...,1]-half[:,1],-start[...,1]-half[:,1]),-1)
    b=torch.stack((delta[...,0],-delta[...,0],delta[...,1],-delta[...,1]),-1)
    i,j=torch.triu_indices(4,4,1,device=future.device)
    den=b[...,i]-b[...,j];parallel=den.abs()<=1e-12
    frac=(a[...,j]-a[...,i])/torch.where(parallel,torch.ones_like(den),den)
    valid=(~parallel)&torch.isfinite(frac)&(frac>=0)&(frac<=1)
    frac=torch.where(valid,frac,torch.zeros_like(frac))
    ends=torch.stack((torch.zeros_like(frac[...,0]),torch.ones_like(frac[...,0])),-1)
    locations=torch.cat((ends,frac),-1)
    choices=(a[...,None,:]+b[...,None,:]*locations[...,None]).amax(-1)
    valid=torch.cat((torch.ones_like(ends,dtype=torch.bool),valid),-1)
    return choices.masked_fill(~valid,torch.inf).min()


def encounter_support_loss(future,dimensions,current_exact_pet,target_pet,*,eligible=True,ego_index=0,margin_m=.02):
    """Recruit only capped outputs with q<4; escape only zero outputs with q>0."""
    if not math.isfinite(margin_m) or margin_m<0:raise ValueError('nonnegative spatial training margin required')
    current=float(current_exact_pet);target=float(torch.as_tensor(target_pet).detach())
    if not math.isfinite(current) or not 0<=current<=4 or not math.isfinite(target) or not 0<=target<=4:
        raise ValueError('finite exact/target PET in[0,4] required')
    zero=future.sum()*0.
    recruit=bool(eligible) and current==4. and target<4.
    escape=bool(eligible) and current==0. and target>0.
    gap=None;loss=zero
    if recruit:
        gap=target_lag_gap(future,dimensions,target,ego_index=ego_index)
        loss=torch.relu(gap+margin_m)+zero
    elif escape:
        gap=continuous_same_time_gap(future,dimensions,ego_index=ego_index)
        loss=torch.relu(margin_m-gap)+zero
    return dict(loss=loss,active=recruit or escape,recruit=recruit,escape=escape,
        signed_gap_m=None if gap is None else float(gap.detach()),margin_m=margin_m,
        exact_PET_or_target_changed=False,not_an_exact_PET_gradient=True,
        all_geometric_configurations_guaranteed_trainable=False)


def batch_encounter_losses(coefficients,decoders,dimensions,agent_mask,current_exact_pet,target_pet,
                           eligible_present,*,margin_m=.02):
    """First-ego FIT roster, explicit padding/presence, no CDF density masking."""
    b=len(coefficients)
    if (coefficients.ndim!=4 or agent_mask.shape!=coefficients.shape[:2] or agent_mask.dtype!=torch.bool
            or len(decoders)!=b or len(dimensions)!=b or current_exact_pet.shape!=(b,) or target_pet.shape!=(b,)
            or eligible_present.shape!=(b,) or eligible_present.dtype!=torch.bool):
        raise ValueError('aligned variable-N scenes and explicit request eligibility required')
    rows=[]
    for i in range(b):
        future=decoders[i](coefficients[i,agent_mask[i]])
        rows.append(encounter_support_loss(future,dimensions[i],current_exact_pet[i],target_pet[i],
            eligible=bool(eligible_present[i]),ego_index=0,margin_m=margin_m))
    return dict(per_scene_loss=torch.stack([r['loss'] for r in rows]),
        support_mask=torch.tensor([r['active'] for r in rows],dtype=torch.bool,device=coefficients.device),
        recruit_count=sum(r['recruit'] for r in rows),escape_count=sum(r['escape'] for r in rows),
        diagnostics=[{k:v for k,v in r.items() if k!='loss'} for r in rows])
