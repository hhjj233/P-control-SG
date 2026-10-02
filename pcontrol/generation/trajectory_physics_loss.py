"""Training-only physical violation costs; never repairs a sampled trajectory.

Tail mean (upper 10% by default) of linear, unit-normalized hinge violations
keeps sparse multi-agent failures from disappearing in a full-scene mean.
This is a soft penalty, not a dynamics model or a hard feasibility guarantee.
"""
import math

import torch


def tail_mean(nonnegative, fraction=.1):
    if (not isinstance(nonnegative,torch.Tensor) or nonnegative.numel()==0
            or not nonnegative.is_floating_point() or not bool(torch.isfinite(nonnegative).all())
            or bool((nonnegative<0).any()) or not math.isfinite(float(fraction)) or not 0<fraction<=1):
        raise ValueError('finite nonnegative values and tail fraction in (0,1] required')
    flat=nonnegative.flatten();k=max(1,math.ceil(float(fraction)*flat.numel()))
    return torch.topk(flat,k,sorted=False).values.mean()


def scene_physics_loss(future,dimensions,road_boundaries,*,tail_fraction=.1,
                       road_scale_m=1.,speed_scale_mps=1.):
    """One unpadded physical F[T,N,xyvxvy], every actor, t>0 only.

The detached observed t0 supplies a per-actor, per-side allowance for existing
outside footprints and a per-actor allowance for existing negative vx. Only
increases beyond that allowance are optimized. Raw violations are separately
reported without that allowance, so official evaluation is not redefined.
At max/hinge/top-k ties autograd selects a subgradient; no smoothness claim.
"""
    if (not isinstance(future,torch.Tensor) or future.ndim!=3 or future.shape[0]<2
            or future.shape[1]<1 or future.shape[2]!=4 or not future.is_floating_point()
            or not bool(torch.isfinite(future).all())):
        raise ValueError('finite unpadded physical future[T>=2,N,4] required')
    device=future.device
    dims=torch.as_tensor(dimensions,dtype=torch.float64,device=device).detach()
    road=torch.as_tensor(road_boundaries,dtype=torch.float64,device=device).detach()
    if (dims.shape!=(future.shape[1],2) or road.ndim!=1 or road.numel()<2
            or not bool(torch.isfinite(dims).all()) or not bool(torch.isfinite(road).all())
            or bool((dims<=0).any()) or not bool((road[1:]>road[:-1]).all())):
        raise ValueError('positive dimensions and strictly increasing physical road boundaries required')
    if any(not math.isfinite(float(v)) or v<=0 for v in (road_scale_m,speed_scale_mps)):
        raise ValueError('positive finite physical unit scales required')
    f=future.to(torch.float64);anchor=f[0].detach();width=dims[:,1]
    allowance_left=torch.relu(road[0]-(anchor[:,1]-width/2))
    allowance_right=torch.relu(anchor[:,1]+width/2-road[-1])
    allowance_reverse=torch.relu(-anchor[:,2])
    left=torch.relu(road[0]-(f[1:,:,1]-width[None]/2))
    right=torch.relu(f[1:,:,1]+width[None]/2-road[-1])
    reverse=torch.relu(-f[1:,:,2])
    extra_road=torch.maximum(torch.relu(left-allowance_left[None]),torch.relu(right-allowance_right[None]))
    extra_reverse=torch.relu(reverse-allowance_reverse[None])
    road_loss=tail_mean(extra_road/float(road_scale_m),tail_fraction)
    speed_loss=tail_mean(extra_reverse/float(speed_scale_mps),tail_fraction)
    raw_road=torch.maximum(left,right)
    return dict(road_loss=road_loss,speed_loss=speed_loss,loss=road_loss+speed_loss,
        diagnostics=dict(actors=f.shape[1],future_actor_frames=extra_road.numel(),
            t0_road_outside=bool(((allowance_left>0)|(allowance_right>0)).any()),
            t0_negative_vx=bool((allowance_reverse>0).any()),
            future_raw_road_outside=bool((raw_road>0).any()),future_raw_negative_vx=bool((reverse>0).any()),
            maximum_raw_road_depth_m=float(raw_road.detach().max()),maximum_raw_reverse_mps=float(reverse.detach().max()),
            maximum_excess_road_depth_m=float(extra_road.detach().max()),maximum_excess_reverse_mps=float(extra_reverse.detach().max()),
            road_has_training_violation=bool((extra_road>0).any()),speed_has_training_violation=bool((extra_reverse>0).any()),
            tail_fraction=float(tail_fraction),t0_excluded_from_decision=True,
            existing_t0_violation_allowance_detached=True,all_real_actors_included=True,
            proposal_modified=False,hard_feasibility_guaranteed=False))


def batch_physics_loss(coefficients,decoders,dimensions,roads,agent_mask,condition_present,**kwargs):
    b=coefficients.shape[0]
    if (coefficients.ndim!=4 or agent_mask.shape!=coefficients.shape[:2] or agent_mask.dtype!=torch.bool
            or agent_mask.device!=coefficients.device or condition_present.shape!=(b,)
            or condition_present.dtype!=torch.bool or condition_present.device!=coefficients.device
            or any(len(value)!=b for value in (decoders,dimensions,roads))):
        raise ValueError('paired masked batch, static geometry, decoders and explicit condition presence required')
    results=[scene_physics_loss(decode(coefficients[i,agent_mask[i]]),dimensions[i],roads[i],**kwargs)
             for i,decode in enumerate(decoders)]
    road=torch.stack([r['road_loss'] for r in results]);speed=torch.stack([r['speed_loss'] for r in results])
    present=condition_present.detach();count=int(present.sum())
    road_loss=torch.where(present,road,torch.zeros_like(road)).sum()/max(count,1)
    speed_loss=torch.where(present,speed,torch.zeros_like(speed)).sum()/max(count,1)
    return dict(road_loss=road_loss,speed_loss=speed_loss,loss=road_loss+speed_loss,
        total_scenes=b,present_scenes=count,diagnostics=[r['diagnostics'] for r in results],
        raw_all_request_road_mean=float(road.detach().mean()),raw_all_request_speed_mean=float(speed.detach().mean()))
