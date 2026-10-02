"""Explicit P-conditioned, frozen-risk-guided DDIM; no observed future input.

Guidance acts on predicted clean coefficients INSIDE each declared late DDIM
step, including its final reverse transition. No best-of-K, line search, or
correction after the sampler returns. Local PET Newton steps are not global
feasibility guarantees. Capped/zero PET can use separately labelled spatial
bootstrap gradients, never passed off as exact PET gradients.
"""
import math
import numpy as np
import torch
from .direct_p_cfg import _CFGPrediction
from .diffusion import ddim_sample
from .sampling_geometry import SamplingGeometry
from .encounter_support_loss import target_lag_gap,continuous_same_time_gap


class PercentileSamplingGuidance:
    def __init__(self,decoder,dimensions,anchors,rank_fn,requested_p,target_pet,*,
                 total_steps=50,last_steps=20,inner_steps=2,strength=.5,
                 position_step_cap_m=.5,p_stop_tolerance=.01,bootstrap=True,ego_index=0):
        if (not 0<=requested_p<=1 or not 0<=target_pet<=4 or total_steps!=50
                or type(last_steps) is not int or not 0<=last_steps<=total_steps
                or type(inner_steps) is not int or inner_steps<1
                or any(not math.isfinite(float(x)) for x in [strength,position_step_cap_m,p_stop_tolerance])
                or strength<0 or position_step_cap_m<=0 or not 0<=p_stop_tolerance<=.05):
            raise ValueError('finite target/guidance settings and declared DDIM50 budget required')
        self.decoder=decoder;self.dimensions=np.asarray(dimensions,dtype=np.float64)
        anchors=np.asarray(anchors,dtype=np.float64);n=len(anchors)
        if type(ego_index) is not int or not 0<=ego_index<n:raise ValueError('valid ego index required')
        self.order=[ego_index]+[i for i in range(n) if i!=ego_index]
        self.geometry=SamplingGeometry(self.dimensions[self.order],anchors[self.order])
        self.rank_fn=rank_fn;self.requested_p=float(requested_p);self.target=float(target_pet)
        self.total_steps=total_steps;self.last_steps=last_steps;self.inner_steps=inner_steps
        self.strength=float(strength);self.position_cap=float(position_step_cap_m)
        self.tolerance=float(p_stop_tolerance);self.bootstrap=bool(bootstrap);self.ego_index=ego_index
        self.calls=0;self.rank_calls=0;self.trace=[]

    def __call__(self,x0,timesteps,x_t):
        if x0.shape[0]!=1 or x0.shape[1]!=len(self.dimensions):raise ValueError('one complete unpadded scene required')
        step=self.calls;self.calls+=1
        if self.strength==0 or step<self.total_steps-self.last_steps:return x0
        current=x0[0].detach().double().clone()
        for inner in range(self.inner_steps):
            with torch.enable_grad():
                state=current.detach().requires_grad_(True);future=self.decoder(state)
                before=self.geometry.metric_calls;active=self.geometry.active_witness_pet(future[:,self.order])
                score=active['exact_score'];pet=float(score['pet_seconds'])
                achieved=float(self.rank_fn(pet));self.rank_calls+=1
                if not math.isfinite(achieved) or not 0<=achieved<=1:raise ValueError('finite estimated percentile required')
                row=dict(step=step,inner=inner,timestep=int(timesteps[0]),PET=pet,achieved_P=achieved,
                    P_error=abs(achieved-self.requested_p),target_PET=self.target,
                    exact_geometry_supported=bool(active['supported']),geometry_reason=active['reason'])
                signal=None;error=None;kind='unsupported'
                if row['P_error']<=self.tolerance:kind='within_target_band'
                elif active['supported']:
                    signal=active['value'];error=signal.detach()-self.target;kind='local_exact_PET'
                elif self.bootstrap and pet==4. and self.target<4.:
                    signal=target_lag_gap(future,self.dimensions,self.target,ego_index=self.ego_index)
                    error=torch.relu(signal.detach()+.02);kind='spatial_recruit_not_PET_gradient'
                elif self.bootstrap and pet==0. and self.target>0.:
                    signal=continuous_same_time_gap(future,self.dimensions,ego_index=self.ego_index)
                    error=-torch.relu(.02-signal.detach());kind='spatial_escape_not_PET_gradient'
                if signal is not None:
                    gradient=torch.autograd.grad(signal,state)[0];norm2=gradient.square().sum()
                    if not bool(torch.isfinite(gradient).all()):raise FloatingPointError('nonfinite sampling gradient')
                    if float(norm2)>1e-14:
                        delta=-self.strength*error*gradient/norm2
                        # K8 decoder is affine: this bounds the actual xy change
                        # at all175 native frames of this inner update.
                        change=self.decoder(state.detach()+delta.detach())[...,:2]-future.detach()[...,:2]
                        maximum=change.norm(dim=-1).max()
                        factor=torch.clamp(maximum.new_tensor(self.position_cap)/maximum.clamp_min(1e-15),max=1.)
                        delta=delta*factor
                        if not bool(torch.isfinite(delta).all()):raise FloatingPointError('nonfinite guidance update')
                        current=state.detach()+delta.detach()
                        row.update(coefficient_step_L2=float(delta.norm()),max_position_step_m=float(maximum*factor))
                    else:kind='flat_coefficient_gradient'
                row.update(kind=kind,exact_metric_calls=self.geometry.metric_calls-before)
                self.trace.append(row)
            if kind in ('within_target_band','unsupported','flat_coefficient_gradient'):break
        return current[None].to(x0.dtype)

    def report(self):
        return dict(callback_invocations=self.calls,active_inner_checks=len(self.trace),
            exact_metric_calls=self.geometry.metric_calls,CDF_rank_queries=self.rank_calls,
            risk_steps=sum(r['kind']=='local_exact_PET' for r in self.trace),
            bootstrap_steps=sum(r['kind'].startswith('spatial_') for r in self.trace),
            trace=self.trace,observed_future_input=False,post_sampler_correction=False,
            guidance_in_final_reverse_transition=True,no_best_of_K=True)


def percentile_guided_sample(model,schedule,features,p,initial_noise,guide,*,scale=2.5,steps=50):
    if steps!=50 or guide.total_steps!=steps or float(p[0])!=float(torch.tensor(guide.requested_p,dtype=p.dtype)):
        raise ValueError('same requested P and DDIM50 schedule required')
    if model.training or any(v.requires_grad for v in model.parameters()):raise ValueError('frozen eval-mode generator required')
    adapter=_CFGPrediction(model,features,p,float(scale))
    sample=ddim_sample(adapter,schedule,features,initial_noise,steps=steps,x0_callback=guide,prediction_type='v')
    return dict(sample=sample,network_evaluations=adapter.network_evaluations,guidance=guide.report(),
        CFG_scale=float(scale),P_condition_retained=True,inference_risk_queries_enabled=True)
