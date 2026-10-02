"""Road-constrained single-path guidance with feasible one-sided PET probes.

All operations occur inside declared DDIM callbacks, never after the sampler
returns. Initial clean estimates may be projected onto road bounds; risk
increments then remain within that convex coefficient set. Queries are on
generated states only; the one-sided difference is not a full PET gradient.
"""
import math
import torch
from .directional_percentile_guidance import DirectionalPercentileGuidance
from .road_envelope import RoadEnvelope
from .encounter_support_loss import target_lag_gap,continuous_same_time_gap


class RoadConstrainedGuidance(DirectionalPercentileGuidance):
    def __init__(self,*args,road_boundaries,road_inset_m=.002,guidance_stride=1,dense_last_steps=5,**kwargs):
        super().__init__(*args,**kwargs)
        if type(guidance_stride) is not int or guidance_stride<1 or not 0<=dense_last_steps<=50:raise ValueError('valid fixed guidance schedule required')
        self.envelope=RoadEnvelope(self.decoder,self.dimensions,road_boundaries,self.inverse_metric[1],inset_m=road_inset_m)
        self.stride=guidance_stride;self.dense_last_steps=dense_last_steps;self.road_trace=[]

    def __call__(self,x0,timesteps,x_t):
        if x0.shape[0]!=1 or x0.shape[1]!=len(self.dimensions) or self.calls>=50:raise ValueError('one fresh complete sampling path required')
        step=self.calls;self.calls+=1;start=50-self.last_steps
        if self.strength==0 or step<start:return x0
        if step<50-self.dense_last_steps and (step-start)%self.stride!=0:return x0
        current,repair=self.envelope.project(x0[0].detach().double())
        self.road_trace.append(dict(step=step,phase='clean_estimate_envelope',**repair))
        for inner in range(self.inner_steps):
            with torch.enable_grad():
                state=current.detach().requires_grad_(True);future=self.decoder(state);before=self.geometry.metric_calls
                active=self.geometry.local_candidate(future[:,self.order]);pet=float(active['exact_score']['pet_seconds'])
                achieved=float(self.rank_fn(pet));self.rank_calls+=1
                if not math.isfinite(achieved) or not 0<=achieved<=1:raise ValueError('finite rank required')
                row=dict(step=step,inner=inner,timestep=int(timesteps[0]),PET=pet,achieved_P=achieved,
                    P_error=abs(achieved-self.requested_p),target_PET=self.target,local_candidate_available=active['available'],geometry_reason=active['reason'])
                delta=None;kind='unsupported'
                if row['P_error']<=self.tolerance:kind='within_target_band'
                elif active['available']:
                    gradient=torch.autograd.grad(active['value'],state)[0]
                    direction=self._direction(gradient)*(-1. if pet>self.target else 1.)
                    change=self.decoder(state.detach()+direction.detach())[...,:2]-future.detach()[...,:2]
                    size=float(change.norm(dim=-1).max())
                    if math.isfinite(size) and size>1e-14:
                        probe_state,projection=self.envelope.project(state.detach()+direction.detach()*(self.probe_position_m/size))
                        probe=probe_state-state.detach();next_pet=self.geometry.score_future(self.decoder(probe_state)[:,self.order])['pet_seconds']
                        derivative=next_pet-pet
                        row.update(probe_PET=next_pet,directional_PET_difference=derivative,probe_projected_actors=projection['projected_actors'])
                        if math.isfinite(derivative) and abs(derivative)>1e-10 and (pet-self.target)*derivative<0:
                            delta=-self.strength*(pet-self.target)*probe/derivative;kind='feasible_one_sided_direction'
                        else:kind='no_feasible_descent_measured'
                elif self.bootstrap and ((pet==4. and self.target<4.) or (pet==0. and self.target>0.) or (pet>self.target and self.target<4.)):
                    recruit=pet>self.target
                    signal=(target_lag_gap(future,self.dimensions,self.target,ego_index=self.ego_index) if recruit
                        else continuous_same_time_gap(future,self.dimensions,ego_index=self.ego_index))
                    error=torch.relu(signal.detach()+.02) if recruit else -torch.relu(.02-signal.detach())
                    gradient=torch.autograd.grad(signal,state)[0];direction=self._direction(gradient);denominator=(gradient*direction).sum()
                    if float(denominator)>1e-14:
                        delta=-self.strength*error*direction/denominator
                        kind='spatial_recruit_not_PET_gradient' if recruit else 'spatial_escape_not_PET_gradient'
                if delta is not None:
                    if not bool(torch.isfinite(delta).all()):raise FloatingPointError('nonfinite constrained guidance')
                    raw_change=self.decoder(state.detach()+delta.detach())[...,:2]-future.detach()[...,:2]
                    raw_maximum=raw_change.norm(dim=-1).max()
                    delta=delta*(raw_maximum.new_tensor(self.position_cap)/raw_maximum.clamp_min(1e-15)).clamp(max=1.)
                    projected,repair=self.envelope.project(state.detach()+delta.detach());delta=projected-state.detach()
                    change=self.decoder(projected)[...,:2]-future.detach()[...,:2]
                    maximum=change.norm(dim=-1).max();factor=(maximum.new_tensor(self.position_cap)/maximum.clamp_min(1e-15)).clamp(max=1.)
                    delta=delta*factor;current=state.detach()+delta.detach()
                    row.update(coefficient_step_L2=float(delta.norm()),max_position_step_m=float(maximum*factor))
                    self.road_trace.append(dict(step=step,inner=inner,phase='risk_increment_envelope',**repair))
                    if repair['failed_actors']:
                        current=state.detach();delta=None;kind='road_projection_failed'
                row.update(kind=kind,exact_metric_calls=self.geometry.metric_calls-before);self.trace.append(row)
            if delta is None:break
        return current[None].to(x0.dtype)

    def report(self):
        result=super().report()
        result.update(risk_steps=sum(r['kind']=='feasible_one_sided_direction' for r in self.trace),
            one_sided_feasible_probe=True,road_envelope=self.envelope.report(),road_trace=self.road_trace,
            guidance_stride=self.stride,dense_last_steps=self.dense_last_steps,
            initial_clean_projection_not_subject_to_risk_step_cap=True)
        return result
