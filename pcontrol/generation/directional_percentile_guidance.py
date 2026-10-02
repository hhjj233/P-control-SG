"""Single-path DDIM guidance using measured coefficient-direction derivatives.

Local witness derivatives only PROPOSE directions. Two complete-scene PET
probes of physical coefficient perturbations determine the step derivative.
No ambient-coordinate differentiability or full-gradient guarantee is made.
The fixed trajectory metric preconditions directions; it is not a trained
model, risk oracle shortcut, candidate selector or final-output repair.
"""
import math
import torch
from .percentile_sampling_guidance import PercentileSamplingGuidance
from .directional_sampling_geometry import DirectionalSamplingGeometry
from .trajectory_basis import basis_matrices
from .encounter_support_loss import target_lag_gap,continuous_same_time_gap


class DirectionalPercentileGuidance(PercentileSamplingGuidance):
    def __init__(self,*args,precondition=True,probe_position_m=1e-4,**kwargs):
        super().__init__(*args,**kwargs)
        if not math.isfinite(probe_position_m) or probe_position_m<=0:raise ValueError('positive physical FD probe required')
        self.geometry=DirectionalSamplingGeometry(self.dimensions[self.order],self.decoder.anchors.detach().cpu().numpy()[self.order])
        self.probe_position_m=float(probe_position_m);self.precondition=bool(precondition)
        decoder=self.decoder
        ba=torch.tensor(basis_matrices(decoder.basis.times,decoder.basis.modes,decoder.basis.horizon)[2],dtype=torch.float64,device=decoder.bp.device)
        operators=[]
        for axis in range(2):
            p=decoder.bp*decoder.scale[:,axis];v=decoder.bv*decoder.scale[:,axis];a=ba*decoder.scale[:,axis]
            gram=(p.T@p+.25*(v.T@v)+.0625*(a.T@a))/len(p)
            eigen,rotation=torch.linalg.eigh(gram);floor=eigen.max()*1e-4
            operators.append((rotation/eigen.clamp_min(floor)[None])@rotation.T)
        self.inverse_metric=torch.stack(operators)

    def _direction(self,gradient):
        if not self.precondition:return gradient
        return torch.stack([gradient[:,:,a]@self.inverse_metric[a].T for a in range(2)],-1)

    def __call__(self,x0,timesteps,x_t):
        if x0.shape[0]!=1 or x0.shape[1]!=len(self.dimensions):raise ValueError('one complete scene required')
        if self.calls>=self.total_steps:raise ValueError('a guide cannot be reused across sampling paths')
        step=self.calls;self.calls+=1
        if self.strength==0 or step<self.total_steps-self.last_steps:return x0
        current=x0[0].detach().double().clone()
        for inner in range(self.inner_steps):
            with torch.enable_grad():
                state=current.detach().requires_grad_(True);future=self.decoder(state);before=self.geometry.metric_calls
                active=self.geometry.local_candidate(future[:,self.order]);pet=float(active['exact_score']['pet_seconds'])
                achieved=float(self.rank_fn(pet));self.rank_calls+=1
                if not math.isfinite(achieved) or not 0<=achieved<=1:raise ValueError('finite rank required')
                row=dict(step=step,inner=inner,timestep=int(timesteps[0]),PET=pet,achieved_P=achieved,
                    P_error=abs(achieved-self.requested_p),target_PET=self.target,local_candidate_available=active['available'],
                    geometry_reason=active['reason']);delta=None;kind='unsupported'
                if row['P_error']<=self.tolerance:kind='within_target_band'
                elif active['available']:
                    gradient=torch.autograd.grad(active['value'],state)[0];direction=self._direction(gradient)
                    change=self.decoder(state.detach()+direction.detach())[...,:2]-future.detach()[...,:2]
                    size=float(change.norm(dim=-1).max())
                    if math.isfinite(size) and size>1e-14:
                        probe=direction.detach()*(self.probe_position_m/size)
                        plus=self.geometry.score_future(self.decoder(state.detach()+probe)[:,self.order])['pet_seconds']
                        minus=self.geometry.score_future(self.decoder(state.detach()-probe)[:,self.order])['pet_seconds']
                        derivative=(plus-minus)/2.
                        row.update(probe_plus_PET=plus,probe_minus_PET=minus,directional_PET_difference=derivative,
                            analytic_directional_prediction=float((gradient*probe).sum()))
                        if math.isfinite(derivative) and abs(derivative)>1e-10:
                            delta=-self.strength*(pet-self.target)*probe/derivative;kind='measured_coefficient_direction'
                        else:kind='flat_measured_direction'
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
                    if not bool(torch.isfinite(delta).all()):raise FloatingPointError('nonfinite directional guidance')
                    change=self.decoder(state.detach()+delta.detach())[...,:2]-future.detach()[...,:2]
                    maximum=change.norm(dim=-1).max();factor=(maximum.new_tensor(self.position_cap)/maximum.clamp_min(1e-15)).clamp(max=1.)
                    delta=delta*factor;current=state.detach()+delta.detach()
                    row.update(coefficient_step_L2=float(delta.norm()),max_position_step_m=float(maximum*factor))
                row.update(kind=kind,exact_metric_calls=self.geometry.metric_calls-before);self.trace.append(row)
            if delta is None:break
        return current[None].to(x0.dtype)

    def report(self):
        result=super().report()
        result.update(risk_steps=sum(r['kind']=='measured_coefficient_direction' for r in self.trace),
            coefficient_directional_probes=True,full_gradient_verified=False,precondition=self.precondition,
            probe_position_m=self.probe_position_m)
        return result
