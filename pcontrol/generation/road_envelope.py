"""Sampling-time lateral K8 coefficient projection onto native-frame roads.

The small convex problem uses no observed future. Initial outside footprints
receive their fixed t0 allowance; raw evaluation still counts them. Bounds
are enforced at the175 native frames, NOT certified continuously in time.
Failures are reported and the input is retained, never dropped or selected.
"""
import numpy as np
import torch
from scipy.optimize import LinearConstraint,minimize


class RoadEnvelope:
    def __init__(self,decoder,dimensions,road_boundaries,inverse_metric_y,*,inset_m=.002):
        self.decoder=decoder
        dims=np.asarray(dimensions,dtype=np.float64);road=np.asarray(road_boundaries,dtype=np.float64)
        anchors=decoder.anchors.detach().cpu().numpy()
        if (dims.shape!=(len(anchors),2) or road.ndim!=1 or len(road)<2 or not np.isfinite(road).all()
                or not np.all(np.diff(road)>0) or not np.isfinite(dims).all() or np.any(dims<=0)
                or not np.isfinite(inset_m) or inset_m<0):raise ValueError('valid fixed geometry and nonnegative inset required')
        lower=road[0]+dims[:,1]/2;upper=road[-1]-dims[:,1]/2
        if np.any(lower>=upper):raise ValueError('road must fit each vehicle width')
        allowance_left=np.maximum(lower-anchors[:,1],0.);allowance_right=np.maximum(anchors[:,1]-upper,0.)
        self.lower=lower-allowance_left;self.upper=upper+allowance_right
        clearance=np.minimum(anchors[:,1]-self.lower,self.upper-anchors[:,1])
        margin=np.minimum(float(inset_m),np.maximum(clearance,0.)*.5)
        self.lower=self.lower+margin;self.upper=self.upper-margin
        metric=inverse_metric_y.detach().cpu().numpy()
        eigen,vectors=np.linalg.eigh(metric)
        if np.any(eigen<=0):raise ValueError('positive metric required')
        self.whitening=(vectors*np.sqrt(eigen)[None])@vectors.T
        self.basis=(decoder.bp*decoder.scale[:,1]).detach().cpu().numpy()
        self.A=self.basis[1:]@self.whitening
        self.checks=0;self.qp_calls=0;self.failures=0;self.max_shift_m=0.

    def project(self,coefficients):
        if coefficients.ndim!=3 or coefficients.shape[0]!=len(self.lower):raise ValueError('complete unpadded coefficients required')
        self.checks+=1;original=coefficients.detach();future=self.decoder(original).detach().cpu().numpy()
        c=original.cpu().numpy().astype(np.float64).copy();changed=0;failed=[]
        for actor in range(len(c)):
            y=future[1:,actor,1];lo=self.lower[actor]-y;hi=self.upper[actor]-y
            if np.max(np.maximum(lo,-hi))<=1e-10:continue
            self.qp_calls+=1
            result=minimize(lambda z:.5*np.dot(z,z),np.zeros(c.shape[1]),jac=lambda z:z,
                constraints=[LinearConstraint(self.A,lo,hi)],method='SLSQP',options=dict(ftol=1e-10,maxiter=80,disp=False))
            step=self.whitening@result.x;new_y=y+self.basis[1:]@step
            violation=float(np.maximum(self.lower[actor]-new_y,new_y-self.upper[actor]).max())
            if not np.isfinite(step).all() or violation>1e-7:
                self.failures+=1;failed.append(actor);continue
            c[actor,:,1]+=step;changed+=1
        answer=torch.as_tensor(c,dtype=coefficients.dtype,device=coefficients.device)
        shift=float((self.decoder(answer)[...,:2]-self.decoder(original)[...,:2]).norm(dim=-1).max())
        self.max_shift_m=max(self.max_shift_m,shift)
        return answer,dict(projected_actors=changed,failed_actors=failed,max_position_shift_m=shift,
            native_frame_constraints_only=True,t0_allowance_kept=True)

    def report(self):
        return dict(projection_checks=self.checks,QP_solves=self.qp_calls,QP_failures=self.failures,
            maximum_projection_position_shift_m=self.max_shift_m,observed_future_input=False,
            continuous_time_road_certified=False)
