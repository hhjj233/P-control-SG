"""Exact mixed-CDF inverse with frozen physical pieces and query gradients.

This adds no estimator, fitted parameter, future observation, or data access.
It compiles the SAME calibrated physical pieces as SceneCDFWarp.quantile,
including within-bin warp crossings. Only requested quantile/P can be active.
Derivatives at jumps/knots are branch conventions, not smoothness guarantees.
"""
import numpy as np
import torch

from .scene_calibration import _physical_pieces
from .torch_frozen_cdf import FrozenTorchCDF


PIECES_KEY='frozen_cdf_physical_pieces'


def quantile_from_pieces(pieces,u):
    """Frozen [B,72,3] physical/left/right table; differentiable u[B]."""
    if (not isinstance(pieces,torch.Tensor) or pieces.ndim!=3 or pieces.shape[1:]!=(72,3)
            or pieces.dtype!=torch.float64 or pieces.requires_grad
            or not isinstance(u,torch.Tensor) or u.dtype not in (torch.float32,torch.float64)
            or u.shape!=(len(pieces),) or u.device!=pieces.device
            or not bool(torch.isfinite(u).all()) or bool(((u<0)|(u>1)).any())):
        raise ValueError('frozen float64[B,72,3] pieces and finite float u[B] required')
    x,left,right=pieces.unbind(-1)
    if (not bool(torch.isfinite(pieces).all()) or bool(((x<0)|(x>4)).any())
            or bool(((left<0)|(left>1)|(right<0)|(right>1)).any())
            or bool((left>right+1e-12).any()) or bool((torch.diff(x,dim=1)<0).any())
            or bool((torch.diff(right,dim=1)<-1e-12).any())
            or bool((left[:,1:]<right[:,:-1]-1e-12).any())
            or bool((x[:,0]!=0).any()) or bool((x[:,-1]!=4).any())
            or bool((left[:,0]!=0).any()) or bool((right[:,-1]!=1).any())):
        raise ValueError('invalid compiled mixed-CDF pieces')
    q=u.double()[:,None]
    index=torch.argmax((right>=q.detach()).to(torch.int64),dim=1,keepdim=True)
    previous=(index-1).clamp_min(0)
    lo=right.gather(1,previous);hi=left.gather(1,index)
    x0=x.gather(1,previous);x1=x.gather(1,index)
    value=x0+(q-lo)/torch.where(hi>lo,hi-lo,torch.ones_like(hi))*(x1-x0)
    value=torch.where((index==0)|(q>hi),x1,value)
    value=torch.where(q<=right[:,:1],torch.zeros_like(value),value)
    value=torch.where(q==0,torch.zeros_like(value),torch.where(q==1,torch.full_like(value,4.),value))
    return value[:,0]


class FrozenTorchInverseCDF(FrozenTorchCDF):
    VERSION='frozen_torch_exact_scene_CDF_and_inverse_v1'

    def __init__(self,masses,counts,warp=None,base_knots=None,*,row_nodes=None):
        super().__init__(masses,counts,warp=warp,base_knots=base_knots,row_nodes=row_nodes)
        mass=self.masses.cpu().numpy();knots=self.base_knots.cpu().numpy()
        nodes=self.row_nodes.cpu().numpy();u=self.u_knots.cpu().numpy()
        capacity=len(knots)+len(u)-2
        x=np.full((len(mass),capacity),4.,dtype=np.float64)
        left=np.ones_like(x);right=np.ones_like(x);valid=np.zeros_like(x,dtype=bool)
        for i in range(len(mass)):
            physical,lb,rb=_physical_pieces(mass[i],knots,u)
            n=len(physical)
            x[i,:n]=physical;left[i,:n]=lb@nodes[i];right[i,:n]=rb@nodes[i];valid[i,:n]=True
        for name,values in dict(inverse_x=x,inverse_left=left,inverse_right=right,inverse_valid=valid).items():
            self.register_buffer(name,torch.tensor(values,device=self.masses.device))

    def quantile(self,u):
        query,q=self._query(u)
        if not bool(torch.isfinite(q).all()) or bool(((q<0)|(q>1)).any()):
            raise ValueError('finite quantile levels in [0,1] required')
        # Detach only discrete piece selection, not interpolation within it.
        index=torch.argmax((self.inverse_right[:,None]>=q.detach()[...,None]).to(torch.int64),dim=-1)
        previous=(index-1).clamp_min(0)
        lo=self.inverse_right.gather(1,previous);hi=self.inverse_left.gather(1,index)
        x0=self.inverse_x.gather(1,previous);x1=self.inverse_x.gather(1,index)
        denominator=torch.where(hi>lo,hi-lo,torch.ones_like(hi))
        value=x0+(q-lo)/denominator*(x1-x0)
        value=torch.where((index==0)|(q>hi),x1,value)
        value=torch.where(q<=self.inverse_right[:,:1],torch.zeros_like(value),value)
        value=torch.where(q==0,torch.zeros_like(value),torch.where(q==1,torch.full_like(value,4.),value))
        return value.reshape(query.shape)

    def target_pet(self,p):
        if not isinstance(p,torch.Tensor) or p.dtype not in (torch.float32,torch.float64):
            raise TypeError('p must be a float32/float64 tensor')
        return self.quantile(1.-p.to(torch.float64))

    def compiled_pieces(self):
        return torch.stack((self.inverse_x,self.inverse_left,self.inverse_right),dim=-1).detach()
