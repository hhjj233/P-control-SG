"""Differentiable branchwise version of the audited scalar midrank target.

CDF tables remain detached. Discrete candidate selection is detached, like the
existing generalized inverse's piece selection; gradients within the selected
continuous branch remain connected to the requested P. No future is read.
"""
import torch

from pcontrol.reference.torch_frozen_inverse import quantile_from_pieces


def _rank_at(pieces,y):
    """One exact scalar query per row, retaining the first real cap knot."""
    x,left,right=pieces.unbind(-1)
    index=torch.argmax((x>=y[:,None]).to(torch.int64),dim=1,keepdim=True)
    previous=(index-1).clamp_min(0)
    hi=x.gather(1,index)[:,0];lo=x.gather(1,previous)[:,0]
    lhi=left.gather(1,index)[:,0];rhi=right.gather(1,index)[:,0];rlo=right.gather(1,previous)[:,0]
    interpolated=rlo+(lhi-rlo)*(y-lo)/torch.where(hi>lo,hi-lo,torch.ones_like(hi))
    return torch.where(y==hi,1.-.5*(lhi+rhi),1.-interpolated)


def midrank_target_from_pieces(pieces,p,*,inward_margin_seconds=.001,excess_rank_tolerance=1e-5,return_details=False):
    """Return PET[B], differentiable in p away from branch changes.

    Equivalent to select_midrank_target's candidate set and tie policy. This
    is a scalar-reference projection, not a physical reachability guarantee.
    It does not replace the true CDF table with an invented inverse curve.
    """
    if not 0<inward_margin_seconds<=.1 or not 0<excess_rank_tolerance<=1e-3:
        raise ValueError('bounded positive target tolerances required')
    canonical=quantile_from_pieces(pieces,1.-p.double())  # Also validates the table/query contract.
    x,left,right=pieces.unbind(-1);dx=x[:,1:]-x[:,:-1]
    repeat=dx==0
    if bool((repeat&((x[:,1:]!=4)|(left[:,1:]!=1)|(right[:,1:]!=1))).any()):
        raise ValueError('only trailing[4,1,1] inverse padding allowed')
    interior=(x>0)&(x<4)
    if bool((interior&((left-right).abs()>1e-12)).any()):raise ValueError('endpoint atoms only')
    nodes_valid=torch.cat((torch.ones_like(x[:,:1],dtype=torch.bool),dx>0),dim=1)
    x0,x1=x[:,:-1],x[:,1:];u0,u1=right[:,:-1],left[:,1:]
    safe_dx=torch.where(dx>0,dx,torch.ones_like(dx));du=u1-u0
    density=du.abs()/safe_dx
    first=torch.nextafter(x0,x1);last=torch.nextafter(x1,x0)
    valid=(dx>0)&(first<x1)&(last>x0)&(first<=last)
    margin=torch.minimum(torch.full_like(dx,inward_margin_seconds),dx/4.)
    rank_margin=excess_rank_tolerance/torch.where(density>0,density,torch.ones_like(density))
    margin=torch.minimum(margin,torch.where(density>0,rank_margin,torch.full_like(dx,torch.inf)))
    a=torch.maximum(first,x0+margin);b=torch.minimum(last,x1-margin)
    inverted=a>b;a=torch.where(inverted,first,a);b=torch.where(inverted,last,b)
    query=1.-p.double()[:,None]
    root=x0+(query-u0)/torch.where(du>0,du,torch.ones_like(du))*dx
    root_valid=valid&(du>0)&(query>u0)&(query<u1)&(root>x0)&(root<x1)
    middle=.5*(x0+x1);flat_valid=valid&(du==0)
    def rank_open(y):return 1.-(u0+du*(y-x0)/safe_dx)
    ys=torch.cat((canonical[:,None],x,a,b,root,middle),dim=1)
    ranks=torch.cat((_rank_at(pieces,canonical)[:,None],1.-.5*(left+right),rank_open(a),rank_open(b),rank_open(root),1.-u0),dim=1)
    masks=torch.cat((torch.ones_like(canonical[:,None],dtype=torch.bool),nodes_valid,valid,valid,root_valid,flat_valid),dim=1)
    # Exclude every padded/invalid candidate before min/tie selection. Selection
    # itself is a discrete branch; only the selected target carries p gradients.
    with torch.no_grad():
        error=(ranks.detach()-p.detach().double()[:,None]).abs()
        error=torch.where(masks,error,torch.full_like(error,torch.inf))
        best=error.min(1,keepdim=True).values
        tied=masks&(error<=best+1e-12)
        distance=(ys.detach()-canonical.detach()[:,None]).abs()
        distance=torch.where(tied,distance,torch.full_like(distance,torch.inf))
        shortest=distance.min(1,keepdim=True).values
        tied=tied&(distance==shortest)
        ys_tied=torch.where(tied,ys.detach(),torch.full_like(ys,torch.inf))
        index=ys_tied.argmin(1,keepdim=True)
        # NumPy favors the canonical candidate on exact distance/value ties.
        index=torch.where(tied[:,:1],torch.zeros_like(index),index)
    selected=ys.gather(1,index)[:,0]
    if not bool(torch.isfinite(selected).all()) or bool(((selected<0)|(selected>4)).any()):raise FloatingPointError('invalid scalar target')
    if not return_details:return selected
    return selected,dict(canonical_target=canonical.detach(),selected_rank=_rank_at(pieces,selected).detach(),
        selected_error=error.gather(1,index)[:,0],target_changed=(selected.detach()!=canonical.detach()),
        selection_index=index[:,0],physical_reachability_proven=False,reference_modified=False)
