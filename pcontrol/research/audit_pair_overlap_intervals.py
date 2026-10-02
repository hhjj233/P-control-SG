"""Independent closed-AABB overlap intervals under PL interpolation of xy.

This audit intersects x and y slab time intervals directly. It does not reuse
the generation-time lateral-window detector or claim smooth-K8 certification.
"""
import numpy as np


def pair_overlap_intervals(future,dimensions,ego_index):
    f=np.asarray(future,dtype=np.float64);d=np.asarray(dimensions,dtype=np.float64)
    if f.shape!=(175,len(d),4) or not np.isfinite(f).all() or not np.isfinite(d).all() or np.any(d<=0):raise ValueError('complete finite native trajectories required')
    output=[]
    for i in range(len(d)):
        for j in range(i+1,len(d)):
            r=f[:,j,:2]-f[:,i,:2];s=np.diff(r,axis=0);half=(d[i]+d[j])/2
            lo=np.zeros(174);hi=np.ones(174);valid=np.ones(174,bool)
            for axis in range(2):
                moving=s[:,axis]!=0.;den=np.where(moving,s[:,axis],1.)
                a=(-half[axis]-r[:-1,axis])/den;b=(half[axis]-r[:-1,axis])/den
                lo=np.maximum(lo,np.where(moving,np.minimum(a,b),0.))
                hi=np.minimum(hi,np.where(moving,np.maximum(a,b),1.))
                valid&=moving|(np.abs(r[:-1,axis])<=half[axis])
            valid&=lo<=hi
            if not valid.any():continue
            intervals=[]
            for k in np.flatnonzero(valid):
                start=(k+lo[k])*.04;end=(k+hi[k])*.04
                if intervals and start<=intervals[-1][1]+1e-12:intervals[-1][1]=max(intervals[-1][1],end)
                else:intervals.append([float(start),float(end)])
            native=(np.abs(r)<=half).all(-1)
            output.append(dict(pair=[i,j],background_pair=ego_index not in (i,j),initial_overlap=bool(native[0]),
                native_overlap_frames=int(native.sum()),intervals_seconds=intervals,
                PL_overlap_duration_seconds=float(sum(b-a for a,b in intervals))))
    return output
