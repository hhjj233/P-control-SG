"""Scalar targets for the EXISTING mixed-CDF midpoint-risk definition.

No data/model loading, CDF fitting, trajectory alteration or rank jitter.
The generalized inverse can disagree with point-midrank control inside an
endpoint atom. We compare actual knot ranks and open-interval candidates, and
separately report the distribution-only error infimum. Scalar attainability
does not establish physical/dynamical attainability of any generated future.
"""
import numpy as np

VERSION='mixed_CDF_atom_aware_midrank_target_v1'


def physical_curve(pieces):
    a=np.array(pieces,dtype=np.float64,copy=True)
    if a.ndim!=2 or a.shape[1]!=3 or len(a)<2 or not np.isfinite(a).all():raise ValueError('finite physical/left/right CDF pieces required')
    x,left,right=a.T;tol=1e-12
    if (x[0]!=0. or x[-1]!=4. or np.any(np.diff(x)<0) or left[0]!=0. or right[-1]!=1.
            or np.any((left<0)|(right>1)|(left>right+tol))
            or np.any(left[1:]<right[:-1]-tol) or np.any(np.diff(right)<-tol)):
        raise ValueError('invalid mixed CDF; no clipping or reference repair is performed')
    cap=int(np.flatnonzero(x==4.)[0])
    # Compiled inverse tables pad AFTER the real4-second knot. Retain the first
    # cap knot, whose left value carries the atom, not the padded[4,1,1] rows.
    if cap+1<len(a) and not np.all(a[cap+1:]==np.array([4.,1.,1.])):raise ValueError('invalid inverse padding')
    a=a[:cap+1]
    if np.any(np.diff(a[:,0])<=0):raise ValueError('distinct physical knots required')
    if np.any(abs(a[1:-1,1]-a[1:-1,2])>tol):raise ValueError('this reference supports endpoint atoms only')
    a.setflags(write=False)
    return a


def _rank(curve,y):
    x,left,right=curve.T
    if not np.isfinite(y) or not 0<=y<=4:raise ValueError('finite scalar PET inside[0,4] required')
    j=int(np.searchsorted(x,y,side='left'))
    if x[j]==y:return float(1.-.5*(left[j]+right[j]))
    u=right[j-1]+(left[j]-right[j-1])*(y-x[j-1])/(x[j]-x[j-1])
    return float(1.-u)


def _quantile(curve,u):
    x,left,right=curve.T
    if u==0:return 0.
    if u==1:return 4.  # Same endpoint convention as the existing interface.
    j=int(np.flatnonzero(right>=u)[0])
    if j==0 or u>left[j]:return float(x[j])
    lo,hi=right[j-1],left[j]
    if hi<=lo:return float(x[j])
    return float(x[j-1]+(u-lo)/(hi-lo)*(x[j]-x[j-1]))


def select_midrank_target(pieces,requested_p,*,inward_margin_seconds=.001,
                          excess_rank_tolerance=1e-5,fine_tolerance=.05):
    """Select a scalar target under the original p, rather than changing p.

    Open-interval endpoint probes move inward by at most1ms and at most the
    specified extra rank error. For very narrow pieces, finite floating-point
    representability can exceed this tolerance; the returned flag discloses it.
    All original CDF knots and the original quantile remain candidates. Thus
    the selected target cannot materially worsen the canonical scalar error.
    """
    p=float(requested_p)
    if not np.isfinite(p) or not 0<=p<=1:raise ValueError('requested P must remain inside[0,1]')
    if (not np.isfinite([inward_margin_seconds,excess_rank_tolerance,fine_tolerance]).all()
            or not 0<inward_margin_seconds<=.1 or not 0<excess_rank_tolerance<=1e-3
            or not 0<fine_tolerance<=.1):raise ValueError('bounded positive target tolerances required')
    curve=physical_curve(pieces);x,left,right=curve.T
    q=_quantile(curve,1.-p);canonical_p=_rank(curve,q)
    candidates=[(q,'canonical_quantile')]+[(float(y),'physical_knot') for y in x]
    lower=min(abs(_rank(curve,float(y))-p) for y in x)
    unrepresentable=0
    for i in range(len(x)-1):
        x0,x1=x[i:i+2];u0,u1=right[i],left[i+1]
        r0,r1=1.-u0,1.-u1;low,high=min(r0,r1),max(r0,r1)
        lower=min(lower,max(low-p,p-high,0.))
        first,last=np.nextafter(x0,x1),np.nextafter(x1,x0)
        if first>=x1 or last<=x0 or first>last:
            unrepresentable+=1;continue
        density=abs(u1-u0)/(x1-x0)
        margin=min(inward_margin_seconds,(x1-x0)/4.)
        if density>0:margin=min(margin,excess_rank_tolerance/density)
        a=max(first,x0+margin);b=min(last,x1-margin)
        if a>b:a,b=first,last
        candidates.extend([(float(a),'open_piece_left'),(float(b),'open_piece_right')])
        if u1>u0 and u0<1.-p<u1:
            y=x0+((1.-p-u0)/(u1-u0))*(x1-x0)
            if x0<y<x1:candidates.append((float(y),'continuous_inverse'))
        elif u1==u0:candidates.append((float(.5*(x0+x1)),'zero_density_plateau'))
    measured=[dict(PET=y,kind=kind,rank=_rank(curve,y),error=abs(_rank(curve,y)-p)) for y,kind in candidates]
    best=min(r['error'] for r in measured)
    tied=[r for r in measured if r['error']<=best+1e-12]
    chosen=min(tied,key=lambda r:(abs(r['PET']-q),r['kind']!='canonical_quantile',r['PET']))
    if chosen['error']>abs(canonical_p-p)+1e-12:raise RuntimeError('target worsened the retained canonical alternative')
    gap=max(chosen['error']-lower,0.)
    return dict(protocol=VERSION,requested_p=p,control_target_pet_seconds=chosen['PET'],
        selected_target_midrank=chosen['rank'],selected_target_midrank_error=chosen['error'],
        selected_target_kind=chosen['kind'],canonical_quantile_pet_seconds=q,
        canonical_target_midrank=canonical_p,canonical_target_midrank_error=abs(canonical_p-p),
        unrestricted_scalar_error_infimum=float(lower),selected_error_above_infimum=float(gap),
        selected_scalar_target_inside_Fine=bool(chosen['error']<=fine_tolerance),
        Fine_impossible_from_error_infimum=bool(lower>fine_tolerance),
        within_excess_rank_tolerance=bool(gap<=excess_rank_tolerance+1e-12),
        target_changed=bool(chosen['PET']!=q),physical_dynamical_attainability_proven=False,
        request_or_reference_or_observation_modified=False,finite_candidate_count=len(measured),
        unrepresentable_open_intervals=unrepresentable,inward_margin_seconds=float(inward_margin_seconds),
        excess_rank_tolerance=float(excess_rank_tolerance))
