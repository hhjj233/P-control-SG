"""Background longitudinal separation inside a diffusion clean-state update.

Lateral-overlap windows are computed continuously for piecewise-linear native
states. Holding y fixed, ordered x-separation at every window endpoint is a
linear coefficient constraint. Orders are local to connected lateral windows,
so passing while laterally separated is not globally forbidden. No ego
coefficient, observed future, risk target, or model parameter is modified here.
"""
import numpy as np
import torch
from scipy.optimize import LinearConstraint,minimize


def lateral_slabs(future,dimensions,pairs):
    """Fractional frame intervals where each pair's closed y footprints meet."""
    if not len(pairs):return np.empty((174,0)),np.empty((174,0)),np.empty((174,0),bool)
    i,j=pairs[:,0],pairs[:,1];relative=future[:,j,1]-future[:,i,1];delta=np.diff(relative,axis=0)
    half=(dimensions[i,1]+dimensions[j,1])/2;parallel=delta==0.
    denominator=np.where(parallel,1.,delta)
    a=(-half[None]-relative[:-1])/denominator;b=(half[None]-relative[:-1])/denominator
    lower=np.maximum(np.minimum(a,b),0.);upper=np.minimum(np.maximum(a,b),1.)
    lower=np.where(parallel,0.,lower);upper=np.where(parallel,1.,upper)
    valid=(lower<=upper)&((~parallel)|(np.abs(relative[:-1])<=half[None]))
    return lower,upper,valid


def background_conflicts(future,dimensions,ego_index,*,margin=0.):
    """Complete-scene PL background conflict scan; includes between-frame hits."""
    f=np.asarray(future,dtype=np.float64);d=np.asarray(dimensions,dtype=np.float64);n=len(d)
    if (f.shape!=(175,n,4) or not np.isfinite(f).all() or d.shape!=(n,2) or not np.isfinite(d).all()
            or np.any(d<=0) or not 0<=ego_index<n or not np.isfinite(margin) or margin<0):raise ValueError('finite complete native scene required')
    pairs=np.array([(i,j) for i in range(n) for j in range(i+1,n) if ego_index not in (i,j)],dtype=np.int64).reshape(-1,2)
    lo,hi,valid=lateral_slabs(f,d,pairs)
    if not len(pairs):return dict(pairs=pairs,lower=lo,upper=hi,lateral=valid,hit=valid)
    i,j=pairs[:,0],pairs[:,1];x=f[:,j,0]-f[:,i,0];dx=np.diff(x,axis=0)
    a=x[:-1]+dx*lo;b=x[:-1]+dx*hi;half=(d[i,0]+d[j,0])/2+float(margin)
    hit=valid&(np.minimum(a,b)<=half[None])&(np.maximum(a,b)>=-half[None])
    return dict(pairs=pairs,lower=lo,upper=hi,lateral=valid,hit=hit)


class BackgroundEnvelope:
    def __init__(self,decoder,dimensions,inverse_metric_x,ego_index,*,clearance_m=.1,max_passes=4):
        self.decoder=decoder;self.dimensions=np.asarray(dimensions,dtype=np.float64);self.ego=int(ego_index)
        self.anchors=decoder.anchors.detach().cpu().numpy();n=len(self.anchors)
        if (self.dimensions.shape!=(n,2) or not 0<=self.ego<n or not np.isfinite(clearance_m) or clearance_m<=0
                or type(max_passes) is not int or max_passes<1):raise ValueError('valid scene and positive projection settings required')
        self.clearance=float(clearance_m);self.max_passes=max_passes
        matrix=inverse_metric_x.detach().cpu().numpy();eigen,rotation=np.linalg.eigh(matrix)
        if np.any(eigen<=0):raise ValueError('positive trajectory metric required')
        self.L=(rotation*np.sqrt(eigen)[None])@rotation.T
        self.B=(decoder.bp*decoder.scale[:,0]).detach().cpu().numpy()
        self.V=(decoder.bv*decoder.scale[:,0]).detach().cpu().numpy()
        self.BW=self.B@self.L;self.VW=self.V@self.L
        self.initial=[]
        for i in range(n):
            for j in range(i+1,n):
                if self.ego not in (i,j) and np.all(np.abs(self.anchors[j,:2]-self.anchors[i,:2])<=(self.dimensions[i]+self.dimensions[j])/2):self.initial.append((i,j))
        self.checks=0;self.solves=0;self.failures=0;self.scan_calls=0;self.max_shift=0.;self.maximum_active_actors=0

    def _scan(self,future,margin):
        self.scan_calls+=1;return background_conflicts(future,self.dimensions,self.ego,margin=margin)

    def _windows(self,original,scan,pair_index):
        i,j=map(int,scan['pairs'][pair_index]);windows=[]
        for k in np.flatnonzero(scan['lateral'][:,pair_index]):
            lo=float(scan['lower'][k,pair_index]);hi=float(scan['upper'][k,pair_index]);start=k+lo;end=k+hi
            if windows and start<=windows[-1][-1][0]+windows[-1][-1][2]+1e-9:windows[-1].append((int(k),lo,hi))
            else:windows.append([(int(k),lo,hi)])
        answer=[]
        for window in windows:
            k,u,_=window[0];dx0=original[k,j,0]-original[k,i,0];dv=(original[k+1,j,0]-original[k+1,i,0])-dx0
            separation=dx0+u*dv
            if abs(separation)<1e-10:
                separation=self.anchors[j,0]-self.anchors[i,0]
                if abs(separation)<1e-10:separation=self.anchors[j,1]-self.anchors[i,1]
            sign=1. if separation>=0 else -1.
            half=float((self.dimensions[i,0]+self.dimensions[j,0])/2);margin=self.clearance
            if k+u<=1e-10:
                initial_gap=sign*(self.anchors[j,0]-self.anchors[i,0])-half
                margin=min(margin,max(initial_gap,0.)*.5)
            for k,lo,hi in window:
                for u in (lo,hi):
                    basis=(1-u)*self.BW[k]+u*self.BW[k+1]
                    dx=(1-u)*(original[k,j,0]-original[k,i,0])+u*(original[k+1,j,0]-original[k+1,i,0])
                    answer.append((i,j,sign,basis,half+margin-sign*dx))
        return answer

    def project(self,coefficients):
        if coefficients.shape!=(len(self.dimensions),8,2):raise ValueError('complete K8 scene required')
        self.checks+=1;original=coefficients.detach();f0=self.decoder(original).detach().cpu().numpy();future=f0.copy()
        current=original.cpu().numpy().astype(np.float64).copy();active=set();constraints=[];pass_count=0;failed=False
        for _ in range(self.max_passes):
            scan=self._scan(future,max(self.clearance-1e-6,0.));new=[]
            for pidx in np.flatnonzero(scan['hit'].any(0)):
                pair=tuple(map(int,scan['pairs'][pidx]))
                if pair not in active and pair not in self.initial:new.append((pidx,pair))
            if not new:break
            for pidx,pair in new:
                active.add(pair);constraints.extend(self._windows(f0,scan,int(pidx)))
            actors=sorted({i for pair in active for i in pair});mapping={actor:j for j,actor in enumerate(actors)}
            variables=len(actors)*8;rows=[];bounds=[]
            for i,j,sign,basis,bound in constraints:
                a=np.zeros(variables);a[mapping[i]*8:(mapping[i]+1)*8]=-sign*basis;a[mapping[j]*8:(mapping[j]+1)*8]=sign*basis
                rows.append(a);bounds.append(bound)
            for actor in actors:
                a=np.zeros((174,variables));a[:,mapping[actor]*8:(mapping[actor]+1)*8]=self.VW[1:]
                rows.extend(a);bounds.extend(np.minimum(self.anchors[actor,2],0.)-f0[1:,actor,2])
            A=np.asarray(rows);b=np.asarray(bounds);self.solves+=1;pass_count+=1;self.maximum_active_actors=max(self.maximum_active_actors,len(actors))
            result=minimize(lambda z:.5*np.dot(z,z),np.zeros(variables),jac=lambda z:z,
                constraints=[LinearConstraint(A,b,np.full_like(b,np.inf))],method='SLSQP',options=dict(ftol=1e-10,maxiter=120,disp=False))
            if not np.isfinite(result.x).all() or float((b-A@result.x).max())>1e-7:
                self.failures+=1;failed=True;break
            current=original.cpu().numpy().astype(np.float64).copy()
            for actor in actors:current[actor,:,0]+=self.L@result.x[mapping[actor]*8:(mapping[actor]+1)*8]
            candidate=torch.as_tensor(current,dtype=original.dtype,device=original.device);future=self.decoder(candidate).detach().cpu().numpy()
        answer=torch.as_tensor(current,dtype=original.dtype,device=original.device)
        remaining=self._scan(future,0.);remaining_pairs=[tuple(map(int,p)) for p in remaining['pairs'][remaining['hit'].any(0)]]
        shift=float(np.linalg.norm(future[...,:2]-f0[...,:2],axis=-1).max());self.max_shift=max(self.max_shift,shift)
        if not torch.equal(answer[self.ego],original[self.ego]) or not torch.equal(answer[:,:,1],original[:,:,1]):raise RuntimeError('background projection changed ego or lateral coefficients')
        return answer,dict(active_pairs=[list(p) for p in sorted(active)],passes=pass_count,solver_failed=failed,
            remaining_background_pairs=[list(p) for p in remaining_pairs],initial_background_pairs=[list(p) for p in self.initial],
            max_position_shift_m=shift,ego_coefficients_unchanged=True,lateral_coefficients_unchanged=True,
            certificate_scope='piecewise-linear interpolation of175 native AABB states, not smooth K8 dynamics')

    def report(self):
        return dict(projection_checks=self.checks,QP_solves=self.solves,QP_failures=self.failures,background_scans=self.scan_calls,
            initial_background_pairs=[list(p) for p in self.initial],maximum_active_actors=self.maximum_active_actors,
            maximum_projection_position_shift_m=self.max_shift,clearance_m=self.clearance,observed_future_input=False)
