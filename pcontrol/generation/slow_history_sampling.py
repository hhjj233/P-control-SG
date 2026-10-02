"""FIT history-only speed strata and reproducible auxiliary coverage cycles."""
import hashlib
import numpy as np


def speed_strata(history,agent_mask,roles):
    if history.ndim!=4 or history.shape[-1]!=4 or history.shape[0]!=len(roles) or agent_mask.shape!=(history.shape[0],history.shape[2]):
        raise ValueError('batched history and actor mask required')
    if not np.all(np.asarray(roles)=='FIT') or agent_mask.dtype!=np.bool_ or not agent_mask.any(1).all():
        raise PermissionError('complete FIT histories only')
    speed=np.where(agent_mask,history[:,-1,:,2],np.inf).min(1)
    if not np.isfinite(speed).all():raise ValueError('nonfinite observed minimum speed')
    return speed,np.where(speed<=2.,'creep',np.where(speed<=5.,'slow','ordinary'))


def fixed_probe_indices(scene_ids,recordings,strata,per_group=12,salt='natural_speed_probe_v1'):
    groups={}
    for group in ('creep','slow','ordinary'):
        buckets={}
        for i,s in enumerate(strata):
            if s==group:buckets.setdefault(str(recordings[i]),[]).append(i)
        for rows in buckets.values():rows.sort(key=lambda i:hashlib.sha256((salt+'|'+str(scene_ids[i])).encode()).hexdigest())
        records=sorted(buckets,key=lambda r:hashlib.sha256((salt+'|'+group+'|'+r).encode()).hexdigest())
        selected=[]
        while len(selected)<per_group:
            added=False
            for rec in records:
                if buckets[rec] and len(selected)<per_group:selected.append(buckets[rec].pop(0));added=True
            if not added:raise ValueError('insufficient history-only probe pool')
        groups[group]=selected
    return [groups[g][j] for j in range(per_group) for g in groups]


class SlowHistoryCycle:
    """First two original random rows + one creep + one slow, no future access.

The slow stratum cycles recordings before reusing one; creep currently has one
FIT recording, a declared data limitation rather than manufactured diversity.
"""
    def __init__(self,recordings,strata,seed=20261301):
        self.rng=np.random.default_rng(seed);self.pools={};self.queues={};self.record_orders={};self.positions={}
        for group in ('creep','slow'):
            buckets={str(r):np.flatnonzero((np.asarray(recordings)==r)&(np.asarray(strata)==group)) for r in sorted(set(recordings))}
            buckets={r:rows for r,rows in buckets.items() if len(rows)}
            if not buckets:raise ValueError('missing slow auxiliary stratum')
            self.pools[group]=buckets;self.queues[group]={r:[] for r in buckets}
            self.record_orders[group]=list(self.rng.permutation(sorted(buckets)));self.positions[group]=0

    def _pick(self,group,excluded):
        # Exclusion only prevents a duplicate history in this four-history batch.
        for _ in range(10000):
            order=self.record_orders[group];pos=self.positions[group];rec=order[pos%len(order)];self.positions[group]=pos+1
            if not self.queues[group][rec]:self.queues[group][rec]=list(self.rng.permutation(self.pools[group][rec]))
            row=int(self.queues[group][rec].pop())
            if row not in excluded:return row
        raise ValueError('cannot find a distinct history in the speed stratum')

    def select(self,base_rows):
        rows=list(map(int,base_rows[:2]))
        if len(rows)!=2 or len(set(rows))!=2:raise ValueError('two distinct ordinary base rows required')
        rows.append(self._pick('creep',set(rows)));rows.append(self._pick('slow',set(rows)))
        return np.asarray(rows,dtype=np.int64)
