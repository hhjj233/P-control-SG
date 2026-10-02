"""All-request descriptive response, diversity and paired-recording summaries."""
from collections import Counter,defaultdict
import itertools
import numpy as np


def risk_response(rows,p_grid,noise_indices,*,rank_tolerance=1e-7,strict_gap=1e-6):
    groups=defaultdict(list);histories=set()
    for row in rows:
        groups[row['scene_id'],row['noise_index']].append(row);histories.add(row['scene_id'])
    if set(groups)!={(sid,z) for sid in histories for z in noise_indices}:raise ValueError('missing H/z response sweep')
    summaries=[]
    for (sid,z),items in sorted(groups.items()):
        items=sorted(items,key=lambda r:r['requested_p'])
        if [r['requested_p'] for r in items]!=list(p_grid):raise ValueError('missing/duplicate P in response sweep')
        achieved=np.array([r['estimated_rank']['p_mid'] for r in items]);pet=np.array([r['pet_seconds'] for r in items])
        errors=np.array([r['p_mid_absolute_error'] for r in items]);delta=np.diff(achieved)
        if not np.isfinite(np.r_[achieved,pet,errors]).all() or np.max(abs(abs(achieved-np.array(p_grid))-errors))>2e-10:
            raise ValueError('response must use original achieved ranks/errors')
        summaries.append(dict(scene_id=sid,noise_index=z,requested_p=list(p_grid),achieved_p=achieved.tolist(),PET=pet.tolist(),
            nondecreasing=bool((delta>=-rank_tolerance).all()),strictly_increasing=bool((delta>strict_gap).all()),
            adjacent_decreases=int((delta < -rank_tolerance).sum()),adjacent_ties=int((abs(delta)<=strict_gap).sum()),
            high_minus_low=float(achieved[-1]-achieved[0]),all_requests_Fine=bool((errors<=.05).all())))
    return dict(sweeps=len(summaries),histories=len(histories),rows=summaries,
        nondecreasing_sweeps=sum(r['nondecreasing'] for r in summaries),strictly_increasing_sweeps=sum(r['strictly_increasing'] for r in summaries),
        all_P_Fine_sweeps=sum(r['all_requests_Fine'] for r in summaries),
        adjacent_pairs=len(summaries)*(len(p_grid)-1),adjacent_decreases=sum(r['adjacent_decreases'] for r in summaries),
        adjacent_ties=sum(r['adjacent_ties'] for r in summaries),mean_high_minus_low=float(np.mean([r['high_minus_low'] for r in summaries])),
        rank_tolerance=rank_tolerance,strict_gap=strict_gap,ties_not_counted_as_strict_response=True)


def pairwise_noise_diversity(futures,ego_mask):
    """All three noise outputs, not only hits; Euclidean distance in metres."""
    values=np.asarray(futures,dtype=np.float64);ego=np.asarray(ego_mask)
    if (values.ndim!=4 or values.shape[0]!=3 or values.shape[1]!=175 or values.shape[2]<3 or values.shape[3]!=4
            or ego.shape!=(values.shape[2],) or ego.dtype!=bool or ego.sum()!=1 or not np.isfinite(values).all()
            or not np.array_equal(values[0,0],values[1,0]) or not np.array_equal(values[0,0],values[2,0])):
        raise ValueError('three complete, same-anchor, same-roster futures required')
    pairs=list(itertools.combinations(range(3),2));out={}
    for label,mask in [('all',np.ones(len(ego),bool)),('ego',ego),('background',~ego)]:
        distances=np.stack([np.linalg.norm(values[i,1:,mask,:2]-values[j,1:,mask,:2],axis=-1) for i,j in pairs])
        # Advanced boolean indexing may move the actor axis; mean below is
        # over all future frames and actors irrespective of that order.
        terminal=np.stack([np.linalg.norm(values[i,-1,mask,:2]-values[j,-1,mask,:2],axis=-1) for i,j in pairs])
        out[label+'_pairwise_mean_displacement_m']=float(distances.mean())
        out[label+'_pairwise_RMS_displacement_m']=float(np.sqrt(np.mean(distances**2)))
        out[label+'_pairwise_final_displacement_m']=float(terminal.mean())
    out.update(noise_outputs=3,noise_pairs=3,t0_excluded=True,all_noise_outputs_retained=True,
        not_a_plausibility_score=True,not_against_observed_future=True)
    return out


def failure_label(row):
    if row['p_mid_absolute_error']<=.05:return 'Fine_hit'
    if row['scalar_error_infimum']>.05:return 'scalar_midrank_gap'
    scalar=row['atom_target_diagnostics']
    target_error=scalar['canonical_target_midrank_error'] if row['arm']=='canonical' else scalar['selected_target_midrank_error']
    if target_error>.05:return 'nominal_scalar_target_outside_Fine'
    if row['pet_seconds']==4.:return 'miss_at_cap'
    if row['pet_seconds']==0.:return 'miss_at_zero'
    return 'continuous_outcome_miss'


def paired_recording_summary(left,right,*,repetitions=10000,seed=20260916):
    """Right minus left; recording-cluster intervals are exploratory with four."""
    def key(r):return r['scene_id'],r['noise_index'],r['requested_p']
    l={key(r):r for r in left};rr={key(r):r for r in right}
    if len(l)!=len(left) or len(rr)!=len(right) or set(l)!=set(rr):raise ValueError('exact paired H/P/z keys required')
    groups=defaultdict(list)
    for k in sorted(l):
        a,b=l[k],rr[k]
        if a['recording_id']!=b['recording_id'] or a['noise_sha256']!=b['noise_sha256']:raise ValueError('recording/noise mismatch')
        groups[a['recording_id']].append((b['p_mid_absolute_error']-a['p_mid_absolute_error'],
            float(b['p_mid_absolute_error']<=.05)-float(a['p_mid_absolute_error']<=.05)))
    if len(groups)<2:raise ValueError('at least two recording clusters required')
    records=sorted(groups);means=np.array([np.mean(groups[r],axis=0) for r in records]);counts=np.array([len(groups[r]) for r in records])
    rng=np.random.default_rng(seed);indices=rng.integers(0,len(records),size=(repetitions,len(records)))
    weighted=(means[indices]*counts[indices][...,None]).sum(1)/counts[indices].sum(1)[:,None];macro=means[indices].mean(1)
    metrics={}
    for j,name in enumerate(('P_MAE','Fine_rate')):
        metrics[name]=dict(delta_request_weighted=float(np.average(means[:,j],weights=counts)),delta_recording_macro=float(means[:,j].mean()),
            recording_bootstrap_weighted_95pct=np.quantile(weighted[:,j],[.025,.975]).tolist(),
            recording_bootstrap_macro_95pct=np.quantile(macro[:,j],[.025,.975]).tolist())
    return dict(direction='right_minus_left',metrics=metrics,recordings=len(records),requests=len(left),repetitions=repetitions,seed=seed,
        by_recording={r:dict(requests=int(counts[i]),delta_P_MAE=float(means[i,0]),delta_Fine_rate=float(means[i,1])) for i,r in enumerate(records)},
        exploratory_few_recordings=True,independent_training_seeds=False,multiple_comparison_adjusted_significance_claim=False)
