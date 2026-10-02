"""History-count guards for checkpoint selection; no fitting or data access."""
import numpy as np

GROUPS=('N3_5','N6_8','N9_plus')


def grouped_scores(crps,num_agents,records):
    values=np.asarray(crps,dtype=float);n=np.asarray(num_agents);records=np.asarray(records)
    if values.ndim!=1 or n.shape!=values.shape or records.shape!=values.shape or not np.isfinite(values).all():
        raise ValueError('finite paired scores and history metadata required')
    masks=[n<6,(n>=6)&(n<9),n>=9]
    if not masks[-1].any():raise ValueError('declared highN STOP support required')
    groups={name:dict(rows=int(mask.sum()),CRPS_seconds=float(values[mask].mean()) if mask.any() else None) for name,mask in zip(GROUPS,masks)}
    overall=float(values.mean());high=groups['N9_plus']['CRPS_seconds']
    return dict(overall=overall,highN=high,selection=.5*(overall+high),groups=groups,
        by_recording={str(rec):dict(rows=int((records==rec).sum()),CRPS_seconds=float(values[records==rec].mean())) for rec in np.unique(records)})


def eligible(scores,initial,policy,rule):
    if rule not in ('unguarded','group_guarded'):raise ValueError('declared rule required')
    cfg=policy[rule];eps=policy['numerical_tolerance_seconds']
    checks=dict(overall=scores['overall']<=cfg['overall_ratio_to_stage1']*initial['overall']+eps,
                highN=scores['highN']<=cfg['highN_ratio_to_stage1']*initial['highN']+eps)
    if rule=='group_guarded':
        for name in GROUPS:
            base=initial['groups'][name];current=scores['groups'][name]
            if current['rows']!=base['rows']:raise ValueError('STOP population changed')
            if base['rows']>=policy['minimum_group_rows']:
                checks[name]=current['CRPS_seconds']<=cfg['all_supported_N_groups_ratio_to_stage1']*base['CRPS_seconds']+eps
    return bool(all(checks.values())),checks
