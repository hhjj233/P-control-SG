"""Pure matched calibration scoring/selection. No files, roles or model fitting data are chosen here."""
import numpy as np

from .scene_calibration import precompute_crps_quadratic
from .scene_context_calibration import (
    interval_quadratic,diagnostic_arrays,summarize_diagnostics,fit_context_warp,
)
from pcontrol.time_attention_pipeline.common import count_context,StableCountWarp

SELECTORS=('legacy_global','count_balanced')


def subset_quadratic(q,mask):
    mask=np.asarray(mask)
    if mask.dtype!=bool or mask.shape!=(len(q['c']),) or not mask.any():raise ValueError('explicit nonempty paired row mask required')
    return dict(A=q['A'][mask],b=q['b'][mask],c=q['c'][mask],cap_seconds=q['cap_seconds'])


def fit_cached(q,counts,mask,*,family,ridge):
    if family not in ('global','count'):raise ValueError('only declared global/count models')
    if len(counts)!=len(q['c']):raise ValueError('paired history counts required')
    warp,report=fit_context_warp(subset_quadratic(q,mask),count_context(np.asarray(counts)[mask]),family=family,ridge=ridge)
    if not report['success']:raise RuntimeError('CAL optimizer did not converge; do not silently discard')
    return StableCountWarp(warp.family,warp.node_values),report


def selector_scores(metrics,cfg):
    overall=metrics['overall'];groups=metrics['groups']['N']
    supported=[g for g,item in groups.items() if item['scenes']>=cfg['minimum_group_rows']]
    local=overall['local_component_score']
    balanced=.5*local+.5*np.mean([groups[g]['local_component_score'] for g in supported]) if supported else local
    return dict(legacy_global=float(overall['expected_PIT_grid_KS']+2*overall['threshold_MAE']),
                count_balanced=float(balanced)),supported


def eligibility(metrics,raw,cfg,selector):
    if selector not in SELECTORS:raise ValueError('unknown selection rule')
    rule=cfg['selectors'][selector];a,b=metrics['overall'],raw['overall']
    checks=dict(overall_CRPS=a['CRPS_seconds']<=rule['overall_CRPS_ratio']*b['CRPS_seconds'])
    if selector=='legacy_global':
        if raw['groups']['N']['N9_plus']['scenes']:
            checks['highN_CRPS']=metrics['groups']['N']['N9_plus']['CRPS_seconds']<=rule['highN_CRPS_ratio']*raw['groups']['N']['N9_plus']['CRPS_seconds']
    else:
        for key in ('twCRPS_1s','twCRPS_2s'):checks[key]=a[key]<=rule['tail_CRPS_ratio']*b[key]
        for g,item in raw['groups']['N'].items():
            if item['scenes']>=cfg['minimum_group_rows']:
                checks[g+'_CRPS']=metrics['groups']['N'][g]['CRPS_seconds']<=rule['supported_group_CRPS_ratio']*item['CRPS_seconds']
    return bool(all(checks.values())),checks


def select_candidate(candidates,selector):
    if selector not in SELECTORS or 'identity' not in candidates:raise ValueError('declared selector and identity fallback required')
    eligible=[name for name,v in candidates.items() if v['eligible'][selector]]
    if not eligible:raise ValueError('identity must remain eligible')
    def key(name):
        v=candidates[name]
        return (v['selection_scores'][selector],0 if name=='identity' else 1 if v['family']=='global' else 2,-(v['ridge'] or 0.))
    return min(eligible,key=key)


class CalibrationScoreCache:
    """Reuse exact per-row proper-score quadratics, not cross-row statistics."""
    def __init__(self,raw,cfg):
        self.raw=raw;self.cfg=cfg
        m,y,n=raw['joint_masses'],raw['target'],raw['num_agents']
        if len(y)!=len(m) or np.shape(n)!=(len(y),):raise ValueError('unaligned prediction rows')
        self.context=count_context(n)
        self.quadratics=dict(CRPS_seconds=precompute_crps_quadratic(m,n,y,family='global'),
            twCRPS_1s=interval_quadratic(m,n,y,1.),twCRPS_2s=interval_quadratic(m,n,y,2.))
        self.metric_policy=dict(cfg,diagnostic_groups=dict(minimum_rows=cfg['minimum_group_rows']))

    def score(self,nodes):
        raw=self.raw
        a=diagnostic_arrays(raw['joint_masses'],self.context,raw['target'],nodes,self.metric_policy,self.quadratics)
        m=summarize_diagnostics(a,raw['target'],self.context,raw['recording_id'],self.metric_policy)
        # No speed/gap measurements are inferred from the unused zero columns.
        m.pop('context_selection_score');m.pop('groups_used_for_selection');m['groups']={'N':m['groups']['N']}
        scores,supported=selector_scores(m,self.cfg);m.update(selection_scores=scores,supported_N_groups=supported)
        a.update(scene_id=raw['scene_id'],recording_id=raw['recording_id'],num_agents=raw['num_agents'],target=raw['target'],
                 joint_masses=raw['joint_masses'],effective_nodes=np.asarray(nodes),role=raw['role'])
        return a,m
