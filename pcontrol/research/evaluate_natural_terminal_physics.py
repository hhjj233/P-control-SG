#!/usr/bin/env python3
"""Dedicated evaluator using SHA + canonical path equivalence, not path spelling.

Frozen training driver/weights stay unchanged. All existing role, provenance,
paired-stream, snapshot and metric gates are retained.
"""
import argparse
import copy
import json
import os
import time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from pcontrol.research import pilot_natural_terminal_physics as training

old=training.old
ROOT=training.ROOT
PROTOCOL=training.PROTOCOL
CODE=training.CODE+('pcontrol/research/evaluate_natural_terminal_physics.py',)
read_policy=training.read_policy
validate_reports=training.validate_reports


def same_artifact(first,second):
    if (set(first)!={'path','sha256'} or set(second)!={'path','sha256'}
            or first['sha256']!=second['sha256']):
        return False
    return old.verify_binding(first).resolve()==old.verify_binding(second).resolve()


def evaluate(binding):
    policy=read_policy(binding);device=old.configure(policy);root=old.resolve(policy['output_root'])
    report_bindings=dict(policy['previous_training'])
    report_bindings.update({r:dict(path=str(root/r/'result.json'),sha256=old.sha256(root/r/'result.json')) for r in policy['recipes']})
    reports={r:old.bound_json(b) for r,b in report_bindings.items()}
    traces={r:[json.loads(s) for s in old.verify_binding(report['epochs']).read_text().splitlines()] for r,report in reports.items()}
    previous_policy=old.read_policy(policy['base_policy'])
    old.validate_traces({r:reports[r] for r in previous_policy['recipes']},{r:traces[r] for r in previous_policy['recipes']},previous_policy,policy['base_policy'])
    validate_reports(reports,traces,policy,binding)
    for report in reports.values():old.check_codes(report['code_sha256'])
    previous=old.bound_json(policy['previous_STOP'])
    if (not same_artifact(previous['policy'],policy['base_policy'])
            or set(previous['training'])!=set(policy['previous_training'])
            or any(not same_artifact(previous['training'][r],b) for r,b in policy['previous_training'].items())):
        raise ValueError('prior STOP does not match fixed previous training')
    parent=old.bound_json(policy['parent_STOP']);cases=old.existing_stop_cases(parent)
    base=old.bound_json(policy['base_training']);data=old.bound_json(base['data']);base_policy=old.bound_json(base['policy'])
    cn,hn=old.bound_json(data['coefficient_normalizer']),old.bound_json(data['history_normalizer'])
    plugin=old.FrozenRiskPlugin.from_refinement_binding(base_policy['risk_plugin_result'],device='cpu')
    refs={c['scene_id']:plugin.condition(c['history'],c['dimensions'],c['road_boundaries'],c['ego_mask'],c['agent_mask']) for c in cases}
    grid={'parent':dict(recipe='parent',epoch=0,checkpoint=policy['parent_checkpoint'])}
    for recipe in policy['recipes']:
        for epoch in policy['evaluation']['candidate_epochs']:
            grid[recipe+'_e'+str(epoch)]=dict(recipe=recipe,epoch=epoch,checkpoint=reports[recipe]['checkpoints'][str(epoch)])
    output=root/'STOP_evaluation';output.mkdir(exist_ok=False);codes={path:old.sha256(ROOT/path) for path in CODE}
    old.write_json(output/'freeze_before_sampling.json',dict(protocol=PROTOCOL,policy=binding,resolved_policy=policy,training=report_bindings,
        new_candidates=grid,previous_STOP=policy['previous_STOP'],CDF=base_policy['risk_plugin_result'],code_sha256=codes,
        same_H_and_noise=True,role='STOP',CAL_AUDIT_accessed=False))
    summaries={};parent_summary=None;basis=old.TrajectoryBasis(8)
    old_rows={(r['scene_id'],r['requested_p']):r for r in parent['rows']}
    for name,candidate in grid.items():
        ck=torch.load(old.verify_binding(candidate['checkpoint']),map_location='cpu')
        if name!='parent':
            report=reports[candidate['recipe']]
            if (ck['protocol']!=PROTOCOL or ck['policy']!=binding or ck['smoke'] or not ck['EMA']
                    or ck['epoch']!=candidate['epoch'] or ck['recipe']!=candidate['recipe']
                    or ck['code_sha256']!=report['code_sha256'] or ck['data']!=base['data']
                    or ck['labels_manifest']!=base['labels_manifest'] or ck['parent_checkpoint']!=policy['parent_checkpoint']):
                raise ValueError('new candidate checkpoint contract mismatch')
        model=old.warmed(ck,device).eval().requires_grad_(False);schedule=old.CosineDiffusionSchedule(100).to(device)
        target=output/name;target.mkdir();rows=[];started=time.perf_counter();replay=0.
        for number,case in enumerate(cases):
            features=old.model_features(case,hn,device);z=torch.from_numpy(case['initial_noise'][None]).to(device)
            arrays={k:case[k] for k in ('history','dimensions','road_boundaries','ego_mask','agent_ids','initial_noise')}
            arrays['future_observed']=case['future'];futures={}
            for p in policy['evaluation']['p_grid']:
                c=old.cfg_sample(model,schedule,features,torch.tensor([p],device=device),z,scale=4.,steps=50)
                physical=c[0].cpu().numpy().astype(np.float64)*np.asarray(cn['scale'])+np.asarray(cn['mean'])
                futures[p]=basis.decode(physical,case['history'][-1]);arrays['generated_p'+str(p).replace('.','_')]=futures[p]
            path=target/('case_%02d.npz'%number)
            with path.open('xb') as handle:np.savez_compressed(handle,**arrays)
            artifact=dict(path=str(path),sha256=old.sha256(path));ref=refs[case['scene_id']]
            for p,future in futures.items():
                scored=ref.score_future(future);rank=scored['estimated_rank'];spec=ref.target_spec(p)
                row={k:case[k] for k in ('scene_id','recording_id','num_agents','role','stratum','noise_seed')}
                row.update(candidate_id=name,method='GP_direct_P',requested_p=p,pet_seconds=scored['pet_seconds'],estimated_rank=rank,
                    target_spec=spec,p_mid_absolute_error=abs(rank['p_mid']-p),p_interval_error=old.interval_error(p,rank),
                    PET_target_absolute_error_seconds=abs(scored['pet_seconds']-spec['target_pet_seconds']),
                    quality=old.quality_metrics(future,case),trajectory_artifact=artifact,array_key='generated_p'+str(p).replace('.','_'),
                    K=1,network_evaluations=100,post_correction=False,external_risk_gradient_guidance=False)
                if name=='parent':
                    prior=old_rows[(case['scene_id'],p)];replay=max(replay,abs(rank['p_mid']-prior['estimated_rank']['p_mid']),abs(scored['pet_seconds']-prior['pet_seconds']))
                rows.append(row)
        if name=='parent' and replay>policy['evaluation']['parent_rank_replay_tolerance']:raise ValueError('parent replay mismatch')
        summary=old.additional_summary(rows)
        rb=old.write_json(target/'results.json',dict(protocol=PROTOCOL,status='complete',candidate=candidate,rows=rows,summary=summary,
            parent_replay_max_error=replay if name=='parent' else None))
        if name=='parent':parent_summary=summary
        summaries[name]=dict(candidate,summary=summary,result=rb,guard=old.eligibility(summary,parent_summary,policy))
        print(json.dumps(dict(candidate_complete=name,MAE=summary['p_mid_MAE'],Fine=summary['Fine_at_0_05'],p50=summary['by_requested_p']['0.5'],
            eligible=summaries[name]['guard']['eligible'],seconds=time.perf_counter()-started)),flush=True)
    old.check_codes(codes)
    candidates={k:v for k,v in summaries.items() if k!='parent'}
    for name,value in previous['candidates'].items():
        value=copy.deepcopy(value);value['guard']=old.eligibility(value['summary'],parent_summary,policy);value['reused_previous_result']=True
        candidates[name]=value
    result=dict(protocol=PROTOCOL,status='complete',policy=binding,resolved_policy=policy,training=report_bindings,candidates=candidates,
        parent=summaries['parent'],parent_summary=parent_summary,selected=old.choose(candidates,parent_summary,policy),role='STOP',
        all_four_training_randomness_matched=True,CAL_AUDIT_accessed=False,estimated_reference_not_true_CDF=True,
        production_default_changed=False,code_sha256=codes,old_candidates_reused_not_resampled=True)
    print(json.dumps(dict(evaluation_complete=old.write_json(output/'results.json',result),selected=result['selected'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',default='configs/natural_percentile/terminal_physics_pilot_v2.json')
    parser.add_argument('--policy-sha256',default=training.SCALED_POLICY_SHA)
    args=parser.parse_args()
    evaluate(dict(path=args.policy,sha256=args.policy_sha256))
