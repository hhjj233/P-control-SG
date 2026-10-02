#!/usr/bin/env python3
"""Natural-only, matched-parent physics and terminal+physics pilot.

Imports frozen source helpers but never mutates old modules or their policies.
All new training uses the original parent, not the failed terminal candidate.
"""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.research import pilot_natural_terminal_value as old
from pcontrol.generation.trajectory_physics_loss import batch_physics_loss
from pcontrol.research.probe_natural_terminal_support import fixed_selection,SALT

PROTOCOL='natural_terminal_physics_paired_training_v1'
POLICY_SHA='95f1d388e081a37d6d393e9ecdbce0813de5a2e7c8b4fb1ff2f76cdb171e0a61'
SCALED_POLICY_SHA='c8b5f58840198675b6f64f31ebd36f8d6674b3a98cf36cdfd2c3e937f37dcdcd'
CODE=old.CODE+('pcontrol/research/pilot_natural_terminal_physics.py','pcontrol/generation/trajectory_physics_loss.py')
PROBE_SELECTION=dict(path='outputs/natural_percentile/terminal_support_probe_v2_20260912/freeze_before_probe.json',
    sha256='c7bed6f3fb7645db6d861db5597ab222fc408d148dd6fdd01eb2f27109661c4a')


def read_policy(binding):
    if binding.get('sha256') not in (POLICY_SHA,SCALED_POLICY_SHA):raise ValueError('only frozen physics pilot policy allowed')
    patch=old.bound_json(binding)
    if patch['protocol']!=PROTOCOL or patch['recipes']!=['physics','terminal_physics']:
        raise ValueError('wrong physics policy/recipes')
    base=old.read_policy(patch['base_policy']);policy=copy.deepcopy(base)
    policy.update(protocol=PROTOCOL,recipes=patch['recipes'],output_root=patch['output_root'],
        physics=patch['physics'],previous_training=patch['previous_training'],previous_STOP=patch['previous_STOP'],
        base_policy=patch['base_policy'],scale_probe=patch['scale_probe'])
    policy['training']['terminal_weight']=patch['terminal_weights'];policy['training']['additional_road_loss']=True
    for binding in patch['previous_training'].values():old.verify_binding(binding)
    old.verify_binding(patch['previous_STOP'])
    if 'prior_scale_probe' in patch:
        prior=old.bound_json(patch['prior_scale_probe'])
        if prior.get('status')!='complete' or prior.get('optimizer_steps')!=0 or not prior.get('all_gradients_finite'):
            raise ValueError('weight scaling must reference a completed no-update FIT probe')
        policy.update(prior_scale_probe=patch['prior_scale_probe'],weight_decision=patch['weight_decision'])
    return policy


class PhysicsTeacher(old.TerminalFITTeacher):
    def physical_batch(self,rows):
        dims=[];roads=[]
        for row in rows:
            mask=self.pack['agent_mask'][row];rm=self.physical['road_boundary_mask'][row]
            dims.append(self.physical['dimensions'][row,mask]);roads.append(self.physical['road_boundaries'][row,rm])
        return dims,roads


def compute_auxiliary(model,schedule,features,p,noise,presence,teacher_batch,physical_batch,cfg,physics,terminal_weight):
    sample=old.training_ddim_suffix(model,schedule,features,p,noise,steps=cfg['terminal_DDIM_steps'],
        grad_last_steps=cfg['grad_last_steps'],cfg_scale=cfg['terminal_CFG_scale'])
    reference,decoders,geometries=teacher_batch
    term=None
    if terminal_weight>0:
        term=old.terminal_batch_loss(sample['sample'],p,reference,decoders,geometries,features['agent_mask'],presence,
            beta=cfg['terminal_beta'],minimum_density=cfg['minimum_density'])
    dims,roads=physical_batch
    physical=batch_physics_loss(sample['sample'],decoders,dims,roads,features['agent_mask'],presence,
        tail_fraction=physics['tail_fraction'],road_scale_m=physics['road_scale_m'],speed_scale_mps=physics['speed_scale_mps'])
    loss=physics['road_weight']*physical['road_loss']+physics['speed_weight']*physical['speed_loss']
    if term is not None:loss=loss+terminal_weight*term['loss']
    return loss,term,physical,sample


def add_auxiliary_backward(model,schedule,features,p,noise,presence,teacher_batch,physical_batch,cfg,physics,terminal_weight):
    mode=model.training
    try:
        model.eval()
        loss,term,physical,sample=compute_auxiliary(model,schedule,features,p,noise,presence,teacher_batch,
            physical_batch,cfg,physics,terminal_weight)
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite auxiliary loss')
        loss.backward()
        return dict(total=physical['total_scenes'],present=physical['present_scenes'],
            road_loss=float(physical['road_loss'].detach()),speed_loss=float(physical['speed_loss'].detach()),
            road_events=sum(d['road_has_training_violation'] for d in physical['diagnostics']),
            speed_events=sum(d['speed_has_training_violation'] for d in physical['diagnostics']),
            supported=0 if term is None else term['supported_scenes'],
            terminal_loss=None if term is None else float(term['loss'].detach()),
            geometry_supported=0 if term is None else term['geometry_supported_scenes'],
            reasons={} if term is None else term['geometry_reasons'],
            point_sum=0. if term is None else term['all_request_point_MAE']*term['total_scenes'],
            forward_NFE=sample['network_evaluations'],backward_NFE=sample['backward_recompute_network_evaluations'])
    finally:model.train(mode)


def train_epoch(model,ema,schedule,optimizer,pack,p_values,cache,teacher,streams,terminal_rng,cfg,physics,recipe,device,
                *,progress=None,max_updates=None):
    model.train();order=streams.order(len(p_values));hashes={k:hashlib.sha256() for k in ('base_noise','drop','presence','terminal_plan')}
    stat=dict(base_loss_scenes=0,updates=0,base_v_sum=0.,projection_support=0,jacobian_support=0,p_dropped=0,
        auxiliary_requests=0,auxiliary_present=0,road_events=0,speed_events=0,road_loss_present_sum=0.,speed_loss_present_sum=0.,
        terminal_requests=0,terminal_supported=0,terminal_geometry_supported=0,terminal_point_error_sum=0.,
        terminal_value_supported_sum=0.,forward_NFE=0,backward_NFE=0,gradient_norm_max=0.)
    reasons=Counter();started=time.perf_counter();weight=cfg['terminal_weight'][recipe]
    for start in range(0,len(order),cfg['batch_size']):
        if max_updates is not None and stat['updates']>=max_updates:break
        rows=order[start:start+cfg['batch_size']]
        features,clean,p=old.direct.tensor_batch(pack,p_values,rows,device);p=p.detach().requires_grad_(True)
        times,noise,u,present=streams.draw(clean.shape,schedule.steps,cfg['p_dropout_probability'])
        for a in (np.asarray(clean.shape,dtype=np.int64),times.numpy(),noise.numpy()):hashes['base_noise'].update(a.tobytes())
        hashes['drop'].update(u.tobytes());hashes['presence'].update(present.tobytes())
        presence=torch.from_numpy(present).to(device);t=times.to(device);optimizer.zero_grad(set_to_none=True)
        base=old.cfg_training_loss(model,schedule,clean,features,p,presence,timesteps=t,noise=noise.to(device))
        extra=old.natural_tangent_losses(base['model_prediction'],base['prediction_target'],base['noisy_coefficients'],p,
            schedule.alpha_bar(t,base['model_prediction']),features['agent_mask'],condition_present=presence,
            projection_weight=cfg['loss_weights']['projection'],jacobian_weight=cfg['loss_weights']['jacobian'],
            min_jacobian_sigma=cfg['min_jacobian_sigma'],**old.tangent.cache_batch(cache,rows,clean.shape[1],device))
        if extra['diagnostics']['jacobian_evaluated'] and extra['diagnostics']['p_graph_connected'] is False:
            raise RuntimeError('disconnected base response gradient')
        loss=base['loss']+extra['loss']
        if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite base loss')
        loss.backward();stat['base_v_sum']+=float(base['loss'].detach())*len(rows)
        stat['projection_support']+=extra['support_counts']['projection'];stat['jacobian_support']+=extra['support_counts']['jacobian']
        del base,extra,loss
        selected=rows[:min(cfg['terminal_per_batch'],len(rows))]
        f,c,q=old.direct.tensor_batch(pack,p_values,selected,device);z=torch.randn(tuple(c.shape),generator=terminal_rng)
        for a in (selected.astype(np.int64),np.asarray(z.shape,dtype=np.int64),z.numpy()):hashes['terminal_plan'].update(a.tobytes())
        aux=add_auxiliary_backward(model,schedule,f,q,z.to(device),presence[:len(selected)],teacher.batch(selected),
            teacher.physical_batch(selected),cfg,physics,weight)
        stat['auxiliary_requests']+=aux['total'];stat['auxiliary_present']+=aux['present']
        stat['road_events']+=aux['road_events'];stat['speed_events']+=aux['speed_events']
        stat['road_loss_present_sum']+=aux['road_loss']*aux['present'];stat['speed_loss_present_sum']+=aux['speed_loss']*aux['present']
        stat['terminal_supported']+=aux['supported'];stat['terminal_geometry_supported']+=aux['geometry_supported']
        stat['terminal_point_error_sum']+=aux['point_sum'];reasons.update(aux['reasons'])
        if weight>0:
            stat['terminal_requests']+=aux['total'];stat['terminal_value_supported_sum']+=aux['terminal_loss']*aux['supported']
        stat['forward_NFE']+=aux['forward_NFE'];stat['backward_NFE']+=aux['backward_NFE']
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
        if not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite combined parameter gradient')
        stat['gradient_norm_max']=max(stat['gradient_norm_max'],float(norm))
        optimizer.step();old.direct.prior.update_ema(ema,model,cfg['EMA_decay'])
        stat['updates']+=1;stat['base_loss_scenes']+=len(rows);stat['p_dropped']+=int((~present).sum())
        if progress and stat['updates']%10==0:progress(dict(recipe=recipe,updates=stat['updates'],seconds=time.perf_counter()-started))
    stat.update(randomness={k:h.hexdigest() for k,h in hashes.items()},order_sha256=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest(),
        all_base_loss_rows_retained=stat['base_loss_scenes']==len(p_values),geometry_reasons=dict(reasons),
        terminal_all_request_point_MAE=stat['terminal_point_error_sum']/stat['terminal_requests'] if stat['terminal_requests'] else None,
        wall_seconds=time.perf_counter()-started)
    return stat


def load_inputs(policy,device):
    base=old.bound_json(policy['base_training']);parent=old.bound_json(policy['parent_training'])
    if old.bound_json(policy['parent_selection'])['choice']['selected']!='tangent_e40_s4':raise ValueError('wrong parent selection')
    checkpoint=torch.load(old.verify_binding(policy['parent_checkpoint']),map_location='cpu')
    if (checkpoint['recipe']!='tangent' or checkpoint['epoch']!=40 or checkpoint['data']!=base['data']
            or checkpoint['labels_manifest']!=base['labels_manifest']):raise ValueError('wrong original parent')
    manifest=old.bound_json(policy['risk_cache']);prepared=old.bound_json(manifest['prepared'])
    pack=old.fit_arrays(prepared['FIT_generator_pack']);labels=old.fit_arrays(prepared['FIT_p_labels'])
    p,join=old.direct.join_percentile_labels(pack,labels,role='FIT')
    if len(p)!=9913 or len(set(pack['recording_id']))!=13:raise ValueError('FIT population changed')
    cache,cache_join=old.tangent.join_risk_cache(pack,p,old.load_tangent_cache(policy['risk_cache']))
    teacher=PhysicsTeacher(manifest,prepared,pack,labels,device)
    codes=dict(parent['code_sha256']);codes.update(manifest['code_sha256']);old.check_codes(codes)
    codes.update({path:old.sha256(ROOT/path) for path in CODE});old.check_codes(codes)
    return dict(base=base,parent=parent,checkpoint=checkpoint,pack=pack,p=p,join=join,cache=cache,cache_join=cache_join,
        teacher=teacher,codes=codes,data=old.bound_json(base['data']))


def train(binding,recipe,*,smoke=False):
    policy=read_policy(binding)
    if binding['sha256']!=SCALED_POLICY_SHA:raise ValueError('formal and smoke training use the gradient-informed v2 policy')
    if recipe not in policy['recipes']:raise ValueError('unregistered recipe')
    device=old.configure(policy);cfg=policy['training'];torch.manual_seed(cfg['seed']);source=load_inputs(policy,device)
    model=old.warmed(source['checkpoint'],device).requires_grad_(True);initial=old.state_hash(model)
    ema=copy.deepcopy(model).requires_grad_(False);schedule=old.CosineDiffusionSchedule(100).to(device)
    if schedule.as_dict()!=source['checkpoint']['schedule']:raise ValueError('schedule changed')
    output=old.resolve(policy['output_root'])/(recipe+'_smoke' if smoke else recipe);output.mkdir(parents=True,exist_ok=False)
    freeze=old.write_json(output/'freeze_before_training.json',dict(protocol=PROTOCOL,policy=binding,resolved_policy=policy,
        recipe=recipe,smoke=smoke,initial_state_sha256=initial,join=source['join'],cache_join=source['cache_join'],
        code_sha256=source['codes'],teacher_original_p_replay_max_error=source['teacher'].replay_max_error,
        STOP_CAL_AUDIT_observations_accessed=False))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    streams=old.PairedStreams(cfg['seed'],cfg['p_dropout_seed']);rng=torch.Generator().manual_seed(cfg['terminal_noise_seed'])
    checkpoints={};started=time.perf_counter()
    def save(epoch):
        path=output/('ema_epoch_%03d.pt'%epoch);data=source['data'];base=source['base']
        value=dict(protocol=PROTOCOL,policy=binding,recipe=recipe,epoch=epoch,smoke=smoke,EMA=True,
            state_dict={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},architecture=ema.architecture_config(),
            schedule=schedule.as_dict(),prediction_type='v',data=base['data'],labels_manifest=base['labels_manifest'],
            parent_checkpoint=policy['parent_checkpoint'],risk_cache=policy['risk_cache'],basis=data['basis'],
            coefficient_normalizer=data['coefficient_normalizer'],history_normalizer=data['history_normalizer'],code_sha256=source['codes'])
        with path.open('xb') as handle:torch.save(value,handle)
        checkpoints[str(epoch)]=dict(path=str(path),sha256=old.sha256(path))
    save(0)
    with (output/'epochs.jsonl').open('x') as log:
        for epoch in range(1,(1 if smoke else cfg['epochs'])+1):
            record=train_epoch(model,ema,schedule,optimizer,source['pack'],source['p'],source['cache'],source['teacher'],
                streams,rng,cfg,policy['physics'],recipe,device,max_updates=1 if smoke else None,
                progress=lambda r:print(json.dumps(dict(epoch=epoch,**r)),flush=True))
            record.update(epoch=epoch,recipe=recipe);log.write(json.dumps(record,sort_keys=True)+'\n');log.flush()
            if epoch in cfg['snapshot_epochs'] or smoke:save(epoch)
            print(json.dumps(record,sort_keys=True),flush=True)
    old.check_codes(source['codes'])
    result=dict(protocol=PROTOCOL,status='smoke_complete' if smoke else 'complete',policy=binding,recipe=recipe,smoke=smoke,
        epochs_completed=1 if smoke else cfg['epochs'],checkpoints=checkpoints,freeze=freeze,code_sha256=source['codes'],
        architecture=ema.architecture_config(),data=source['base']['data'],labels_manifest=source['base']['labels_manifest'],
        epochs=dict(path=str(output/'epochs.jsonl'),sha256=old.sha256(output/'epochs.jsonl')),wall_seconds=time.perf_counter()-started,
        parent_checkpoint=policy['parent_checkpoint'],production_default_changed=False,STOP_CAL_AUDIT_observations_accessed=False)
    print(json.dumps(dict(training_complete=old.write_json(output/'result.json',result))),flush=True)


def probe(binding):
    policy=read_policy(binding);torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);device=torch.device('cpu');source=load_inputs(policy,device)
    model=old.warmed(source['checkpoint'],device).eval();initial=old.state_hash(model)
    teacher=source['teacher'];pack=source['pack'];cfg=policy['training'];schedule=old.CosineDiffusionSchedule()
    frozen=old.bound_json(PROBE_SELECTION);rows=[item['row'] for item in frozen['selected']]
    expected=fixed_selection(pack['scene_id'],pack['recording_id'],pack['agent_mask'].sum(1),8)
    if rows!=expected:raise ValueError('fixed FIT probe selection changed')
    output=old.resolve(policy['output_root'])/'scale_probe';output.mkdir(parents=True,exist_ok=False)
    old.write_json(output/'freeze.json',dict(policy=binding,selection=PROBE_SELECTION,device='cpu',optimizer_steps=0,
        code_sha256=source['codes'],physics=policy['physics'],target='natural_OOF_p_only'))
    def grads():
        return torch.cat([(p.grad.detach().double() if p.grad is not None else torch.zeros_like(p,dtype=torch.float64)).flatten()
                          for p in model.parameters()])
    answers=[];started=time.perf_counter()
    with (output/'requests.jsonl').open('x') as log:
        for row in rows:
            sid=str(pack['scene_id'][row]);indices=np.array([row]);f,c,p=old.direct.tensor_batch(pack,source['p'],indices,device)
            seed=int(hashlib.sha256((SALT+'|noise|'+sid).encode()).hexdigest()[:8],16)
            z=torch.randn(tuple(c.shape),generator=torch.Generator().manual_seed(seed))
            model.zero_grad(set_to_none=True)
            loss,term,physical,_=compute_auxiliary(model,schedule,f,p,z,torch.ones(1,dtype=torch.bool),teacher.batch(indices),
                teacher.physical_batch(indices),cfg,policy['physics'],1.)
            term['loss'].backward(retain_graph=True);gt=grads();model.zero_grad(set_to_none=True)
            (policy['physics']['road_weight']*physical['road_loss']+policy['physics']['speed_weight']*physical['speed_loss']).backward()
            gp=grads();tn=float(gt.norm());pn=float(gp.norm())
            if not torch.isfinite(gt).all() or not torch.isfinite(gp).all():raise FloatingPointError('nonfinite scale-probe gradient')
            answer=dict(scene_id=sid,row=row,num_agents=int(f['agent_mask'].sum()),terminal_supported=term['supported_scenes'],
                terminal_loss=float(term['loss'].detach()),road_loss=float(physical['road_loss'].detach()),speed_loss=float(physical['speed_loss'].detach()),
                terminal_grad_norm=tn,physics_grad_norm=pn,physics_to_terminal_ratio=pn/tn if tn>0 else None,
                cosine=float(torch.dot(gt,gp)/(tn*pn)) if tn>0 and pn>0 else None,physical=physical['diagnostics'][0])
            answers.append(answer);log.write(json.dumps(answer,sort_keys=True)+'\n');log.flush()
            print(json.dumps(dict(probe_completed=len(answers),N=answer['num_agents'],physics_gradient=pn)),flush=True)
    if old.state_hash(model)!=initial:raise ValueError('probe changed weights')
    ratios=[r['physics_to_terminal_ratio'] for r in answers if r['physics_grad_norm']>0 and r['physics_to_terminal_ratio'] is not None]
    result=dict(status='complete',scenes=len(answers),nonzero_physics_gradients=sum(r['physics_grad_norm']>0 for r in answers),
        road_events=sum(r['physical']['road_has_training_violation'] for r in answers),
        speed_events=sum(r['physical']['speed_has_training_violation'] for r in answers),
        nonzero_ratio_median=float(np.median(ratios)) if ratios else None,nonzero_ratio_max=max(ratios) if ratios else None,
        all_gradients_finite=True,weights_unchanged=True,optimizer_steps=0,seconds=time.perf_counter()-started,
        source_selection=PROBE_SELECTION,policy=binding,code_sha256=source['codes'],weight_search_performed=False)
    print(json.dumps(dict(scale_probe=old.write_json(output/'result.json',result),summary=result)),flush=True)


def validate_reports(reports,traces,policy,binding):
    for recipe in policy['recipes']:
        report=reports[recipe]
        if (report.get('protocol')!=PROTOCOL or report.get('status')!='complete' or report.get('smoke') is not False
                or report.get('recipe')!=recipe or report.get('policy')!=binding or report.get('epochs_completed')!=3
                or set(report['checkpoints'])!={'0','1','3'} or report['parent_checkpoint']!=policy['parent_checkpoint']):
            raise ValueError('new training report is incomplete or mismatched')
    for recipe,trace in traces.items():
        if [row['epoch'] for row in trace]!=[1,2,3]:raise ValueError('missing three-epoch trace')
        for row in trace:
            if row['base_loss_scenes']!=9913 or row['updates']!=78 or not row['all_base_loss_rows_retained']:
                raise ValueError('base training exposure changed')
    baseline=traces['continue']
    for recipe,trace in traces.items():
        for a,b in zip(baseline,trace):
            if a['order_sha256']!=b['order_sha256'] or a['randomness']!=b['randomness']:
                raise ValueError('unpaired order, base noise, dropout or terminal plan')


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
    if previous['policy']!=policy['base_policy'] or previous['training']!=policy['previous_training']:
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
    parser.add_argument('--policy-sha256',default=SCALED_POLICY_SHA)
    parser.add_argument('--stage',choices=['probe','train','evaluate'],required=True)
    parser.add_argument('--recipe',choices=['physics','terminal_physics']);parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();binding=dict(path=args.policy,sha256=args.policy_sha256)
    if args.stage=='probe':probe(binding)
    elif args.stage=='train':train(binding,args.recipe,smoke=args.smoke)
    else:
        if args.smoke:raise ValueError('smoke results cannot enter evaluation')
        evaluate(binding)
