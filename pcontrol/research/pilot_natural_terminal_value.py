#!/usr/bin/env python3
"""Paired FIT-only training and original STOP12 evaluation of terminal loss.

The estimator, OOF labels, base tangent objective, architecture and inference
policy remain frozen. Terminal oracle calls occur only during training.
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

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from pcontrol.data.complete_scene_view import verify_binding, resolve, sha256
from pcontrol.generation.direct_p_cfg import ClassifierFreePercentileDenoiser, cfg_training_loss, cfg_sample
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.risk_tangent import load_tangent_cache
from pcontrol.generation.risk_tangent_loss import natural_tangent_losses
from pcontrol.generation.risk_guidance import TorchTrajectoryDecoder
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.training_ddim_suffix import training_ddim_suffix
from pcontrol.generation.terminal_training import terminal_batch_loss
from pcontrol.reference.scene_calibration import SceneCDFWarp
from pcontrol.reference.torch_frozen_cdf import FrozenTorchCDF
from pcontrol.plugins.risk_plugin import FrozenRiskPlugin
from pcontrol.generation.evaluation import quality_metrics
from pcontrol.research import train_natural_direct_p as direct
from pcontrol.research import train_natural_risk_tangent as tangent
from pcontrol.research.refine_natural_direct_p import PairedStreams, state_hash
from pcontrol.research.probe_natural_terminal_support import fit_arrays
from pcontrol.research.probe_natural_risk_directions import GeometryOnlyAdapter
from pcontrol.research.evaluate_natural_diffusion import bound_json, write_json, model_features
from pcontrol.research.evaluate_natural_direct_p import interval_error
from pcontrol.research.pilot_relative_direct_p import existing_stop_cases, additional_summary, eligibility, choose

POLICY_SHA = '7b7ffa041980c18a2b76437bf3538f46e81d910a6507c839a515065181f5dd3a'
PROTOCOL = 'natural_terminal_value_paired_training_v1'
ARCH_KEYS = ('coefficient_dim','hidden_dim','heads','layers','feedforward_dim')
CODE = ('pcontrol/research/pilot_natural_terminal_value.py','pcontrol/generation/terminal_training.py',
    'pcontrol/generation/training_ddim_suffix.py','pcontrol/generation/terminal_percentile_loss.py',
    'pcontrol/reference/torch_frozen_cdf.py','pcontrol/research/probe_natural_terminal_support.py',
    'pcontrol/research/train_natural_risk_tangent.py','pcontrol/research/refine_natural_direct_p.py',
    'pcontrol/research/pilot_relative_direct_p.py','pcontrol/research/evaluate_natural_direct_p_refinement.py',
    'pcontrol/research/evaluate_natural_direct_p.py','pcontrol/research/evaluate_natural_diffusion.py',
    'pcontrol/generation/evaluation.py','pcontrol/plugins/risk_plugin.py')


def read_policy(binding):
    if binding.get('sha256') != POLICY_SHA: raise ValueError('only frozen terminal pilot policy allowed')
    policy = bound_json(binding)
    if policy['protocol'] != PROTOCOL or policy['recipes'] != ['continue','terminal']:
        raise ValueError('wrong policy protocol or arms')
    for key in ('parent_selection','parent_training','parent_checkpoint','parent_STOP','base_training','risk_cache'):
        verify_binding(policy[key])
    return policy


def check_codes(codes):
    for path, digest in codes.items(): verify_binding(dict(path=path,sha256=digest))


def configure(policy):
    torch.set_num_threads(policy['runtime']['threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    return torch.device(policy['runtime']['device'])


def warmed(checkpoint, device):
    model = ClassifierFreePercentileDenoiser(**{k:checkpoint['architecture'][k] for k in ARCH_KEYS})
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    if model.architecture_config()!=checkpoint['architecture']: raise ValueError('architecture changed')
    if not torch.equal(model.null_p_embedding.detach(),checkpoint['state_dict']['null_p_embedding']):
        raise ValueError('learned null changed')
    return model.to(device)


class TerminalFITTeacher:
    """Identity-bound FIT geometry and per-row own-recording-excluded CDFs."""
    def __init__(self, manifest, prepared, pack, labels, device):
        self.pack, self.device = pack, device
        self.physical = fit_arrays(prepared['FIT_physical_pack'])
        for source in (labels,self.physical):
            for key in ('scene_id','recording_id','role'):
                if not np.array_equal(source[key],pack[key]): raise ValueError('FIT identity mismatch: '+key)
        if not np.array_equal(self.physical['agent_mask'],pack['agent_mask']): raise ValueError('roster changed')
        mask=pack['agent_mask']; counts=mask.sum(1); n=len(counts)
        if not np.array_equal(self.physical['history'][:,-1][mask],pack['anchors'][mask]):
            raise ValueError('physical t0 mismatch')
        if not np.all(pack['ego_mask'][:,0]) or not np.all(pack['ego_mask'].sum(1)==1):
            raise ValueError('geometry adapters require explicit first-ego roster')
        self.normalizer=bound_json(prepared['coefficient_normalizer'])
        self.masses=np.zeros((n,66),np.float64); self.nodes=np.zeros((n,9),np.float64)
        seen=np.zeros(n,bool); fit_records=set(map(str,pack['recording_id']))
        for fold, source in sorted(manifest['oof_fold_sources'].items()):
            if (set(source['held_recordings']) & set(source['train_recordings'])
                    or not set(source['held_recordings']) <= fit_records):
                raise PermissionError('invalid FIT recording exclusion')
            prediction=fit_arrays(source['held_predictions'],held_recordings=source['held_recordings'])
            rows=np.flatnonzero(labels['fold']==int(fold))
            lookup={(str(s),str(r)):j for j,(s,r) in enumerate(zip(prediction['scene_id'],prediction['recording_id']))}
            if len(lookup)!=len(rows) or len(lookup)!=len(prediction['scene_id']): raise ValueError('OOF fold size/identity mismatch')
            order=np.array([lookup[(str(pack['scene_id'][i]),str(pack['recording_id'][i]))] for i in rows])
            if (not set(map(str,pack['recording_id'][rows]))<=set(source['held_recordings'])
                    or not np.array_equal(prediction['num_agents'][order],counts[rows])
                    or not np.array_equal(prediction['pet_seconds'][order],labels['pet_seconds'][rows])):
                raise ValueError('wrong own-fold predictions')
            warp=SceneCDFWarp.from_dict(bound_json(source['warp']))
            self.masses[rows]=prediction['joint_masses'][order]; self.nodes[rows]=warp.row_nodes(counts[rows]); seen[rows]=True
        if not seen.all(): raise ValueError('missing OOF CDF rows')
        reference=FrozenTorchCDF(self.masses,counts,row_nodes=self.nodes)
        replay=reference.rank(torch.from_numpy(labels['pet_seconds']))['p_mid'].numpy()
        self.replay_max_error=float(np.max(np.abs(replay-labels['p_mid'])))
        if self.replay_max_error>1e-12: raise ValueError('own-fold rank replay failed')

    def batch(self, rows):
        mask=self.pack['agent_mask'][rows]; counts=mask.sum(1)
        reference=FrozenTorchCDF(torch.tensor(self.masses[rows],dtype=torch.float64,device=self.device),counts,
                                 row_nodes=self.nodes[rows])
        decoders, geometries=[],[]
        for row in rows:
            valid=self.pack['agent_mask'][row]; anchors=self.pack['anchors'][row,valid]
            decoders.append(TorchTrajectoryDecoder(TrajectoryBasis(8),self.normalizer,anchors,device=self.device))
            geometries.append(GeometryOnlyAdapter(self.physical['dimensions'][row,valid],anchors))
        return reference,decoders,geometries


def add_terminal_backward(model,schedule,features,p,noise,presence,teacher_batch,cfg,weight):
    """Accumulate after base backward; keep eval mode throughout recomputation."""
    mode=model.training
    try:
        model.eval()
        sample=training_ddim_suffix(model,schedule,features,p,noise,steps=cfg['terminal_DDIM_steps'],
            grad_last_steps=cfg['grad_last_steps'],cfg_scale=cfg['terminal_CFG_scale'])
        reference,decoders,geometries=teacher_batch
        term=terminal_batch_loss(sample['sample'],p,reference,decoders,geometries,features['agent_mask'],presence,
            beta=cfg['terminal_beta'],minimum_density=cfg['minimum_density'])
        loss=weight*term['loss']
        if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite terminal loss')
        loss.backward()
        return dict(loss=float(term['loss'].detach()),supported=term['supported_scenes'],
            present=term['condition_present_scenes'],total=term['total_scenes'],
            geometry_supported=term['geometry_supported_scenes'],geometry_reasons=term['geometry_reasons'],
            all_request_point_error_sum=term['all_request_point_MAE']*term['total_scenes'],
            forward_NFE=sample['network_evaluations'],backward_recompute_NFE=sample['backward_recompute_network_evaluations'])
    finally:
        model.train(mode)


def train_epoch(model,ema,schedule,optimizer,pack,p_values,cache,teacher,streams,terminal_rng,cfg,recipe,device,*,progress=None,max_updates=None):
    model.train(); order=streams.order(len(p_values)); hashes={k:hashlib.sha256() for k in ('base_noise','drop','presence','terminal_plan')}
    stat=dict(base_loss_scenes=0,updates=0,base_v_sum=0.,projection_support=0,jacobian_support=0,
        jacobian_disconnected=0,p_dropped=0,terminal_requests=0,terminal_present=0,terminal_supported=0,
        terminal_geometry_supported=0,terminal_value_supported_sum=0.,terminal_point_error_sum=0.,
        terminal_forward_NFE=0,terminal_backward_recompute_NFE=0,terminal_geometry_reasons={},gradient_norm_max=0.)
    reasons=Counter(); started=time.perf_counter(); weight=cfg['terminal_weight'][recipe]
    for start in range(0,len(order),cfg['batch_size']):
        if max_updates is not None and stat['updates']>=max_updates: break
        rows=order[start:start+cfg['batch_size']]
        features,clean,p=direct.tensor_batch(pack,p_values,rows,device); p=p.detach().requires_grad_(True)
        times,noise,u,present=streams.draw(clean.shape,schedule.steps,cfg['p_dropout_probability'])
        for a in (np.asarray(clean.shape,dtype=np.int64),times.numpy(),noise.numpy()): hashes['base_noise'].update(a.tobytes())
        hashes['drop'].update(u.tobytes()); hashes['presence'].update(present.tobytes())
        presence=torch.from_numpy(present).to(device); t=times.to(device)
        optimizer.zero_grad(set_to_none=True)
        base=cfg_training_loss(model,schedule,clean,features,p,presence,timesteps=t,noise=noise.to(device))
        extra=natural_tangent_losses(base['model_prediction'],base['prediction_target'],base['noisy_coefficients'],p,
            schedule.alpha_bar(t,base['model_prediction']),features['agent_mask'],condition_present=presence,
            projection_weight=cfg['loss_weights']['projection'],jacobian_weight=cfg['loss_weights']['jacobian'],
            min_jacobian_sigma=cfg['min_jacobian_sigma'],**tangent.cache_batch(cache,rows,clean.shape[1],device))
        if extra['diagnostics']['jacobian_evaluated'] and extra['diagnostics']['p_graph_connected'] is False:
            raise RuntimeError('base p-Jacobian disconnected')
        base_loss=base['loss']+extra['loss']
        if not bool(torch.isfinite(base_loss)): raise FloatingPointError('nonfinite original loss')
        base_loss.backward()
        stat['base_v_sum']+=float(base['loss'].detach())*len(rows)
        stat['projection_support']+=extra['support_counts']['projection'];stat['jacobian_support']+=extra['support_counts']['jacobian']
        del base_loss,base,extra
        selected=rows[:min(cfg['terminal_per_batch'],len(rows))]
        terminal_features,terminal_clean,terminal_p=direct.tensor_batch(pack,p_values,selected,device)
        terminal_noise=torch.randn(tuple(terminal_clean.shape),generator=terminal_rng)
        for a in (selected.astype(np.int64),np.asarray(terminal_noise.shape,dtype=np.int64),terminal_noise.numpy()):
            hashes['terminal_plan'].update(a.tobytes())
        terminal_present=presence[:len(selected)]
        if weight>0:
            terminal=add_terminal_backward(model,schedule,terminal_features,terminal_p,terminal_noise.to(device),
                terminal_present,teacher.batch(selected),cfg,weight)
            stat['terminal_requests']+=terminal['total'];stat['terminal_present']+=terminal['present']
            stat['terminal_supported']+=terminal['supported'];stat['terminal_geometry_supported']+=terminal['geometry_supported']
            stat['terminal_value_supported_sum']+=terminal['loss']*terminal['supported']
            stat['terminal_point_error_sum']+=terminal['all_request_point_error_sum'];reasons.update(terminal['geometry_reasons'])
            stat['terminal_forward_NFE']+=terminal['forward_NFE'];stat['terminal_backward_recompute_NFE']+=terminal['backward_recompute_NFE']
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
        if not bool(torch.isfinite(norm)): raise FloatingPointError('nonfinite combined parameter gradient')
        stat['gradient_norm_max']=max(stat['gradient_norm_max'],float(norm))
        optimizer.step();direct.prior.update_ema(ema,model,cfg['EMA_decay'])
        stat['updates']+=1;stat['base_loss_scenes']+=len(rows);stat['p_dropped']+=int((~present).sum())
        if progress and stat['updates']%10==0: progress(dict(recipe=recipe,updates=stat['updates'],wall_seconds=time.perf_counter()-started))
    stat.update(randomness={k:h.hexdigest() for k,h in hashes.items()},
        order_sha256=hashlib.sha256(order.astype(np.int64).tobytes()).hexdigest(),terminal_geometry_reasons=dict(reasons),
        wall_seconds=time.perf_counter()-started,all_base_loss_rows_retained=stat['base_loss_scenes']==len(p_values),
        terminal_value_supported_mean=stat['terminal_value_supported_sum']/max(stat['terminal_supported'],1),
        terminal_all_request_point_MAE=stat['terminal_point_error_sum']/max(stat['terminal_requests'],1))
    return stat


def train(binding,recipe,*,smoke=False):
    policy=read_policy(binding)
    if recipe not in policy['recipes']: raise ValueError('unknown recipe')
    device=configure(policy); cfg=policy['training']; torch.manual_seed(cfg['seed'])
    base,parent=bound_json(policy['base_training']),bound_json(policy['parent_training'])
    if bound_json(policy['parent_selection'])['choice']['selected']!='tangent_e40_s4': raise ValueError('wrong parent selection')
    checkpoint=torch.load(verify_binding(policy['parent_checkpoint']),map_location='cpu')
    if (checkpoint['recipe']!='tangent' or checkpoint['epoch']!=40 or checkpoint['data']!=base['data']
            or checkpoint['labels_manifest']!=base['labels_manifest']): raise ValueError('wrong parent data/labels')
    manifest=bound_json(policy['risk_cache']); prepared=bound_json(manifest['prepared'])
    pack=fit_arrays(prepared['FIT_generator_pack']); labels=fit_arrays(prepared['FIT_p_labels'])
    p_values,join=direct.join_percentile_labels(pack,labels,role='FIT')
    if len(p_values)!=9913 or len(set(pack['recording_id']))!=13: raise ValueError('FIT population changed')
    cache,cache_join=tangent.join_risk_cache(pack,p_values,load_tangent_cache(policy['risk_cache']))
    teacher=TerminalFITTeacher(manifest,prepared,pack,labels,device)
    model=warmed(checkpoint,device); model.requires_grad_(True)
    initial=state_hash(model); ema=copy.deepcopy(model).requires_grad_(False)
    schedule=CosineDiffusionSchedule(100).to(device)
    if schedule.as_dict()!=checkpoint['schedule']: raise ValueError('schedule changed')
    codes=dict(parent['code_sha256']);codes.update(manifest['code_sha256']);check_codes(codes)
    codes.update({path:sha256(ROOT/path) for path in CODE});check_codes(codes)
    output=resolve(policy['output_root'])/(recipe+'_smoke' if smoke else recipe);output.mkdir(parents=True,exist_ok=False)
    freeze=write_json(output/'freeze_before_training.json',dict(protocol=PROTOCOL,policy=binding,recipe=recipe,smoke=smoke,
        initial_state_sha256=initial,join=join,cache_join=cache_join,code_sha256=codes,
        teacher_original_p_replay_max_error=teacher.replay_max_error,training=cfg,
        all_FIT_sources_only=True,STOP_CAL_AUDIT_observations_accessed=False,optimizer_state_restored=False))
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    streams=PairedStreams(cfg['seed'],cfg['p_dropout_seed']);terminal_rng=torch.Generator().manual_seed(cfg['terminal_noise_seed'])
    checkpoints={};data=bound_json(base['data']);start=time.perf_counter()
    def save(epoch):
        path=output/('ema_epoch_%03d.pt'%epoch)
        state=dict(protocol=PROTOCOL,policy=binding,recipe=recipe,epoch=epoch,smoke=smoke,EMA=True,
            state_dict={k:v.detach().cpu().clone() for k,v in ema.state_dict().items()},
            architecture=ema.architecture_config(),schedule=schedule.as_dict(),prediction_type='v',
            data=base['data'],labels_manifest=base['labels_manifest'],parent_checkpoint=policy['parent_checkpoint'],
            risk_cache=policy['risk_cache'],basis=data['basis'],coefficient_normalizer=data['coefficient_normalizer'],
            history_normalizer=data['history_normalizer'],code_sha256=codes)
        with path.open('xb') as handle:torch.save(state,handle)
        checkpoints[str(epoch)]=dict(path=str(path),sha256=sha256(path))
    save(0)
    with (output/'epochs.jsonl').open('x') as log:
        for epoch in range(1,(1 if smoke else cfg['epochs'])+1):
            record=train_epoch(model,ema,schedule,optimizer,pack,p_values,cache,teacher,streams,terminal_rng,cfg,recipe,device,
                progress=lambda row:print(json.dumps(dict(epoch=epoch,**row)),flush=True),max_updates=1 if smoke else None)
            record.update(epoch=epoch,recipe=recipe);log.write(json.dumps(record,sort_keys=True)+'\n');log.flush()
            if epoch in cfg['snapshot_epochs'] or smoke:save(epoch)
            print(json.dumps(record,sort_keys=True),flush=True)
    check_codes(codes)
    result=dict(protocol=PROTOCOL,status='smoke_complete' if smoke else 'complete',policy=binding,recipe=recipe,
        smoke=smoke,epochs_completed=1 if smoke else cfg['epochs'],checkpoints=checkpoints,freeze=freeze,
        code_sha256=codes,architecture=ema.architecture_config(),data=base['data'],labels_manifest=base['labels_manifest'],
        epochs=dict(path=str(output/'epochs.jsonl'),sha256=sha256(output/'epochs.jsonl')),
        wall_seconds=time.perf_counter()-start,parent_checkpoint=policy['parent_checkpoint'],
        STOP_CAL_AUDIT_observations_accessed=False,production_default_changed=False)
    print(json.dumps(dict(training_complete=write_json(output/'result.json',result))),flush=True)


def validate_traces(reports, traces, policy, binding):
    if set(reports)!=set(policy['recipes']) or set(traces)!=set(policy['recipes']):
        raise ValueError('both paired recipes required')
    for recipe,report in reports.items():
        if (report.get('protocol')!=PROTOCOL or report.get('status')!='complete'
                or report.get('smoke') is not False or report.get('recipe')!=recipe
                or report.get('policy')!=binding or report.get('epochs_completed')!=policy['training']['epochs']
                or set(report.get('checkpoints',{}))!=set(map(str,policy['training']['snapshot_epochs']))
                or report.get('parent_checkpoint')!=policy['parent_checkpoint']):
            raise ValueError('incomplete, mismatched, or smoke training cannot enter comparison')
        expected=range(1,policy['training']['epochs']+1)
        if [r['epoch'] for r in traces[recipe]]!=list(expected): raise ValueError('missing fixed epochs')
        for row in traces[recipe]:
            if row['base_loss_scenes']!=9913 or row['updates']!=78 or not row['all_base_loss_rows_retained']:
                raise ValueError('training population or update count changed')
    for first,second in zip(traces['continue'],traces['terminal']):
        if first['randomness']!=second['randomness'] or first['order_sha256']!=second['order_sha256']:
            raise ValueError('paired data/drop/noise/terminal plan mismatch')


def evaluate(binding):
    policy=read_policy(binding);device=configure(policy);root=resolve(policy['output_root'])
    report_bindings={r:dict(path=str(root/r/'result.json'),sha256=sha256(root/r/'result.json')) for r in policy['recipes']}
    reports={r:bound_json(b) for r,b in report_bindings.items()}
    traces={r:[json.loads(line) for line in verify_binding(v['epochs']).read_text().splitlines()] for r,v in reports.items()}
    validate_traces(reports,traces,policy,binding)
    for report in reports.values():check_codes(report['code_sha256'])
    parent=bound_json(policy['parent_STOP']);cases=existing_stop_cases(parent)
    base=bound_json(policy['base_training']);data=bound_json(base['data']);base_policy=bound_json(base['policy'])
    cn,hn=bound_json(data['coefficient_normalizer']),bound_json(data['history_normalizer'])
    plugin=FrozenRiskPlugin.from_refinement_binding(base_policy['risk_plugin_result'],device='cpu')
    references={case['scene_id']:plugin.condition(case['history'],case['dimensions'],case['road_boundaries'],
        case['ego_mask'],case['agent_mask']) for case in cases}
    grid={'parent':dict(recipe='parent',epoch=0,checkpoint=policy['parent_checkpoint'])}
    for recipe in policy['recipes']:
        for epoch in policy['evaluation']['candidate_epochs']:
            grid[recipe+'_e'+str(epoch)]=dict(recipe=recipe,epoch=epoch,checkpoint=reports[recipe]['checkpoints'][str(epoch)])
    output=root/'STOP_evaluation';output.mkdir(exist_ok=False);codes={path:sha256(ROOT/path) for path in CODE}
    write_json(output/'freeze_before_sampling.json',dict(protocol=PROTOCOL,policy=binding,training=report_bindings,
        candidates=grid,code_sha256=codes,parent_STOP=policy['parent_STOP'],CDF=base_policy['risk_plugin_result'],
        role='STOP',same_noise_and_history=True,K=1,CAL_AUDIT_accessed=False))
    summaries={};parent_summary=None;basis=TrajectoryBasis(8)
    old_rows={(r['scene_id'],r['requested_p']):r for r in parent['rows']}
    for name,candidate in grid.items():
        checkpoint=torch.load(verify_binding(candidate['checkpoint']),map_location='cpu')
        if name!='parent':
            report=reports[candidate['recipe']]
            if (checkpoint['protocol']!=PROTOCOL or checkpoint['policy']!=binding or checkpoint['smoke']
                    or not checkpoint['EMA'] or checkpoint['epoch']!=candidate['epoch']
                    or checkpoint['recipe']!=candidate['recipe'] or checkpoint['code_sha256']!=report['code_sha256']
                    or checkpoint['data']!=base['data'] or checkpoint['labels_manifest']!=base['labels_manifest']
                    or checkpoint['parent_checkpoint']!=policy['parent_checkpoint']):
                raise ValueError('candidate checkpoint provenance mismatch')
        model=warmed(checkpoint,device).eval().requires_grad_(False);schedule=CosineDiffusionSchedule(100).to(device)
        target=output/name;target.mkdir();rows=[];started=time.perf_counter();replay_error=0.
        for number,case in enumerate(cases):
            features=model_features(case,hn,device);z=torch.from_numpy(case['initial_noise'][None]).to(device)
            arrays={k:case[k] for k in ('history','dimensions','road_boundaries','ego_mask','agent_ids','initial_noise')}
            arrays['future_observed']=case['future'];futures={}
            for p in policy['evaluation']['p_grid']:
                coeff=cfg_sample(model,schedule,features,torch.tensor([p],device=device),z,scale=4.,steps=50)
                physical=coeff[0].cpu().numpy().astype(np.float64)*np.asarray(cn['scale'])+np.asarray(cn['mean'])
                future=basis.decode(physical,case['history'][-1]);futures[p]=future
                arrays['generated_p'+str(p).replace('.','_')]=future
            path=target/('case_%02d.npz'%number)
            with path.open('xb') as handle:np.savez_compressed(handle,**arrays)
            artifact=dict(path=str(path),sha256=sha256(path));reference=references[case['scene_id']]
            for p,future in futures.items():
                score=reference.score_future(future);rank=score['estimated_rank'];spec=reference.target_spec(p)
                row={k:case[k] for k in ('scene_id','recording_id','num_agents','role','stratum','noise_seed')}
                row.update(candidate_id=name,method='GP_direct_P',requested_p=p,pet_seconds=score['pet_seconds'],
                    estimated_rank=rank,target_spec=spec,p_mid_absolute_error=abs(rank['p_mid']-p),
                    p_interval_error=interval_error(p,rank),PET_target_absolute_error_seconds=abs(score['pet_seconds']-spec['target_pet_seconds']),
                    quality=quality_metrics(future,case),trajectory_artifact=artifact,
                    array_key='generated_p'+str(p).replace('.','_'),K=1,network_evaluations=100,
                    post_correction=False,external_risk_gradient_guidance=False)
                if name=='parent':
                    old=old_rows[(case['scene_id'],p)]
                    replay_error=max(replay_error,abs(rank['p_mid']-old['estimated_rank']['p_mid']),abs(score['pet_seconds']-old['pet_seconds']))
                rows.append(row)
        if name=='parent' and replay_error>policy['evaluation']['parent_rank_replay_tolerance']:
            raise ValueError('parent does not replay saved STOP; no candidate comparison allowed')
        summary=additional_summary(rows)
        result=write_json(target/'results.json',dict(protocol=PROTOCOL,status='complete',candidate=candidate,summary=summary,rows=rows,
            parent_replay_max_rank_or_PET_error=replay_error if name=='parent' else None))
        if name=='parent':parent_summary=summary
        summaries[name]=dict(candidate,summary=summary,result=result,guard=eligibility(summary,parent_summary,policy))
        print(json.dumps(dict(candidate_complete=name,MAE=summary['p_mid_MAE'],Fine=summary['Fine_at_0_05'],
            p50=summary['by_requested_p']['0.5'],eligible=summaries[name]['guard']['eligible'],seconds=time.perf_counter()-started)),flush=True)
    check_codes(codes);candidates={k:v for k,v in summaries.items() if k!='parent'}
    paired={str(e):{metric:candidates['terminal_e'+str(e)]['summary'][metric]-candidates['continue_e'+str(e)]['summary'][metric]
        for metric in ('p_mid_MAE','Fine_at_0_05','road_outside_scene_rate','all_pair_overlap_scene_rate')}
        for e in policy['evaluation']['candidate_epochs']}
    result=dict(protocol=PROTOCOL,status='complete',policy=binding,training=report_bindings,candidates=candidates,
        parent=summaries['parent'],parent_summary=parent_summary,paired_terminal_minus_continue=paired,
        selected=choose(candidates,parent_summary,policy),role='STOP',all_paired_randomness_matched=True,
        CAL_AUDIT_accessed=False,estimated_reference_not_true_CDF=True,production_default_changed=False,code_sha256=codes)
    print(json.dumps(dict(evaluation_complete=write_json(output/'results.json',result),selected=result['selected'])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',default='configs/natural_percentile/terminal_value_pilot_v1.json')
    parser.add_argument('--policy-sha256',default=POLICY_SHA)
    parser.add_argument('--stage',choices=['train','evaluate'],required=True)
    parser.add_argument('--recipe',choices=['continue','terminal'])
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args();binding=dict(path=args.policy,sha256=args.policy_sha256)
    if args.stage=='train':train(binding,args.recipe,smoke=args.smoke)
    else:
        if args.smoke:raise ValueError('smoke checkpoint cannot enter STOP comparison')
        evaluate(binding)
