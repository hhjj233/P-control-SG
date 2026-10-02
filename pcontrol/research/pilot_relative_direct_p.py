#!/usr/bin/env python3
"""Bounded same-ruler continuation vs observed-relative denoiser pilot.

Natural FIT training and the existing STOP illustration set only. The CDF,
OOF labels, tangent cache, coefficient basis, p/seed/CFG budgets stay fixed.
No CAL/AUDIT observations or protected raw data are accessed. No production
default is changed. This pilot is not a capacity-matched novelty proof.
"""
import argparse
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pcontrol.data.complete_scene_view import resolve, verify_binding, sha256
from pcontrol.generation.direct_p_cfg import ClassifierFreePercentileDenoiser, cfg_sample
from pcontrol.generation.relative_direct_p import RelativeInteractionPercentileDenoiser
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.risk_tangent import load_tangent_cache
from pcontrol.generation.evaluation import quality_metrics
from pcontrol.plugins.risk_plugin import FrozenRiskPlugin
from pcontrol.research import train_natural_direct_p as direct
from pcontrol.research import train_natural_risk_tangent as tangent
from pcontrol.research.refine_natural_direct_p import PairedStreams, state_hash
from pcontrol.research.evaluate_natural_direct_p import summarize, interval_error
from pcontrol.research.evaluate_natural_direct_p_refinement import quality_guards
from pcontrol.research.evaluate_natural_diffusion import bound_json, write_json, model_features

POLICY_SHA = '0de3fb954fbe97bda4548353abd4ba41af0ac81d631c0474550667eec08fa09a'
PROTOCOL = 'natural_relative_direct_P_architecture_pilot_v1'
CODE = ('pcontrol/research/pilot_relative_direct_p.py', 'pcontrol/generation/relative_direct_p.py',
        'pcontrol/research/evaluate_natural_direct_p.py', 'pcontrol/research/evaluate_natural_direct_p_refinement.py',
        'pcontrol/research/evaluate_natural_diffusion.py', 'pcontrol/generation/evaluation.py')
ARCH_KEYS = ('coefficient_dim', 'hidden_dim', 'heads', 'layers', 'feedforward_dim')
RNG_KEYS = {'order_sha256', 'diffusion_t_and_noise_sha256', 'p_dropout_uniform_sha256',
            'effective_p_presence_mask_sha256', 'diffusion_rng_end_state_sha256'}


def read_policy(binding):
    if binding.get('sha256') != POLICY_SHA:
        raise ValueError('only the frozen relative pilot policy is allowed')
    policy = bound_json(binding)
    if policy['protocol'] != PROTOCOL or policy['recipes'] != ['continue', 'relative']:
        raise ValueError('wrong pilot recipes')
    for key in ('parent_selection', 'parent_training', 'parent_checkpoint', 'parent_STOP', 'base_training', 'risk_cache'):
        verify_binding(policy[key])
    return policy


def model_from_checkpoint(checkpoint, recipe, device, *, warm=False):
    config = {key: checkpoint['architecture'][key] for key in ARCH_KEYS}
    cls = ClassifierFreePercentileDenoiser if recipe == 'continue' else RelativeInteractionPercentileDenoiser
    model = cls(**config)
    if recipe == 'relative' and warm:
        model.load_from_parent_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    for name, value in checkpoint['state_dict'].items():
        if not torch.equal(model.state_dict()[name].cpu(), value.cpu()):
            raise ValueError('warm start or reload changed a source tensor: '+name)
    return model.to(device)


def check_codes(codes):
    for name, checksum in codes.items():
        verify_binding(dict(path=name, sha256=checksum))


def snapshot(output, epoch, recipe, model, schedule, policy_binding, policy, base, data, codes):
    path = output/('ema_epoch_%03d.pt' % epoch)
    value = dict(protocol=PROTOCOL, policy=policy_binding, recipe=recipe, epoch=epoch,
        state_dict={key: tensor.detach().cpu().clone() for key, tensor in model.state_dict().items()},
        architecture=model.architecture_config(), schedule=schedule.as_dict(), prediction_type='v',
        data=base['data'], labels_manifest=base['labels_manifest'], risk_cache=policy['risk_cache'],
        parent_checkpoint=policy['parent_checkpoint'], basis=data['basis'],
        coefficient_normalizer=data['coefficient_normalizer'], history_normalizer=data['history_normalizer'],
        code_sha256=codes, EMA=True, labels_changed=False, CDF_changed=False, AUDIT_decoded=False)
    with path.open('xb') as handle:
        torch.save(value, handle)
    return dict(path=str(path), sha256=sha256(path))


def train(policy_binding, recipe):
    policy = read_policy(policy_binding)
    if recipe not in policy['recipes']:
        raise ValueError('unregistered recipe')
    base, parent = bound_json(policy['base_training']), bound_json(policy['parent_training'])
    selected = bound_json(policy['parent_selection'])
    if selected['choice']['selected'] != 'tangent_e40_s4':
        raise ValueError('wrong frozen warm-start choice')
    source = direct.prepare_inputs(base['data'], base['labels_manifest'], base['policy'])
    cache, cache_join = tangent.join_risk_cache(source['fit'], source['fit_p'], load_tangent_cache(policy['risk_cache']))
    cfg = policy['training']; device = torch.device(policy['runtime']['device'])
    torch.set_num_threads(policy['runtime']['threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(cfg['seed'])
    parent_checkpoint = torch.load(verify_binding(policy['parent_checkpoint']), map_location='cpu')
    if (parent_checkpoint['recipe'] != 'tangent' or parent_checkpoint['epoch'] != 40
            or parent_checkpoint['data'] != parent['data'] or parent['data'] != base['data']
            or parent_checkpoint['labels_manifest'] != parent['labels_manifest']):
        raise ValueError('wrong parent training/data/labels')
    model = model_from_checkpoint(parent_checkpoint, recipe, device, warm=True)
    schedule = CosineDiffusionSchedule(100).to(device)
    if schedule.as_dict() != parent_checkpoint['schedule']:
        raise ValueError('schedule changed')
    original = model_from_checkpoint(parent_checkpoint, 'continue', device)
    features, clean, p = direct.tensor_batch(source['fit'], source['fit_p'], slice(0, 4), device)
    with torch.no_grad():
        steps = torch.tensor([0, 25, 50, 99], device=device)
        if not torch.equal(original(clean, steps, features, p), model(clean, steps, features, p)):
            raise ValueError('zero-initialized new model does not exactly replay parent')
    del original, parent_checkpoint
    codes = dict(parent['code_sha256']); check_codes(codes)
    codes.update({path: sha256(ROOT/path) for path in CODE}); check_codes(codes)
    output = resolve(policy['output_root'])/recipe; output.mkdir(parents=True, exist_ok=False)
    freeze = write_json(output/'freeze_before_training.json', dict(protocol=PROTOCOL, policy=policy_binding,
        recipe=recipe, code_sha256=codes, parent_checkpoint=policy['parent_checkpoint'],
        initial_state_sha256=state_hash(model), initial_parent_function_exact=True,
        architecture=model.architecture_config(), cache_join=cache_join, training=cfg,
        data=base['data'], labels_manifest=base['labels_manifest'], CAL_AUDIT_decoded=False))
    ema = copy.deepcopy(model).requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'], weight_decay=cfg['weight_decay'])
    streams = PairedStreams(cfg['seed'], cfg['p_dropout_seed'])
    checkpoints = {'0': snapshot(output, 0, recipe, ema, schedule, policy_binding, policy, base, source['data'], codes)}
    validation = {'0': direct.fixed_validation(ema, schedule, source['stop'], source['stop_p'], policy, device)}
    started = time.perf_counter()
    with (output/'epochs.jsonl').open('x') as log:
        for epoch in range(1, cfg['epochs']+1):
            record = tangent.train_epoch(model, ema, schedule, optimizer, source['fit'], source['fit_p'],
                cache, streams, cfg, cfg['loss_weights'], device, min_sigma=cfg['min_jacobian_sigma'])
            record.update(epoch=epoch, recipe=recipe, wall_seconds=time.perf_counter()-started)
            if epoch % cfg['validation_every'] == 0:
                validation[str(epoch)] = direct.fixed_validation(ema, schedule, source['stop'], source['stop_p'], policy, device)
                record['STOP_v_MSE_diagnostic'] = validation[str(epoch)]
            if epoch in cfg['snapshot_epochs']:
                checkpoints[str(epoch)] = snapshot(output, epoch, recipe, ema, schedule, policy_binding,
                    policy, base, source['data'], codes)
                record['checkpoint'] = checkpoints[str(epoch)]
            log.write(json.dumps(record, sort_keys=True)+'\n'); log.flush()
            print(json.dumps(record, sort_keys=True), flush=True)
    check_codes(codes)
    result = dict(protocol=PROTOCOL, status='complete', policy=policy_binding, recipe=recipe,
        parent_checkpoint=policy['parent_checkpoint'],
        checkpoints=checkpoints, architecture=ema.architecture_config(), data=base['data'],
        labels_manifest=base['labels_manifest'], risk_cache=policy['risk_cache'],
        code_sha256=codes, freeze=freeze, epochs_completed=cfg['epochs'],
        epochs=dict(path=str(output/'epochs.jsonl'), sha256=sha256(output/'epochs.jsonl')),
        STOP_v_MSE_diagnostics=validation, wall_seconds=time.perf_counter()-started,
        labels_changed=False, CDF_changed=False, CAL_decoded=False, AUDIT_decoded=False,
        performance_improvement_claimed=False, production_default_changed=False)
    result_binding = write_json(output/'result.json', result)
    print(json.dumps(dict(training_complete=True, recipe=recipe, result=result_binding)), flush=True)


def additional_summary(rows):
    value = summarize(rows)
    value['by_requested_p'] = {format(p, 'g'): float(np.mean([r['p_mid_absolute_error'] for r in rows if r['requested_p']==p]))
                               for p in (.1, .5, .9)}
    return value


def eligibility(summary, parent, policy):
    guard = quality_guards(summary, parent, policy)
    for key, value, maximum in (
        ('negative_Fine', -summary['Fine_at_0_05'], -parent['Fine_at_0_05']),
        ('p50_MAE', summary['by_requested_p']['0.5'], parent['by_requested_p']['0.5'])):
        guard['checks'][key] = dict(value=value, maximum=maximum, passed=value<=maximum+1e-12)
    guard['eligible'] = all(item['passed'] for item in guard['checks'].values())
    return guard


def choose(candidates, parent, policy):
    eligible = [key for key, row in candidates.items() if row['guard']['eligible']
                and row['summary']['p_mid_MAE'] < parent['p_mid_MAE']-policy['evaluation']['min_improvement']]
    return min(eligible, key=lambda k: (candidates[k]['summary']['p_mid_MAE'], candidates[k]['epoch'], k)) if eligible else 'parent'


def validate_training_reports(reports, policy_binding, policy, parent):
    if set(reports) != set(policy['recipes']):
        raise ValueError('all and only registered arms are required')
    for recipe, report in reports.items():
        if (report.get('protocol') != PROTOCOL or report.get('status') != 'complete'
                or report.get('recipe') != recipe or report.get('epochs_completed') != policy['training']['epochs']
                or report.get('policy') != policy_binding
                or report.get('parent_checkpoint') != policy['parent_checkpoint']
                or report.get('data') != parent['data'] or report.get('labels_manifest') != parent['labels_manifest']
                or report.get('risk_cache') != policy['risk_cache']
                or set(report.get('checkpoints', {})) != {'0', '10', '20'}
                or any(report.get(key) is not False for key in
                    ('labels_changed','CDF_changed','CAL_decoded','AUDIT_decoded','production_default_changed'))):
            raise ValueError('completed training report or source contract changed')
        check_codes(report['code_sha256'])


def validate_paired_traces(traces, *, epochs=20):
    if set(traces) != {'continue', 'relative'}:
        raise ValueError('both paired traces required')
    for recipe, rows in traces.items():
        if len(rows) != epochs or [r.get('epoch') for r in rows] != list(range(1,epochs+1)):
            raise ValueError('trace must contain every fixed epoch exactly once in order')
        for row in rows:
            if (row.get('recipe') != recipe or row.get('base_loss_scenes') != 9913
                    or row.get('updates') != 78 or row.get('p_disconnected_batches') != 0
                    or set(row.get('randomness', {})) != RNG_KEYS):
                raise ValueError('trace role/population/update/randomness schema changed')
            if any(not isinstance(v,str) or len(v)!=64 or any(c not in '0123456789abcdef' for c in v)
                   for v in row['randomness'].values()):
                raise ValueError('randomness checksums must be exact SHA256 values')
    for first, second in zip(traces['continue'], traces['relative']):
        if first['randomness'] != second['randomness']:
            raise ValueError('training order/noise/presence differs across arms')


def validate_candidate_checkpoint(checkpoint, candidate, report, policy_binding, policy, data):
    if (checkpoint.get('protocol') != PROTOCOL or checkpoint.get('policy') != policy_binding
            or checkpoint.get('recipe') != candidate['recipe'] or checkpoint.get('epoch') != candidate['epoch']
            or checkpoint.get('code_sha256') != report['code_sha256']
            or checkpoint.get('data') != report['data'] or checkpoint.get('labels_manifest') != report['labels_manifest']
            or checkpoint.get('risk_cache') != policy['risk_cache']
            or checkpoint.get('parent_checkpoint') != policy['parent_checkpoint']
            or checkpoint.get('architecture') != report['architecture']
            or checkpoint.get('basis') != data['basis']
            or checkpoint.get('coefficient_normalizer') != data['coefficient_normalizer']
            or checkpoint.get('history_normalizer') != data['history_normalizer']
            or checkpoint.get('prediction_type') != 'v' or checkpoint.get('EMA') is not True
            or checkpoint.get('schedule') != CosineDiffusionSchedule(100).as_dict()
            or any(checkpoint.get(key) is not False for key in ('labels_changed','CDF_changed','AUDIT_decoded'))):
        raise ValueError('candidate checkpoint differs from its fixed training/data/schedule contract')


def existing_stop_cases(parent):
    """Only already-saved parent STOP artifacts; no raw trajectory loader."""
    seen = set(); cases = []
    for row in parent['rows']:
        if row['role'] != 'STOP':
            raise PermissionError('only parent STOP may be opened')
        if row['scene_id'] in seen:
            continue
        seen.add(row['scene_id'])
        binding = row['trajectory_artifact']
        with np.load(verify_binding(binding), allow_pickle=False) as archive:
            case = {k: archive[k].copy() for k in
                    ('history', 'dimensions', 'road_boundaries', 'ego_mask', 'agent_ids', 'initial_noise', 'future_observed')}
        case.update({k: row[k] for k in ('scene_id', 'recording_id', 'num_agents', 'role', 'stratum', 'noise_seed')})
        case.update(future=case.pop('future_observed'), agent_mask=np.ones(row['num_agents'], bool), future_dt=.04)
        cases.append(case)
    if len(cases) != 12:
        raise ValueError('exact parent paired STOP12 required')
    return cases


def evaluate(policy_binding):
    policy = read_policy(policy_binding); source = resolve(policy['output_root'])
    training = {r: dict(path=str(source/r/'result.json'), sha256=sha256(source/r/'result.json')) for r in policy['recipes']}
    reports = {r: bound_json(b) for r, b in training.items()}
    base = bound_json(policy['base_training'])
    validate_training_reports(reports, policy_binding, policy, base)
    traces = {r: [json.loads(line) for line in verify_binding(v['epochs']).read_text().splitlines()] for r, v in reports.items()}
    validate_paired_traces(traces, epochs=policy['training']['epochs'])
    parent = bound_json(policy['parent_STOP']); parent_summary = additional_summary(parent['rows'])
    base_policy = bound_json(base['policy']); data = bound_json(base['data'])
    grid = {r+'_e'+str(e)+'_s4': dict(recipe=r, epoch=e, checkpoint=reports[r]['checkpoints'][str(e)])
            for r in policy['recipes'] for e in policy['evaluation']['candidate_epochs']}
    checkpoint_headers = {}
    for key, candidate in grid.items():
        ck = torch.load(verify_binding(candidate['checkpoint']), map_location='cpu')
        validate_candidate_checkpoint(ck, candidate, reports[candidate['recipe']], policy_binding, policy, data)
        checkpoint_headers[key] = ck
    output = source/'STOP_evaluation'; output.mkdir(exist_ok=False)
    codes = {p: sha256(ROOT/p) for p in CODE}
    write_json(output/'freeze_before_sampling.json', dict(protocol=PROTOCOL, policy=policy_binding,
        training=training, candidates=grid, code_sha256=codes, role='STOP', parent=policy['parent_STOP'],
        CDF=base_policy['risk_plugin_result'], same_H_and_initial_noise=True, K=1, CAL_AUDIT_decoded=False))
    torch.set_num_threads(policy['runtime']['threads']); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    device = torch.device(policy['runtime']['device']); cases = existing_stop_cases(parent)
    plugin = FrozenRiskPlugin.from_refinement_binding(base_policy['risk_plugin_result'], device='cpu')
    cn, hn = bound_json(data['coefficient_normalizer']), bound_json(data['history_normalizer'])
    basis = TrajectoryBasis(8); summaries = {}
    for key, candidate in grid.items():
        target = output/key; target.mkdir()
        ck = checkpoint_headers[key]
        model = model_from_checkpoint(ck, candidate['recipe'], device).eval().requires_grad_(False)
        schedule = CosineDiffusionSchedule(100).to(device); rows=[]; started=time.perf_counter()
        for number, case in enumerate(cases):
            features = model_features(case, hn, device)
            z = torch.from_numpy(case['initial_noise'][None]).to(device)
            arrays = {k:case[k] for k in ('history','dimensions','road_boundaries','ego_mask','agent_ids','initial_noise')}
            arrays['future_observed'] = case['future']; futures = {}
            for p in policy['evaluation']['p_grid']:
                c = cfg_sample(model, schedule, features, torch.tensor([p],device=device), z, scale=4., steps=50)
                physical = c[0].cpu().numpy().astype(np.float64)*np.asarray(cn['scale'])+np.asarray(cn['mean'])
                futures[p] = basis.decode(physical, case['history'][-1]); arrays['generated_p'+str(p).replace('.','_')] = futures[p]
            reference = plugin.condition(case['history'],case['dimensions'],case['road_boundaries'],case['ego_mask'],case['agent_mask'])
            path = target/('case_%02d.npz'%number)
            with path.open('xb') as handle:np.savez_compressed(handle,**arrays)
            artifact = dict(path=str(path),sha256=sha256(path))
            for p, future in futures.items():
                scored = reference.score_future(future); spec=reference.target_spec(p); rank=scored['estimated_rank']
                row = {k:case[k] for k in ('scene_id','recording_id','num_agents','role','stratum','noise_seed')}
                row.update(candidate_id=key,method='GP_direct_P',requested_p=p,pet_seconds=scored['pet_seconds'],
                    estimated_rank=rank,target_spec=spec,p_mid_absolute_error=abs(rank['p_mid']-p),
                    p_interval_error=interval_error(p,rank),PET_target_absolute_error_seconds=abs(scored['pet_seconds']-spec['target_pet_seconds']),
                    quality=quality_metrics(future,case),trajectory_artifact=artifact,array_key='generated_p'+str(p).replace('.','_'),
                    K=1,network_evaluations=100,post_correction=False,external_risk_gradient_guidance=False)
                rows.append(row)
        summary = additional_summary(rows)
        result = write_json(target/'results.json',dict(protocol=PROTOCOL,status='complete',candidate=candidate,rows=rows,summary=summary))
        summaries[key] = dict(candidate,summary=summary,guard=eligibility(summary,parent_summary,policy),result=result)
        print(json.dumps(dict(candidate_complete=key,point_MAE=summary['p_mid_MAE'],Fine=summary['Fine_at_0_05'],
            p50_MAE=summary['by_requested_p']['0.5'],eligible=summaries[key]['guard']['eligible'],wall_seconds=time.perf_counter()-started)),flush=True)
    check_codes(codes)
    result = dict(protocol=PROTOCOL,status='complete',policy=policy_binding,training=training,candidates=summaries,
        parent_result=policy['parent_STOP'],parent_summary=parent_summary,selected=choose(summaries,parent_summary,policy),
        code_sha256=codes,all_20_epoch_randomness_matched=True,role='STOP',CAL_AUDIT_decoded=False,
        estimated_reference_not_true_CDF=True,production_default_changed=False,capacity_matched_control=False)
    binding=write_json(output/'results.json',result)
    print(json.dumps(dict(evaluation_complete=True,result=binding,selected=result['selected'])),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',default='configs/natural_percentile/relative_direct_p_pilot_v1.json')
    parser.add_argument('--policy-sha256',default=POLICY_SHA)
    parser.add_argument('--stage',choices=('train','evaluate'),required=True)
    parser.add_argument('--recipe',choices=('continue','relative'))
    args=parser.parse_args(); binding=dict(path=args.policy,sha256=args.policy_sha256)
    if args.stage=='train':train(binding,args.recipe)
    else:evaluate(binding)
