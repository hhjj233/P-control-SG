#!/usr/bin/env python3
"""Fixed FIT-only, no-update coverage probe for exact terminal value gradients.

Never filters scene selection using future PET, rank, or old/new gradient support.
The balanced count/recording sample is diagnostic, not population-representative.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pcontrol.generation.direct_p_cfg import ClassifierFreePercentileDenoiser, cfg_training_loss
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.training_ddim_suffix import training_ddim_suffix
from pcontrol.generation.terminal_percentile_loss import terminal_percentile_value_loss
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.risk_guidance import TorchTrajectoryDecoder
from pcontrol.reference.scene_calibration import SceneCDFWarp
from pcontrol.reference.torch_frozen_cdf import FrozenTorchCDF
from pcontrol.research.probe_natural_risk_directions import GeometryOnlyAdapter

SALT = 'natural_terminal_support_count_recording_balanced_v1'
MANIFEST = dict(path='outputs/natural_percentile/natural_risk_tangent_cache_v1_20260910/manifest.json',
    sha256='91393995f81419ca620065a38d5def6aa239f31d984bb2921e2668221a834f2b')
PARENT = dict(path='outputs/natural_percentile/natural_risk_tangent_training_v1_20260910/tangent/ema_epoch_040.pt',
    sha256='f6e1f0f718429a050d003d89dd3d3a899d36fe61cd6ab77e05791d1dfe8454f7')


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def bound(binding):
    path = Path(binding['path'])
    if not path.is_absolute(): path = ROOT/path
    if hashlib.sha256(path.read_bytes()).hexdigest() != binding['sha256']:
        raise ValueError('source binding changed: '+str(path))
    return path


def read_json(binding):
    return json.loads(bound(binding).read_text())


def write_new(path, value):
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def fit_arrays(binding, *, held_recordings=None):
    with np.load(bound(binding), allow_pickle=False) as archive:
        if 'role' in archive.files:
            if not np.all(archive['role'] == 'FIT'):
                raise PermissionError('only FIT observations may be decoded')
        elif held_recordings is None:
            raise PermissionError('role-less arrays require hash-bound own-fold FIT recordings')
        if held_recordings is not None:
            actual = set(map(str, archive['recording_id']))
            if not actual or not actual <= set(held_recordings):
                raise PermissionError('prediction recordings not in frozen held-FIT fold')
        return {key: archive[key] for key in archive.files}


def count_bin(n):
    if n < 3: raise ValueError('at least three real actors required')
    return 'N3_5' if n <= 5 else 'N6_8' if n <= 8 else 'N9_plus'


def fixed_selection(scene_ids, recordings, counts, per_bin):
    """H count/recording-only selection, stable to input row order."""
    if len(set(map(str, scene_ids))) != len(scene_ids) or per_bin < 1:
        raise ValueError('unique scenes and positive bin size required')
    by_bin = {}
    for group in ('N3_5', 'N6_8', 'N9_plus'):
        buckets = {}
        for i, n in enumerate(counts):
            if count_bin(int(n)) == group:
                buckets.setdefault(str(recordings[i]), []).append(i)
        for rows in buckets.values():
            rows.sort(key=lambda i: (digest(SALT+'|'+str(scene_ids[i])), str(scene_ids[i])))
        order = sorted(buckets, key=lambda rec: digest(SALT+'|'+group+'|'+rec))
        selected = []
        while len(selected) < per_bin:
            added = False
            for rec in order:
                if buckets[rec] and len(selected) < per_bin:
                    selected.append(buckets[rec].pop(0)); added = True
            if not added: raise ValueError('insufficient scenes in '+group)
        by_bin[group] = selected
    return [by_bin[group][j] for j in range(per_bin) for group in by_bin]


def state_hash(model):
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def grad_vector(model):
    return torch.cat([(p.grad.detach().double() if p.grad is not None else torch.zeros_like(p, dtype=torch.float64)).flatten()
                      for p in model.parameters()])


def summarize(rows):
    if not rows: return dict(requests=0)
    supported = [r for r in rows if r['usable_parameter_gradient']]
    return dict(requests=len(rows), geometry_supported=sum(r['geometry_supported'] for r in rows),
        CDF_smooth_supported=sum(r['CDF_smooth_supported'] for r in rows),
        usable_parameter_gradients=len(supported), usable_fraction=len(supported)/len(rows),
        all_request_estimated_p_MAE=float(np.mean([r['point_error'] for r in rows])),
        all_request_Fine_at_005=sum(r['point_error'] <= .05 for r in rows)/len(rows),
        all_request_interval_MAE=float(np.mean([r['interval_error'] for r in rows])),
        geometry_reasons=dict(Counter(r['geometry_reason'] for r in rows)),
        overlap_scenes=sum(r['all_pair_overlap_scene'] for r in rows),
        road_outside_scenes=sum(r['road_outside_scene'] for r in rows),
        negative_vx_scenes=sum(r['negative_vx_scene'] for r in rows),
        terminal_gradient_L2_median=float(np.median([r['terminal_gradient_L2'] for r in supported])) if supported else None,
        terminal_to_base_gradient_ratio_median=float(np.median([r['terminal_to_base_gradient_ratio'] for r in supported])) if supported else None,
        terminal_base_cosine_median=float(np.median([r['terminal_base_cosine'] for r in supported])) if supported else None,
        request_seconds_median=float(np.median([r['elapsed_seconds'] for r in rows])))


def run(output, per_bin=8, budget_seconds=180):
    started = time.monotonic()
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True); torch.manual_seed(20260912)
    output.mkdir(parents=True, exist_ok=False)
    manifest = read_json(MANIFEST); prepared = read_json(manifest['prepared'])
    g = fit_arrays(prepared['FIT_generator_pack'])
    h = fit_arrays(prepared['FIT_physical_pack']); labels = fit_arrays(prepared['FIT_p_labels'])
    for source in (h, labels):
        for key in ('scene_id', 'recording_id', 'role'):
            if not np.array_equal(g[key], source[key]): raise ValueError('FIT join mismatch: '+key)
    if not np.array_equal(g['agent_mask'], h['agent_mask']): raise ValueError('physical roster mismatch')
    counts = g['agent_mask'].sum(1)
    chosen = fixed_selection(g['scene_id'], g['recording_id'], counts, per_bin)
    selection = [dict(row=int(i), scene_id=str(g['scene_id'][i]), recording_id=str(g['recording_id'][i]),
                      num_agents=int(counts[i]), count_bin=count_bin(int(counts[i]))) for i in chosen]
    policy = dict(protocol='natural_terminal_support_no_update_v1', selected=selection,
        selection_salt=SALT, selection_uses_only_scene_id_recording_and_history_count=True,
        selection_uses_PET_p_or_gradient_support=False, requests=['natural_OOF_p','midpoint_0.5'],
        same_noise_between_requests=True, DDIM_steps=50, grad_last_steps=5, CFG_scale=4.,
        parent=PARENT, manifest=MANIFEST, source_prepared=manifest['prepared'],
        device='cpu', optimizer_steps=0, STOP_CAL_AUDIT_observations_accessed=False,
        budget_seconds=budget_seconds, count_recording_balanced_not_population_representative=True,
        base_gradient_diagnostic='natural clean v-MSE, natural OOF p, t=50, one fixed independent noise',
        code_sha256={name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('pcontrol/research/probe_natural_terminal_support.py', 'pcontrol/generation/training_ddim_suffix.py',
             'pcontrol/reference/torch_frozen_cdf.py', 'pcontrol/generation/terminal_percentile_loss.py')})
    write_new(output/'freeze_before_probe.json', policy)
    folds = {}
    for fold in sorted(set(int(labels['fold'][i]) for i in chosen)):
        source = manifest['oof_fold_sources'][str(fold)]
        if set(source['held_recordings']) & set(source['train_recordings']):
            raise ValueError('overlapping OOF train/held recordings')
        if not set(source['held_recordings']) <= set(map(str,g['recording_id'])):
            raise PermissionError('held predictions contain non-FIT recordings')
        prediction = fit_arrays(source['held_predictions'], held_recordings=source['held_recordings'])
        lookup = {(str(sid), str(rec)): j for j,(sid,rec) in enumerate(zip(prediction['scene_id'], prediction['recording_id']))}
        if len(lookup) != len(prediction['scene_id']): raise ValueError('duplicate OOF identity')
        folds[fold] = (prediction, lookup, SceneCDFWarp.from_dict(read_json(source['warp'])), source)
    checkpoint = torch.load(bound(PARENT), map_location='cpu')
    if checkpoint['recipe'] != 'tangent' or checkpoint['epoch'] != 40: raise ValueError('wrong parent')
    model = ClassifierFreePercentileDenoiser(**{k:checkpoint['architecture'][k] for k in
        ('coefficient_dim','hidden_dim','heads','layers','feedforward_dim')})
    model.load_state_dict(checkpoint['state_dict'], strict=True); model.requires_grad_(True); model.eval()
    before = state_hash(model)
    schedule = CosineDiffusionSchedule(100)
    if schedule.as_dict() != checkpoint['schedule']: raise ValueError('wrong schedule')
    normalizer = read_json(prepared['coefficient_normalizer']); rows = []
    with (output/'requests.jsonl').open('x') as log:
        for item in selection:
            if time.monotonic()-started > budget_seconds: break
            i = item['row']; mask = g['agent_mask'][i]; n = item['num_agents']; rm = g['road_boundary_mask'][i]
            anchors = g['anchors'][i,mask]
            if not np.array_equal(anchors,h['history'][i,-1,mask]): raise ValueError('anchor mismatch')
            if not g['ego_mask'][i,mask][0] or g['ego_mask'][i,mask].sum() != 1: raise ValueError('adapter assumes ego first')
            features = dict(history=torch.from_numpy(g['history'][i][:,mask][None]),
                dimensions=torch.from_numpy(g['dimensions'][i,mask][None]),
                road_boundaries=torch.from_numpy(g['road_boundaries'][i,rm][None]),
                road_boundary_mask=torch.ones((1,int(rm.sum())),dtype=torch.bool),
                ego_mask=torch.from_numpy(g['ego_mask'][i,mask][None]),agent_mask=torch.ones((1,n),dtype=torch.bool))
            fold = int(labels['fold'][i]); pred, lookup, warp, source = folds[fold]
            rec = item['recording_id']; j = lookup[(item['scene_id'],rec)]
            if rec not in source['held_recordings'] or rec in source['train_recordings']: raise ValueError('not own OOF')
            if int(pred['num_agents'][j]) != n or pred['pet_seconds'][j] != labels['pet_seconds'][i]: raise ValueError('OOF label mismatch')
            ref = FrozenTorchCDF(pred['joint_masses'][j:j+1], [n], warp)
            natural_p = float(labels['p_mid'][i])
            replay = ref.rank(torch.tensor([float(labels['pet_seconds'][i])],dtype=torch.float64))['p_mid'].item()
            if abs(replay-natural_p) > 1e-12: raise ValueError('OOF rank does not replay')
            seed = int(digest(SALT+'|noise|'+item['scene_id'])[:8],16)
            noise = torch.randn((1,n,8,2),generator=torch.Generator().manual_seed(seed))
            decoder = TorchTrajectoryDecoder(TrajectoryBasis(8),normalizer,anchors,device='cpu')
            dims = h['dimensions'][i,mask]; bounds = h['road_boundaries'][i,h['road_boundary_mask'][i]]
            geometry = GeometryOnlyAdapter(dims,anchors)
            model.zero_grad(set_to_none=True)
            base = cfg_training_loss(model,schedule,torch.from_numpy(g['coef_clean'][i,mask][None]),features,
                torch.tensor([natural_p],dtype=torch.float32),torch.ones(1,dtype=torch.bool),
                timesteps=torch.tensor([50]),noise=torch.randn(noise.shape,generator=torch.Generator().manual_seed(seed+1)))
            base['loss'].backward(); base_grad = grad_vector(model); base_norm = float(base_grad.norm())
            for request, requested in [('natural_OOF_p',natural_p),('midpoint_0.5',.5)]:
                if time.monotonic()-started > budget_seconds: break
                tick = time.monotonic(); model.zero_grad(set_to_none=True)
                p = torch.tensor([requested],dtype=torch.float32)
                out = training_ddim_suffix(model,schedule,features,p,noise,steps=50,grad_last_steps=5,cfg_scale=4.)
                future = decoder(out['sample'][0]); active = geometry.active_witness_pet(future)
                score = active.get('exact_score')
                if score is None: score = geometry.score_future(future)
                y = active['value'].reshape(1) if active['supported'] else torch.tensor([score['pet_seconds']],dtype=torch.float64)
                term = terminal_percentile_value_loss(y,p,ref,supported_geometry=torch.tensor([bool(active['supported'])]),graph_anchor=out['sample'])
                term['loss'].backward(); terminal_grad = grad_vector(model); norm = float(terminal_grad.norm())
                finite = bool(torch.isfinite(terminal_grad).all()); usable = term['supported_scenes']==1 and finite and norm>0
                data = future.detach().numpy(); a,b = np.triu_indices(n,1)
                overlap = np.all(np.abs(data[:,a,:2]-data[:,b,:2]) <= .5*(dims[a]+dims[b])[None],axis=-1)
                outside = (data[...,1]-dims[None,:,1]/2 < bounds[0]) | (data[...,1]+dims[None,:,1]/2 > bounds[-1])
                row = dict(item,request=request,requested_p_float32=float(p[0]),original_OOF_p=natural_p,
                    noise_seed=seed,fold=fold,ownfold_prediction_row=j,PET_seconds=float(score['pet_seconds']),
                    estimated_p_mid=float(term['estimated_rank']['p_mid'][0]),point_error=term['all_request_point_MAE'],
                    interval_error=term['all_request_interval_MAE'],geometry_supported=bool(active['supported']),
                    geometry_reason=active['reason'],CDF_smooth_supported=bool(ref.density(y)['valid'][0]),
                    usable_parameter_gradient=usable,terminal_gradient_L2=norm if finite else None,
                    terminal_gradient_all_finite=finite,base_v_MSE=float(base['loss'].detach()),base_gradient_L2=base_norm,
                    terminal_to_base_gradient_ratio=norm/max(base_norm,1e-30) if finite else None,
                    terminal_base_cosine=float(torch.dot(terminal_grad,base_grad)/max(norm*base_norm,1e-30)) if finite else None,
                    loss=float(term['loss'].detach()),all_pair_overlap_scene=bool(overlap.any()),
                    road_outside_scene=bool(outside.any()),negative_vx_scene=bool((data[...,2]<0).any()),
                    elapsed_seconds=time.monotonic()-tick)
                rows.append(row); log.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n'); log.flush()
                print(json.dumps(dict(completed_requests=len(rows),scene=item['scene_id'],request=request,
                    N=n,supported=usable,reason=active['reason'],elapsed_seconds=time.monotonic()-started)),flush=True)
    if before != state_hash(model): raise ValueError('probe changed model weights')
    result = dict(status='complete' if len(rows)==len(selection)*2 else 'time_budget_partial',
        expected_requests=len(selection)*2,completed_requests=len(rows),summary=summarize(rows),
        by_request={key:summarize([r for r in rows if r['request']==key]) for key in ('natural_OOF_p','midpoint_0.5')},
        by_count={key:summarize([r for r in rows if r['count_bin']==key]) for key in ('N3_5','N6_8','N9_plus')},
        by_recording={key:summarize([r for r in rows if r['recording_id']==key]) for key in sorted({r['recording_id'] for r in rows})},
        elapsed_seconds=time.monotonic()-started,parent_parameters_unchanged=True,parent_state_sha256=before,
        optimizer_steps=0,training_performed=False,performance_improvement_claimed=False,
        notes=['balanced diagnostic sample, not FIT population or blind test',
               'both scores and gradients use estimated own-fold reference, not known true CDF',
               'base gradient is one fixed natural v-MSE draw, not the full tangent training objective',
               'all-pair overlap is discrete footprint overlap, not verified collisions',
               'all completed requests retained; incomplete requests are not successes or failures'])
    write_new(output/'result.json', result)
    print(json.dumps(result,sort_keys=True),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--per-bin',type=int,default=8)
    parser.add_argument('--budget-seconds',type=float,default=180)
    args = parser.parse_args()
    run(args.output,args.per_bin,args.budget_seconds)
