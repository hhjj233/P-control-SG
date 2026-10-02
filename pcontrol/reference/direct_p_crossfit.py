"""Recording-out-of-fold natural estimated p labels for direct-P supervision.

FIT labels use fresh teachers that never trained/normalized on their recording.
Each teacher has its OWN fixed-family CAL-only warp. STOP uses the separately
frozen full-FIT reference. No actual future trajectory is decoded by this module;
only existing history/static observations and the capped scalar PET are needed.
These remain estimated ranks, not known true conditional percentiles.
"""
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from pcontrol.data.complete_scene_view import verify_binding, sha256, resolve, collate_reference_examples
from pcontrol.data.expanded_scene_view import ExpandedSceneReferenceView
from pcontrol.reference.scene_models import SceneCDFReference
from pcontrol.reference.scene_calibration import SceneCDFWarp, fit_scene_calibration
from pcontrol.reference.scores import crps_from_params
from pcontrol.plugins.risk_plugin import FrozenRiskPlugin
from pcontrol.research.train_natural_scene_reference import _batch


ROOT=Path(__file__).resolve().parents[2]
POLICY_SHA='c5d52262816e9e7b298d7108960735009359e4abb57e26adacf73b29ac7d3ab7'
PROTOCOL='natural_recording_OOF_direct_P_labels_v1'
PREPARE_PROTOCOL='natural_recording_OOF_direct_P_prepared_v1'
TEACHER_PROTOCOL='natural_recording_OOF_direct_P_teacher_v1'
LABEL_KEYS=frozenset(('scene_id','recording_id','role','pet_seconds','p_mid','p_low','p_up','atom_width','fold'))
FEATURE_KEYS=('history','dimensions','road_boundaries','road_boundary_mask','ego_mask','agent_mask')
CODE=('pcontrol/reference/direct_p_crossfit.py','pcontrol/research/prepare_natural_direct_p_labels.py',
      'pcontrol/reference/scene_models.py','pcontrol/reference/mixed_cdf.py',
      'pcontrol/reference/scores.py','pcontrol/reference/scene_calibration.py',
      'pcontrol/plugins/risk_plugin.py','pcontrol/data/complete_scene_view.py',
      'pcontrol/data/expanded_scene_view.py','pcontrol/research/train_natural_scene_reference.py')


def _json(binding):return json.loads(verify_binding(binding).read_text())


def _write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False)
    return dict(path=str(Path(path).resolve()),sha256=sha256(path))


def _write_npz(path,arrays):
    with Path(path).open('xb') as handle:np.savez_compressed(handle,**arrays)
    return dict(path=str(Path(path).resolve()),sha256=sha256(path))


def code_bindings():return {name:sha256(ROOT/name) for name in CODE}


def _verify_codes(expected):
    if expected!=code_bindings():raise ValueError('OOF producer/dependency code changed')


def read_policy(binding):
    if binding.get('sha256')!=POLICY_SHA:raise ValueError('only the frozen direct-P policy is authorized')
    policy=_json(binding);o=policy['oof']
    if (policy['protocol']!='natural_recording_OOF_direct_P_diffusion_pilot_v1'
            or o['folds']!=5 or o['architecture']!='M2' or o['base_epochs']!=20 or o['highN_epochs']!=7
            or o['device']!='cuda:0' or o['threads']!=2 or o['random_within_atom_labels'] is not False
            or o['FIT_relabelled_by_full_reference'] is not False or o['AUDIT_decoded'] is not False):
        raise ValueError('frozen recording-OOF protocol mismatch')
    return policy


def recording_folds(records,salt):
    if len(records)!=13 or len(set(records))!=13:raise ValueError('exactly13 distinct FIT recordings required')
    ordered=sorted(records,key=lambda r:hashlib.sha256((salt+'|'+r).encode()).hexdigest())
    return {str(fold):sorted(ordered[fold::5]) for fold in range(5)}


def _pack_physical(examples):
    batch=collate_reference_examples(examples)
    result=dict(batch['features'],target=batch['target'])
    for key in ('scene_id','recording_id','role'):result[key]=np.asarray([e['metadata'][key] for e in examples])
    return result


def _load_pack(binding,role):
    path=verify_binding(binding)
    with np.load(path,allow_pickle=False) as archive:
        declared=archive['role']
        if not np.all(declared==role):raise PermissionError('physical context cache role mismatch')
        expected=set(FEATURE_KEYS)|{'target','scene_id','recording_id','role'}
        if set(archive.files)!=expected:raise ValueError('unexpected physical cache fields')
        result={key:archive[key] for key in expected}
    verify_binding(binding,path)
    return result


def fit_fold_normalizer(pack,train_rows,held_records,fold):
    rows=np.asarray(train_rows,dtype=np.int64)
    if len(rows)==0 or not np.all(pack['role'][rows]=='FIT'):
        raise ValueError('fold normalizer requires nonempty FIT-only rows')
    records=sorted(set(pack['recording_id'][rows].tolist()))
    if set(records)&set(held_records):raise ValueError('held recording entered fold normalization')
    mask=pack['agent_mask'][rows]
    h=pack['history'][rows].transpose(0,2,1,3)[mask].reshape(-1,4)
    dims=pack['dimensions'][rows][mask]
    if not np.isfinite(h).all() or not np.isfinite(dims).all():raise ValueError('nonfinite fold history/static data')
    return dict(protocol='natural_direct_P_fold_FIT_scale_only_v1',fold=int(fold),
        training_recordings=records,excluded_held_recordings=sorted(held_records),
        fit_scene_count=len(rows),history_valid_state_count=len(h),dimension_actor_occurrences=len(dims),
        history_scale=np.maximum(h.std(0),1.).tolist(),dimension_scale=np.maximum(np.sqrt((dims**2).mean(0)),1.).tolist(),
        history_centering=False,road_scale_source='history_scale_y',targets_used=False,roles_used=['FIT'],
        future_or_CAL_or_AUDIT_used=False,
        FIT_scene_id_sha256=hashlib.sha256('\n'.join(pack['scene_id'][rows].tolist()).encode()).hexdigest())


def normalized_subset(pack,rows,normalizer):
    rows=np.asarray(rows,dtype=np.int64)
    out={key:values[rows].copy() for key,values in pack.items()}
    out['history']=(out['history']/np.asarray(normalizer['history_scale'])).astype(np.float32)
    out['dimensions']=(out['dimensions']/np.asarray(normalizer['dimension_scale'])).astype(np.float32)
    out['road_boundaries']=(out['road_boundaries']/normalizer['history_scale'][1]).astype(np.float32)
    return out


def _state_hash(model):
    digest=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):digest.update(key.encode());digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_fresh_teacher(train,config,*,device,epoch_callback=None):
    """Fresh model and fixed epoch schedules; no held/STOP/CAL arguments exist."""
    if not np.all(train['role']=='FIT'):raise PermissionError('teacher optimizer only accepts fold FIT')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    torch.manual_seed(config['seed'])
    model=SceneCDFReference('M2',torch.linspace(0.,4.,65,dtype=torch.float64),
        zero_atom_enabled=True,hidden_dim=config['hidden_dim'],heads=config['heads']).to(device)
    initial=_state_hash(model);trace=[]
    counts=train['agent_mask'].sum(1)
    raw=1.+(counts>=config['highN_threshold'])*(config['highN_weight_multiplier']-1.)
    high_weights=raw/raw.mean()
    for stage,epochs,lr in (('base',config['base_epochs'],config['base_learning_rate']),
                            ('highN',config['highN_epochs'],config['highN_learning_rate'])):
        optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=config['weight_decay'])
        rng=np.random.default_rng(config['seed'])
        weights=np.ones(len(counts)) if stage=='base' else high_weights
        for epoch in range(1,epochs+1):
            model.train();order=rng.permutation(len(counts));total=0.
            for start in range(0,len(order),config['batch_size']):
                idx=order[start:start+config['batch_size']];features,target=_batch(train,idx)
                features={key:value.to(device) for key,value in features.items()};target=target.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss=(crps_from_params(model(features),target,normalized=True)*torch.as_tensor(weights[idx],device=device)).mean()
                if not bool(torch.isfinite(loss)):raise FloatingPointError('nonfinite fold teacher CRPS')
                loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip_norm'])
                if not bool(torch.isfinite(norm)):raise FloatingPointError('nonfinite teacher gradients')
                optimizer.step();total+=float(loss.detach())*len(idx)
            row=dict(stage=stage,epoch=epoch,training_objective_normalized_CRPS=total/len(counts),rows=len(counts))
            trace.append(row)
            if epoch_callback is not None:epoch_callback(row)
    model.eval();model.requires_grad_(False)
    return model,dict(initial_state_sha256=initial,final_state_sha256=_state_hash(model),epochs=trace,
                     initialization='fresh_random_no_checkpoint_loaded',held_or_STOP_selection=False,
                     precision=dict(encoder='float32',CDF_and_CRPS='float64',cuda_matmul_TF32=False,
                                    cudnn_TF32=False,cudnn_benchmark=False,automatic_mixed_precision=False),
                     highN_weight_min=float(high_weights.min()),highN_weight_max=float(high_weights.max()))


def predict_masses(model,pack,batch_size=128):
    device=next(model.parameters()).device;values=[]
    with torch.inference_mode():
        for start in range(0,len(pack['target']),batch_size):
            features,_target=_batch(pack,slice(start,start+batch_size))
            values.append(model({k:v.to(device) for k,v in features.items()}).joint_masses.cpu().numpy())
    return np.concatenate(values)


def rank_labels(masses,counts,pet,identity,warp,fold):
    rank=warp.rank(masses,counts,pet)
    arrays={key:np.asarray(identity[key]).copy() for key in ('scene_id','recording_id','role')}
    arrays.update(pet_seconds=np.asarray(pet,dtype=np.float64),
        p_low=np.asarray(rank['p_low'],dtype=np.float64),p_up=np.asarray(rank['p_up'],dtype=np.float64),
        p_mid=np.asarray(rank['p_mid'],dtype=np.float64),
        atom_width=np.asarray(rank['p_up']-rank['p_low'],dtype=np.float64),fold=np.full(len(pet),fold,np.int64))
    _validate_arrays(arrays)
    return arrays


def _validate_arrays(arrays):
    if set(arrays)!=LABEL_KEYS:raise ValueError('label array schema mismatch')
    n=len(arrays['scene_id'])
    if any(np.shape(v)!=(n,) for v in arrays.values()) or len(set(arrays['scene_id'].tolist()))!=n:
        raise ValueError('label identity/shape/uniqueness mismatch')
    for key in ('pet_seconds','p_low','p_up','p_mid','atom_width'):
        if arrays[key].dtype!=np.float64 or not np.isfinite(arrays[key]).all():raise ValueError('labels require finite float64')
    lo,up,mid=arrays['p_low'],arrays['p_up'],arrays['p_mid']
    if (np.any(lo<0) or np.any(up>1) or np.any(up<lo) or np.any(arrays['pet_seconds']<0)
            or np.any(arrays['pet_seconds']>4) or not np.allclose(mid,.5*(lo+up),rtol=0.,atol=2e-15)
            or np.any(mid<lo-2e-15) or np.any(mid>up+2e-15)
            or not np.array_equal(arrays['atom_width'],up-lo)):
        raise ValueError('rank interval/midpoint/PET contract changed')


def validate_label_manifest(binding):
    """Authenticate JSON/hash lineage only; never decode CAL/AUDIT observations."""
    manifest=_json(binding)
    if manifest.get('protocol')!=PROTOCOL or manifest.get('status')!='complete':raise ValueError('incomplete OOF label manifest')
    policy=read_policy(manifest['policy']);prepared=_json(manifest['prepared'])
    _verify_codes(manifest['code_sha256'])
    if prepared['code_sha256']!=manifest['code_sha256'] or prepared['policy']!=manifest['policy']:
        raise ValueError('prepared/source policy mismatch')
    fit=set(prepared['role_recordings']['FIT']);cal=set(prepared['role_recordings']['CAL'])
    stop=set(prepared['role_recordings']['STOP'])
    folds=policy['oof']['heldout_recordings']
    if (folds!=recording_folds(sorted(fit),policy['oof']['fold_salt']) or fit&cal or fit&stop or cal&stop
            or set(manifest['fold_artifacts'])!=set(folds) or set(manifest['roles'])!={'FIT','STOP'}
            or manifest.get('random_within_atom_labels') is not False or manifest.get('AUDIT_decoded') is not False):
        raise ValueError('OOF roles/folds/label semantics changed')
    initials=[]
    for fold,held in folds.items():
        result=_json(manifest['fold_artifacts'][fold]);header=_json(result['checkpoint_header'])
        norm=_json(result['normalizer']);cal_report=_json(result['calibration_report'])
        if (result.get('protocol')!=TEACHER_PROTOCOL or result.get('status')!='complete'
                or result['fold']!=int(fold) or result['policy']!=manifest['policy'] or result['prepared']!=manifest['prepared']
                or result['code_sha256']!=manifest['code_sha256']
                or result['held_recordings']!=held or set(result['train_recordings'])!=fit-set(held)
                or header['held_recordings']!=held or header['train_recordings']!=result['train_recordings']
                or header['initialization']!='fresh_random_no_checkpoint_loaded'
                or header['normalizer']!=result['normalizer'] or header['checkpoint']!=result['checkpoint']
                or header['fixed_base_epochs']!=20 or header['fixed_highN_epochs']!=7
                or header['held_or_STOP_model_selection'] is not False
                or norm['training_recordings']!=result['train_recordings'] or norm['excluded_held_recordings']!=held
                or norm['targets_used'] is not False or norm['future_or_CAL_or_AUDIT_used'] is not False
                or set(cal_report['calibration_recordings'])!=cal or cal_report['role']!='CAL'
                or cal_report['family']!='count' or cal_report['ridge']!=.1 or cal_report['rows']!=504
                or cal_report['fit_report']['success'] is not True or cal_report['fullFIT_warp_reused'] is not False):
            raise ValueError('teacher held-record exclusion/fresh normalization/CAL provenance failed')
        for key in ('checkpoint','labels','warp','training_trace'):verify_binding(result[key])
        warp=SceneCDFWarp.from_dict(_json(result['warp']))
        if warp.family!='count' or cal_report['fit_report']['final_model']!=warp.as_dict():raise ValueError('fold warp differs from fitted CAL model')
        initials.append(header['initial_state_sha256'])
    if len(set(initials))!=1:raise ValueError('same-seed same-architecture fresh initial states differ')
    for role,count in (('FIT',9913),('STOP',538)):
        if manifest['roles'][role]['rows']!=count:raise ValueError('label role denominator changed')
        verify_binding(manifest['roles'][role]['artifact'])
    if (manifest['roles']['FIT']['label_source']!='recording_OOF_teacher_with_own_CAL_warp'
            or manifest['roles']['STOP']['label_source']!='frozen_fullFIT_reference'
            or manifest['full_reference']!=policy['risk_plugin_result']):
        raise ValueError('FIT or STOP label source was replaced')
    return manifest


def load_role_labels(binding,role):
    """Decode only one requested FIT/STOP label file, joining is caller by ID."""
    if role not in ('FIT','STOP'):raise PermissionError('generator labels are FIT or STOP only')
    manifest=validate_label_manifest(binding);prepared=_json(manifest['prepared'])
    artifact=manifest['roles'][role]['artifact'];path=verify_binding(artifact)
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=LABEL_KEYS:raise ValueError('unexpected generator-label fields')
        if not np.all(archive['role']==role):raise PermissionError('role checked before label values')
        arrays={key:archive[key] for key in LABEL_KEYS}
    verify_binding(artifact,path);_validate_arrays(arrays)
    expected=prepared['role_identities'][role]
    observed=set(zip(arrays['scene_id'].tolist(),arrays['recording_id'].tolist()))
    if observed!=set(zip(expected['scene_id'],expected['recording_id'])):
        raise ValueError('role label coverage differs from all original identities')
    if role=='FIT':
        policy=read_policy(manifest['policy'])
        mapping={rec:int(fold) for fold,records in policy['oof']['heldout_recordings'].items() for rec in records}
        if not np.array_equal(arrays['fold'],[mapping[r] for r in arrays['recording_id']]):raise ValueError('FIT row was labelled by a teacher that saw its recording')
    elif not np.all(arrays['fold']==-1):raise ValueError('STOP must use only the frozen final reference')
    return arrays


def prepare(policy_binding):
    policy_binding=dict(path=str(resolve(policy_binding['path'])),sha256=policy_binding['sha256'])
    policy=read_policy(policy_binding);output=resolve(policy['oof']['output_root'])
    if output.exists():raise FileExistsError('never overwrite an OOF label run')
    code=code_bindings();packs={};identities={};records={};counts={}
    physical={}
    for role,purpose,expected in (('FIT','fit',9913),('STOP','stop',538),('CAL','calibrate',504)):
        view=ExpandedSceneReferenceView(policy['data_manifest']['path'],policy['data_manifest']['sha256'],purpose=purpose)
        examples=[view[i] for i in range(len(view))]
        if len(examples)!=expected or any(e['metadata']['role']!=role for e in examples):raise ValueError('immutable reference-role denominator changed')
        physical[role]=_pack_physical(examples)
        identities[role]={key:physical[role][key].tolist() for key in ('scene_id','recording_id')}
        records[role]=sorted(set(identities[role]['recording_id']));counts[role]=expected
    if (set(records['FIT'])&set(records['CAL']) or set(records['FIT'])&set(records['STOP'])
            or set(records['STOP'])&set(records['CAL'])
            or recording_folds(records['FIT'],policy['oof']['fold_salt'])!=policy['oof']['heldout_recordings']):
        raise ValueError('frozen fold/role separation mismatch')
    _verify_codes(code);output.mkdir(parents=True,exist_ok=False)
    for role,pack in physical.items():packs[role]=_write_npz(output/(role+'_physical_reference.npz'),pack)
    result=dict(protocol=PREPARE_PROTOCOL,status='prepared',policy=policy_binding,source_data=policy['data_manifest'],
        code_sha256=code,physical_packs=packs,role_identities=identities,role_recordings=records,counts=counts,
        folds=policy['oof']['heldout_recordings'],AUDIT_decoded=False,native_future_trajectories_decoded=False,
        native_future_visibility_masks_used_only_for_existing_E_full_membership=True,
        data_fields='observed_history_static_geometry_capped_PET_identity_only',generator_data_modified=False)
    return _write_json(output/'prepared.json',result)


def _read_prepared(binding):
    prepared=_json(binding)
    if prepared.get('protocol')!=PREPARE_PROTOCOL or prepared.get('status')!='prepared':raise ValueError('wrong OOF preparation')
    policy=read_policy(prepared['policy']);_verify_codes(prepared['code_sha256'])
    return prepared,policy


def run_fold(prepared_binding,fold):
    prepared_binding=dict(path=str(resolve(prepared_binding['path'])),sha256=prepared_binding['sha256'])
    prepared,policy=_read_prepared(prepared_binding);cfg=policy['oof']
    if type(fold) is not int or str(fold) not in cfg['heldout_recordings']:raise ValueError('fold must be0..4')
    output=resolve(cfg['output_root'])/('fold_'+str(fold));output.mkdir(exist_ok=False)
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    torch.set_num_threads(cfg['threads'])
    if torch.get_num_interop_threads()!=1:torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():raise RuntimeError('frozen teacher policy requires explicit cuda:0; no fallback')
    started=time.perf_counter();fit=_load_pack(prepared['physical_packs']['FIT'],'FIT')
    held=cfg['heldout_recordings'][str(fold)];is_held=np.isin(fit['recording_id'],held)
    train_rows,held_rows=np.flatnonzero(~is_held),np.flatnonzero(is_held)
    if not len(train_rows) or not len(held_rows):raise ValueError('empty fold train/held population')
    normalizer=fit_fold_normalizer(fit,train_rows,held,fold)
    norm_binding=_write_json(output/'normalizer.json',normalizer)
    train=normalized_subset(fit,train_rows,normalizer)
    with (output/'epochs.jsonl').open('x',encoding='utf-8') as logfile:
        def progress(row):
            logfile.write(json.dumps(row,sort_keys=True)+'\n');logfile.flush()
            print(json.dumps(dict(fold=fold,**row)),flush=True)
        model,training=train_fresh_teacher(train,cfg,device=cfg['device'],epoch_callback=progress)
    training_binding=_write_json(output/'training_report.json',training)
    header=dict(protocol=TEACHER_PROTOCOL,fold=fold,policy=prepared['policy'],prepared=prepared_binding,
        train_recordings=normalizer['training_recordings'],held_recordings=held,
        training_rows=len(train_rows),held_rows=len(held_rows),normalizer=norm_binding,
        initialization=training['initialization'],initial_state_sha256=training['initial_state_sha256'],
        seed=cfg['seed'],fixed_base_epochs=cfg['base_epochs'],fixed_highN_epochs=cfg['highN_epochs'],
        held_or_STOP_model_selection=False,teacher_frozen_before_CAL_fit=True,
        precision=training['precision'],
        code_sha256=prepared['code_sha256'],architecture=model.architecture_config())
    checkpoint_path=output/'teacher.pt'
    with checkpoint_path.open('xb') as handle:
        torch.save(dict(header=header,state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),handle)
    checkpoint=dict(path=str(checkpoint_path),sha256=sha256(checkpoint_path))
    header_binding=_write_json(output/'checkpoint_header.json',dict(header,checkpoint=checkpoint))
    # CAL is first opened here, after the complete fixed-budget base is frozen.
    cal_raw=_load_pack(prepared['physical_packs']['CAL'],'CAL')
    cal=normalized_subset(cal_raw,np.arange(len(cal_raw['target'])),normalizer)
    cal_mass=predict_masses(model,cal,cfg['batch_size']);cal_counts=cal['agent_mask'].sum(1)
    cal_predictions=_write_npz(output/'CAL_predictions.npz',dict(joint_masses=cal_mass,num_agents=cal_counts,
        pet_seconds=cal['target'],scene_id=cal['scene_id'],recording_id=cal['recording_id'],role=cal['role']))
    calibrated=fit_scene_calibration(cal_mass,cal_counts,cal['target'],family='count',ridge=.1)
    if not calibrated.report['success']:raise RuntimeError('fold CAL optimizer failed; do not label held data')
    warp=calibrated.warp;warp_binding=_write_json(output/'CAL_warp.json',warp.as_dict())
    calibration_binding=_write_json(output/'calibration_report.json',dict(role='CAL',family='count',ridge=.1,
        rows=len(cal['target']),calibration_recordings=prepared['role_recordings']['CAL'],
        checkpoint=checkpoint,normalizer=norm_binding,predictions=cal_predictions,fullFIT_warp_reused=False,
        family_or_ridge_reselected=False,fit_report=calibrated.report))
    held_pack=normalized_subset(fit,held_rows,normalizer)
    held_mass=predict_masses(model,held_pack,cfg['batch_size']);held_counts=held_pack['agent_mask'].sum(1)
    held_labels=rank_labels(held_mass,held_counts,held_pack['target'],held_pack,warp,fold)
    labels_binding=_write_npz(output/'held_FIT_labels.npz',held_labels)
    held_predictions=_write_npz(output/'held_FIT_predictions.npz',dict(joint_masses=held_mass,num_agents=held_counts,
        pet_seconds=held_pack['target'],scene_id=held_pack['scene_id'],recording_id=held_pack['recording_id']))
    # STOP only supplies a descriptive coordinate comparison, never any selection.
    stop_raw=_load_pack(prepared['physical_packs']['STOP'],'STOP')
    stop=normalized_subset(stop_raw,np.arange(len(stop_raw['target'])),normalizer)
    stop_mass=predict_masses(model,stop,cfg['batch_size']);stop_n=stop['agent_mask'].sum(1)
    stop_labels=rank_labels(stop_mass,stop_n,stop['target'],stop,warp,fold)
    stop_diagnostic=_write_npz(output/'STOP_teacher_diagnostic.npz',dict(stop_labels,
        quantiles=warp.quantile(stop_mass,stop_n,np.array([[.1,.3,.5,.7,.9]])),
        cap_mass=1.-warp.cdf(stop_mass,stop_n,4.,side='left')))
    _verify_codes(prepared['code_sha256'])
    result=dict(protocol=TEACHER_PROTOCOL,status='complete',fold=fold,policy=prepared['policy'],prepared=prepared_binding,
        code_sha256=prepared['code_sha256'],train_recordings=header['train_recordings'],held_recordings=held,
        train_rows=len(train_rows),held_rows=len(held_rows),normalizer=norm_binding,checkpoint=checkpoint,
        checkpoint_header=header_binding,training_trace=training_binding,calibration_report=calibration_binding,
        calibration_predictions=cal_predictions,warp=warp_binding,labels=labels_binding,held_predictions=held_predictions,
        STOP_diagnostic=stop_diagnostic,STOP_diagnostic_used_for_selection=False,AUDIT_decoded=False,
        held_labels_used_for_training_or_calibration=False,fullFIT_weights_or_warp_reused=False,
        wall_seconds=time.perf_counter()-started,device=cfg['device'])
    binding=_write_json(output/'result.json',result)
    print(json.dumps(dict(fold_complete=fold,result=binding,held_rows=len(held_rows))),flush=True)
    return binding


def _load_arrays(binding):
    path=verify_binding(binding)
    with np.load(path,allow_pickle=False) as archive:result={key:archive[key] for key in archive.files}
    verify_binding(binding,path);return result


def aggregate(prepared_binding):
    prepared_binding=dict(path=str(resolve(prepared_binding['path'])),sha256=prepared_binding['sha256'])
    prepared,policy=_read_prepared(prepared_binding);output=resolve(policy['oof']['output_root'])
    if (output/'manifest.json').exists():raise FileExistsError('never replace frozen direct-P labels')
    artifacts={};results={}
    # All five completed teachers must exist before any final label aggregation.
    for fold in range(5):
        path=output/('fold_'+str(fold))/'result.json'
        binding=dict(path=str(path),sha256=sha256(path));result=_json(binding)
        if (result.get('status')!='complete' or result.get('protocol')!=TEACHER_PROTOCOL
                or result['fold']!=fold or result['prepared']!=prepared_binding
                or result['code_sha256']!=prepared['code_sha256']):raise ValueError('all-five fold completion barrier failed')
        artifacts[str(fold)]=binding;results[str(fold)]=result
    parts=[_load_arrays(results[str(fold)]['labels']) for fold in range(5)]
    for part in parts:_validate_arrays(part)
    joined={key:np.concatenate([part[key] for part in parts]) for key in LABEL_KEYS}
    _validate_arrays(joined)
    lookup={str(sid):i for i,sid in enumerate(joined['scene_id'])}
    expected=prepared['role_identities']['FIT']['scene_id']
    if set(lookup)!=set(expected):raise ValueError('held-FIT labels do not cover every source scene exactly once')
    order=np.array([lookup[sid] for sid in expected]);joined={key:value[order] for key,value in joined.items()}
    fit_binding=_write_npz(output/'FIT_labels.npz',joined)
    stop=_load_pack(prepared['physical_packs']['STOP'],'STOP')
    plugin=FrozenRiskPlugin.from_refinement_binding(policy['risk_plugin_result'],device='cpu')
    lo=[];up=[];mid=[];quantiles=[];caps=[]
    for row in range(len(stop['target'])):
        mask=stop['agent_mask'][row];roadmask=stop['road_boundary_mask'][row]
        reference=plugin.condition(stop['history'][row],stop['dimensions'][row],
            stop['road_boundaries'][row,roadmask],stop['ego_mask'][row],mask)
        rank=reference.rank(stop['target'][row]);lo.append(rank['p_low']);up.append(rank['p_up']);mid.append(rank['p_mid'])
        quantiles.append(reference.quantile(np.array([.1,.3,.5,.7,.9])));caps.append(1.-reference.cdf(4.,side='left'))
    stop_labels={key:stop[key].copy() for key in ('scene_id','recording_id','role')}
    stop_labels.update(pet_seconds=stop['target'].copy(),p_low=np.asarray(lo),p_up=np.asarray(up),p_mid=np.asarray(mid),
                       atom_width=np.asarray(up)-np.asarray(lo),fold=np.full(len(lo),-1,np.int64))
    _validate_arrays(stop_labels);stop_binding=_write_npz(output/'STOP_labels.npz',stop_labels)
    disagreements={}
    for fold,result in results.items():
        teacher=_load_arrays(result['STOP_diagnostic'])
        if not np.array_equal(teacher['scene_id'],stop_labels['scene_id']):raise ValueError('STOP diagnostic identity mismatch')
        diff=np.abs(teacher['p_mid']-stop_labels['p_mid'])
        qdiff=np.abs(teacher['quantiles']-np.asarray(quantiles));cdiff=np.abs(teacher['cap_mass']-np.asarray(caps))
        disagreements[fold]=dict(rows=len(diff),p_mid_absolute_difference_mean=float(diff.mean()),
            p_mid_absolute_difference_p95=float(np.quantile(diff,.95)),p_mid_absolute_difference_max=float(diff.max()),
            quantile_levels=[.1,.3,.5,.7,.9],quantile_absolute_difference_mean_seconds=qdiff.mean(0).tolist(),
            cap_mass_absolute_difference_mean=float(cdiff.mean()),used_for_teacher_or_hyperparameter_selection=False)
    disagreement_binding=_write_json(output/'STOP_reference_disagreement.json',dict(protocol=PROTOCOL,
        full_reference=policy['risk_plugin_result'],folds=disagreements,descriptive_only=True))
    manifest=dict(protocol=PROTOCOL,status='complete',policy=prepared['policy'],prepared=prepared_binding,
        code_sha256=prepared['code_sha256'],source_data=policy['data_manifest'],fold_artifacts=artifacts,
        fold_assignment=policy['oof']['heldout_recordings'],full_reference=policy['risk_plugin_result'],
        roles=dict(FIT=dict(artifact=fit_binding,rows=len(joined['scene_id']),label_source='recording_OOF_teacher_with_own_CAL_warp'),
                   STOP=dict(artifact=stop_binding,rows=len(stop_labels['scene_id']),label_source='frozen_fullFIT_reference')),
        p_definition='midpoint_of_estimated_adversity_rank_interval_no_atom_randomization',
        formula='p_mid=1-0.5*(F_left(Y)+F_right(Y))',random_within_atom_labels=False,
        teacher_and_normalizer_recording_exclusion=True,CAL_used_only_for_teacher_warps=True,
        CAL_generator_examples=False,AUDIT_decoded=False,native_future_trajectories_decoded=False,
        fullFIT_reference_used_for_FIT_labels=False,generator_pack_modified=False,
        known_true_percentiles=False,STOP_disagreement=disagreement_binding)
    binding=_write_json(output/'manifest.json',manifest)
    validate_label_manifest(binding)
    load_role_labels(binding,'FIT');load_role_labels(binding,'STOP')
    return binding
