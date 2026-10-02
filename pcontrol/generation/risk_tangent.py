"""Natural FIT-only PET/CDF tangents for optional direct-P training losses.

g = d PET / d normalized coefficients. h = f_OOF(Y)*g = d CDF / d coefficients;
the ADVERSITY-RANK tangent is -h because p=1-F(Y). Invalid geometry/density has
explicit independent masks. No p label is changed, no CDF network is loaded,
and no STOP/AUDIT/CAL observations enter the cache.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from pcontrol.data.complete_scene_view import verify_binding,resolve,sha256
from pcontrol.generation.data import load_training_pack
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.risk_guidance import TorchTrajectoryDecoder
from pcontrol.reference.scene_calibration import SceneCDFWarp
from pcontrol.research.probe_natural_risk_directions import GeometryOnlyAdapter


ROOT=Path(__file__).resolve().parents[2]
PROTOCOL='natural_FIT_subset_OOF_risk_tangent_cache_v1'
DATA_SHA='b10d412b54cf6a94c82639362515b4fff62305d8e448c5c6dbcd3eaa00fad408'
LABEL_SHA='6c8281879edcf6331b2a57f26dc6df04ee123df14510f2c18fcee873a30eee87'
OOF_PREPARED_SHA='b65c5afc88c4ab97310dd6c1b1078e1abd0a339fc6e52fa67eeebda95b2bfed6'
POLICY_SHA='237ec4f3d1645d0dda94810822ab9b9facea59146fa760e32732f5f89eb1c342'
SUBSET_SALT='natural_direct_p_tangent_FIT_v1'
CODE=('pcontrol/generation/risk_tangent.py','pcontrol/research/prepare_natural_risk_tangent.py',
      'pcontrol/research/probe_natural_risk_directions.py','pcontrol/plugins/risk_plugin.py',
      'pcontrol/data/scene_pet.py','pcontrol/generation/risk_guidance.py',
      'pcontrol/generation/trajectory_basis.py','pcontrol/generation/data.py',
      'pcontrol/reference/scene_calibration.py','pcontrol/data/complete_scene_view.py')


def _json(binding):return json.loads(verify_binding(binding).read_text())


def _write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False)
    return dict(path=str(Path(path).resolve()),sha256=sha256(path))


def _write_npz(path,value):
    with Path(path).open('xb') as handle:np.savez_compressed(handle,**value)
    return dict(path=str(Path(path).resolve()),sha256=sha256(path))


def _npz(binding,keys=None,role=None):
    path=verify_binding(binding)
    with np.load(path,allow_pickle=False) as archive:
        if role is not None and not np.all(archive['role']==role):raise PermissionError('non-FIT cache input refused before numeric decode')
        arrays={key:archive[key] for key in (archive.files if keys is None else keys)}
    verify_binding(binding,path)
    return arrays


def code_bindings():return {path:sha256(ROOT/path) for path in CODE}


def _verify_codes(expected):
    if code_bindings()!=expected:raise ValueError('risk tangent code changed after preparation')


def select_subset(scene_ids,recordings,roles,*,size=2048,shards=8,salt=SUBSET_SALT):
    ids=np.asarray(scene_ids).astype(str);rec=np.asarray(recordings).astype(str);roles=np.asarray(roles).astype(str)
    if (ids.ndim!=1 or rec.shape!=ids.shape or roles.shape!=ids.shape or not np.all(roles=='FIT')
            or len(set(ids))!=len(ids) or type(size) is not int or not 1<=size<=len(ids)
            or type(shards) is not int or shards<1 or not isinstance(salt,str) or not salt):
        raise ValueError('unique FIT identities and a bounded fixed subset required')
    order=sorted(range(len(ids)),key=lambda i:(hashlib.sha256((salt+'|'+ids[i]).encode()).hexdigest(),ids[i]))[:size]
    return [dict(source_index=int(index),scene_id=str(ids[index]),recording_id=str(rec[index]),role='FIT',
                 subset_rank=rank,shard=rank%shards,
                 selection_hash=hashlib.sha256((salt+'|'+ids[index]).encode()).hexdigest()) for rank,index in enumerate(order)]


def oof_density(warp,masses,count,original_pet,decoded_pet,*,epsilon=1e-6,
                relative_tolerance=.01,minimum_density=1e-6,cdf_difference_limit=.05):
    """Two-sided secants of the SAME cached held-FIT OOF CDF, not a new teacher.

    A density-invalid value is diagnostic only; masking it must not disable a
    valid geometry-projection loss. h remains density*g, but Jacobian loss must
    intersect valid_geom AND valid_density and the trainer's p-present mask.
    """
    y,d=float(original_pet),float(decoded_pet)
    if not np.isfinite([y,d]).all() or not 0<=y<=4 or not 0<=d<=4:
        raise ValueError('finite original/decoded capped PET in[0,4] required')
    m=np.asarray(masses,dtype=np.float64).reshape(1,-1)
    center=float(warp.cdf(m,[count],y)[0]);decoded=float(warp.cdf(m,[count],d)[0])
    difference=abs(decoded-center)
    answer=dict(density=0.,density_left=0.,density_right=0.,density_relative_difference=0.,
        cdf_reconstruction_difference=difference,valid_density=False,
        density_reason='original_atom_or_endpoint_neighborhood')
    if not epsilon<y<4.-epsilon:return answer
    left=float(warp.cdf(m,[count],y-epsilon)[0]);right=float(warp.cdf(m,[count],y+epsilon)[0])
    fl,fr=(center-left)/epsilon,(right-center)/epsilon
    density=.5*(fl+fr);relative=abs(fl-fr)/max(abs(fl),abs(fr),1e-12)
    if not np.isfinite([fl,fr,density,relative]).all():raise FloatingPointError('nonfinite OOF density diagnostic')
    answer.update(density=density,density_left=fl,density_right=fr,density_relative_difference=relative)
    if min(fl,fr)<0:reason='nonmonotone_numeric_secant'
    elif density<=minimum_density:reason='density_too_small'
    elif relative>relative_tolerance:reason='CDF_density_kink'
    elif difference>cdf_difference_limit:reason='CDF_reconstruction_mismatch'
    else:reason='valid_smooth_aligned_OOF_density'
    answer.update(valid_density=reason=='valid_smooth_aligned_OOF_density',density_reason=reason)
    return answer


def geometry_tangent(coefficient,anchors,dimensions,normalizer,original_pet,*,pet_tolerance=.01):
    c=np.asarray(coefficient,dtype=np.float64);n,k,_=c.shape
    decoder=TorchTrajectoryDecoder(TrajectoryBasis(k),normalizer,anchors,device='cpu')
    adapter=GeometryOnlyAdapter(dimensions,anchors)
    variable=torch.tensor(c,dtype=torch.float64,requires_grad=True);future=decoder(variable)
    answer=dict(g=np.zeros_like(c),unit_g=np.zeros_like(c),valid_geom=False,
                reason='original_PET_atom',originalPET=float(original_pet),decodedPET=0.,gradient_norm=0.)
    if not np.isfinite(original_pet) or not 0<=original_pet<=4:raise ValueError('invalid original natural PET')
    if original_pet in (0.,4.):
        answer['decodedPET']=float(adapter.score_future(future)['pet_seconds']);return answer
    active=adapter.active_witness_pet(future)
    score=active.get('exact_score')
    if score is None:score=adapter.score_future(future)
    decoded=float(score['pet_seconds']);answer.update(decodedPET=decoded,reason=active['reason'])
    if abs(decoded-original_pet)>pet_tolerance:
        answer['reason']='PET_reconstruction_mismatch';return answer
    if not active['supported']:return answer
    g=torch.autograd.grad(active['value'],variable)[0].detach().numpy();norm=float(np.linalg.norm(g))
    if not np.isfinite(g).all() or norm<=1e-12:
        answer['reason']='flat_or_nonfinite_coefficient_gradient';return answer
    answer.update(g=g,unit_g=g/norm,valid_geom=True,gradient_norm=norm,
                  reason='valid_natural_reconstruction_geometry')
    return answer


def _same(a,b):return a['sha256']==b['sha256'] and resolve(a['path'])==resolve(b['path'])


def read_policy(binding):
    if binding.get('sha256')!=POLICY_SHA:raise ValueError('only the frozen risk-tangent policy allowed')
    policy=_json(binding);c=policy['cache']
    if (policy['protocol']!='natural_direct_P_risk_tangent_training_v1'
            or policy['generator_data']['sha256']!=DATA_SHA or policy['labels_manifest']['sha256']!=LABEL_SHA
            or policy['physical_prepared']['sha256']!=OOF_PREPARED_SHA
            or c['role']!='FIT' or c['selected_scenes']!=2048 or c['shards']!=8 or c['hash_salt']!=SUBSET_SALT
            or c['risk_rank_tangent_sign']!=-1 or c['new_p_labels_created'] is not False
            or c['CDF_network_loaded'] is not False or c['STOP_CAL_AUDIT_observations_decoded'] is not False):
        raise ValueError('risk-tangent cache scope mismatch')
    return policy


def _identity_map(arrays):
    fields=('scene_id','recording_id','role')
    keys=list(zip(*(np.asarray(arrays[name]).astype(str).tolist() for name in fields)))
    if len(set(keys))!=len(keys) or len(set(arrays['scene_id'].astype(str)))!=len(keys):
        raise ValueError('duplicate scene identity')
    return {key:index for index,key in enumerate(keys)}


def _load_sources(policy):
    gen_meta=_json(policy['generator_data']);labels_meta=_json(policy['labels_manifest']);prepared=_json(policy['physical_prepared'])
    if (gen_meta['counts']['FIT']!=9913 or labels_meta.get('status')!='complete'
            or labels_meta['roles']['FIT']['rows']!=9913
            or labels_meta['roles']['FIT']['label_source']!='recording_OOF_teacher_with_own_CAL_warp'
            or not _same(labels_meta['prepared'],policy['physical_prepared'])
            or not _same(prepared['source_data'],gen_meta['dataset_manifest'])):
        raise ValueError('generator/OOF/physical source populations differ')
    pack=load_training_pack(gen_meta['packs']['FIT'],role='FIT')
    labels=_npz(labels_meta['roles']['FIT']['artifact'],role='FIT')
    physical=_npz(prepared['physical_packs']['FIT'],
        keys=('scene_id','recording_id','role','history','dimensions','agent_mask','ego_mask','target'),role='FIT')
    lookup=_identity_map(pack);lm=_identity_map(labels);pm=_identity_map(physical)
    if len(lookup)!=9913 or set(lookup)!=set(lm) or set(lookup)!=set(pm):
        raise ValueError('all9913 FIT identities must match exactly across sources')
    ordered=sorted(lookup,key=lookup.get)
    labels={key:value[[lm[k] for k in ordered]] for key,value in labels.items()}
    physical={key:value[[pm[k] for k in ordered]] for key,value in physical.items()}
    if (not np.array_equal(labels['pet_seconds'],physical['target'])
            or not np.array_equal(pack['agent_mask'],physical['agent_mask'])
            or not np.array_equal(pack['anchors'],physical['history'][:,-1])
            or not np.array_equal(pack['ego_mask'],physical['ego_mask'])
            or not np.all(pack['ego_mask'][:,0]) or not np.all(pack['ego_mask'].sum(1)==1)):
        raise ValueError('natural scalar PET, all-actor masks or physical anchors changed')
    masses=np.zeros((9913,66),np.float64);seen=np.zeros(9913,bool);folds={};fit_records=set(pack['recording_id'])
    for fold,binding in labels_meta['fold_artifacts'].items():
        result=_json(binding)
        if (result['status']!='complete' or result['fold']!=int(fold)
                or not _same(result['prepared'],policy['physical_prepared'])
                or set(result['train_recordings'])&set(result['held_recordings'])
                or set(result['train_recordings'])|set(result['held_recordings'])!=fit_records
                or result['fullFIT_weights_or_warp_reused'] is not False):
            raise ValueError('invalid recording-excluded OOF teacher source')
        predictions=_npz(result['held_predictions'],keys=('scene_id','recording_id','joint_masses','num_agents','pet_seconds'))
        if predictions['joint_masses'].shape!=(len(predictions['scene_id']),66):raise ValueError('OOF head shape changed')
        warp=SceneCDFWarp.from_dict(_json(result['warp']))
        if warp.family!='count':raise ValueError('expected each teacher own frozen count warp')
        for i,(sid,rec) in enumerate(zip(predictions['scene_id'].astype(str),predictions['recording_id'].astype(str))):
            key=(sid,rec,'FIT')
            if key not in lookup or rec not in result['held_recordings']:raise ValueError('OOF prediction outside held recording')
            row=lookup[key]
            if (seen[row] or int(labels['fold'][row])!=int(fold)
                    or predictions['pet_seconds'][i]!=labels['pet_seconds'][row]
                    or int(predictions['num_agents'][i])!=int(pack['agent_mask'][row].sum())):
                raise ValueError('OOF prediction identity/label/count mismatch')
            masses[row]=predictions['joint_masses'][i];seen[row]=True
        folds[fold]=dict(result=binding,held_predictions=result['held_predictions'],warp=result['warp'],
                         held_recordings=result['held_recordings'],train_recordings=result['train_recordings'])
    if set(folds)!=set(map(str,range(5))) or not seen.all():raise ValueError('OOF five-fold predictions do not cover all FIT exactly once')
    return gen_meta,labels_meta,prepared,pack,labels,physical,masses,folds


def prepare_cache(policy_binding):
    policy_binding=dict(path=str(resolve(policy_binding['path'])),sha256=policy_binding['sha256'])
    policy=read_policy(policy_binding);cfg=policy['cache'];output=resolve(cfg['output_root'])
    if output.exists():raise FileExistsError('never overwrite a tangent cache run')
    codes=code_bindings()
    gm,lm,pm,pack,labels,physical,masses,folds=_load_sources(policy)
    selected=select_subset(pack['scene_id'],pack['recording_id'],pack['role'],size=2048,shards=8,salt=cfg['hash_salt'])
    _verify_codes(codes);output.mkdir(parents=True,exist_ok=False)
    identities={key:pack[key].copy() for key in ('scene_id','recording_id','role','agent_mask')}
    identities.update(originalPET=labels['pet_seconds'].copy(),p_mid=labels['p_mid'].copy(),fold=labels['fold'].copy())
    identity_binding=_write_npz(output/'FIT_identity.npz',identities)
    shard_inputs={}
    for shard in range(8):
        items=[row for row in selected if row['shard']==shard];idx=np.array([row['source_index'] for row in items])
        arrays={key:pack[key][idx].copy() for key in ('scene_id','recording_id','role','agent_mask','anchors','ego_mask')}
        arrays.update(source_index=idx,subset_rank=np.array([row['subset_rank'] for row in items],np.int64),
            coef_clean=pack['coef_clean'][idx].astype(np.float64),
            history_t0=physical['history'][idx,-1].copy(),dimensions=physical['dimensions'][idx].copy(),
            originalPET=labels['pet_seconds'][idx].copy(),p_mid=labels['p_mid'][idx].copy(),fold=labels['fold'][idx].copy(),
            oof_masses=masses[idx])
        shard_inputs[str(shard)]=_write_npz(output/('input_shard_'+str(shard)+'.npz'),arrays)
    prepared=dict(protocol=PROTOCOL,status='prepared',policy=policy_binding,code_sha256=codes,
        generator_data=policy['generator_data'],labels_manifest=policy['labels_manifest'],physical_prepared=policy['physical_prepared'],
        FIT_generator_pack=gm['packs']['FIT'],FIT_physical_pack=pm['physical_packs']['FIT'],
        FIT_p_labels=lm['roles']['FIT']['artifact'],coefficient_normalizer=gm['coefficient_normalizer'],
        full_FIT_identity=identity_binding,selected=selected,selected_count=2048,total_FIT_rows=9913,
        shards=8,shard_inputs=shard_inputs,oof_fold_sources=folds,
        source_roles_decoded=['FIT'],CAL_observations_decoded=False,STOP_AUDIT_observations_decoded=False,
        native_future_arrays_decoded=False,CDF_network_loaded=False,existing_p_labels_modified=False,
        h_definition='h=f_OOF(Y)*g=dCDF_dnormalized_coefficients',risk_rank_tangent_sign=-1,
        labels_and_density_source='same_scene_own_recording_OOF_teacher_and_own_CAL_warp',
        tensor_layout='B,Nmax,8,xy',coefficient_coordinates='unchanged_global_FIT_normalized_clean_coefficients',
        policy_parent_training_or_evaluation_artifacts_not_opened=True)
    return _write_json(output/'prepared.json',prepared)


def _read_prepared(binding):
    prepared=_json(binding)
    if prepared.get('protocol')!=PROTOCOL or prepared.get('status')!='prepared':raise ValueError('wrong tangent preparation')
    policy=read_policy(prepared['policy']);_verify_codes(prepared['code_sha256'])
    return prepared,policy


VECTOR_KEYS=('g','unit_g','h')
FLOAT_KEYS=('density','density_candidate','density_left','density_right','density_relative_difference',
            'cdf_reconstruction_difference','originalPET','decodedPET','gradient_norm')
CACHE_KEYS=frozenset(VECTOR_KEYS+FLOAT_KEYS+('scene_id','recording_id','role','agent_mask','fold','p_mid',
    'selected_subset_mask','valid_geom','valid_density','reason','density_reason'))


def run_worker(prepared_binding,shard):
    prepared_binding=dict(path=str(resolve(prepared_binding['path'])),sha256=prepared_binding['sha256'])
    prepared,policy=_read_prepared(prepared_binding);cfg=policy['cache']
    if type(shard) is not int or not 0<=shard<8:raise ValueError('shard must be0..7')
    output=resolve(cfg['output_root']);result_path=output/('shard_'+str(shard)+'.json')
    array_path=output/('shard_'+str(shard)+'.npz')
    if result_path.exists() or array_path.exists():raise FileExistsError('never overwrite a tangent worker')
    torch.set_num_threads(cfg['threads_per_worker'])
    data=_npz(prepared['shard_inputs'][str(shard)],role='FIT')
    expected=[row for row in prepared['selected'] if row['shard']==shard]
    if data['scene_id'].tolist()!=[row['scene_id'] for row in expected]:raise ValueError('worker identities differ from frozen hash subset')
    normalizer=_json(prepared['coefficient_normalizer'])
    warps={fold:SceneCDFWarp.from_dict(_json(source['warp'])) for fold,source in prepared['oof_fold_sources'].items()}
    nrows,nmax=data['agent_mask'].shape
    out={key:data[key].copy() for key in ('source_index','subset_rank','scene_id','recording_id','role','agent_mask','fold','p_mid')}
    out.update(valid_geom=np.zeros(nrows,bool),valid_density=np.zeros(nrows,bool),
               reason=np.full(nrows,'pending',dtype='U128'),density_reason=np.full(nrows,'pending',dtype='U128'))
    for key in VECTOR_KEYS:out[key]=np.zeros((nrows,nmax,8,2),np.float64)
    for key in FLOAT_KEYS:out[key]=np.zeros(nrows,np.float64)
    started=time.perf_counter()
    for row in range(nrows):
        mask=data['agent_mask'][row];count=int(mask.sum())
        if (not mask[0] or data['ego_mask'][row].sum()!=1 or not data['ego_mask'][row,0]
                or not np.array_equal(data['anchors'][row,mask],data['history_t0'][row,mask])):
            raise ValueError('worker changed ego role or actual t0')
        geometry=geometry_tangent(data['coef_clean'][row,mask],data['anchors'][row,mask],data['dimensions'][row,mask],
            normalizer,data['originalPET'][row],pet_tolerance=cfg['reconstruction_PET_error_max_seconds'])
        density=oof_density(warps[str(int(data['fold'][row]))],data['oof_masses'][row],count,
            geometry['originalPET'],geometry['decodedPET'],epsilon=cfg['density_difference_step_seconds'],
            relative_tolerance=cfg['density_left_right_relative_tolerance'],minimum_density=cfg['density_minimum'],
            cdf_difference_limit=cfg['cdf_reconstruction_difference_max'])
        out['valid_geom'][row]=geometry['valid_geom'];out['valid_density'][row]=density['valid_density']
        out['reason'][row]=geometry['reason'];out['density_reason'][row]=density['density_reason']
        for key in ('originalPET','decodedPET','gradient_norm'):out[key][row]=geometry[key]
        for key in ('density_left','density_right','density_relative_difference','cdf_reconstruction_difference'):
            out[key][row]=density[key]
        out['density_candidate'][row]=density['density']
        out['density'][row]=density['density'] if density['valid_density'] else 0.
        out['g'][row,mask]=geometry['g'];out['unit_g'][row,mask]=geometry['unit_g']
        out['h'][row]=out['density'][row]*out['g'][row]
        if (row+1)%32==0:
            print(json.dumps(dict(shard=shard,processed=row+1,total=nrows,valid_geom=int(out['valid_geom'][:row+1].sum()),
                valid_jac=int((out['valid_geom'][:row+1]&out['valid_density'][:row+1]).sum()),wall_seconds=time.perf_counter()-started)),flush=True)
    _verify_codes(prepared['code_sha256']);binding=_write_npz(array_path,out)
    result=dict(protocol=PROTOCOL,status='complete',shard=shard,prepared=prepared_binding,policy=prepared['policy'],
        code_sha256=prepared['code_sha256'],artifact=binding,rows=nrows,scene_ids=out['scene_id'].tolist(),
        valid_geom=int(out['valid_geom'].sum()),valid_density=int(out['valid_density'].sum()),
        valid_jac=int((out['valid_geom']&out['valid_density']).sum()),reasons=dict(Counter(out['reason'])),
        density_reasons=dict(Counter(out['density_reason'])),wall_seconds=time.perf_counter()-started,
        CDF_network_loaded=False,nonFIT_observations_decoded=False,p_labels_changed=False,
        h_definition='dCDF_dc=f*g',risk_rank_tangent_sign=-1)
    return _write_json(result_path,result)


def _validate_cache_arrays(arrays,manifest):
    if set(arrays)!=CACHE_KEYS:raise ValueError('unexpected tangent cache fields')
    n=manifest['rows'];mask=arrays['agent_mask'];selected=arrays['selected_subset_mask']
    if (len(arrays['scene_id'])!=n or len(_identity_map(arrays))!=n or not np.all(arrays['role']=='FIT')
            or mask.dtype!=np.bool_ or mask.shape[0]!=n or selected.shape!=(n,) or selected.dtype!=np.bool_
            or selected.sum()!=manifest['selected_count'] or arrays['valid_geom'].dtype!=np.bool_
            or arrays['valid_density'].dtype!=np.bool_):raise ValueError('invalid full-FIT tangent identity/masks')
    geom=arrays['valid_geom'];density_valid=arrays['valid_density']
    if np.any((geom|density_valid)&~selected):raise ValueError('unselected rows acquired auxiliary supervision')
    for key in VECTOR_KEYS:
        value=arrays[key]
        if (value.shape!=mask.shape+(8,2) or value.dtype!=np.float64 or not np.isfinite(value).all()
                or np.any(value[~mask]!=0) or np.any(value[~geom]!=0)):
            raise ValueError('tangent vectors must preserve all actors and zero unsupported geometry/padding')
    if (not np.isfinite(arrays['density']).all() or np.any(arrays['density'][~density_valid]!=0)
            or not np.allclose(arrays['h'],arrays['g']*arrays['density'][:,None,None,None],rtol=1e-12,atol=1e-12)):
        raise ValueError('masked density/CDF tangent mismatch')
    if geom.any():
        norm=np.sqrt((arrays['g'][geom]**2).sum((1,2,3)))
        if (np.any(norm<=1e-12) or not np.allclose(arrays['unit_g'][geom],arrays['g'][geom]/norm[:,None,None,None],rtol=1e-12,atol=1e-12)
                or np.any(arrays['originalPET'][geom]<=0) or np.any(arrays['originalPET'][geom]>=4)
                or np.any(np.abs(arrays['decodedPET'][geom]-arrays['originalPET'][geom])>.01)):
            raise ValueError('geometry eligibility or unit tangent inconsistent')
    if (np.any(arrays['density'][density_valid]<=1e-6)
            or np.any(arrays['density_relative_difference'][density_valid]>.01)
            or np.any(arrays['cdf_reconstruction_difference'][density_valid]>.05)):
        raise ValueError('Jacobian density/smoothness/alignment mask violates frozen policy')


def aggregate_cache(prepared_binding):
    prepared_binding=dict(path=str(resolve(prepared_binding['path'])),sha256=prepared_binding['sha256'])
    prepared,policy=_read_prepared(prepared_binding);output=resolve(policy['cache']['output_root'])
    if (output/'manifest.json').exists() or (output/'FIT_tangents.npz').exists():raise FileExistsError('never overwrite tangent aggregation')
    result_bindings={};results={}
    for shard in range(8):
        path=output/('shard_'+str(shard)+'.json');binding=dict(path=str(path),sha256=sha256(path));result=_json(binding)
        expected=[row['scene_id'] for row in prepared['selected'] if row['shard']==shard]
        if (result.get('protocol')!=PROTOCOL or result.get('status')!='complete' or result['shard']!=shard
                or not _same(result['prepared'],prepared_binding) or result['code_sha256']!=prepared['code_sha256']
                or result['scene_ids']!=expected or result['rows']!=len(expected)):
            raise ValueError('all-eight exact-shard completion barrier failed')
        result_bindings[str(shard)]=binding;results[shard]=result
    identity=_npz(prepared['full_FIT_identity'],role='FIT');n=len(identity['scene_id']);nmax=identity['agent_mask'].shape[1]
    arrays={key:value.copy() for key,value in identity.items()}
    arrays.update(selected_subset_mask=np.zeros(n,bool),valid_geom=np.zeros(n,bool),valid_density=np.zeros(n,bool),
        reason=np.full(n,'not_selected',dtype='U128'),density_reason=np.full(n,'not_selected',dtype='U128'))
    for key in VECTOR_KEYS:arrays[key]=np.zeros((n,nmax,8,2),np.float64)
    for key in FLOAT_KEYS:
        if key not in arrays:arrays[key]=np.zeros(n,np.float64)
    seen=np.zeros(n,bool)
    for shard,result in results.items():
        part=_npz(result['artifact'],role='FIT');indices=part['source_index']
        expected=[row for row in prepared['selected'] if row['shard']==shard]
        if (part['scene_id'].tolist()!=result['scene_ids']
                or indices.tolist()!=[row['source_index'] for row in expected]):
            raise ValueError('shard arrays do not match their exact frozen source-index roster')
        if np.any(indices<0) or np.any(indices>=n) or seen[indices].any():raise ValueError('duplicated or invalid tangent shard rows')
        for key in ('scene_id','recording_id','role','agent_mask','p_mid','fold','originalPET'):
            if not np.array_equal(arrays[key][indices],part[key]):raise ValueError('shard altered source identity/mask/label/PET')
        for key in VECTOR_KEYS+FLOAT_KEYS+('valid_geom','valid_density','reason','density_reason'):
            arrays[key][indices]=part[key]
        seen[indices]=True
    expected_indices=np.array([row['source_index'] for row in prepared['selected']])
    arrays['selected_subset_mask'][expected_indices]=True
    if not np.array_equal(seen,arrays['selected_subset_mask']):raise ValueError('2048 selected IDs are not covered exactly once')
    summary=dict(rows=n,selected_count=int(seen.sum()),valid_geom=int(arrays['valid_geom'].sum()),
        valid_density=int(arrays['valid_density'].sum()),valid_jac=int((arrays['valid_geom']&arrays['valid_density']).sum()),
        reasons=dict(Counter(arrays['reason'][seen])),density_reasons=dict(Counter(arrays['density_reason'][seen])))
    _validate_cache_arrays(arrays,summary);_verify_codes(prepared['code_sha256'])
    artifact=_write_npz(output/'FIT_tangents.npz',arrays)
    manifest=dict(protocol=PROTOCOL,status='complete',policy=prepared['policy'],prepared=prepared_binding,
        generator_data=prepared['generator_data'],labels_manifest=prepared['labels_manifest'],physical_prepared=prepared['physical_prepared'],
        FIT_array=artifact,shard_results=result_bindings,code_sha256=prepared['code_sha256'],
        coefficient_normalizer=prepared['coefficient_normalizer'],oof_fold_sources=prepared['oof_fold_sources'],
        **summary,tensor_layout='B,Nmax,8,xy',p_labels_modified=False,all_base_training_rows_retained=True,
        geometry_tangent_scope='supported_clean_natural_coefficient_reconstruction_local_only',
        h_definition='h=f*g=dCDF_dc',risk_rank_tangent_sign=-1,
        density='effective_f_zero_when_invalid; density_candidate_preserves_diagnostic_secant_mean',
        projection_mask='selected_subset_mask & valid_geom & trainer_p_present',
        jacobian_mask='selected_subset_mask & valid_geom & valid_density & trainer_p_present & trainer_sigma>=0.5',
        CDF_network_loaded=False,nonFIT_observations_decoded=False,native_future_arrays_decoded=False)
    binding=_write_json(output/'manifest.json',manifest)
    load_tangent_cache(binding)
    return binding


def validate_tangent_manifest(binding):
    """Validate hash/JSON-only lineage; no teacher weights or CAL arrays decoded."""
    manifest=_json(binding)
    if (manifest.get('protocol')!=PROTOCOL or manifest.get('status')!='complete' or manifest.get('rows')!=9913
            or manifest.get('selected_count')!=2048 or manifest.get('risk_rank_tangent_sign')!=-1
            or manifest.get('p_labels_modified') is not False or manifest.get('CDF_network_loaded') is not False
            or manifest.get('nonFIT_observations_decoded') is not False):raise ValueError('wrong tangent cache population/scope')
    policy=read_policy(manifest['policy']);prepared=_json(manifest['prepared']);_verify_codes(manifest['code_sha256'])
    if (not _same(manifest['generator_data'],policy['generator_data']) or not _same(manifest['labels_manifest'],policy['labels_manifest'])
            or not _same(manifest['physical_prepared'],policy['physical_prepared'])
            or prepared['code_sha256']!=manifest['code_sha256'] or set(manifest['shard_results'])!=set(map(str,range(8)))):
        raise ValueError('cache source or shard binding mismatch')
    identities=[]
    for shard,binding in manifest['shard_results'].items():
        result=_json(binding);expected=[row['scene_id'] for row in prepared['selected'] if row['shard']==int(shard)]
        if (result['status']!='complete' or result['shard']!=int(shard) or result['scene_ids']!=expected
                or not _same(result['prepared'],manifest['prepared']) or result['code_sha256']!=manifest['code_sha256']):
            raise ValueError('incomplete or mixed tangent worker lineage')
        verify_binding(result['artifact']);identities.extend(expected)
    if len(identities)!=2048 or len(set(identities))!=2048:raise ValueError('subset is not exactly2048 unique scenes')
    for source in manifest['oof_fold_sources'].values():
        if set(source['held_recordings'])&set(source['train_recordings']):raise ValueError('OOF density source has held-record leakage')
        for key in ('result','held_predictions','warp'):verify_binding(source[key])
    verify_binding(manifest['FIT_array']);return manifest


def load_tangent_cache(binding):
    manifest=validate_tangent_manifest(binding)
    path=verify_binding(manifest['FIT_array'])
    with np.load(path,allow_pickle=False) as archive:
        if set(archive.files)!=CACHE_KEYS:raise ValueError('unexpected tangent fields refused before numeric decode')
    arrays=_npz(manifest['FIT_array'],keys=CACHE_KEYS,role='FIT')
    _validate_cache_arrays(arrays,manifest)
    prepared=_json(manifest['prepared']);identity=_npz(prepared['full_FIT_identity'],role='FIT')
    for key in ('scene_id','recording_id','role','agent_mask','fold','p_mid','originalPET'):
        if not np.array_equal(arrays[key],identity[key]):raise ValueError('full-FIT source identity/mask/original label changed')
    expected=np.zeros(manifest['rows'],bool)
    expected[[row['source_index'] for row in prepared['selected']]]=True
    if not np.array_equal(arrays['selected_subset_mask'],expected):raise ValueError('auxiliary subset differs from frozen hash selection')
    return arrays
