#!/usr/bin/env python3
"""Expand only FIT's H-selected hash queue; retain only E_full observations.

The4096 H keys/actor rosters freeze before native future eligibility is checked.
Incomplete scenes stay in the eligibility ledger and never trigger replacement
or PET measurement. STOP/CAL/AUDIT reuse the exact old complete-view artifacts.
"""
from collections import Counter
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
for _name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[_name]='1'
import numpy as np
from pcontrol.data.highd import load_train_recording
from pcontrol.data.clips import extract_context
from pcontrol.data.scenes import select_scene_contexts,selection_cohort_rows,scene_id,extract_scene,scene_arrays
from pcontrol.data.scene_pet import scene_occupancy_pet,PROTOCOL as METRIC_PROTOCOL
from pcontrol.data.complete_scene_view import ELIGIBILITY,ESTIMAND,verify_binding,resolve,sha256,complete_eligibility,validate_complete_labels

CONFIG_PROTOCOL='natural_complete_scene_FIT_expansion_config_v1'
RECORD_PROTOCOL='natural_expanded_complete_scene_recording_v1'
VIEW_PROTOCOL='natural_expanded_complete_scene_reference_view_v1'
CODE=('pcontrol/research/expand_natural_complete_scenes.py',
    'pcontrol/data/scenes.py','pcontrol/data/scene_pet.py','pcontrol/data/clips.py',
    'pcontrol/data/highd.py','pcontrol/data/pair_pet.py','pcontrol/data/pet.py',
    'pcontrol/data/road_semantics.py','pcontrol/data/cut_in_pet.py','pcontrol/data/complete_scene_view.py')


def bound_json(binding):return json.loads(verify_binding(binding).read_text())


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False)
    return dict(path=str(Path(path).resolve()),sha256=sha256(path))


def safe(value):
    if isinstance(value,np.ndarray):return safe(value.tolist())
    if isinstance(value,np.generic):return safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {key:safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [safe(item) for item in value]
    return value


def validate(config):
    required={'protocol','input','raw_root','FIT_recordings','selection','eligibility','estimand','measurement','runtime',
        'loss_choice','training_executed_by_expansion','generator_training','incomplete_PET_measured','replace_incomplete_scenes','output_root'}
    if set(config)!=required or config['protocol']!=CONFIG_PROTOCOL:raise ValueError('only frozen FIT expansion schema accepted')
    if set(config['input'])!={'scene_manifest','complete_view','original_split','roles'}:raise ValueError('all four source bindings required')
    source=bound_json(config['input']['scene_manifest']);view=bound_json(config['input']['complete_view'])
    roles=bound_json(config['input']['roles']);original=bound_json(config['input']['original_split'])
    if source['protocol']!='natural_ego_environment_scene_dataset_v1' or source['status']!='complete' or view['protocol']!='natural_complete_ego_scene_reference_view_v1':raise ValueError('wrong frozen source dataset/view')
    if source['input_bindings']['roles']!=dict(path=str(resolve(config['input']['roles']['path'])),sha256=config['input']['roles']['sha256']) or source['input_bindings']['original_split']!=dict(path=str(resolve(config['input']['original_split']['path'])),sha256=config['input']['original_split']['sha256']):raise ValueError('original source authorization differs')
    if view['source_manifest']['sha256']!=config['input']['scene_manifest']['sha256'] or config['eligibility']!=ELIGIBILITY or config['estimand']!=ESTIMAND:raise ValueError('complete-scene estimand changed')
    if config['FIT_recordings']!=roles['FIT'] or len(roles['FIT'])!=13 or not set(roles['FIT'])<=set(original['splits']['train']):raise PermissionError('only original FIT13 may expand')
    if config['selection']!={'hash_salt':'natural_ego_scene_v1','history_quota_per_recording':4096,'nested_original_prefix':512}:raise ValueError('nested H-only expansion policy changed')
    if config['measurement']!={'sample_period':.04,'window':[0.,6.96],'cap_seconds':4.}:raise ValueError('original window/PET changed')
    if config['runtime']!={'maximum_parallel_recording_workers':4,'threads_per_worker':1} or config['loss_choice']!='CRPS':raise ValueError('runtime/loss changed')
    if any(config[key] is not False for key in ('training_executed_by_expansion','generator_training','incomplete_PET_measured','replace_incomplete_scenes')):raise ValueError('expansion may not train, fill, replace or label incomplete scenes')
    if any(view['by_role'][role]['complete_scenes']!=expected for role,expected in (('STOP',538),('CAL',504),('AUDIT',487))):raise ValueError('nonFIT frozen complete-view denominators changed')
    for path,checksum in source['code_sha256'].items():verify_binding(dict(path=path,sha256=checksum))
    if resolve(config['output_root'])==resolve(config['raw_root']) or resolve(config['raw_root']) in resolve(config['output_root']).parents:raise ValueError('outputs must be separate from raw observations')
    return source,view,roles


def expanded_contexts(recording,old_cohort,*,salt='natural_ego_scene_v1',quota=4096):
    if quota<512 or quota>4096:raise ValueError('expanded quota must retain the original512 prefix')
    prefix,audit=select_scene_contexts(recording,salt=salt,max_scenes=512)
    if len(prefix)!=512 or selection_cohort_rows(prefix)!=old_cohort['scenes']:
        raise ValueError('original512 scene IDs/H/actor rosters or hash order changed')
    grid=recording.tracks.loc[recording.tracks.frame%25==0]
    ordered=sorted((hashlib.sha256(f'{salt}|{recording.recording_id}|{int(ego)}|{int(t0)}'.encode()).hexdigest(),int(ego),int(t0))
        for t0,ego in grid[['frame','id']].itertuples(index=False,name=None))
    frames={int(t0):rows.set_index('id') for t0,rows in grid.groupby('frame',sort=False)}
    selected=list(prefix);failures=Counter();attempts=0
    for rank in range(prefix[-1]['candidate_rank']+1,len(ordered)):
        if len(selected)==quota:break
        digest,ego,t0=ordered[rank];attempts+=1
        context,reason=extract_context(recording,t0,ego,frame_rows=frames[t0])
        if context is None:failures[str(reason)]+=1;continue
        selected.append(dict(context=context,scene_id=scene_id(recording.recording_id,ego,t0),selection_hash=digest,candidate_rank=rank))
    return selected,dict(original_prefix_exact=True,original_prefix_rows=512,expanded_H_selected=len(selected),quota=quota,
        original_selection_audit=audit,additional_H_attempts=attempts,additional_H_rejections=dict(failures),
        future_used_for_selection=False,incomplete_replacement=False)


def build_recording(config,config_sha,rec):
    source,view,roles=validate(config)
    if rec not in roles['FIT']:raise PermissionError('raw expansion is authorized only for FIT13 recordings')
    directory=resolve(config['output_root'])/'FIT';directory.mkdir(parents=True,exist_ok=True)
    if any((directory/f'{rec}.{suffix}').exists() for suffix in ('claim.json','cohort.json','eligibility.jsonl','npz','result.json')):raise FileExistsError('never overwrite or resume an expansion recording')
    write_json(directory/f'{rec}.claim.json',dict(protocol=RECORD_PROTOCOL,recording_id=rec,config_sha256=config_sha))
    old_entry=source['recordings'][rec];old_result=bound_json(old_entry['result_binding'])
    old_cohort=bound_json(old_entry['artifacts']['selection_cohort']);raw=old_result['source_csv_bindings']
    snapshots={}
    if set(raw)!={'tracks','tracksMeta','recordingMeta'}:raise ValueError('three raw FIT recording bindings required')
    for key,binding in raw.items():
        path=verify_binding(binding,resolve(config['raw_root'])/f'{rec}_{key}.csv');stat=path.stat();snapshots[key]=(stat.st_size,stat.st_mtime_ns)
    code={path:sha256(ROOT/path) for path in CODE};started=time.perf_counter()
    recording=load_train_recording(resolve(config['raw_root']),rec,split_path=resolve(config['input']['original_split']['path']))
    selected,audit=expanded_contexts(recording,old_cohort,salt=config['selection']['hash_salt'],quota=4096)
    cohort_rows=selection_cohort_rows(selected)
    cohort_binding=write_json(directory/f'{rec}.cohort.json',dict(protocol=RECORD_PROTOCOL,recording_id=rec,role='FIT',
        config_sha256=config_sha,input_bindings=config['input'],code_sha256=code,source_csv_bindings=raw,
        original512_cohort=old_entry['artifacts']['selection_cohort'],selection_audit=audit,scenes=cohort_rows,
        all_H_keys_frozen_before_future_extraction=True,future_extraction_started=False))
    old_data_path=verify_binding(old_entry['artifacts']['data'])
    with np.load(old_data_path,allow_pickle=False) as archive:
        old_ids=archive['scene_id'];old_complete=archive['complete_horizon'];old_values=archive['pet_value']
    complete_scenes=[];labels=[];complete_selected=[];cohort_indices=[];counts=Counter();by_N={};times=np.arange(175)*.04
    ledger_path=directory/f'{rec}.eligibility.jsonl'
    with ledger_path.open('x',encoding='utf-8') as ledger:
        for i,item in enumerate(selected):
            context=item['context'];scene,reason=extract_scene(recording,context.t0_frame,context.ego_id,context=context)
            if scene is None:raise RuntimeError('a selected H scene was discarded: '+str(reason))
            complete=bool(scene.future_observed_mask.all());label=None
            if i<512 and (scene.scene_id!=old_ids[i] or complete!=bool(old_complete[i])):raise ValueError('nested512 eligibility changed')
            if complete:
                label=scene_occupancy_pet(scene.future_native,scene.future_observed_mask,scene.dimensions,times=times,ego_index=0,sample_period=.04,window=(0.,6.96),cap_seconds=4.)
                if not label['complete'] or not label['point_identified'] or not np.isfinite(label['pet_value_seconds']):raise ValueError('complete scene lacks a valid point PET; no row may be dropped')
                if i<512 and label['pet_value_seconds']!=old_values[i]:raise ValueError('nested512 complete PET changed')
                complete_scenes.append(scene);labels.append(label);complete_selected.append(item);cohort_indices.append(i)
            record=dict(cohort_rows[i],cohort_row=i,complete_horizon=complete,PET_computed=complete,
                observation_audit=scene.observation_audit,selected_row_replaced=False,label=safe(label),
                exclusion_reason=None if complete else 'not_all_selected_actors_observed_for_full175_native_frames')
            ledger.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n');ledger.flush()
            for counter in (counts,by_N.setdefault(str(len(scene.agent_ids)),Counter())):
                counter['H_selected_scenes']+=1;counter['complete_scenes']+=complete;counter['excluded_incomplete']+=not complete
                if complete:counter[label['status']]+=1
            if (i+1)%256==0:print(json.dumps(dict(recording=rec,checked=i+1,H_selected=len(selected),complete=counts['complete_scenes'])),flush=True)
    for key,binding in raw.items():
        stat=resolve(binding['path']).stat()
        if (stat.st_size,stat.st_mtime_ns)!=snapshots[key]:raise RuntimeError('raw FIT source changed during expansion')
    metadata=dict(protocol=RECORD_PROTOCOL,metric_version=METRIC_PROTOCOL,recording_id=rec,role='FIT',config_sha256=config_sha,
        input_bindings=config['input'],code_sha256=code,source_csv_bindings=raw,selection_cohort=cohort_binding,
        eligibility=ELIGIBILITY,estimand=ESTIMAND,complete_only=True,natural_only=True,simulation_or_generated_futures=False,
        focal_vehicle_input=False,partial_zero_included=False,incomplete_PET_computed=False,model_training=False,generator_training=False)
    arrays=scene_arrays(complete_scenes,labels,complete_selected,role='FIT',metadata=metadata)
    arrays['cohort_row_index']=np.asarray(cohort_indices,dtype=np.int64)
    eligible=complete_eligibility(arrays);validate_complete_labels(arrays,eligible)
    if not eligible.all():raise RuntimeError('incomplete scene entered the expansion shard')
    path=directory/f'{rec}.npz'
    with path.open('xb') as handle:np.savez_compressed(handle,**arrays)
    result=dict(protocol=RECORD_PROTOCOL,status='complete',recording_id=rec,role='FIT',config_sha256=config_sha,
        input_bindings=config['input'],code_sha256=code,source_csv_bindings=raw,cohort=cohort_binding,
        eligibility_ledger=dict(path=str(ledger_path),sha256=sha256(ledger_path)),data=dict(path=str(path),sha256=sha256(path)),
        counts=dict(counts),by_N={key:dict(value) for key,value in by_N.items()},complete_scene_ids=arrays['scene_id'].tolist(),
        complete_cohort_rows=cohort_indices,original512_nested_ids_history_rosters_eligibility_and_PET_exact=True,
        complete_only=True,incomplete_rows_replaced=False,incomplete_PET_computed=False,wall_seconds=time.perf_counter()-started)
    write_json(directory/f'{rec}.result.json',result);print(json.dumps(dict(recording_complete=rec,counts=dict(counts))),flush=True)
    return result


def finalize(config,config_sha):
    source,old_view,roles=validate(config);root=resolve(config['output_root'])
    if (root/'manifest.json').exists():raise FileExistsError('unified expanded view already exists')
    entries={};totals={role:Counter() for role in ('FIT','STOP','CAL','AUDIT')};jobs={};code={path:sha256(ROOT/path) for path in CODE}
    for rec,original in old_view['recordings'].items():
        if original['role']!='FIT':
            entry=copy.deepcopy(original);entry['source_kind']='unchanged_original_complete_view';entries[rec]=entry
            totals[original['role']]['complete_scenes']+=len(entry['selected_rows']);continue
        path=root/'FIT'/f'{rec}.result.json';result=json.loads(path.read_text())
        if result.get('status')!='complete' or result.get('protocol')!=RECORD_PROTOCOL or result.get('config_sha256')!=config_sha or result.get('input_bindings')!=config['input'] or result.get('code_sha256')!=code or not result.get('original512_nested_ids_history_rosters_eligibility_and_PET_exact'):
            raise ValueError('expanded FIT worker source/protocol/prefix verification differs')
        for name in ('data','cohort','eligibility_ledger'):verify_binding(result[name])
        with np.load(verify_binding(result['data'],root/'FIT'/f'{rec}.npz'),allow_pickle=False) as archive:
            required=('offsets','future_observed_mask_agents','complete_horizon','point_identified','pet_value','label_status','label_interval_seconds','scene_id','recording_id','role','cohort_row_index')
            arrays={key:archive[key] for key in required}
        full=complete_eligibility(arrays);validate_complete_labels(arrays,full)
        if not full.all() or arrays['scene_id'].tolist()!=result['complete_scene_ids'] or arrays['cohort_row_index'].tolist()!=result['complete_cohort_rows'] or not np.all(arrays['role']=='FIT') or not np.all(arrays['recording_id']==rec):raise ValueError('expanded FIT complete rows or source identity differ')
        n=len(full);entries[rec]=dict(source_kind='expanded_FIT_complete_only',role='FIT',data=result['data'],selected_rows=list(range(n)),
            selected_scene_ids=arrays['scene_id'].tolist(),counts=dict(complete_scenes=n,source_scenes=result['counts']['H_selected_scenes']),
            result_binding=dict(path=str(path),sha256=sha256(path)),cohort=result['cohort'],eligibility_ledger=result['eligibility_ledger'])
        totals['FIT']['complete_scenes']+=n;totals['FIT']['source_scenes']+=result['counts']['H_selected_scenes'];jobs[rec]=entries[rec]['result_binding']
    all_ids=[sid for entry in entries.values() for sid in entry['selected_scene_ids']]
    if len(set(all_ids))!=len(all_ids):raise ValueError('duplicate selected scene across unified view')
    for role,n in (('STOP',538),('CAL',504),('AUDIT',487)):
        if totals[role]['complete_scenes']!=n:raise ValueError('frozen nonFIT denominator changed')
    view_code=dict(code);view_code['pcontrol/data/expanded_scene_view.py']=sha256(ROOT/'pcontrol/data/expanded_scene_view.py')
    manifest=dict(protocol=VIEW_PROTOCOL,status='complete',base_complete_view=config['input']['complete_view'],source_scene_manifest=config['input']['scene_manifest'],
        expansion_config_sha256=config_sha,input_bindings=config['input'],code_sha256=view_code,expansion_worker_code_sha256=code,recordings=entries,by_role={key:dict(value) for key,value in totals.items()},
        eligibility=ELIGIBILITY,estimand=ESTIMAND,partial_zero_included=False,loss='CRPS',FIT_history_quota=4096,
        nonFIT_views_bitwise_unchanged=True,nested_original512_verified=True,future_availability_is_model_input=False,
        no_future_critical_actor_input=True,simulation_or_generated_futures=False,generator_training=False,
        target_cohort_is_all_original_scenes=False,missing_data_population_recovery=False,model_training_executed_by_expansion=False,
        FIT_worker_results=jobs,complete_scenes=len(all_ids))
    binding=write_json(root/'manifest.json',manifest);print(json.dumps(dict(unified_manifest=binding,by_role=manifest['by_role'])),flush=True)
    return manifest,binding


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--recording');parser.add_argument('--finalize',action='store_true');parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if sha256(args.config)!=args.config_sha256:raise ValueError('expansion config changed')
    config=json.loads(args.config.read_text());validate(config)
    if args.validate_only:print(json.dumps(dict(status='config_valid',raw_opened=False)));return
    if args.finalize:
        if args.recording:parser.error('finalize cannot specify recording')
        return finalize(config,args.config_sha256)
    if not args.recording:parser.error('recording required for a FIT worker')
    return build_recording(config,args.config_sha256,args.recording)


if __name__=='__main__':main()
