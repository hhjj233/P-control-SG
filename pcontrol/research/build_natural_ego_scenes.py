#!/usr/bin/env python3
"""One recording worker: freeze H-only cohort, export native futures and scene PET."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
from pcontrol.data.highd import load_train_recording,load_split_assignment,recording_id
from pcontrol.data.scenes import select_scene_contexts,selection_cohort_rows,extract_scene,scene_arrays
from pcontrol.data.scene_pet import scene_occupancy_pet,PROTOCOL as METRIC_VERSION

PROTOCOL='natural_highd_ego_scene_build_v1'
RESULT_PROTOCOL='natural_highd_ego_scene_recording_result_v1'
POLICY_SHA='7ac3114317970a545962b4dc49c4a6dacead64422d9d7b05046c38edf9628968'
CODE=('pcontrol/research/build_natural_ego_scenes.py','pcontrol/data/scenes.py','pcontrol/data/scene_pet.py',
    'pcontrol/data/highd.py','pcontrol/data/clips.py','pcontrol/data/pair_pet.py',
    'pcontrol/data/pet.py','pcontrol/data/road_semantics.py','pcontrol/data/cut_in_pet.py')


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def bound_json(binding):
    path=Path(binding['path']).resolve()
    if sha(path)!=binding['sha256']:raise ValueError('input SHA256 differs: '+str(path))
    return json.loads(path.read_text())


def write_json(path,value):
    with Path(path).open('x',encoding='utf-8') as handle:json.dump(value,handle,indent=2,sort_keys=True,allow_nan=False)
    return dict(path=str(Path(path).resolve()),sha256=sha(path))


def json_safe(value):
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {key:json_safe(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(item) for item in value]
    return value


def validate(config,recording):
    if set(config)!={'protocol','raw_root','input','recordings','selection','measurement','runtime','output_root'} or config['protocol']!=PROTOCOL:
        raise ValueError('only frozen ego-scene build schema accepted')
    if set(config['input'])!={'source_manifest','original_split','roles','policy'} or config['input']['policy']['sha256']!=POLICY_SHA:
        raise ValueError('four exact input bindings and immutable scene policy required')
    policy=bound_json(config['input']['policy']);source=bound_json(config['input']['source_manifest'])
    roles=bound_json(config['input']['roles']);original=bound_json(config['input']['original_split'])
    rec=recording_id(recording);splits=load_split_assignment(Path(config['input']['original_split']['path']))
    train=set(splits['train'])
    if len(train)!=29 or len(config['recordings'])!=29 or set(config['recordings'])!=train or rec not in train:
        raise PermissionError('only all29 original-train records may be selected; worker recording must be authorized')
    if rec in set(splits['val'])|set(splits['test'])|{'18'}:raise PermissionError('protected recording forbidden')
    if config['selection']!={'hash_salt':'natural_ego_scene_v1','max_scenes_per_recording':512} or config['measurement']!={'sample_period':.04,'window':[0.,6.96],'cap_seconds':4.}:
        raise ValueError('sampling/native time/cap differs from preregistered protocol')
    if config['runtime']!={'maximum_parallel_recording_workers':8,'threads_per_worker':1}:raise ValueError('runtime policy differs')
    if not Path(config['output_root']).is_absolute() or Path(config['output_root']).resolve()==Path(config['raw_root']).resolve() or Path(config['raw_root']).resolve() in Path(config['output_root']).resolve().parents:
        raise ValueError('new absolute outputs must remain outside raw data')
    if source.get('protocol')!='natural_highd_variable_context_event_clips_v2' or source.get('natural_only') is not True or source.get('allow_simulated_futures') is not False or source.get('split_binding')!=config['input']['original_split']:
        raise ValueError('raw source manifest is not the authenticated natural original-train source')
    for relative in ('pcontrol/data/highd.py','pcontrol/data/clips.py'):
        path=(ROOT/relative).resolve()
        if source.get('code_sha256',{}).get(str(path))!=sha(path):raise ValueError('frozen raw/history extraction code changed')
    extraction=bound_json(source['data_protocol_binding'])
    if Path(extraction['raw_data_root']).resolve()!=Path(config['raw_root']).resolve():raise ValueError('raw source root changed')
    if roles.get('protocol')!='natural_ego_scene_recording_roles_v1' or roles.get('original_split')!=config['input']['original_split']:
        raise ValueError('scene role provenance differs')
    parent=bound_json(roles['parent_role_roster']);names=('FIT','STOP','CAL','AUDIT')
    assigned=[value for name in names for value in roles[name]]
    if len(assigned)!=29 or set(assigned)!=train or any(roles[name]!=parent[name] for name in names):raise ValueError('scene recording roles must preserve the entire v5 partition')
    if any(roles.get(key) is not False for key in ('old_pair_event_counts_reused','old_calibration_results_reused','model_training','generator_training')):
        raise ValueError('data-only scene build cannot reuse old outcomes or train')
    role=next(name for name in names if rec in roles[name])
    if policy.get('metric',{}).get('aggregation')!='minimum_over_all_ego_other_pairs_no_background_background_primary_target':raise ValueError('ego-scene metric aggregation differs')
    return rec,role,source


def build_recording(config,config_sha256,recording):
    rec,role,source=validate(config,recording);output=Path(config['output_root'])
    paths={kind:output/f'{rec}.{suffix}' for kind,suffix in (('selection','selection.json'),('events','events.jsonl'),('data','npz'),('result','result.json'),('claim','claim.json'))}
    output.mkdir(parents=True,exist_ok=True)
    if any(path.exists() for path in paths.values()):raise FileExistsError('never overwrite/retry an existing scene recording worker')
    write_json(paths['claim'],dict(protocol=PROTOCOL,recording_id=rec,config_sha256=config_sha256))
    raw_bindings=source['input_bindings'][rec];snapshots={}
    if set(raw_bindings)!={'tracks','tracksMeta','recordingMeta'}:raise ValueError('three raw recording CSV bindings required')
    for key,binding in raw_bindings.items():
        path=Path(binding['path']).resolve();expected=Path(config['raw_root']).resolve()/f'{rec}_{key}.csv'
        if path!=expected or sha(path)!=binding['sha256']:raise ValueError('raw source identity/SHA mismatch')
        stat=path.stat();snapshots[key]=(stat.st_size,stat.st_mtime_ns)
    started=time.perf_counter();code={path:sha(ROOT/path) for path in CODE}
    recording_data=load_train_recording(Path(config['raw_root']),rec,split_path=Path(config['input']['original_split']['path']))
    selected,selection_audit=select_scene_contexts(recording_data,salt=config['selection']['hash_salt'],max_scenes=512)
    cohort=selection_cohort_rows(selected)
    cohort_binding=write_json(paths['selection'],dict(protocol=PROTOCOL,recording_id=rec,role=role,config_sha256=config_sha256,
        input_bindings=config['input'],code_sha256=code,source_csv_bindings=raw_bindings,selection_audit=selection_audit,scenes=cohort,
        all_selected_keys_frozen_before_future_extraction=True,future_extraction_started=False))
    scenes=[];labels=[];counts=Counter();by_n={};times=np.arange(175,dtype=np.float64)*.04
    with paths['events'].open('x',encoding='utf-8') as ledger:
        for position,item in enumerate(selected):
            context=item['context'];scene,reason=extract_scene(recording_data,context.t0_frame,context.ego_id,context=context)
            if scene is None:raise RuntimeError('a frozen H-qualified scene was discarded: '+str(reason))
            try:
                label=scene_occupancy_pet(scene.future_native,scene.future_observed_mask,scene.dimensions,times=times,
                    ego_index=0,sample_period=.04,window=(0.,6.96),cap_seconds=4.)
            except (ValueError,FloatingPointError) as error:
                label=dict(metric_version=METRIC_VERSION,status='invalid_input',complete=bool(scene.future_observed_mask.all()),point_identified=False,
                    pet_value_seconds=None,label_interval_seconds=[np.nan,np.nan],observed_min_seconds=np.nan,observed_min_capped_seconds=np.nan,
                    critical_other_index=None,observed_critical_other_index=None,pair_results=[],error_type=type(error).__name__,error=str(error))
            for key in ('critical_other_index','observed_critical_other_index'):
                idx=label.get(key);label[key.replace('_index','_actor_id')]=None if idx is None else int(scene.agent_ids[idx])
            label['observed_min_is_positive_infinity']=bool(np.isposinf(label['observed_min_seconds']))
            label['critical_vehicle_is_label_diagnostic_not_input']=True
            counter=by_n.setdefault(str(len(scene.agent_ids)),Counter())
            for target in (counts,counter):
                target['selected_scenes']+=1;target[label['status']]+=1
                target['complete_horizon']+=bool(scene.future_observed_mask.all());target['point_identified']+=bool(label['point_identified'])
                target['censored_scenes']+=label['status']=='partial_observation_interval';target['invalid_scenes']+=label['status']=='invalid_input'
            entry=dict(cohort[position],role=role,cohort_position=position,observation_audit=scene.observation_audit,label=json_safe(label),
                future_only_for_labeling=True,future_critical_actor_used_in_history_inputs=False,selected_row_replaced=False)
            ledger.write(json.dumps(entry,sort_keys=True,allow_nan=False)+'\n');ledger.flush()
            scenes.append(scene);labels.append(label)
            if (position+1)%32==0:print(json.dumps(dict(recording=rec,measured_scenes=position+1,selected=len(selected))),flush=True)
    for key,binding in raw_bindings.items():
        stat=Path(binding['path']).stat()
        if (stat.st_size,stat.st_mtime_ns)!=snapshots[key]:raise RuntimeError('raw source changed during scene build')
    for key in ('selected_scenes','complete_horizon','point_identified','censored_scenes','invalid_scenes','exact_zero','certified_zero_with_missing','complete_finite','complete_capped','complete_no_shared_occupancy'):
        counts.setdefault(key,0)
    metadata=dict(protocol=PROTOCOL,metric_version=METRIC_VERSION,recording_id=rec,role=role,config_sha256=config_sha256,
        input_bindings=config['input'],code_sha256=code,source_csv_bindings=raw_bindings,selection_cohort=cohort_binding,
        source_origin='natural_observation',all_selected_scenes_retained=True,ego_index=0,focal_vehicle_input=False,
        other_agent_order='ascending_raw_ID',history_dt_seconds=.08,future_native_dt_seconds=.04,
        future_labels_do_not_change_history_membership=True,model_training=False,generator_training=False,
        complete_horizon_and_point_identified_are_distinct=True,no_default_training_queue_created=True,
        original_protected_val_test_or18_opened=False,old_pair_results_not_scene_evidence=True)
    arrays=scene_arrays(scenes,labels,selected,role=role,metadata=metadata)
    with paths['data'].open('xb') as handle:np.savez_compressed(handle,**arrays)
    data_binding=dict(path=str(paths['data']),sha256=sha(paths['data']));event_binding=dict(path=str(paths['events']),sha256=sha(paths['events']))
    if counts['selected_scenes']!=len(cohort):raise RuntimeError('selected cohort denominator changed after future extraction')
    result=dict(protocol=RESULT_PROTOCOL,status='completed',recording_id=rec,role=role,config_sha256=config_sha256,input_bindings=config['input'],
        code_sha256=code,source_csv_bindings=raw_bindings,selection_cohort=cohort_binding,events=event_binding,data=data_binding,
        counts=dict(counts),by_N={key:dict(value) for key,value in by_n.items()},selection_audit=selection_audit,
        all_selected_scenes_retained=True,model_training=False,generator_training=False,wall_seconds=time.perf_counter()-started)
    write_json(paths['result'],result);print(json.dumps(dict(recording_complete=rec,counts=dict(counts)),sort_keys=True),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--config-sha256',required=True);parser.add_argument('--recording',required=True);parser.add_argument('--validate-only',action='store_true')
    args=parser.parse_args()
    if sha(args.config)!=args.config_sha256:raise ValueError('scene config SHA256 differs')
    config=json.loads(args.config.read_text());rec,role,_source=validate(config,args.recording)
    if args.validate_only:
        print(json.dumps(dict(status='config_valid',recording_id=rec,role=role,raw_CSV_opened=False,outputs_created=False)));return
    build_recording(config,args.config_sha256,rec)


if __name__=='__main__':main()
