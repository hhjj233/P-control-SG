"""Complete natural scenes for the final protocol, behind an explicit read gate.

Original train-only loaders are unchanged. Production final CSV access starts
only after user approval AND a completed execution freeze. Pure extraction can
be tested on already authorized development recordings without such a grant.
"""
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

from pcontrol.data.highd import (
    RawRecording,TrackMetadata,TRACK_COLUMNS,META_COLUMNS,HighDContractError,
    recording_id,_lane_markings)
from pcontrol.data.clips import extract_context
from pcontrol.data.scenes import scene_id,selection_cohort_rows,extract_scene,scene_arrays
from pcontrol.data.scene_pet import scene_occupancy_pet,PROTOCOL as PET_PROTOCOL
from pcontrol.data.complete_scene_view import complete_eligibility,validate_complete_labels
from pcontrol.publication_pipeline.final_validation_protocol import (
    require_approval,verified_json,resolve,bind)

DATA_PROTOCOL='final_complete_natural_recording_v1'


def load_authorized_final_recording(policy_binding, freeze_binding, approval_binding, *, scope, recording):
    permission=require_approval(policy_binding,freeze_binding,approval_binding,scope=scope)
    rec=recording_id(recording)
    if rec not in permission.recording_ids: raise PermissionError('recording outside approved evaluation scope')
    p=verified_json(policy_binding);root=resolve(p['raw_root'])
    paths={key:root/f'{rec}_{key}.csv' for key in ('tracks','tracksMeta','recordingMeta')}
    # No code above this line may stat or read a raw highD file.
    sources={key:bind(path) for key,path in paths.items()}
    with paths['recordingMeta'].open(newline='',encoding='utf-8-sig') as handle: rows=list(csv.DictReader(handle))
    if len(rows)!=1 or recording_id(rows[0]['id'])!=rec or float(rows[0]['frameRate'])!=25.:
        raise HighDContractError('one matching 25Hz recording metadata row required')
    record=rows[0]
    tracks=pd.read_csv(paths['tracks'],usecols=list(TRACK_COLUMNS))
    table=pd.read_csv(paths['tracksMeta'],usecols=list(META_COLUMNS))
    if table['id'].duplicated().any():raise HighDContractError('duplicate tracksMeta actor')
    metadata={}
    for row in table.to_dict('records'):
        ints=np.asarray([row[k] for k in ('id','initialFrame','finalFrame','drivingDirection')],dtype=np.float64)
        if not np.isfinite(ints).all() or not np.equal(ints,np.floor(ints)).all():
            raise HighDContractError('integer actor/frame/direction metadata required')
        actor,first,last,direction=map(int,ints);length,width=float(row['width']),float(row['height'])
        if actor<=0 or first<=0 or last<first or direction not in (1,2):raise HighDContractError('invalid actor metadata')
        if not np.isfinite([length,width]).all() or min(length,width)<=0:raise HighDContractError('invalid dimensions')
        metadata[actor]=TrackMetadata(first,last,length,width,str(row['class']),direction)
    result=RawRecording(rec,tracks,metadata,25.,_lane_markings(record['upperLaneMarkings']),
        _lane_markings(record['lowerLaneMarkings']),{k:str(v) for k,v in paths.items()},
        'explicitly_authorized_final_highd_csv_no_training_permission')
    for key,path in paths.items():
        if bind(path)!=sources[key]:raise RuntimeError('source CSV changed during decode')
    return result,sources,permission


def select_history_cohort(recording, *, quota, salt):
    """Same H-only identity hash/grid as development, without a 512-row ceiling."""
    if type(quota) is not int or not 1<=quota<=4096 or not isinstance(salt,str) or not salt:
        raise ValueError('explicit fixed quota in 1..4096 and nonempty salt required')
    grid=recording.tracks.loc[recording.tracks['frame']%25==0]
    ordered=[]
    for frame,ego in grid[['frame','id']].itertuples(index=False,name=None):
        frame,ego=int(frame),int(ego)
        score=hashlib.sha256(f'{salt}|{recording.recording_id}|{ego}|{frame}'.encode()).hexdigest()
        ordered.append((score,ego,frame))
    ordered.sort();frames={int(k):v.set_index('id') for k,v in grid.groupby('frame',sort=False)}
    selected=[];failures=Counter();actors=Counter();attempted=0
    for rank,(score,ego,t0) in enumerate(ordered):
        attempted+=1
        context,reason=extract_context(recording,t0,ego,frame_rows=frames[t0],audit_counts=actors)
        if context is None:failures[str(reason)]+=1;continue
        selected.append(dict(context=context,scene_id=scene_id(recording.recording_id,ego,t0),
            selection_hash=score,candidate_rank=rank))
        if len(selected)==quota:break
    return selected,dict(recording_id=recording.recording_id,quota=quota,salt=salt,
        grid_candidates=len(ordered),attempted=attempted,selected=len(selected),
        unexamined_candidates=len(ordered)-attempted,history_rejections=dict(failures),actor_rejections=dict(actors),
        raw_decoder_may_have_decoded_future_rows=True,future_values_or_completeness_used_for_selection=False,
        PET_or_model_predictions_used_for_selection=False,replacement_after_incomplete=False)


def json_safe(value):
    if isinstance(value,np.generic):return json_safe(value.item())
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    return value


def write_once(path,value):
    with Path(path).open('x') as handle:json.dump(json_safe(value),handle,indent=2,sort_keys=True,allow_nan=False)
    return bind(path)


def extract_complete_recording(recording, selected, *, output_dir, role, selection_audit, provenance):
    """Commit H cohort before native-future eligibility/labels, never refill it."""
    if role not in ('VAL','TEST','R18','STOP'):raise ValueError('final role or explicit development rehearsal required')
    directory=Path(output_dir);directory.mkdir(parents=True,exist_ok=False)
    cohort=selection_cohort_rows(selected)
    cohort_binding=write_once(directory/'history_cohort_before_future.json',dict(protocol=DATA_PROTOCOL,
        role=role,recording_id=recording.recording_id,scenes=cohort,selection=selection_audit,
        provenance=provenance,future_eligibility_or_PET_started=False))
    complete=[];labels=[];kept=[];indices=[];counts=Counter();ledger=[]
    for index,item in enumerate(selected):
        context=item['context']
        scene,reason=extract_scene(recording,context.t0_frame,context.ego_id,context=context)
        if scene is None:raise RuntimeError('history-selected scene disappeared: '+str(reason))
        eligible=bool(scene.future_observed_mask.all());label=None
        if eligible:
            label=scene_occupancy_pet(scene.future_native,scene.future_observed_mask,scene.dimensions,
                times=np.arange(175)*.04,ego_index=0,sample_period=.04,window=(0.,6.96),cap_seconds=4.)
            if not label['complete'] or not label['point_identified'] or not np.isfinite(label['pet_value_seconds']):
                raise RuntimeError('complete scene has no valid point target; do not drop it')
            complete.append(scene);labels.append(label);kept.append(item);indices.append(index)
        counts['H_selected']+=1;counts['complete']+=eligible;counts['incomplete']+=not eligible
        if eligible:counts[label['status']]+=1
        ledger.append(dict(cohort[index],cohort_index=index,complete_horizon=eligible,PET_computed=eligible,
            natural_observation_audit=scene.observation_audit,label=json_safe(label),replaced=False))
    ledger_binding=write_once(directory/'eligibility.json',ledger)
    metadata=dict(protocol=DATA_PROTOCOL,role=role,source_recording=recording.recording_id,
        provenance=provenance,history_cohort=cohort_binding,metric_protocol=PET_PROTOCOL,
        natural_only=True,simulated_or_filled_future=False,reference_or_generator_training=False,
        complete_population_only=True,incomplete_PET_measured=False)
    arrays=scene_arrays(complete,labels,kept,role=role,metadata=metadata)
    arrays['cohort_row_index']=np.asarray(indices,dtype=np.int64)
    eligible=complete_eligibility(arrays);validate_complete_labels(arrays,eligible)
    if not eligible.all():raise RuntimeError('incomplete future entered final point-label data')
    file=directory/'complete_scenes.npz'
    with file.open('xb') as handle:np.savez_compressed(handle,**arrays)
    report=dict(protocol=DATA_PROTOCOL,status='complete',role=role,recording_id=recording.recording_id,
        data=bind(file),history_cohort=cohort_binding,eligibility=ledger_binding,counts=dict(counts),
        complete_scene_ids=arrays['scene_id'].tolist(),complete_cohort_rows=indices,
        num_agents=arrays['num_agents'].tolist(),provenance=provenance,
        all_incomplete_candidates_retained_in_ledger=True,selected_rows_replaced=False,
        final_performance_measurement=False)
    report_binding=write_once(directory/'recording_result.json',report)
    return report,report_binding
