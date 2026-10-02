"""History-selected ego scenes and native observed futures, without focal roles.

Selection finishes using history only. Native future observations are then
read for the fixed actors; missing/invalid observations stay NaN with a mask.
This module never generates, interpolates, fits a model, or replaces a scene
because its future or downstream PET label is incomplete.
"""
from collections import Counter
from dataclasses import dataclass
import hashlib

import numpy as np
import pandas as pd

from .clips import extract_context, _centres_and_states, RAW_COLUMNS


PROTOCOL='natural_history_selected_ego_scene_native_future_v1'


def scene_id(recording,ego_id,t0_frame):
    return f'highd{recording}_ego{int(ego_id)}_t0{int(t0_frame)}'


def select_scene_contexts(recording,*,salt,max_scenes=512):
    """Return chosen H-only contexts plus a complete selection-use ledger.

    The total count of history-eligible contexts is unknown after the quota is
    reached; unexamined candidates are reported rather than counted eligible.
    """
    if not isinstance(salt,str) or not salt or type(max_scenes) is not int or not 1<=max_scenes<=512:
        raise ValueError('explicit nonempty salt and quota in1..512 are required')
    grid=recording.tracks.loc[recording.tracks['frame']%25==0]
    ordered=[]
    for t0,ego in grid[['frame','id']].itertuples(index=False,name=None):
        t0,ego=int(t0),int(ego)
        digest=hashlib.sha256(f'{salt}|{recording.recording_id}|{ego}|{t0}'.encode()).hexdigest()
        ordered.append((digest,ego,t0))
    ordered.sort()
    frame_rows={int(frame):rows.set_index('id') for frame,rows in grid.groupby('frame',sort=False)}
    selected=[];failures=Counter();actor_rejections=Counter();attempted=0
    for rank,(digest,ego,t0) in enumerate(ordered):
        attempted+=1
        context,reason=extract_context(recording,t0,ego,frame_rows=frame_rows[t0],audit_counts=actor_rejections)
        if context is None:failures[str(reason)]+=1;continue
        if context.source_agent_ids[0]!=ego or list(context.source_agent_ids[1:])!=sorted(context.source_agent_ids[1:].tolist()):
            raise RuntimeError('scene identities must be ego0 and ascending observed background IDs')
        selected.append(dict(context=context,scene_id=scene_id(recording.recording_id,ego,t0),selection_hash=digest,candidate_rank=rank))
        if len(selected)==max_scenes:break
    audit=dict(recording_id=recording.recording_id,salt=salt,max_scenes=max_scenes,
        hash_template='salt|two_digit_recording|decimal_ego_id|decimal_t0_frame',
        candidate_grid='positive_raw_frame_divisible_by25',grid_actor_time_candidates=len(ordered),
        history_context_attempts=attempted,history_context_rejections=dict(failures),
        context_actor_rejections=dict(actor_rejections),selected_history_scenes=len(selected),
        quota_reached=len(selected)==max_scenes,unexamined_grid_candidates=len(ordered)-attempted,
        future_values_used_for_selection=False,future_completeness_used_for_selection=False,
        future_semantics_or_PET_used_for_selection=False,future_critical_actor_used_in_inputs=False,
        selected_future_failure_triggers_replacement=False,
        raw_csv_decoder_may_have_decoded_all_recording_rows=True)
    return selected,audit


@dataclass(frozen=True)
class ObservedEgoScene:
    scene_id: str
    recording_id: str
    t0_frame: int
    ego_id: int
    agent_ids: np.ndarray
    history: np.ndarray  # [13,N,4], physical centres; ego0, remaining raw IDs ascending
    dimensions: np.ndarray  # [N,2], actual longitudinal length/lateral width
    road_boundaries: np.ndarray
    history_frame_ids: np.ndarray
    future_frame_ids: np.ndarray
    future_native: np.ndarray  # [175,N,4], NaN at all unobserved/invalid samples
    future_observed_mask: np.ndarray  # [175,N]
    observation_audit: dict


def extract_scene(recording,t0_frame,ego_id,*,context=None):
    """Return (scene,None), or an H-only rejection; never reject missing future.

    Call with a selected context to avoid repeating its history read. Reindexing
    the native raw grid inserts NaNs for absent rows, not synthetic states.
    """
    t0,ego=int(t0_frame),int(ego_id)
    if context is None:
        context,reason=extract_context(recording,t0,ego)
        if context is None:return None,reason
    if (context.recording_id,context.t0_frame,context.ego_id)!=(recording.recording_id,t0,ego):
        raise ValueError('selected context does not match the requested scene')
    ids=context.source_agent_ids
    if ids[0]!=ego or len(ids)<3 or len(set(ids))!=len(ids) or ids[1:].tolist()!=sorted(ids[1:].tolist()):
        raise ValueError('scene must contain fixed ego0 plus all history-selected sorted IDs, no focal')
    frames=np.arange(t0,t0+175,dtype=np.int64);n=len(ids)
    requested=pd.MultiIndex.from_product([ids,frames],names=['id','frame'])
    positions=recording._by_id_frame.index.get_indexer(requested)
    exists=positions>=0;raw=np.full((n*175,6),np.nan,dtype=np.float64)
    raw[exists]=recording._by_id_frame.iloc[positions[exists]][RAW_COLUMNS].to_numpy(dtype=np.float64)
    raw=raw.reshape(n,175,6).transpose(1,0,2);exists=exists.reshape(n,175).T
    finite=np.isfinite(raw).all(-1)
    # Match the frozen H-only extractor's metadata-consistency tolerance.
    # This is not a geometry expansion and does not alter positions or sizes.
    dimensions_ok=np.isclose(raw[...,2:4],context.dimensions[None],rtol=0,atol=1e-9).all(-1)
    mask=exists&finite&dimensions_ok
    states=_centres_and_states(raw,context.raw_anchor_center,context.forward_sign)
    states[~mask]=np.nan
    if not mask[0].all() or not np.array_equal(states[0],context.history[-1]):
        raise RuntimeError('native t0 does not exactly reproduce the frozen all-actor history t0')
    actors=[]
    for j,actor in enumerate(ids):
        metadata=recording.track_metadata[int(actor)];missing=~exists[:,j]
        counts=dict(before_source_track=int((missing&(frames<metadata.initial_frame)).sum()),
            after_source_track=int((missing&(frames>metadata.final_frame)).sum()),
            missing_source_frame=int((missing&(frames>=metadata.initial_frame)&(frames<=metadata.final_frame)).sum()),
            nonfinite_source_state=int((exists[:,j]&~finite[:,j]).sum()),
            inconsistent_dimensions=int((exists[:,j]&finite[:,j]&~dimensions_ok[:,j]).sum()))
        actors.append(dict(actor_id=int(actor),scene_actor_index=j,observed_frames=int(mask[:,j].sum()),
            requested_frames=175,missing_reason_counts=counts,first_source_frame=metadata.initial_frame,last_source_frame=metadata.final_frame))
    audit=dict(actors=actors,all_actor_future_complete=bool(mask.all()),observed_actor_frames=int(mask.sum()),
        requested_actor_frames=int(mask.size),ego_future_complete=bool(mask[:,0].all()),
        fixed_t0_actor_ids=True,future_critical_actor_not_selected_as_input=True,
        history_dt_seconds=.08,future_native_dt_seconds=.04,future_duration_seconds=6.96,
        interpolation=False,extrapolation=False,future_missing_values='NaN_with_false_mask')
    return ObservedEgoScene(scene_id(recording.recording_id,ego,t0),recording.recording_id,t0,ego,
        ids.copy(),context.history.copy(),context.dimensions.copy(),context.canonical_boundaries.copy(),
        context.history_frame_ids.copy(),frames,states,mask,audit),None


def selection_cohort_rows(selected):
    """JSON-safe H-only cohort rows; persist these before extracting futures."""
    return [dict(scene_id=row['scene_id'],recording_id=row['context'].recording_id,t0_frame=row['context'].t0_frame,
        ego_id=row['context'].ego_id,agent_ids=row['context'].source_agent_ids.tolist(),
        actual_context_agents=row['context'].num_agents,history_frame_ids=row['context'].history_frame_ids.tolist(),
        history_sha256=hashlib.sha256(row['context'].history.tobytes()).hexdigest(),
        selection_hash=row['selection_hash'],candidate_rank=row['candidate_rank'],
        membership_and_selection_use_history_only=True) for row in selected]


def scene_arrays(scenes,labels,selected,*,role,metadata):
    """All selected scenes, including censored/invalid labels, in ragged arrays."""
    if len(scenes)!=len(labels) or len(scenes)!=len(selected):raise ValueError('scene/label/cohort rows must align exactly')
    n=len(scenes);offsets=np.r_[0,np.cumsum([len(scene.agent_ids) for scene in scenes])].astype(np.int64)
    def concatenate(field,empty_shape,transform=lambda value:value):
        return np.concatenate([transform(getattr(scene,field)) for scene in scenes]) if n else np.empty(empty_shape,np.float64)
    max_boundaries=max((len(scene.road_boundaries) for scene in scenes),default=0)
    boundaries=np.zeros((n,max_boundaries),np.float64);boundary_mask=np.zeros((n,max_boundaries),bool);road=[]
    for i,scene in enumerate(scenes):
        values=scene.road_boundaries;boundaries[i,:len(values)]=values;boundary_mask[i,:len(values)]=True
        lane=int(np.searchsorted(values,scene.history[-1,0,1],side='right')-1)
        if lane<0 or lane>=len(values)-1:raise ValueError('ego t0 is outside the observed carriageway')
        road.append([values[0],values[lane],values[lane+1],values[-1]])
    import json
    result=dict(scene_id=np.asarray([scene.scene_id for scene in scenes],dtype='U96'),
        recording_id=np.asarray([scene.recording_id for scene in scenes],dtype='U2'),
        ego_id=np.asarray([scene.ego_id for scene in scenes],dtype=np.int64),t0_frame=np.asarray([scene.t0_frame for scene in scenes],dtype=np.int64),
        role=np.full(n,role,dtype='U5'),num_agents=np.diff(offsets),offsets=offsets,
        agent_ids=np.concatenate([scene.agent_ids for scene in scenes]) if n else np.empty(0,np.int64),
        dimensions_agents=concatenate('dimensions',(0,2)),history_agents=concatenate('history',(0,13,4),lambda x:x.transpose(1,0,2)),
        future_native_agents=concatenate('future_native',(0,175,4),lambda x:x.transpose(1,0,2)),
        future_observed_mask_agents=np.concatenate([scene.future_observed_mask.T for scene in scenes]) if n else np.empty((0,175),bool),
        history_frame_ids=np.stack([scene.history_frame_ids for scene in scenes]) if n else np.empty((0,13),np.int64),
        future_frame_ids=np.stack([scene.future_frame_ids for scene in scenes]) if n else np.empty((0,175),np.int64),
        road_geometry=np.asarray(road,dtype=np.float64).reshape(n,4),carriageway_boundaries=boundaries,carriageway_boundary_mask=boundary_mask,
        selection_hash=np.asarray([row['selection_hash'] for row in selected],dtype='U64'),
        selection_candidate_rank=np.asarray([row['candidate_rank'] for row in selected],dtype=np.int64),
        pet_value=np.asarray([float(label['pet_value_seconds']) if label['point_identified'] else np.nan for label in labels],dtype=np.float64),
        point_identified=np.asarray([label['point_identified'] for label in labels],dtype=bool),
        complete_horizon=np.asarray([scene.future_observed_mask.all() for scene in scenes],dtype=bool),
        label_interval_seconds=np.asarray([label['label_interval_seconds'] for label in labels],dtype=np.float64).reshape(n,2),
        observed_min_seconds=np.asarray([label['observed_min_seconds'] for label in labels],dtype=np.float64),
        observed_min_capped_seconds=np.asarray([label['observed_min_capped_seconds'] for label in labels],dtype=np.float64),
        label_status=np.asarray([label['status'] for label in labels],dtype='U64'),
        metadata_json=np.asarray(json.dumps(metadata,sort_keys=True,allow_nan=False)))
    if len(np.unique(result['scene_id']))!=n:raise ValueError('duplicate ego/t0 scene in selected cohort')
    return result
