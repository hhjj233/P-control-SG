"""Fixed identity/N-stratified evaluation and descriptive rollout diagnostics.

No training dependency is changed. Selection reads identity/count members only,
freezes twelve keys in memory, and only then decodes the selected real futures.
These purposeful strata are not an unweighted representative population sample.
Quality diagnostics are not a safety certificate; ADE is against one observed
future and is neither minK nor a measure of multimodal plausibility/diversity.
"""
import hashlib
import json

import numpy as np

from pcontrol.data.complete_scene_view import verify_binding
from pcontrol.data.expanded_scene_view import validate_manifest
from pcontrol.generation.data import NaturalTrajectorySource, DATA_SHA256
from pcontrol.research.natural_scene_visual_data import load_scene_case


PROTOCOL = 'natural_decoupled_diffusion_case_evaluation_v1'
STRATA = (('N3_5',3,5),('N6_8',6,8),('N9_plus',9,np.inf))
SALT = 'natural_decoupled_diffusion_eval_v1'


def _identity_rows(dataset_binding, role):
    if role not in ('STOP','AUDIT'):
        raise PermissionError('evaluation selection is explicitly STOP or AUDIT, never FIT/CAL')
    if dataset_binding.get('sha256') != DATA_SHA256:
        raise ValueError('evaluation requires the unchanged frozen complete-scene source')
    manifest = json.loads(verify_binding(dataset_binding).read_text())
    _base, source = validate_manifest(manifest)
    source_path = verify_binding(manifest['source_scene_manifest'])
    rows = []
    keys = ('scene_id','recording_id','role','offsets','num_agents')
    for recording, entry in sorted(manifest['recordings'].items()):
        if entry['role'] != role:
            continue
        if (entry['source_kind'] != 'unchanged_original_complete_view'
                or entry['data'] != source['recordings'][recording]['artifacts']['data']):
            raise ValueError('selected role must retain original frozen complete rows')
        path = verify_binding(entry['data'],source_path.parent/(recording+'.npz'))
        with np.load(path,allow_pickle=False) as archive:
            arrays = {key:archive[key] for key in keys}
        verify_binding(entry['data'],path)
        if not np.all(arrays['role']==role) or not np.all(arrays['recording_id']==recording):
            raise PermissionError('identity shard role differs from the requested evaluation role')
        for index,sid in zip(entry['selected_rows'],entry['selected_scene_ids']):
            count = int(arrays['offsets'][index+1]-arrays['offsets'][index])
            if str(arrays['scene_id'][index]) != sid or count != int(arrays['num_agents'][index]) or count < 3:
                raise ValueError('frozen identity/count row changed')
            rows.append(dict(scene_id=sid,recording_id=recording,source_row=int(index),num_agents=count,role=role))
    expected = 538 if role=='STOP' else 487
    if len(rows) != expected or len({r['scene_id'] for r in rows}) != expected:
        raise ValueError('evaluation role denominator or scene uniqueness changed')
    return rows


def _select_identities(rows, *, per_stratum, salt):
    if type(per_stratum) is not int or not 1<=per_stratum<=4 or not isinstance(salt,str) or not salt:
        raise ValueError('bounded evaluation requires1..4 scenes per stratum and explicit nonempty salt')
    if len({row['scene_id'] for row in rows}) != len(rows):
        raise ValueError('scene identities must be unique before selection')
    selected = []
    for name,low,high in STRATA:
        eligible = [row for row in rows if low<=row['num_agents']<=high]
        if len(eligible)<per_stratum:
            raise ValueError('insufficient support for the fixed '+name+' evaluation stratum')
        ordered = sorted(eligible,key=lambda row:(hashlib.sha256((salt+'|'+row['scene_id']).encode()).hexdigest(),row['scene_id']))
        for rank,row in enumerate(ordered[:per_stratum]):
            selected.append(dict(row,stratum=name,selection_hash=hashlib.sha256((salt+'|'+row['scene_id']).encode()).hexdigest(),
                selection_rank_within_stratum=rank,eligible_stratum_scenes=len(eligible),selection_salt=salt,
                selection_uses='identity_and_actual_history_agent_count_only',future_used_for_selection=False))
    return selected


def select_role_cases(dataset_binding, role, per_stratum=4, salt=SALT):
    """Return twelve full unpadded cases (or3*per_stratum), never tune on AUDIT.

    This function makes no parameter choice. Its caller is responsible for
    freezing all methods/settings before requesting the AUDIT role. Reading
    metadata for authentication does not decode another role's observations.
    """
    selected = _select_identities(_identity_rows(dataset_binding,role),per_stratum=per_stratum,salt=salt)
    # Barrier: no selected future loader is constructed/called above this line.
    source = NaturalTrajectorySource(dataset_binding,role='STOP') if role=='STOP' else None
    lookup = {} if source is None else {sid:i for i,(_rec,_row,sid) in enumerate(source.rows)}
    cases = []
    for row in selected:
        if role=='STOP':
            example = source[lookup[row['scene_id']]]
            f = example['features']
            case = dict(history=f['history'],future=example['future_observed'],dimensions=f['dimensions'],
                road_boundaries=f['road_boundaries'],ego_mask=f['ego_mask'],agent_ids=example['agent_ids'],
                source_bindings=dict(dataset_manifest=dataset_binding,shard=example['source_binding']))
        else:
            observed = load_scene_case(dataset_binding,row)
            ego_mask = np.zeros(row['num_agents'],bool);ego_mask[0]=True
            case = {key:observed[key] for key in ('history','future','dimensions','road_boundaries','agent_ids','source_bindings')}
            case['ego_mask'] = ego_mask
        n = row['num_agents']
        if (np.shape(case['history']) != (13,n,4) or np.shape(case['future']) != (175,n,4)
                or np.shape(case['dimensions']) != (n,2) or np.shape(case['agent_ids']) != (n,)
                or not np.array_equal(case['history'][-1],case['future'][0])):
            raise ValueError('selected future must preserve all fixed actors and observed t0')
        case.update(row,agent_mask=np.ones(n,bool),future_dt=.04,
                    selected_identity_frozen_before_future_decode=True,
                    future_is_actual_observation=True,generator_used_for_case_selection=False)
        cases.append(case)
    return cases


def quality_metrics(future, case):
    """Descriptive metrics for ONE proposed future, including every real actor.

    Overlap tests are closed AABBs at175 discrete sample times, not verified
    collisions or a continuous-time collision detector. Road outside means the
    width footprint extends beyond the actual outer carriageway boundaries;
    crossing an internal lane line is not itself an off-road violation.
    Acceleration/jerk are native forward differences of the supplied velocity.
    RMS below is sqrt(mean(vector squared magnitude)), not componentwise RMS.
    """
    f, truth = np.asarray(future,dtype=np.float64),np.asarray(case['future'],dtype=np.float64)
    dims,bounds = np.asarray(case['dimensions'],dtype=np.float64),np.asarray(case['road_boundaries'],dtype=np.float64)
    if (f.ndim!=3 or f.shape[0]!=175 or f.shape[2]!=4 or f.shape[1]<2 or truth.shape!=f.shape
            or dims.shape!=(f.shape[1],2) or bounds.ndim!=1 or len(bounds)<2
            or not all(np.isfinite(x).all() for x in (f,truth,dims,bounds)) or np.any(dims<=0)
            or np.any(np.diff(bounds)<=0) or float(case.get('future_dt',.04))!=.04):
        raise ValueError('complete finite matching F175 scenes, physical dimensions and road boundaries required')
    n=f.shape[1]
    if ('agent_mask' in case and (np.shape(case['agent_mask'])!=(n,) or not np.asarray(case['agent_mask']).all())):
        raise ValueError('quality metrics do not silently discard padded or missing actors')
    ego=np.asarray(case.get('ego_mask',np.arange(n)==0))
    if ego.shape!=(n,) or ego.dtype!=np.bool_ or ego.sum()!=1:
        raise ValueError('exactly one ego required for ego-specific displacement diagnostics')
    ei=int(np.flatnonzero(ego)[0]); i,j=np.triu_indices(n,k=1)
    separation=np.abs(f[:,i,:2]-f[:,j,:2]); half=.5*(dims[i]+dims[j])
    overlap=np.all(separation<=half[None],axis=-1)
    overlap_frame=overlap.any(1)
    road_outside=(f[...,1]-dims[None,:,1]/2<bounds[0]) | (f[...,1]+dims[None,:,1]/2>bounds[-1])
    negative=f[...,2]<0
    acceleration=np.diff(f[...,2:4],axis=0)/.04
    jerk=np.diff(acceleration,axis=0)/.04
    anorm=np.linalg.norm(acceleration,axis=-1);jnorm=np.linalg.norm(jerk,axis=-1)
    displacement=np.linalg.norm(f[...,:2]-truth[...,:2],axis=-1)
    interval_velocity=np.diff(f[...,:2],axis=0)/.04
    midpoint_velocity=.5*(f[:-1,:,2:4]+f[1:,:,2:4])
    consistency=np.linalg.norm(interval_velocity-midpoint_velocity,axis=-1)
    ego_pairs=(i==ei)|(j==ei)
    return dict(protocol=PROTOCOL,num_agents=n,frames=175,dt_seconds=.04,
        all_pair_overlap_scene=bool(overlap.any()),
        all_pair_overlap_frame_fraction=float(overlap_frame.mean()),
        all_pair_overlap_pair_frame_fraction=float(overlap.mean()),
        all_pair_overlap_frames=int(overlap_frame.sum()),
        all_pair_overlap_pair_frames=int(overlap.sum()),
        pair_frame_denominator=int(overlap.size),
        ego_overlap_scene=bool(overlap[:,ego_pairs].any()),
        ego_overlap_frame_fraction=float(overlap[:,ego_pairs].any(1).mean()),
        road_outside_scene=bool(road_outside.any()),
        road_outside_frame_actor_fraction=float(road_outside.mean()),
        road_outside_actor_frames=int(road_outside.sum()),frame_actor_denominator=int(road_outside.size),
        negative_vx_scene=bool(negative.any()),negative_vx_frame_actor_fraction=float(negative.mean()),
        acceleration_vector_rms_mps2=float(np.sqrt(np.mean(anorm**2))),acceleration_max_mps2=float(anorm.max()),
        jerk_vector_rms_mps3=float(np.sqrt(np.mean(jnorm**2))),jerk_max_mps3=float(jnorm.max()),
        position_velocity_consistency_rms_mps=float(np.sqrt(np.mean(consistency**2))),
        ADE_m=float(displacement.mean()),FDE_m=float(displacement[-1].mean()),
        ego_ADE_m=float(displacement[:,ei].mean()),ego_FDE_m=float(displacement[-1,ei]),
        anchor_xy_max_error_m=float(np.linalg.norm(f[0,:,:2]-truth[0,:,:2],axis=-1).max()),
        anchor_velocity_max_error_mps=float(np.linalg.norm(f[0,:,2:4]-truth[0,:,2:4],axis=-1).max()),
        ADE_is_minK=False,ADE_reference='one_observed_future_not_plausibility_or_diversity',
        all_pair_overlap_is_verified_collision=False,continuous_time_collision_assessed=False,
        lane_change_internal_boundary_crossing_is_not_offroad=True,
        complete_fixed_roster_used=True,proposal_modified=False)
