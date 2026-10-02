#!/usr/bin/env python3
"""Replay the completed Transformer branch from its saved full trajectories.

Ranks/quantiles and native quality are implemented independently here. PET is
recomputed with the canonical all-actor oracle (not an independent PET
definition); PL overlap uses the slab audit, separate from sampling guidance.
No fitting, trajectory modification, filtering or checkpoint selection occurs.
"""
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.data.scene_pet import scene_occupancy_pet
from pcontrol.generation.cdf_shape_context import CONTEXT_KEY, shape_from_reference
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research.audit_pair_overlap_intervals import pair_overlap_intervals
from pcontrol.research.evaluate_time_attention_pipeline import legacy_reference, score_cell
from pcontrol.research.pilot_relative_direct_p import existing_stop_cases
from pcontrol.research.train_time_attention_generator import make_model
from pcontrol.research.refine_natural_direct_p import state_hash

U = np.array([0., .05, .1, .25, .5, .75, .9, .95, 1.])


def rank(mass, nodes, y):
    """Direct two interpolations, with the two endpoint atoms explicit."""
    x = np.linspace(0., 4., len(mass)-1)
    levels = np.r_[mass[0], mass[0]+np.cumsum(mass[1:-1])]
    levels[-1] = 1.-mass[-1]
    value = float(np.interp(np.interp(y, x, levels), U, nodes))
    left, right = (0. if y == 0 else value), (1. if y == 4 else value)
    return dict(p_low=1.-right, p_up=1.-left, p_mid=1.-.5*(left+right))


def quantile(mass, nodes, u):
    """Explicit scan over the linear CDF pieces; no production inverse call."""
    x = np.linspace(0., 4., len(mass)-1)
    base = np.r_[mass[0], mass[0]+np.cumsum(mass[1:-1])]
    base[-1] = 1.-mass[-1]
    if u <= np.interp(base[0], U, nodes):
        return 0.
    extra = []
    for i in range(len(x)-1):
        if base[i+1] > base[i]:
            for v in U[(U > base[i]) & (U < base[i+1])]:
                extra.append(x[i]+(v-base[i])/(base[i+1]-base[i])*(x[i+1]-x[i]))
    points = np.unique(np.r_[x, extra])
    values = np.interp(np.interp(points, x, base), U, nodes)
    hits = np.flatnonzero(values >= u)
    if not len(hits):
        return 4.
    j = int(hits[0])
    if j == 0:
        return 0.
    return float(points[j-1]+(u-values[j-1])/(values[j]-values[j-1])*(points[j]-points[j-1]))


def midpoint_error_infimum(mass, nodes, requested):
    """Distribution-only infimum, not dynamical reachability or a success excuse."""
    lo = 1.-float(np.interp(1.-mass[-1], U, nodes))
    hi = 1.-float(np.interp(mass[0], U, nodes))
    if hi < lo-1e-12:
        raise ValueError('nonmonotone reference')
    continuous = max(lo-requested, requested-hi, 0.)
    return min(continuous, abs(rank(mass,nodes,0.)['p_mid']-requested),
               abs(rank(mass,nodes,4.)['p_mid']-requested))


def native_checks(f, dims, bounds, ego):
    n = len(dims)
    pairs = [(i,j) for i in range(n) for j in range(i+1,n)]
    overlap = np.stack([np.all(abs(f[:,i,:2]-f[:,j,:2]) <= (dims[i]+dims[j])/2, axis=1)
                        for i,j in pairs], axis=1)
    lower_edge=f[...,1]-dims[None,:,1]/2
    upper_edge=f[...,1]+dims[None,:,1]/2
    outside = np.maximum(bounds[0]-lower_edge,upper_edge-bounds[-1])
    acceleration = np.diff(f[...,2:],axis=0)/.04
    jerk = np.diff(acceleration,axis=0)/.04
    close = []
    for i,j in pairs:
        if ego in (i,j):
            continue
        edge = abs(f[:,i,:2]-f[:,j,:2])-(dims[i]+dims[j])/2
        close.append((edge[:,1] <= 0) & (np.maximum(edge[:,0],0.) < 1.))
    any_close = np.stack(close,axis=1).any(1) if close else np.zeros(175,bool)
    longest = run = 0
    for value in any_close:
        run = run+1 if value else 0
        longest = max(longest,run)
    return dict(all_pair_overlap_scene=bool(overlap.any()),all_pair_overlap_pair_frames=int(overlap.sum()),
        road_outside_scene=bool((outside>0).any()),road_outside_actor_frames=int((outside>0).sum()),
        negative_vx_frame_actor_fraction=float((f[...,2]<0).mean()),
        acceleration_vector_rms_mps2=float(np.sqrt(np.mean(np.sum(acceleration**2,axis=-1)))),
        jerk_vector_rms_mps3=float(np.sqrt(np.mean(np.sum(jerk**2,axis=-1)))),
        road_outside_max_m=float(np.maximum(outside,0.).max()),
        road_excess_over_initial_max_m=float(np.maximum(outside-np.maximum(outside[0],0.),0.).max()),
        background_pairs_long_gap_under1m_frame_fraction=float(any_close.mean()),
        background_pairs_long_gap_under1m_longest_sample_span_seconds=max(longest-1,0)*.04)


def verify_nested(value, seen):
    if isinstance(value,dict):
        if set(value)=={'path','sha256'}:
            key=(value['path'],value['sha256'])
            if key not in seen:
                io.verify_binding(value);seen.add(key)
        else:
            for item in value.values():verify_nested(item,seen)
    elif isinstance(value,list):
        for item in value:verify_nested(item,seen)


def run():
    names = ['reference/manifest.json','crossfit/manifest.json','generator_data/manifest.json',
             'generator/adaptation/result.json','generator/terminal/result.json','generation/results.json',
             'reference_chain_audit.json']
    bindings = {name:c.bind(c.OUTPUT/name) for name in names}
    docs = {name:c.json_file(b) for name,b in bindings.items()}
    seen=set()
    for doc in docs.values():
        verify_nested(doc,seen)
        if isinstance(doc.get('code_sha256'),dict):c.verify_sources(doc['code_sha256'])
    reference, labels, prepared, adapt, terminal, generation, first_audit = [docs[n] for n in names]
    if first_audit['status']!='pass' or first_audit['OOF_rows_replayed']!=9913:
        raise ValueError('complete teacher/CDF audit required')
    if any(x['reference_manifest']!=bindings[names[0]] for x in (labels,prepared,adapt,terminal,generation)):
        raise ValueError('reference version disagreement')
    if any(x['labels_manifest']!=bindings[names[1]] for x in (prepared,adapt,terminal)):
        raise ValueError('label version disagreement')
    if generation['training']!=bindings[names[4]] or terminal['adaptation']!=bindings[names[3]]:
        raise ValueError('training chain mismatch')
    if terminal['smoke'] or terminal['status']!='complete' or terminal['epochs_completed']!=3:
        raise ValueError('three actual terminal epochs required')
    logs=[json.loads(line) for line in (c.OUTPUT/'generator/terminal/epochs.jsonl').read_text().splitlines()]
    if len(logs)!=3 or any(r['updates']!=78 or r['base_loss_scenes']!=9913 or r['auxiliary_requests']!=3744
                           or r['forward_NFE']!=7800 or r['backward_NFE']!=7800 for r in logs):
        raise ValueError('full-FIT/full-DDIM training exposure missing')
    if terminal['initial_state_sha256']==terminal['final_state_sha256']:
        raise ValueError('terminal model did not change')
    if generation['checkpoint']!=terminal['snapshots']['3']['raw']:
        raise ValueError('checkpoint selection changed')
    # Check real checkpoint tensors as well as declared hashes/training traces.
    actual_states={}
    for label,binding in [('adaptation',adapt['checkpoint']),('terminal',generation['checkpoint'])]:
        checkpoint=torch.load(io.verify_binding(binding),map_location='cpu',weights_only=False)
        if checkpoint['reference_manifest']!=bindings[names[0]] or checkpoint['labels_manifest']!=bindings[names[1]]:
            raise ValueError('checkpoint header uses a different reference/label version')
        model=make_model(checkpoint['architecture'],checkpoint['state_dict'],torch.device('cpu'))
        actual_states[label]=state_hash(model)
        del model
    if (actual_states['adaptation']!=terminal['initial_state_sha256']
            or actual_states['terminal']!=terminal['final_state_sha256']):
        raise ValueError('checkpoint tensors do not replay actual training-state hashes')
    adapt_logs=[json.loads(line) for line in (c.OUTPUT/'generator/adaptation/epochs.jsonl').read_text().splitlines()]
    if (len(adapt_logs)!=adapt['epochs_completed'] or any(r['FIT_rows']!=9913 for r in adapt_logs)
            or adapt['optimizer_updates']!=78*len(adapt_logs)):
        raise ValueError('adaptation natural-data exposure changed')
    candidates=[(adapt['initial_validation']['mean'],0)]+[(r['STOP']['mean'],r['epoch']) for r in adapt_logs]
    if min(candidates)[1]!=adapt['best_epoch']:raise ValueError('STOP v-MSE selection does not replay')
    p=c.policy(generation['policy'])
    old=c.json_file(p['old_generation_result']);old_child=c.json_file(old['candidates']['background_guarded']['result'])
    cases={v['scene_id']:v for v in existing_stop_cases(old_child)}
    old_rows={(r['scene_id'],r['requested_p']):r for r in old_child['rows']}
    plugin=c.TimeAttentionRiskPlugin.from_manifest(bindings[names[0]],device='cpu');old_plugin=legacy_reference()
    errors=dict(rank=0.,PET=0.,quantile=0.,quality=0.,reference_forward=0.,condition=0.)
    reports=[];cross={k:[] for k in generation['cross_reference_matrix']};refs={};saved={}
    if len(generation['rows'])!=36 or len({(r['scene_id'],r['requested_p']) for r in generation['rows']})!=36:
        raise ValueError('incomplete generation denominator')
    for row in generation['rows']:
        sid=row['scene_id'];case=cases[sid];q=row['requested_p'];old_row=old_rows[sid,q]
        if sid not in saved:
            a=c.arrays(row['trajectory_artifact']);saved[sid]=a
            for key in ('history','dimensions','road_boundaries','ego_mask','agent_ids','initial_noise'):
                if not np.array_equal(a[key],case[key]):raise ValueError('changed paired '+key)
            if not np.array_equal(a['future_observed'],case['future']):raise ValueError('natural future changed')
            args=[case[k] for k in ('history','dimensions','road_boundaries','ego_mask','agent_mask')]
            ref=plugin.condition(*args);ore=old_plugin.condition(*args);refs[sid]=(ref,ore)
            for prefix,r in (('new',ref),('old',ore)):
                errors['reference_forward']=max(errors['reference_forward'],float(np.max(abs(a[prefix+'_reference_joint_masses']-r._masses[0]))),
                    float(np.max(abs(a[prefix+'_reference_row_nodes']-r._warp.row_nodes([r.num_agents])[0]))))
            pieces=c.StableTorchInverse(ref._masses,[ref.num_agents],warp=ref._warp,base_knots=ref._knots).compiled_pieces()[0].numpy()
            errors['condition']=max(errors['condition'],float(np.max(abs(a[CONTEXT_KEY]-np.asarray(shape_from_reference(ref),dtype=np.float32)))),
                                     float(np.max(abs(a[PIECES_KEY]-pieces))))
        a=saved[sid];f=a[row['array_key']];ref,ore=refs[sid];ei=int(np.flatnonzero(a['ego_mask'])[0])
        if f.shape!=(175,len(a['agent_ids']),4) or not np.isfinite(f).all() or not np.array_equal(f[0],a['history'][-1]):
            raise ValueError('incomplete, nonfinite or reanchored output')
        metric=scene_occupancy_pet(f,np.ones(f.shape[:2],bool),a['dimensions'],times=np.arange(175)*.04,ego_index=ei,
            sample_period=.04,window=(0.,6.96),cap_seconds=4.)
        y=metric['pet_value_seconds'];m=a['new_reference_joint_masses'];node=a['new_reference_row_nodes'];r=rank(m,node,y)
        errors['PET']=max(errors['PET'],abs(y-row['pet_seconds']))
        errors['rank']=max(errors['rank'],*(abs(r[k]-row['estimated_rank'][k]) for k in r))
        target=quantile(m,node,1.-q);errors['quantile']=max(errors['quantile'],abs(target-row['target_spec']['target_pet_seconds']))
        quality=native_checks(f,a['dimensions'],a['road_boundaries'],ei)
        for key,value in quality.items():
            source=row['quality'] if key in row['quality'] else row['scene_quality']
            if key in source:errors['quality']=max(errors['quality'],abs(float(value)-float(source[key])))
        intervals=pair_overlap_intervals(f,a['dimensions'],ei)
        if intervals!=row['PL_overlap_intervals']:raise ValueError('PL overlap audit replay failed')
        om=a['old_reference_joint_masses'];on=a['old_reference_row_nodes']
        old_y=old_row['pet_seconds']
        cross['old_output_old_CDF'].append(abs(rank(om,on,old_y)['p_mid']-q))
        cross['old_output_new_CDF'].append(abs(rank(m,node,old_y)['p_mid']-q))
        cross['new_output_old_CDF'].append(abs(rank(om,on,y)['p_mid']-q))
        cross['new_output_new_CDF'].append(abs(r['p_mid']-q))
        floor=midpoint_error_infimum(m,node,q)
        reports.append(dict(scene_id=sid,P=q,N=len(a['agent_ids']),PET=y,PET_target=target,P_error=abs(r['p_mid']-q),
            Fine=abs(r['p_mid']-q)<=.05,rank=r,distribution_only_error_infimum=floor,
            Fine_impossible_from_CDF_atoms_alone=floor>.05,whole_window_critical_actor=metric['critical_other_index'],
            whole_window_witness=metric['witness'],quality=quality,artifact=row['trajectory_artifact'],array_key=row['array_key']))
        print(json.dumps(dict(audited=sid,P=q,PET=y,P_error=abs(r['p_mid']-q))),flush=True)
    if max(errors.values())>2e-10:raise ValueError('numeric replay failed: '+str(errors))
    cells={k:score_cell(v) for k,v in cross.items()}
    for name,cell in cells.items():
        for key,value in cell.items():
            if abs(value-generation['cross_reference_matrix'][name][key])>2e-10:raise ValueError('cross-reference table mismatch')
    result=dict(status='pass',stage_bindings=bindings,hash_bindings_verified=len(seen),histories=12,requests=36,
        all_actors_preserved=True,paired_H_P_z_preserved=True,fixed_t0_preserved=True,
        new_reference_forward_and_actual_conditions_replayed=True,maximum_errors=errors,
        actual_checkpoint_state_hashes=actual_states,
        terminal_training=dict(epochs=3,updates=sum(r['updates'] for r in logs),natural_row_exposures=3*9913,
            auxiliary_requests=3*3744,full50step_forward_backward=True),
        cross_reference_matrix=cells,rows=reports,
        Fine_impossible_from_CDF_atoms_alone=sum(r['Fine_impossible_from_CDF_atoms_alone'] for r in reports),
        all36_remain_in_primary_denominator=True,
        PET_check='canonical all-actor oracle replay; not a separately defined metric',
        rank_quality_check='independent interpolation/inverse scan and native quality arithmetic',
        PL_check='slab interval audit, independent of sampling envelope; not smooth-spline certification',
        no_new_training_or_resampling=True,code_sha256=io.sha256(__file__))
    io.write_json(c.OUTPUT/'full_pipeline_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','stage_bindings')}),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2);torch.backends.mha.set_fastpath_enabled(False)
    run()
