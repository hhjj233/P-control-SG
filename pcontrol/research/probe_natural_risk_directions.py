#!/usr/bin/env python3
"""Bounded FIT128 geometry-direction precheck, never CDF fitting or training.

Use exactly the previously frozen basis-diagnostic identities. Decode stored
natural clean coefficients, borrow only the frozen active-witness geometry
kernel, and test fixed small coefficient perturbations. Perturbations are
diagnostic counterfactual arrays, not observed traffic or new training labels.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))

import numpy as np
import torch

from pcontrol.data.complete_scene_view import verify_binding,resolve,sha256
from pcontrol.data.expanded_scene_view import ExpandedSceneReferenceView
from pcontrol.data.scene_pet import scene_occupancy_pet
from pcontrol.generation.data import load_training_pack
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.risk_guidance import TorchTrajectoryDecoder
from pcontrol.plugins.risk_plugin import RiskReference


PROTOCOL='natural_fixed_FIT128_local_risk_direction_probe_v1'
DATA_SHA='b10d412b54cf6a94c82639362515b4fff62305d8e448c5c6dbcd3eaa00fad408'
RMS_RADII=(1e-4,1e-3,1e-2)
NOISE_SALT='natural_risk_direction_probe_v1'
CODE=('pcontrol/research/probe_natural_risk_directions.py','pcontrol/data/scene_pet.py',
      'pcontrol/plugins/risk_plugin.py','pcontrol/generation/risk_guidance.py',
      'pcontrol/generation/trajectory_basis.py','pcontrol/generation/data.py',
      'pcontrol/data/expanded_scene_view.py','pcontrol/data/complete_scene_view.py')


class GeometryOnlyAdapter:
    """Duck-type adapter for the existing geometry method, with NO CDF object."""
    active_witness_pet=RiskReference.active_witness_pet

    def __init__(self,dimensions,anchors):
        self._dimensions=np.asarray(dimensions,dtype=np.float64).copy()
        self.anchors=np.asarray(anchors,dtype=np.float64).copy()
        self._container_agents=len(self._dimensions);self._valid_indices=np.arange(self._container_agents)
        self._ego_index=0

    def _candidate(self,candidate):
        if isinstance(candidate,torch.Tensor):candidate=candidate.detach().cpu().numpy()
        values=np.asarray(candidate,dtype=np.float64)
        if values.shape!=(175,self._container_agents,4) or not np.isfinite(values).all():
            raise ValueError('complete finite fixed-roster diagnostic future required')
        error=float(np.max(np.abs(values[0]-self.anchors)))
        if error>1e-5:raise ValueError('diagnostic future changed actual t0')
        return values,error

    def score_future(self,candidate):
        future,_=self._candidate(candidate)
        metric=scene_occupancy_pet(future,np.ones(future.shape[:2],bool),self._dimensions,
                                  times=np.arange(175)*.04,ego_index=0)
        return dict(pet_seconds=metric['pet_value_seconds'],pet_raw_seconds=metric['observed_min_seconds'],
                    metric=metric,CDF_or_rank_computed=False)


def _signature(metric):
    witness=metric.get('witness')
    return None if witness is None else [witness['other_index'],witness['ego_segment_frames'],witness['other_segment_frames']]


def probe_direction(coefficient,anchors,dimensions,normalizer,scene_id):
    coefficient=np.asarray(coefficient,dtype=np.float64)
    decoder=TorchTrajectoryDecoder(TrajectoryBasis(coefficient.shape[1]),normalizer,anchors,device='cpu')
    adapter=GeometryOnlyAdapter(dimensions,anchors)
    state=torch.tensor(coefficient,dtype=torch.float64,requires_grad=True)
    future=decoder(state)
    active=adapter.active_witness_pet(future)
    score=active.get('exact_score')
    if score is None:score=adapter.score_future(future)
    raw=score['pet_raw_seconds'];reconstructed=score['pet_seconds']
    out=dict(supported=bool(active['supported']),reason=active['reason'],
        reconstructed_PET_seconds=float(reconstructed),reconstructed_raw_PET_seconds=float(raw),
        reconstruction_status=score['metric']['status'],witness=score['metric']['witness'],
        coefficient_gradient=None,unit_coefficient_direction=None,perturbations=[],
        geometry_only=True,CDF_model_loaded=False,risk_rank_or_new_p_label_generated=False)
    if not active['supported']:return out
    gradient=torch.autograd.grad(active['value'],state)[0].detach().numpy()
    norm=float(np.linalg.norm(gradient))
    if not np.isfinite(gradient).all() or norm<=1e-12:
        out.update(supported=False,reason='flat_or_nonfinite_coefficient_risk_gradient');return out
    dimensions_count=coefficient.size
    seed=int.from_bytes(hashlib.sha256((NOISE_SALT+'|'+scene_id).encode()).digest()[:8],'little')%(2**32)
    rng=np.random.default_rng(seed);random=rng.standard_normal(coefficient.shape)
    directions=dict(gradient=gradient/norm*np.sqrt(dimensions_count),
                    fixed_random=random/np.linalg.norm(random)*np.sqrt(dimensions_count))
    baseline=future.detach().numpy();signature=_signature(score['metric'])
    for kind,direction in directions.items():
        for radius in RMS_RADII:
            for sign in (-1,1):
                delta=sign*radius*direction
                with torch.no_grad():changed=decoder(torch.tensor(coefficient+delta,dtype=torch.float64)).numpy()
                measured=adapter.score_future(changed);value=measured['pet_raw_seconds']
                predicted_change=float(np.sum(gradient*delta))
                observed_change=float(value-raw) if np.isfinite(value) else None
                informative=abs(predicted_change)>1e-10
                error=None if observed_change is None else abs(observed_change-predicted_change)
                out['perturbations'].append(dict(direction=kind,rms_radius=radius,sign=sign,
                    normalized_coefficient_L2_radius=float(np.linalg.norm(delta)),
                    predicted_delta_PET_seconds=predicted_change,actual_delta_PET_seconds=observed_change,
                    informative_predicted_change=informative,
                    sign_agrees=None if not informative or observed_change is None else bool(observed_change*predicted_change>0),
                    first_order_absolute_error_seconds=error,
                    first_order_relative_error=None if error is None else error/max(abs(predicted_change),1e-8),
                    positive_uncapped_regime_preserved=bool(np.isfinite(value) and 0<value<4),
                    same_witness_actor_and_segments=bool(_signature(measured['metric'])==signature),
                    perturbed_raw_PET_seconds=float(value),perturbed_capped_PET_seconds=measured['pet_seconds'],
                    maximum_position_displacement_m=float(np.linalg.norm(changed[...,:2]-baseline[...,:2],axis=-1).max())))
    out.update(coefficient_gradient=gradient.tolist(),unit_coefficient_direction=(gradient/norm).tolist(),
               coefficient_gradient_L2=norm,active_constraints=active['active_constraints'],
               active_value_replay_error=active['value_replay_error'],
               underlying_full_oracle_coordinate_FD_checks=len(active['finite_difference_checks']))
    return out


def summarize(rows):
    supported=[r for r in rows if r['supported']]
    groups={}
    for direction in ('gradient','fixed_random'):
        for radius in RMS_RADII:
            items=[p for r in supported for p in r['perturbations'] if p['direction']==direction and p['rms_radius']==radius]
            finite=[p for p in items if p['first_order_absolute_error_seconds'] is not None]
            informative=[p for p in finite if p['informative_predicted_change']]
            groups[direction+'_rms_'+format(radius,'g')]=dict(perturbations=len(items),finite=len(finite),
                informative=len(informative),sign_agreement_fraction=None if not informative else float(np.mean([p['sign_agrees'] for p in informative])),
                positive_uncapped_regime_fraction=None if not items else float(np.mean([p['positive_uncapped_regime_preserved'] for p in items])),
                first_order_absolute_error_median_seconds=None if not finite else float(np.median([p['first_order_absolute_error_seconds'] for p in finite])),
                first_order_absolute_error_p95_seconds=None if not finite else float(np.quantile([p['first_order_absolute_error_seconds'] for p in finite],.95)),
                first_order_relative_error_p95=None if not finite else float(np.quantile([p['first_order_relative_error'] for p in finite],.95)),
                max_position_displacement_p95_m=None if not finite else float(np.quantile([p['maximum_position_displacement_m'] for p in finite],.95)))
    errors=[r['cached_GT_vs_reconstructed_PET_absolute_error_seconds'] for r in rows]
    return dict(scenes=len(rows),supported=len(supported),support_fraction=len(supported)/len(rows),
                reasons=dict(Counter(r['reason'] for r in rows)),
                reconstruction_status_counts=dict(Counter(r['reconstruction_status'] for r in rows)),
                GT_vs_reconstruction_PET_MAE_seconds=float(np.mean(errors)),
                GT_vs_reconstruction_PET_p95_seconds=float(np.quantile(errors,.95)),
                perturbation_groups=groups)


def _safe(value):
    if isinstance(value,np.generic):return _safe(value.item())
    if isinstance(value,float) and not np.isfinite(value):return None
    if isinstance(value,dict):return {k:_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [_safe(v) for v in value]
    return value


def _write(path,value):
    with path.open('x',encoding='utf-8') as handle:json.dump(_safe(value),handle,indent=2,sort_keys=True,allow_nan=False)
    return dict(path=str(path.resolve()),sha256=sha256(path))


def run_probe(data_binding,output_root,maximum=128):
    if data_binding.get('sha256')!=DATA_SHA or maximum not in (32,128):
        raise ValueError('only frozen generator data and fixed first32/full128 diagnostic allowed')
    manifest=json.loads(verify_binding(data_binding).read_text())
    diagnostic=json.loads(verify_binding(manifest['basis_diagnostic']).read_text())
    selected=diagnostic['selected_scene_ids']
    if (diagnostic['role']!='FIT' or len(selected)!=128 or len(set(selected))!=128
            or diagnostic['selected_K']!=8 or manifest['basis']['modes']!=8):
        raise ValueError('existing FIT128 basis diagnostic identity selection changed')
    gt={row['scene_id']:row for row in diagnostic['per_scene']['8']}
    if set(gt)!=set(selected):raise ValueError('cached actual PET does not cover the exact128')
    pack=load_training_pack(manifest['packs']['FIT'],role='FIT')
    normalizer=json.loads(verify_binding(manifest['coefficient_normalizer']).read_text())
    view=ExpandedSceneReferenceView(manifest['dataset_manifest']['path'],manifest['dataset_manifest']['sha256'],purpose='fit')
    source_lookup={sid:i for i,(_rec,_row,sid) in enumerate(view.rows)}
    pack_lookup={str(sid):i for i,sid in enumerate(pack['scene_id'])}
    if any(sid not in source_lookup or sid not in pack_lookup for sid in selected):raise ValueError('FIT128 ID missing from bound source/pack')
    output=resolve(output_root)
    if output.exists():raise FileExistsError('never overwrite a direction diagnostic')
    output.mkdir(parents=True,exist_ok=False)
    codes={path:sha256(ROOT/path) for path in CODE}
    freeze=dict(protocol=PROTOCOL,generator_data=data_binding,basis_diagnostic=manifest['basis_diagnostic'],
        FIT_pack=manifest['packs']['FIT'],coefficient_normalizer=manifest['coefficient_normalizer'],
        selected_scene_ids=selected[:maximum],selected_before_outcomes=True,maximum_scenes=maximum,
        perturbation_rms_radii=list(RMS_RADII),directions=['gradient','fixed_random'],signs=[-1,1],
        random_direction_salt=NOISE_SALT,code_sha256=codes,
        CDF_models_or_weights_loaded=False,new_p_labels_created=False,raw_CSV_read=False,
        STOP_CAL_AUDIT_read=False,native_future_trajectory_array_decoded=False,
        source_input_scope='FIT_pack_coefficients_and_anchors; selected_existing_FIT_H_static_PET_only',
        purpose='local_geometry_risk_projection_potential_not_global_gradient_guarantee_or_training')
    freeze_binding=_write(output/'freeze.json',freeze)
    torch.set_num_threads(2);rows=[];started=time.perf_counter()
    with (output/'per_scene.jsonl').open('x',encoding='utf-8') as ledger:
        for number,sid in enumerate(selected[:maximum]):
            idx=pack_lookup[sid];example=view[source_lookup[sid]];features=example['features']
            n=int(pack['agent_mask'][idx].sum())
            if (example['metadata']['role']!='FIT' or example['metadata']['recording_id']!=pack['recording_id'][idx]
                    or n!=features['history'].shape[1] or not np.array_equal(pack['anchors'][idx,:n],features['history'][-1])
                    or example['target']!=gt[sid]['observed_PET']):raise ValueError('selected physical context/cached PET identity mismatch')
            result=probe_direction(pack['coef_clean'][idx,:n].astype(np.float64),pack['anchors'][idx,:n],
                features['dimensions'],normalizer,sid)
            result.update(scene_id=sid,recording_id=example['metadata']['recording_id'],role='FIT',num_agents=n,
                fixed_selection_index=number,cached_GT_PET_seconds=float(example['target']),
                cached_basis64_reconstruction_PET_seconds=gt[sid]['reconstructed_PET'],
                stored_clean_coefficient_dtype='float32_lifted_to_float64_for_derivative',
                cached_GT_vs_reconstructed_PET_absolute_error_seconds=abs(result['reconstructed_PET_seconds']-example['target']))
            rows.append(result);ledger.write(json.dumps(_safe(result),sort_keys=True,allow_nan=False)+'\n');ledger.flush()
            if (number+1)%32==0:
                elapsed=time.perf_counter()-started
                progress=dict(processed=number+1,maximum=maximum,wall_seconds=elapsed,
                    estimated_total_seconds=elapsed/(number+1)*maximum,summary=summarize(rows))
                _write(output/('progress_%03d.json'%(number+1)),progress)
                print(json.dumps(dict(stage='risk_direction_probe',processed=number+1,maximum=maximum,
                    supported=progress['summary']['supported'],wall_seconds=elapsed,
                    estimated_total_seconds=progress['estimated_total_seconds'])),flush=True)
    if {path:sha256(ROOT/path) for path in CODE}!=codes:raise RuntimeError('diagnostic source changed during execution')
    report=dict(freeze,freeze_binding=freeze_binding,status='complete',summary=summarize(rows),
        per_scene=dict(path=str(output/'per_scene.jsonl'),sha256=sha256(output/'per_scene.jsonl')),
        wall_seconds=time.perf_counter()-started,training_executed=False,
        directions_saved_only_for_supported_clean_reconstructions=True,
        training_use_or_loss_hyperparameters_not_authorized_by_this_probe=True,
        caveat='directions valid only locally on supported reconstructions; actual high-noise predictions can switch actors/segments or hit atoms',
        perturbations_are_observations=False,DDPM_noise_level_equivalence_claimed=False)
    return _write(output/'result.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',required=True);parser.add_argument('--data-sha256',required=True)
    parser.add_argument('--output-root',required=True);parser.add_argument('--maximum',type=int,choices=(32,128),default=128)
    args=parser.parse_args()
    print(json.dumps(run_probe(dict(path=args.data,sha256=args.data_sha256),args.output_root,args.maximum),indent=2))
