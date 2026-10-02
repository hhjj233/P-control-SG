"""Frozen reference evaluation: score every complete natural scene, never fit."""
import json
from contextlib import nullcontext
from pathlib import Path
import numpy as np
import torch
from pcontrol.publication_pipeline.final_validation_protocol import verified_json,REFERENCE_ARMS
from pcontrol.publication_pipeline.final_reference_suite import FrozenFinalReference
from pcontrol.publication_pipeline.final_scene_data import write_once
from pcontrol.publication_pipeline.final_execution_data import save_arrays
from pcontrol.reference.publication_calibration import CalibrationScoreCache
from pcontrol.time_attention_pipeline import common as c
from pcontrol.research.pilot_time_attention_cdf import tensor_hash
from pcontrol.research.audit_transformer_reference_validation import numeric_check,own_metrics


def paired_reference(left,right,*,repetitions=10000,seed=20260916):
    for key in ('scene_id','recording_id','target','num_agents'):
        if not np.array_equal(left[key],right[key]):raise ValueError('reference comparison is not paired')
    records=np.unique(left['recording_id']);result={}
    for key in ('CRPS_seconds','twCRPS_1s','twCRPS_2s'):
        delta=left[key]-right[key]
        if not len(delta):result[key]=dict(scenes=0,delta=None);continue
        means=np.array([delta[left['recording_id']==r].mean() for r in records])
        counts=np.array([(left['recording_id']==r).sum() for r in records])
        item=dict(scenes=len(delta),recordings=len(records),delta_scene_weighted=float(delta.mean()),delta_recording_macro=float(means.mean()),
            bootstrap_95pct=None,macro_bootstrap_95pct=None)
        if len(records)>=2:
            rng=np.random.default_rng(seed);indices=rng.integers(0,len(records),(repetitions,len(records)))
            weighted=(means[indices]*counts[indices]).sum(1)/counts[indices].sum(1)
            item.update(bootstrap_95pct=np.quantile(weighted,[.025,.975]).tolist(),macro_bootstrap_95pct=np.quantile(means[indices].mean(1),[.025,.975]).tolist())
        result[key]=item
    return dict(direction='main_TimeAttn_minus_comparator',metrics=result,
        multiple_comparison_adjusted_significance_claim=False,one_recording_is_descriptive_only=len(records)<2)


def evaluate_references(dataset,contract_binding,output_dir,*,device):
    contract=verified_json(contract_binding);p=verified_json(contract['policy'])
    prior=verified_json(p['sources']['guarded_reference_results']);prepared=verified_json(prior['prepared'])
    cfg=verified_json(prepared['calibration_policy'])['calibration']
    root=Path(output_dir);root.mkdir(parents=True,exist_ok=False)
    n=len(dataset);results={};audits={};scored_arrays={}
    if n==0:
        return write_once(root/'result.json',dict(status='no_eligible_complete_scenes',contract=contract_binding,
            dataset=dataset.binding,scope=dataset.manifest['scope'],scenes=0,models={},no_refitting=True))
    for arm in REFERENCE_ARMS:
        plugin=FrozenFinalReference.from_contract(contract_binding,arm,device=device)
        state=tensor_hash(plugin._model.state_dict());masses=[];targets=[]
        for i in range(n):
            item=dataset[i]
            # Keep the saved GRU weights, but avoid the cuDNN fused RNN path
            # whose measured FP32 discrepancy exceeded the prespecified check.
            # This affects inference backend only; original training is untouched.
            context=torch.backends.cudnn.flags(enabled=False,benchmark=False,deterministic=True,allow_tf32=False) \
                if arm=='M2_GRU' and str(device).startswith('cuda') else nullcontext()
            with context:ref=plugin.condition_features(item['features'])
            masses.append(ref._masses[0]);targets.append(item['target'])
        raw=dict(joint_masses=np.asarray(masses),target=np.asarray(targets),
            scene_id=np.array([r['scene_id'] for r in dataset.rows]),recording_id=np.array([r['recording_id'] for r in dataset.rows]),
            num_agents=np.array([r['num_agents'] for r in dataset.rows]),role=np.full(n,dataset.manifest['scope']))
        if not np.isfinite(raw['joint_masses']).all():raise FloatingPointError('reference prediction failed; do not drop rows')
        cache=CalibrationScoreCache(raw,cfg);models={};predictions={};errors={}
        armroot=root/arm;armroot.mkdir()
        for view in ('raw','frozen_calibrated'):
            warp=c.StableCountWarp.identity() if view=='raw' else plugin._warp
            arrays,metrics=cache.score(warp.row_nodes(raw['num_agents']))
            metrics.pop('selection_scores');metrics.pop('supported_N_groups')
            errors[view]=numeric_check(arrays,cfg)
            independent=own_metrics(arrays,cfg)
            for name,value in independent['overall'].items():
                if abs(value-metrics['overall'][name])>2e-10:raise ValueError('reference summary disagreement')
            predictions[view]=save_arrays(armroot/(view+'.npz'),arrays);models[view]=metrics
            scored_arrays[arm,view]=arrays
        if state!=tensor_hash(plugin._model.state_dict()) or any(t.grad is not None or t.requires_grad for t in plugin._model.parameters()):
            raise RuntimeError('evaluation modified reference weights')
        results[arm]=dict(models=models,predictions=predictions,selected_model=contract['reference_models'][arm],
            device=str(device),cuda_GRU_cudnn_fusion_disabled=arm=='M2_GRU' and str(device).startswith('cuda'))
        audits[arm]=errors
        print(json.dumps(dict(stage='reference_evaluated',scope=dataset.manifest['scope'],arm=arm,scenes=n)),flush=True)
    comparisons={}
    for arm in REFERENCE_ARMS:
        if arm=='M2_TimeAttn':continue
        comparisons[arm]={view:paired_reference(scored_arrays['M2_TimeAttn',view],scored_arrays[arm,view],
            repetitions=p['uncertainty']['bootstrap_repetitions'],seed=p['uncertainty']['seed']) for view in ('raw','frozen_calibrated')}
    return write_once(root/'result.json',dict(status='complete',contract=contract_binding,dataset=dataset.binding,
        scope=dataset.manifest['scope'],scenes=n,models=results,comparisons=comparisons,
        independent_numeric_maximum_errors=audits,no_refitting=True,no_final_checkpoint_selection=True,
        software_rehearsal=dataset.manifest['software_rehearsal']))
