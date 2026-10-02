#!/usr/bin/env python3
"""Five fresh47-epoch teachers and their own count-calibrated OOF coordinates."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.publication_pipeline.reference import PROTOCOL as REFERENCE_PROTOCOL
from pcontrol.reference import direct_p_crossfit as data
from pcontrol.reference.scene_calibration import fit_scene_calibration
from pcontrol.research import crossfit_time_attention_reference as old
from pcontrol.research import train_natural_scene_reference as io

PROTOCOL='guarded_time_attention_crossfit_v1'
CODE=('pcontrol/research/crossfit_guarded_reference.py','pcontrol/publication_pipeline/reference.py',*old.CODE)


def policy(pb):
    p=c.json_file(pb);reference=c.json_file(p['reference_manifest'])
    if (p['protocol']!=PROTOCOL or reference['protocol']!=REFERENCE_PROTOCOL or reference['status']!='complete'
            or p['protected_data_access'] or p['CAL_generator_examples'] or p['labels']['random_within_atom']
            or p['teacher_training']['held_or_STOP_selection'] or p['teacher_training']['old55_epoch_teacher_weights_reused']):
        raise ValueError('wrong guarded OOF scope')
    if (p['teacher_training']['base_epochs']!=reference['model_training_epochs']['base']
            or p['teacher_training']['highN_epochs']!=reference['model_training_epochs']['highN']
            or p['calibration']['family']!=reference['selected_family'] or p['calibration']['ridge']!=reference['selected_ridge']):
        raise ValueError('teacher recipe must match selected guarded reference')
    c.verify_sources(reference['code_sha256']);return p,reference,io.resolve(p['output_root'])


def runtime(p):
    torch.set_num_threads(p['runtime']['threads']);torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True);torch.backends.mha.set_fastpath_enabled(False)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False


def prepare(pb):
    p,reference,root=policy(pb);physical=c.json_file(p['physical_prepared'])
    same_data=(physical['source_data']['sha256']==reference['data']['sha256']
               and io.resolve(physical['source_data']['path'])==io.resolve(reference['data']['path']))
    if not same_data or physical['folds']!=p['heldout_recordings'] or physical['counts']!={'FIT':9913,'STOP':538,'CAL':504}:
        raise ValueError('physical data/fold identities changed')
    c.verify_sources(physical['code_sha256'])
    for binding in physical['physical_packs'].values():io.verify_binding(binding)
    root.mkdir(parents=True,exist_ok=False)
    report=dict(protocol=PROTOCOL,status='prepared',policy=pb,reference_manifest=p['reference_manifest'],
        data=reference['data'],physical_prepared=p['physical_prepared'],physical_packs=physical['physical_packs'],
        folds=p['heldout_recordings'],counts=physical['counts'],role_recordings=physical['role_recordings'],
        teacher_training=p['teacher_training'],calibration=p['calibration'],
        code_sha256=c.source_bindings(CODE),no_fullFIT_label_substitution=True)
    io.write_json(root/'prepared.json',report)
    print(json.dumps(dict(stage='guarded_OOF_prepared',epochs=[p['teacher_training']['base_epochs'],p['teacher_training']['highN_epochs']])),flush=True)


def inputs(pb):
    p,reference,root=policy(pb);b=c.bind(root/'prepared.json');f=c.json_file(b)
    if f['policy']!=pb or f['reference_manifest']!=p['reference_manifest']:raise ValueError('preparation drift')
    c.verify_sources(f['code_sha256']);return p,reference,root,b,f


def fold(pb,index):
    p,reference,root,fb,f=inputs(pb);runtime(p)
    if str(index) not in f['folds']:raise ValueError('fold0..4 required')
    out=root/f'fold_{index}';out.mkdir(exist_ok=False)
    physical=data._load_pack(f['physical_packs']['FIT'],'FIT');held=f['folds'][str(index)]
    mask=np.isin(physical['recording_id'],held);train_rows=np.flatnonzero(~mask);held_rows=np.flatnonzero(mask)
    norm=data.fit_fold_normalizer(physical,train_rows,held,index);nb=io.write_json(out/'normalizer.json',norm)
    train=data.normalized_subset(physical,train_rows,norm);device=torch.device(p['runtime']['device'])
    with (out/'epochs.jsonl').open('x') as log:model,stats=old.train_fixed(train,p['teacher_training'],device,log,index)
    header=dict(protocol=PROTOCOL,policy=pb,prepared=fb,fold=index,normalizer=nb,
        train_recordings=norm['training_recordings'],held_recordings=held,training_rows=len(train_rows),held_rows=len(held_rows),
        fixed_base_epochs=p['teacher_training']['base_epochs'],fixed_highN_epochs=p['teacher_training']['highN_epochs'],
        seed=p['teacher_training']['seed'],initialization=stats['initialization'],initial_state_sha256=stats['initial_state_sha256'],
        architecture=model.architecture_config(),held_or_STOP_model_selection=False,code_sha256=f['code_sha256'])
    with (out/'teacher.pt').open('xb') as handle:torch.save(dict(header=header,state_dict={k:v.cpu().clone() for k,v in model.state_dict().items()}),handle)
    checkpoint=c.bind(out/'teacher.pt');io.write_json(out/'training_report.json',stats)
    cal=data._load_pack(f['physical_packs']['CAL'],'CAL')
    masses=data.predict_masses(model,data.normalized_subset(cal,np.arange(len(cal['target'])),norm),p['teacher_training']['batch_size'])
    cfg=p['calibration']
    if cfg['family']=='identity':warp=c.StableCountWarp.identity();fit_report=dict(success=True,identity=True)
    else:
        fit=fit_scene_calibration(masses,cal['agent_mask'].sum(1),cal['target'],family=cfg['family'],ridge=cfg['ridge'])
        if not fit.report['success']:raise RuntimeError('fold calibration failed')
        warp=c.StableCountWarp.from_dict(fit.warp.as_dict());fit_report=fit.report
    wb=io.write_json(out/'calibration_model.json',warp.as_dict())
    cal_b=io.save_pack(out/'CAL_raw_predictions.npz',dict(joint_masses=masses,target=cal['target'],num_agents=cal['agent_mask'].sum(1),
        scene_id=cal['scene_id'],recording_id=cal['recording_id'],role=cal['role']))
    cr=io.write_json(out/'calibration_report.json',dict(role='CAL',rows=504,recordings=sorted(set(cal['recording_id'])),
        family=cfg['family'],ridge=cfg['ridge'],fit=fit_report,fullFIT_nodes_reused=False))
    h={k:v[held_rows] for k,v in physical.items()}
    masses=data.predict_masses(model,data.normalized_subset(physical,held_rows,norm),p['teacher_training']['batch_size'])
    labels,context=old.payload(masses,h,warp,index)
    lb=io.save_pack(out/'labels.npz',labels);cb=io.save_pack(out/'context.npz',context)
    report=dict(header,status='complete',checkpoint=checkpoint,labels=lb,context=cb,calibration_model=wb,
        calibration_report=cr,CAL_raw_predictions=cal_b,training_trace=c.bind(out/'epochs.jsonl'),
        reference_manifest=p['reference_manifest'],native_future_trajectories_decoded=False,CAL_generator_examples=False)
    c.verify_sources(f['code_sha256']);io.write_json(out/'result.json',report)
    print(json.dumps(dict(stage='guarded_fold_complete',fold=index,training_rows=len(train_rows),held_rows=len(held_rows))),flush=True)


def aggregate(pb):
    p,reference,root,fb,f=inputs(pb);runtime(p)
    physical=data._load_pack(f['physical_packs']['FIT'],'FIT');lookup={str(s):i for i,s in enumerate(physical['scene_id'])}
    seen=np.zeros(9913,bool);labels_out=context_out=None;bindings={};initials=[]
    for index in range(5):
        rb=c.bind(root/f'fold_{index}/result.json');r=c.json_file(rb);c.verify_sources(r['code_sha256'])
        if r['policy']!=pb or r['prepared']!=fb or r['status']!='complete' or r['held_recordings']!=f['folds'][str(index)]:raise ValueError('wrong fold completion')
        if set(r['held_recordings'])&set(r['train_recordings']):raise ValueError('fold leakage')
        labels=c.arrays(r['labels']);context=c.arrays(r['context']);data._validate_arrays(labels)
        rows=np.array([lookup[str(v)] for v in labels['scene_id']])
        if seen[rows].any() or not np.array_equal(labels['scene_id'],context['scene_id']) or not np.array_equal(labels['pet_seconds'],physical['target'][rows]):
            raise ValueError('held data identity/outcome changed')
        if labels_out is None:
            labels_out={k:np.empty((9913,)+v.shape[1:],dtype=v.dtype) for k,v in labels.items()}
            context_out={k:np.empty((9913,)+v.shape[1:],dtype=v.dtype) for k,v in context.items()}
        for k,v in labels.items():labels_out[k][rows]=v
        for k,v in context.items():context_out[k][rows]=v
        seen[rows]=True;bindings[str(index)]=rb;initials.append(r['initial_state_sha256'])
    if not seen.all() or len(set(initials))!=1:raise ValueError('all9913 same-seed recording-excluded predictions required')
    fit_l=io.save_pack(root/'FIT_labels.npz',labels_out);fit_c=io.save_pack(root/'FIT_context.npz',context_out)
    stop=data._load_pack(f['physical_packs']['STOP'],'STOP');norm=c.json_file(reference['normalizer'])
    cp=torch.load(io.verify_binding(reference['checkpoint']),map_location='cpu',weights_only=False);model=c.make_reference_model()
    model.load_state_dict(cp['state_dict'],strict=True);model.to(p['runtime']['device']).eval().requires_grad_(False)
    masses=data.predict_masses(model,data.normalized_subset(stop,np.arange(538),norm),p['teacher_training']['batch_size'])
    warp=c.StableCountWarp.from_dict(c.json_file(reference['calibration_model']));labels,context=old.payload(masses,stop,warp,-1)
    stop_l=io.save_pack(root/'STOP_labels.npz',labels);stop_c=io.save_pack(root/'STOP_context.npz',context)
    result=dict(protocol=PROTOCOL,status='complete',policy=pb,prepared=fb,reference_manifest=p['reference_manifest'],data=reference['data'],
        physical_packs=f['physical_packs'],fold_artifacts=bindings,
        roles=dict(FIT=dict(rows=9913,labels=fit_l,context=fit_c,label_source='fresh47epoch_recording_excluded_TimeAttn_own_count_CAL'),
                   STOP=dict(rows=538,labels=stop_l,context=stop_c,label_source='fullFIT_guarded_reference')),
        code_sha256=f['code_sha256'],old55epoch_teacher_weights_reused=False,fullFIT_in_sample_label_substitution=False,
        random_within_atom_labels=False,CAL_generator_examples=False,native_future_trajectories_decoded=False)
    c.verify_sources(f['code_sha256']);io.write_json(root/'manifest.json',result)
    print(json.dumps(dict(stage='guarded_OOF_complete',FIT=9913,STOP=538)),flush=True)


def suite(pb):
    p,_,root,_,_=inputs(pb);logs=root/'runlogs';logs.mkdir(exist_ok=True)
    for index in range(5):
        print(json.dumps(dict(starting_fold=index)),flush=True)
        argv=[sys.executable,str(Path(__file__).resolve()),'fold','--fold',str(index),'--policy',pb['path'],'--policy-sha256',pb['sha256']]
        with (logs/f'fold_{index}.log').open('x') as log:
            subprocess.run(argv,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'),stdout=log,stderr=subprocess.STDOUT,check=True)
        print(json.dumps(dict(completed_fold=index)),flush=True)
    aggregate(pb)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','fold','aggregate','suite'])
    parser.add_argument('--policy',required=True);parser.add_argument('--policy-sha256',required=True);parser.add_argument('--fold',type=int)
    a=parser.parse_args();pb=dict(path=str(io.resolve(a.policy)),sha256=a.policy_sha256)
    if a.command=='fold':fold(pb,a.fold)
    else:{'prepare':prepare,'aggregate':aggregate,'suite':suite}[a.command](pb)
