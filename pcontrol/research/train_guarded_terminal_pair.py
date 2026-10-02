#!/usr/bin/env python3
"""Matched canonical/atom-aware control packages with the same guarded CDF."""
import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.publication_pipeline import generator_data as gd,atom_training
from pcontrol.generation.atom_aware_generator import AtomAwareRiskGenerator
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.generation.wide_risk_sampling import WideRiskHistoryCycle
from pcontrol.generation.slow_history_sampling import speed_strata
from pcontrol.generation.cdf_shape_context import CONTEXT_KEY
from pcontrol.reference.torch_frozen_inverse import PIECES_KEY
from pcontrol.generation.random_stream_state import paired_stream_state,slow_cycle_state
from pcontrol.research import train_natural_scene_reference as io
from pcontrol.research import train_guarded_generator_adaptation as adaptation
from pcontrol.research import train_time_attention_generator as prior
from pcontrol.research import train_wide_coupled_risk_generator as legacy
from pcontrol.research.refine_natural_direct_p import PairedStreams,state_hash

PROTOCOL='guarded_atom_target_paired_terminal_training_v1'
CODE=tuple(dict.fromkeys((*adaptation.CODE,'pcontrol/research/train_guarded_terminal_pair.py',
    'pcontrol/publication_pipeline/atom_training.py','pcontrol/generation/atom_aware_generator.py',
    'pcontrol/generation/terminal_atom_aware.py','pcontrol/generation/excess_rank_loss.py',
    'pcontrol/generation/diffusion.py','pcontrol/generation/direct_p_cfg.py',
    'pcontrol/research/train_natural_direct_p.py','pcontrol/research/train_natural_diffusion.py','pcontrol/research/train_natural_scene_reference.py')))


def prepare(pb):
    p=c.json_file(pb)
    if (p['protocol']!=PROTOCOL or p['arms']!=['canonical','atom_aware'] or p['epochs_per_arm']!=3
            or p['protected_data_access'] or not p['paired_training_streams']):raise ValueError('wrong paired study')
    r=c.json_file(p['adaptation']);s=adaptation.load_source(r['policy'])
    if r['status']!='complete' or r['protocol']!=adaptation.PROTOCOL:raise ValueError('complete common adaptation required')
    cp=torch.load(io.verify_binding(r['checkpoint']),map_location='cpu',weights_only=False)
    if not gd.same_binding(cp['reference_manifest'],s['preparation']['reference_manifest']):raise ValueError('reference mismatch')
    settings,physics,coverage=prior.active_terminal_settings(s['policy'])
    codes=c.source_bindings(CODE);root=io.resolve(p['output_root']);root.mkdir(exist_ok=False)
    manifest=dict(protocol=PROTOCOL,status='prepared',policy=pb,adaptation=p['adaptation'],checkpoint=r['checkpoint'],
        preparation=r['preparation'],adaptation_policy=r['policy'],reference_manifest=r['reference_manifest'],
        labels_manifest=r['labels_manifest'],terminal_settings=settings,physics=physics,coverage=coverage,code_sha256=codes,
        arm_changes=dict(canonical='unchanged_target_and_losses',atom_aware=p['atom_objective']),
        all9913_base_and_all3744_auxiliary_rows_per_epoch=True,generated_evaluation_before_freeze=False)
    io.write_json(root/'manifest.json',manifest);print(json.dumps(dict(stage='paired_terminal_frozen',settings=settings,physics=physics,coverage=coverage)),flush=True)


def inputs(pb):
    p=c.json_file(pb);root=io.resolve(p['output_root']);mb=c.bind(root/'manifest.json');m=c.json_file(mb)
    if not gd.same_binding(m['policy'],pb):raise ValueError('policy drift')
    c.verify_sources(m['code_sha256']);s=adaptation.load_source(m['adaptation_policy'])
    if not gd.same_binding(s['preparation_binding'],m['preparation']):raise ValueError('preparation drift')
    cp=torch.load(io.verify_binding(m['checkpoint']),map_location='cpu',weights_only=False)
    return p,root,mb,m,s,cp


def model_from_checkpoint(cp,arm,target_policy,device):
    if arm=='canonical':return prior.make_model(cp['architecture'],cp['state_dict'],device)
    if arm!='atom_aware':raise ValueError('unknown arm')
    kw={k:cp['architecture'][k] for k in ('coefficient_dim','hidden_dim','heads','layers','feedforward_dim','context_hidden_dim','dynamic_bottleneck')}
    with torch.random.fork_rng(devices=[]):model=AtomAwareRiskGenerator(**kw,**target_policy)
    model.load_state_dict(cp['state_dict'],strict=True)
    if sum(x.numel() for x in model.parameters())!=980368:raise ValueError('neural architecture changed')
    return model.to(device)


def train(pb,arm,smoke=False):
    p,root,mb,m,s,cp=inputs(pb)
    if arm not in p['arms']:raise ValueError('unregistered arm')
    runtime=dict(runtime=p['runtime'],generator=dict(seed=p['seed']));device=prior.runtime(runtime)
    model=model_from_checkpoint(cp,arm,p['target_policy'],device);initial=state_hash(model)
    expected=c.json_file(m['adaptation'])['selected_state_sha256']
    if initial!=expected:raise ValueError('arms must start at identical common weights')
    optimizer=torch.optim.AdamW(model.finetuning_groups(p['core_learning_rate'],p['adapter_learning_rate']),weight_decay=1e-4)
    ema=copy.deepcopy(model).requires_grad_(False);schedule=CosineDiffusionSchedule(100).to(device)
    teacher=gd.TransformerFITTeacher(s,device);_,strata=speed_strata(s['physical']['history'],s['packs']['FIT']['agent_mask'],s['packs']['FIT']['role'])
    cycle=WideRiskHistoryCycle(s['packs']['FIT']['recording_id'],strata,seed=p['seed']+4)
    streams=PairedStreams(p['seed']+10,p['seed']+11);aux=torch.Generator().manual_seed(p['seed']+12)
    settings=m['terminal_settings'];physics=m['physics'];coverage=m['coverage']
    runroot=root/(arm+'_smoke' if smoke else arm);runroot.mkdir(exist_ok=False)
    freeze=io.write_json(runroot/'freeze.json',dict(policy=pb,paired_manifest=mb,arm=arm,smoke=smoke,
        initial_state_sha256=initial,checkpoint=m['checkpoint'],code_sha256=m['code_sha256'],
        full_generator_trainable=True,all_reference_parameters_frozen=True))
    shape=dict(shape=s['contexts']['FIT'][CONTEXT_KEY],pieces=s['contexts']['FIT'][PIECES_KEY])
    epoch_function=legacy.train_epoch if arm=='canonical' else atom_training.train_epoch
    recipe='wide_coupled' if arm=='canonical' else 'atom_aware'
    started=time.monotonic();snapshots={};traces=[];restarts={}
    def save(candidate,epoch,averaged):
        path=runroot/f'{"ema" if averaged else "raw"}_epoch_{epoch:03d}.pt'
        payload=dict(protocol=PROTOCOL,policy=pb,arm=arm,epoch=epoch,smoke=smoke,EMA=averaged,
            state_dict=prior.cpu_state(candidate),architecture=candidate.architecture_config(),
            generator_data=s['policy']['generator_data'],labels_manifest=m['labels_manifest'],reference_manifest=m['reference_manifest'],
            paired_manifest=mb,common_initial_checkpoint=m['checkpoint'],code_sha256=m['code_sha256'],
            basis=s['data']['basis'],coefficient_normalizer=s['data']['coefficient_normalizer'],history_normalizer=s['data']['history_normalizer'])
        with path.open('xb') as f:torch.save(payload,f)
        return c.bind(path)
    with (runroot/'epochs.jsonl').open('x') as log:
        for epoch in range(1,(1 if smoke else p['epochs_per_arm'])+1):
            row=epoch_function(model,ema,schedule,optimizer,s['packs']['FIT'],s['pvalues']['FIT'],s['cache'],teacher,
                streams,aux,settings,physics,coverage,recipe,device,shape_values=shape,slow_cycle=cycle,
                max_updates=1 if smoke else None,progress=lambda r:print(json.dumps(dict(arm=arm,epoch=epoch,**r)),flush=True))
            row.update(epoch=epoch,arm=arm,smoke=smoke);traces.append(row);log.write(json.dumps(row)+'\n');log.flush()
            snapshots[str(epoch)]=dict(raw=save(model,epoch,False),EMA=save(ema,epoch,True))
            # Save recoverable optimizer/stream state, without claiming tested exact restart.
            state=dict(protocol=PROTOCOL,policy=pb,arm=arm,smoke=smoke,epoch=epoch,
                model_state=prior.cpu_state(model),EMA_state=prior.cpu_state(ema),optimizer_state=optimizer.state_dict(),
                paired_streams_state=paired_stream_state(streams),auxiliary_generator_state=aux.get_state(),
                history_cycle_state=slow_cycle_state(cycle),torch_CPU_RNG=torch.get_rng_state(),
                torch_device_RNG=torch.cuda.get_rng_state(device) if device.type=='cuda' else None,code_sha256=m['code_sha256'])
            path=runroot/f'training_state_epoch_{epoch:03d}.pt'
            with path.open('xb') as f:torch.save(state,f)
            restarts[str(epoch)]=c.bind(path);print(json.dumps(dict(stage='terminal_epoch_complete',**row)),flush=True)
    if not smoke and any(r['base_loss_scenes']!=9913 or r['updates']!=78 or r['auxiliary_requests']!=3744
                          or r['forward_NFE']!=7800 or r['backward_NFE']!=7800 for r in traces):
        raise ValueError('incomplete full-DDIM training budget')
    if initial==state_hash(model):raise ValueError('no generator update occurred')
    c.verify_sources(m['code_sha256'])
    result=dict(protocol=PROTOCOL,status='smoke_complete' if smoke else 'complete',policy=pb,paired_manifest=mb,arm=arm,smoke=smoke,
        freeze=freeze,epochs_completed=len(traces),snapshots=snapshots,restart_states=restarts,
        selected_checkpoint=snapshots[str(len(traces))]['raw'],checkpoint_selection=p['checkpoint_selection'],
        initial_state_sha256=initial,final_state_sha256=state_hash(model),epochs=c.bind(runroot/'epochs.jsonl'),
        reference_manifest=m['reference_manifest'],labels_manifest=m['labels_manifest'],code_sha256=m['code_sha256'],
        wall_seconds=time.monotonic()-started,protected_data_access=False,generated_control_performance_evaluated=False)
    io.write_json(runroot/'result.json',result);print(json.dumps(dict(stage='terminal_complete',arm=arm,smoke=smoke)),flush=True)


def suite(pb):
    p,root,mb,m,s,cp=inputs(pb)
    for arm in p['arms']:
        smoke=c.json_file(c.bind(root/(arm+'_smoke')/'result.json'))
        if smoke['status']!='smoke_complete' or not gd.same_binding(smoke['paired_manifest'],mb):raise ValueError('both matching smoke runs required')
    del s,cp
    for arm in p['arms']:
        subprocess.run([sys.executable,__file__,'train','--policy',pb['path'],'--policy-sha256',pb['sha256'],'--arm',arm],check=True)
    print(json.dumps(dict(stage='both_paired_terminal_arms_complete')),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['prepare','train','suite'])
    parser.add_argument('--policy',default='configs/natural_percentile/guarded_atom_terminal_pair_v1.json')
    parser.add_argument('--policy-sha256',required=True);parser.add_argument('--arm',choices=['canonical','atom_aware']);parser.add_argument('--smoke',action='store_true')
    a=parser.parse_args();pb=dict(path=str(io.resolve(a.policy)),sha256=a.policy_sha256)
    if a.command=='train':train(pb,a.arm,a.smoke)
    elif a.command=='suite':suite(pb)
    else:prepare(pb)
