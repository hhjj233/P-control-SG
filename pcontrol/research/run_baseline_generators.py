#!/usr/bin/env python3
"""Fresh CVAE / standard direct-P diffusion, isolated FIT/STOP selection."""
import argparse, copy, json, os, sys, time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.publication_pipeline import generator_data as gd
from pcontrol.generation.data import FEATURE_KEYS
from pcontrol.generation.direct_p import JointPercentileDenoiser, direct_p_training_loss
from pcontrol.generation.diffusion import CosineDiffusionSchedule
from pcontrol.research.baseline_models import JointPCVAE
from pcontrol.research import train_natural_scene_reference as io

OUT=ROOT/'outputs/natural_percentile/baselines_v1'
SEED=20260922

def new_model(arm):
    return JointPCVAE() if arm=='pcvae' else JointPercentileDenoiser(16)

def load_source():
    return gd.load_inputs(c.bind(ROOT/'configs/natural_percentile/guarded_generator_adaptation_v1.json'))

def batch(source, role, rows, device):
    pack=source['packs'][role]; mask=pack['agent_mask'][rows]; n=int(mask.sum(1).max())
    f={}
    for k in FEATURE_KEYS:
        a=pack[k][rows]
        if k=='history': a=a[:,:,:n]
        elif k in ('dimensions','agent_mask','ego_mask'): a=a[:,:n]
        f[k]=torch.as_tensor(a,device=device)
    clean=torch.as_tensor(pack['coef_clean'][rows,:n],device=device)
    p=torch.tensor(source['pvalues'][role][rows],dtype=torch.float32,device=device)
    return f,clean,p

def objective(model, arm, schedule, f, clean, p, rng, beta):
    noise=torch.randn(clean.shape,generator=rng,dtype=torch.float32).to(clean.device)
    if arm=='pcvae': return model.loss(f,p,clean,noise,beta)[0]
    t=torch.randint(100,(len(p),),generator=rng).to(clean.device)
    return direct_p_training_loss(model,schedule,clean,f,p,timesteps=t,noise=noise)['loss']

@torch.no_grad()
def validation(model, arm, source, schedule, device):
    model.eval(); rng=torch.Generator().manual_seed(SEED+1); total=0.
    for start in range(0,538,128):
        rows=np.arange(start,min(start+128,538));f,clean,p=batch(source,'STOP',rows,device)
        total+=float(objective(model,arm,schedule,f,clean,p,rng,.01))*len(rows)
    return total/538

def train(arm, device):
    torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    source=load_source(); out=OUT/arm;out.mkdir(parents=True,exist_ok=False)
    freeze=dict(protocol=c.bind(ROOT/'docs/baseline_protocol.md'),
        source_policy=c.bind(ROOT/'configs/natural_percentile/guarded_generator_adaptation_v1.json'),
        labels=source['labels_binding'],generator_data=source['policy']['generator_data'],
        seed=SEED,arm=arm,FIT=9913,STOP=538,max_epochs=60,patience=12,
        selection='STOP_fixed_noise_beta_0.01_ELBO' if arm=='pcvae' else 'STOP_fixed_noise_v_MSE',
        warm_start=False,TEST_selection=False,post_primary_supplement=True,
        code=c.source_bindings(('pcontrol/research/baseline_models.py',
                               'pcontrol/research/run_baseline_generators.py')))
    io.write_json(out/'freeze.json',freeze)
    model=new_model(arm).to(device); schedule=CosineDiffusionSchedule(100).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
    order_rng=np.random.default_rng(SEED); rng=torch.Generator().manual_seed(SEED+2)
    best=float('inf');best_epoch=0; stale=0; best_state=None;started=time.monotonic()
    with (out/'epochs.jsonl').open('x') as log:
        for epoch in range(1,61):
            model.train();order=order_rng.permutation(9913);total=0.
            for start in range(0,9913,128):
                rows=order[start:start+128];f,clean,p=batch(source,'FIT',rows,device)
                opt.zero_grad(set_to_none=True)
                loss=objective(model,arm,schedule,f,clean,p,rng,.01*min(epoch/10,1))
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite loss; no TEST fallback')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
                total+=float(loss.detach())*len(rows)
            val=validation(model,arm,source,schedule,device)
            if val<best:
                best=val;best_epoch=epoch;stale=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:stale+=1
            row=dict(epoch=epoch,FIT_loss=total/9913,STOP_loss=val,best_epoch=best_epoch,best_STOP=best,
                elapsed_seconds=time.monotonic()-started)
            log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(dict(arm=arm,**row)),flush=True)
            if epoch>=10 and stale>=12:break
    torch.save(dict(state_dict=best_state,arm=arm,epoch=best_epoch,freeze=c.bind(out/'freeze.json')),
               out/'checkpoint.pt')
    io.write_json(out/'training_result.json',dict(status='complete',arm=arm,freeze=c.bind(out/'freeze.json'),
        checkpoint=c.bind(out/'checkpoint.pt'),best_epoch=best_epoch,epochs_completed=epoch,STOP_loss=best,
        parameters=sum(p.numel() for p in model.parameters()),wall_seconds=time.monotonic()-started,
        coefficient_normalizer=source['data']['coefficient_normalizer'],history_normalizer=source['data']['history_normalizer']))
    print(json.dumps(dict(stage='training_complete',arm=arm,best_epoch=best_epoch)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--arm',choices=['pcvae','p_diffusion'],required=True)
    parser.add_argument('--device',default='cuda:0');a=parser.parse_args();train(a.arm,a.device)
