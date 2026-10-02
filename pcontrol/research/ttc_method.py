#!/usr/bin/env python3
"""TTC version of our method trained on pure car-following clips only.

A generator trained only on car-following scenes. Pipeline:
  labels   Cross-fitted TTC percentile labels: the car-following FIT clips are split into four
           folds by recording. For each fold a TTC reference with the architecture and settings of
           pcontrol/research/ttc_reference.py is fitted on the other folds (43 epochs, the best
           epoch of the full TTC reference), and the fold's observed TTC is ranked (p_mid).
  train    A percentile-conditioned joint diffusion model (the classifier-free P-diffusion
           architecture of the paper's conditioning pathway) is trained on the car-following clips
           with these labels. Recordings 05 and 28 are held out for early stopping.
  evaluate The 96 TEST car-following histories and noise draws of ttc_generation.py,
           with four arms: full (conditioning at CFG scale 2.5 plus TTC guidance), condition only,
           guidance only (null branch plus TTC guidance) and no P. TTC guidance uses the settings
           chosen on the DEV recordings (inverse TTC, last 30 steps, 3 inner steps, strength 1, 1 m).
Scoring uses the full TTC reference, as before.
"""
import argparse, hashlib, json, os, sys, time
from multiprocessing import get_context
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
for k, v in (('TTC_SIGNAL', 'inverse'), ('TTC_LAST', '30'), ('TTC_INNER', '3'), ('TTC_STRENGTH', '1.0'), ('TTC_POSCAP', '1.0')):
    os.environ.setdefault(k, v)
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.generation.direct_p_cfg import ClassifierFreePercentileDenoiser, cfg_training_loss, _CFGPrediction
from pcontrol.generation.diffusion import CosineDiffusionSchedule, ddim_sample
from pcontrol.generation.trajectory_basis import TrajectoryBasis
from pcontrol.generation.risk_guidance import TorchTrajectoryDecoder
from pcontrol.reference.scores import crps_from_params
from pcontrol.reference.mixed_cdf import rank_from_params, quantile_from_params
from pcontrol.research.ttc_reference import TTC_CAP, CAP_TAG, scaled, load, normalizer, features as ref_features, make_model as make_reference
from pcontrol.research import ttc_generation as gen
from pcontrol.research.p_path_ablation import GeometryOnlyGuidance, forbidden_rank
from pcontrol.research.evaluate_natural_diffusion import model_features

REV = ROOT / 'outputs/natural_percentile/experiments_v1'
OUT = REV / (f'ttc_method{CAP_TAG}' + ('_bounded' if os.environ.get('TTC_BOUNDED', '0') == '1' else ''))
P_DIFF = ROOT / 'outputs/natural_percentile/baselines_v1/p_diffusion/training_result.json'
QUEUE = ROOT / 'outputs/natural_percentile/transformer_publication_v1_20260916/final_validation_v1/TEST/queue/queue.json'
STOP_RECORDINGS = ('05', '28')
SEED, DROP, FOLDS, REF_EPOCHS = 20260930, .1, 4, 43
GRID = (.1, .3, .5, .7, .9)


def normalizers():
    tr = c.json_file(c.bind(P_DIFF))
    return c.json_file(tr['history_normalizer']), c.json_file(tr['coefficient_normalizer'])


def labels(device='cuda:1'):
    OUT.mkdir(parents=True, exist_ok=True)
    fit = load('FIT'); hs, ds, _ = normalizer(); recs = sorted(set(fit['recording_id'].tolist()), key=int)
    fold_of = {r: i % FOLDS for i, r in enumerate(recs)}
    fold = np.array([fold_of[r] for r in fit['recording_id']])
    p_mid = np.full(len(fold), np.nan)
    for k in range(FOLDS):
        torch.manual_seed(SEED + k); rng = np.random.default_rng(SEED + k)
        model = make_reference().to(device); opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
        train_rows = np.flatnonzero(fold != k)
        ref_epochs = int(torch.load(gen.REF / 'checkpoint.pt', map_location='cpu', weights_only=False)['epoch'])
        for epoch in range(ref_epochs):
            model.train(); order = rng.permutation(train_rows)
            for s in range(0, len(order), 64):
                r = order[s:s + 64]
                y = torch.tensor(scaled(fit['ttc_seconds'][r]), dtype=torch.float64, device=device)
                loss = crps_from_params(model(ref_features(fit, r, hs, ds, device)), y, normalized=True).mean()
                opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
        model.eval().cpu(); rows = np.flatnonzero(fold == k)
        with torch.no_grad():
            for i in rows:
                params = model(ref_features(fit, [i], hs, ds, 'cpu'))
                lo, hi = fit['offsets'][i], fit['offsets'][i + 1]
                h = np.transpose(fit['history'][lo:hi], (1, 0, 2))
                rank, _, _ = gen.bounded_functions(params, gen.ttc0_seconds(h, fit['dimensions'][lo:hi], int(fit['ego'][i]), int(fit['leader'][i])))
                a, b = rank(float(scaled(fit['ttc_seconds'][i]))); p_mid[i] = .5 * (a + b)
        print(json.dumps(dict(fold=k, train=int(len(train_rows)), labelled=int(len(rows)))), flush=True)
    assert np.isfinite(p_mid).all()
    np.savez_compressed(OUT / 'labels.npz', scene_id=fit['scene_id'], recording_id=fit['recording_id'], fold=fold, p_mid=p_mid)
    print(json.dumps(dict(labels=len(p_mid), mean=float(p_mid.mean()), quantiles=np.quantile(p_mid, [.1, .5, .9]).round(3).tolist())), flush=True)


def build_packs():
    fit = load('FIT'); lab = np.load(OUT / 'labels.npz', allow_pickle=False); assert np.array_equal(lab['scene_id'], fit['scene_id'])
    hn, cn = normalizers(); basis = TrajectoryBasis(8); mean, scale = np.asarray(cn['mean']), np.asarray(cn['scale'])
    packs = []
    for i in range(len(fit['scene_id'])):
        lo, hi = fit['offsets'][i], fit['offsets'][i + 1]; n = int(hi - lo)
        history = np.transpose(fit['history'][lo:hi], (1, 0, 2)); future = np.transpose(fit['future'][lo:hi], (1, 0, 2))
        assert np.array_equal(future[0], history[-1])
        ego_mask = np.zeros(n, bool); ego_mask[fit['ego'][i]] = True
        case = dict(history=history, dimensions=fit['dimensions'][lo:hi],
                    road_boundaries=fit['road_boundaries'][fit['road_offsets'][i]:fit['road_offsets'][i + 1]],
                    ego_mask=ego_mask, agent_mask=np.ones(n, bool), num_agents=n)
        f = {k: v[0].numpy() for k, v in model_features(case, hn, torch.device('cpu')).items()}
        coef = (basis.encode(future, history[-1]) - mean) / scale
        packs.append(dict(features=f, coef=coef.astype(np.float32), p=float(lab['p_mid'][i]), stop=fit['recording_id'][i] in STOP_RECORDINGS))
    return packs


def batch(packs, rows, device):
    n = max(packs[i]['coef'].shape[0] for i in rows); r = max(len(packs[i]['features']['road_boundaries']) for i in rows); B = len(rows)
    f = dict(history=np.zeros((B, 13, n, 4), np.float32), dimensions=np.zeros((B, n, 2), np.float32),
             road_boundaries=np.zeros((B, r), np.float32), road_boundary_mask=np.zeros((B, r), bool),
             ego_mask=np.zeros((B, n), bool), agent_mask=np.zeros((B, n), bool))
    coef = np.zeros((B, n, 8, 2), np.float32)
    for b, i in enumerate(rows):
        pk = packs[i]; k = pk['coef'].shape[0]; rb = len(pk['features']['road_boundaries'])
        f['history'][b, :, :k] = pk['features']['history']; f['dimensions'][b, :k] = pk['features']['dimensions']
        f['road_boundaries'][b, :rb] = pk['features']['road_boundaries']; f['road_boundary_mask'][b, :rb] = pk['features']['road_boundary_mask']
        f['ego_mask'][b, :k] = pk['features']['ego_mask']; f['agent_mask'][b, :k] = True; coef[b, :k] = pk['coef']
    t = lambda a: torch.from_numpy(a).to(device)
    return {k: t(v) for k, v in f.items()}, t(coef), torch.tensor([packs[i]['p'] for i in rows], dtype=torch.float32, device=device)


def train(device='cuda:1'):
    torch.set_num_threads(4); torch.manual_seed(SEED); np.random.seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.use_deterministic_algorithms(True)
    packs = build_packs(); train_rows = [i for i, p in enumerate(packs) if not p['stop']]; stop_rows = [i for i, p in enumerate(packs) if p['stop']]
    model = ClassifierFreePercentileDenoiser(16).to(device); schedule = CosineDiffusionSchedule(100).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    order_rng = np.random.default_rng(SEED); rng = torch.Generator().manual_seed(SEED + 2); drop = torch.Generator().manual_seed(SEED + 3)
    def objective(f, clean, p, present, g):
        noise = torch.randn(clean.shape, generator=g, dtype=torch.float32).to(device)
        t = torch.randint(100, (len(p),), generator=g).to(device)
        return cfg_training_loss(model, schedule, clean, f, p, present, timesteps=t, noise=noise)['loss']
    best, best_epoch, stale, best_state, started = float('inf'), 0, 0, None, time.monotonic()
    with (OUT / 'epochs.jsonl').open('x') as log:
        for epoch in range(1, 401):
            model.train(); order = order_rng.permutation(train_rows); total = 0.
            for s in range(0, len(order), 128):
                rows = order[s:s + 128]; f, clean, p = batch(packs, rows, device)
                present = (torch.rand(len(rows), generator=drop) >= DROP).to(device)
                opt.zero_grad(set_to_none=True); loss = objective(f, clean, p, present, rng)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); total += float(loss) * len(rows)
            model.eval(); g = torch.Generator().manual_seed(SEED + 1); val = 0.
            with torch.no_grad():
                for s in range(0, len(stop_rows), 128):
                    rows = stop_rows[s:s + 128]; f, clean, p = batch(packs, rows, device)
                    val += float(objective(f, clean, p, torch.ones(len(rows), dtype=torch.bool, device=device), g)) * len(rows)
            val /= len(stop_rows)
            if val < best: best, best_epoch, stale, best_state = val, epoch, 0, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else: stale += 1
            log.write(json.dumps(dict(epoch=epoch, FIT_loss=total / len(train_rows), STOP_loss=val, best_epoch=best_epoch)) + '\n'); log.flush()
            if epoch % 20 == 0: print(json.dumps(dict(epoch=epoch, STOP=val, best_epoch=best_epoch, seconds=time.monotonic() - started)), flush=True)
            if epoch >= 10 and stale >= 40: break
    torch.save(dict(state_dict=best_state, epoch=best_epoch), OUT / 'checkpoint.pt')
    json.dump(dict(status='complete', best_epoch=best_epoch, epochs=epoch, STOP_loss=best, train_clips=len(train_rows), stop_clips=len(stop_rows),
                   stop_recordings=STOP_RECORDINGS, condition_dropout=DROP, code=c.bind(Path(__file__))),
              open(OUT / 'training_result.json', 'w'), indent=2)
    print(json.dumps(dict(stage='trained', best_epoch=best_epoch, STOP=best)), flush=True)


def init():
    global MODEL, HN, CN, PROFILE, Z, REFM, HS, DS, SCHED, BASIS
    torch.set_num_threads(2)
    HN, CN = normalizers()
    MODEL = ClassifierFreePercentileDenoiser(16); MODEL.load_state_dict(torch.load(OUT / 'checkpoint.pt', map_location='cpu', weights_only=False)['state_dict'])
    MODEL.eval().requires_grad_(False)
    PROFILE = c.json_file(c.json_file(c.bind(QUEUE))['contract'])['generation_profile']
    Z = gen._load_split()
    cp = torch.load(gen.REF / 'checkpoint.pt', map_location='cpu', weights_only=False)
    REFM = make_reference(); REFM.load_state_dict(cp['state_dict']); REFM.eval()
    HS, DS, _ = normalizer(); SCHED = CosineDiffusionSchedule(100); BASIS = TrajectoryBasis(8)


ARMS = {'Full': (True, True), 'Condition only': (True, False), 'Guidance only': (False, True), 'No P': (False, False)}


def run_case(args):
    ci, i = args; z = Z
    lo, hi = z['offsets'][i], z['offsets'][i + 1]; n = int(hi - lo)
    history = np.transpose(z['history'][lo:hi], (1, 0, 2)).copy(); dims = z['dimensions'][lo:hi].copy()
    bounds = z['road_boundaries'][z['road_offsets'][i]:z['road_offsets'][i + 1]].copy(); ego, lead = int(z['ego'][i]), int(z['leader'][i])
    ego_mask = np.zeros(n, bool); ego_mask[ego] = True
    case = dict(history=history, dimensions=dims, road_boundaries=bounds, ego_mask=ego_mask, agent_mask=np.ones(n, bool), num_agents=n)
    feats = model_features(case, HN, torch.device('cpu'))
    with torch.no_grad():
        params = REFM(ref_features(z, [i], HS, DS, 'cpu'))
    rank_fn, target_fn, y0 = gen.bounded_functions(params, gen.ttc0_seconds(history, dims, ego, lead))
    rng = np.random.default_rng(int(hashlib.sha256(f'{gen.SEED}|{z["scene_id"][i]}'.encode()).hexdigest()[:8], 16))
    noises = [rng.standard_normal((n, 8, 2)).astype(np.float32) for _ in range(3)]
    decoder = TorchTrajectoryDecoder(BASIS, CN, history[-1]); rows = []
    for zi, noise in enumerate(noises):
        for arm, (cond, guide_on) in ARMS.items():
            for p in (GRID if (cond or guide_on) else (None,)):
                target = target_fn(p) if p is not None else None
                guide = GeometryOnlyGuidance(decoder, dims, history[-1], forbidden_rank, .5, 2., ego_index=ego, road_boundaries=bounds, **PROFILE)
                if guide_on:
                    guide = gen.TTCGuidance(decoder, dims, history[-1], forbidden_rank, .5, 2., ego_index=ego, road_boundaries=bounds,
                                            **PROFILE).setup(dims, lead, rank_fn, p, target, y0)
                adapter = _CFGPrediction(MODEL, feats, torch.tensor([p if cond else .5], dtype=torch.float32), 2.5 if cond else 0.)
                coef = ddim_sample(adapter, SCHED, feats, torch.tensor(noise[None]), steps=50, x0_callback=guide, prediction_type='v')[0].numpy()
                future = BASIS.decode(coef.astype(np.float64) * np.asarray(CN['scale']) + np.asarray(CN['mean']), history[-1])
                assert np.isfinite(future).all() and np.array_equal(future[0], history[-1])
                ttc = gen.exact_ttc(future, dims, ego, lead)
                for q in (GRID if p is None else (p,)):
                    tq = target_fn(q)
                    lo_r, hi_r = rank_fn(scaled(ttc)); e = max(lo_r - q, q - hi_r, 0.)
                    rows.append(dict(arm=arm, case_index=ci, scene_id=str(z['scene_id'][i]), stratum=gen.stratum(n), noise_index=zi,
                                     requested_p=q, TTC_seconds=ttc, target_TTC_seconds=tq * TTC_CAP / 4., target_atom=tq in (0., 4., y0),
                                     interval_error=e, interval_Fine=e <= .05, TTC_target_abs_error_seconds=abs(ttc - tq * TTC_CAP / 4.)))
    return rows


def evaluate():
    out = OUT / 'TEST'; out.mkdir(exist_ok=False)
    z, chosen = gen.select_cases()
    with get_context('spawn').Pool(24, initializer=init) as pool:
        rows = [r for rr in pool.map(run_case, list(enumerate(chosen)), chunksize=1) for r in rr]
    report = dict(histories=len(chosen), arms={})
    for arm in ARMS:
        mine = [r for r in rows if r['arm'] == arm]
        report['arms'][arm] = dict(all=gen.summarize(mine), by_p={str(p): gen.summarize([r for r in mine if r['requested_p'] == p]) for p in GRID})
        print(arm, json.dumps(report['arms'][arm]['all']), json.dumps({k: round(v['Fine'], 3) for k, v in report['arms'][arm]['by_p'].items()}), flush=True)
    report['code'] = c.bind(Path(__file__))
    (out / 'result.json').write_text(json.dumps(report, indent=2) + '\n'); (out / 'rows.json').write_text(json.dumps(rows) + '\n')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=('labels', 'train', 'evaluate')); ap.add_argument('--device', default='cuda:1')
    a = ap.parse_args()
    {'labels': lambda: labels(a.device), 'train': lambda: train(a.device), 'evaluate': evaluate}[a.cmd]()
