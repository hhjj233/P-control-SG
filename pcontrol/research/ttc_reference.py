#!/usr/bin/env python3
"""History-conditioned reference distribution of minimum TTC for pure car-following clips.

Architecture, inputs and input scales are those of the final PET reference (M2_TimeAttn:
two actorwise time Transformers and the PET-range relation readout on a 65-knot mixed CDF
over [0, 4] with atoms at both ends). Only the target changes: y = 4 * min(TTC, 60 s) / 60 s.
Training uses analytic CRPS on FIT clips, with two FIT recordings held out for early
stopping. TEST clips are only scored after training.
"""
import json, os, sys, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.reference.time_attention_cdf import TimeAttentionSceneCDF
from pcontrol.reference.scores import crps_from_params
from pcontrol.reference.mixed_cdf import rank_from_params, cdf_from_params

NP = ROOT / 'outputs/natural_percentile'
DATA = NP / 'experiments_v1/ttc_data'
OUT = NP / 'experiments_v1/ttc_reference'
CONTRACT = NP / 'transformer_publication_v1_20260916/final_validation_v1_preparation/frozen_contract.json'
TTC_CAP, SEED = float(os.environ.get('TTC_CAP_S', 60)), 20260929
CAP_TAG = '' if TTC_CAP == 60. else f'_cap{int(TTC_CAP)}'
OUT = NP / f'experiments_v1/ttc_reference{CAP_TAG}'


def scaled(ttc):
    return 4. * np.minimum(np.asarray(ttc, dtype=np.float64), TTC_CAP) / TTC_CAP


def load(role):
    z = dict(np.load(DATA / f'{role}.npz', allow_pickle=False))
    return z


def normalizer():
    contract = c.json_file(c.bind(CONTRACT))
    norm = c.json_file(contract['reference_models']['M2_TimeAttn']['descriptor']['normalizer'])
    return np.asarray(norm['history_scale'], dtype=np.float64), np.asarray(norm['dimension_scale'], dtype=np.float64), norm


def features(z, rows, hscale, dscale, device):
    counts = [z['offsets'][i + 1] - z['offsets'][i] for i in rows]
    roads = [z['road_offsets'][i + 1] - z['road_offsets'][i] for i in rows]
    B, N, R = len(rows), max(counts), max(roads)
    h = np.zeros((B, 13, N, 4), np.float32); d = np.zeros((B, N, 2), np.float32)
    road = np.zeros((B, R), np.float32); rmask = np.zeros((B, R), bool)
    ego = np.zeros((B, N), bool); agent = np.zeros((B, N), bool)
    for b, i in enumerate(rows):
        lo, hi = z['offsets'][i], z['offsets'][i + 1]; n = hi - lo
        h[b, :, :n] = np.transpose(z['history'][lo:hi], (1, 0, 2)) / hscale
        d[b, :n] = z['dimensions'][lo:hi] / dscale
        rb = z['road_boundaries'][z['road_offsets'][i]:z['road_offsets'][i + 1]]
        road[b, :len(rb)] = rb / hscale[1]; rmask[b, :len(rb)] = True
        ego[b, z['ego'][i]] = True; agent[b, :n] = True
    t = lambda a: torch.from_numpy(a).to(device)
    return dict(history=t(h), dimensions=t(d), road_boundaries=t(road), road_boundary_mask=t(rmask),
                ego_mask=t(ego), agent_mask=t(agent))


def make_model():
    torch.manual_seed(SEED)
    return TimeAttentionSceneCDF(torch.linspace(0, 4, 65, dtype=torch.float64), hidden_dim=64, heads=4,
                                 temporal_layers=2, temporal_feedforward_dim=128)


@torch.no_grad()
def crps(model, z, rows, hscale, dscale, device, bs=256):
    model.eval(); out = []
    for s in range(0, len(rows), bs):
        r = rows[s:s + bs]
        y = torch.tensor(scaled(z['ttc_seconds'][r]), dtype=torch.float64, device=device)
        out.append(crps_from_params(model(features(z, r, hscale, dscale, device)), y).cpu().numpy())
    return np.concatenate(out) * TTC_CAP / 4.  # seconds on the TTC scale


def train(device='cuda:0'):
    torch.manual_seed(SEED); np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=False)
    fit = load('FIT'); hscale, dscale, norm = normalizer()
    recs = sorted(set(fit['recording_id'].tolist()), key=int)
    stop_recs = recs[::7][:2]
    is_stop = np.isin(fit['recording_id'], stop_recs)
    train_rows, stop_rows = np.flatnonzero(~is_stop), np.flatnonzero(is_stop)
    model = make_model().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.)
    rng = np.random.default_rng(SEED); best, best_epoch, stale, best_state = float('inf'), 0, 0, None
    started = time.monotonic()
    with (OUT / 'epochs.jsonl').open('x') as log:
        for epoch in range(1, 101):
            model.train(); order = rng.permutation(train_rows); total = 0.
            for s in range(0, len(order), 64):
                r = order[s:s + 64]
                y = torch.tensor(scaled(fit['ttc_seconds'][r]), dtype=torch.float64, device=device)
                loss = crps_from_params(model(features(fit, r, hscale, dscale, device)), y, normalized=True).mean()
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite CRPS')
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); total += float(loss) * len(r)
            val = float(crps(model, fit, stop_rows, hscale, dscale, device).mean())
            if val < best:
                best, best_epoch, stale = val, epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else: stale += 1
            row = dict(epoch=epoch, train_CRPS_normalized=total / len(train_rows), STOP_CRPS_seconds=val,
                       best_epoch=best_epoch, elapsed_seconds=time.monotonic() - started)
            log.write(json.dumps(row) + '\n'); log.flush(); print(json.dumps(row), flush=True)
            if stale >= 15: break
    torch.save(dict(state_dict=best_state, epoch=best_epoch, ttc_cap_seconds=TTC_CAP, normalizer=norm,
                    architecture=make_model().architecture_config(), stop_recordings=stop_recs,
                    best_STOP_CRPS_seconds=best, train_clips=int(len(train_rows)), stop_clips=int(len(stop_rows))), OUT / 'checkpoint.pt')
    evaluate(device, started)


def evaluate(device='cuda:0', started=None):
    """TEST scoring after training only, from the saved checkpoint."""
    started = time.monotonic() if started is None else started
    cp = torch.load(OUT / 'checkpoint.pt', map_location='cpu', weights_only=False)
    hscale, dscale, _ = normalizer(); fit = load('FIT')
    train_rows = np.flatnonzero(~np.isin(fit['recording_id'], cp['stop_recordings']))
    model = make_model().to(device); model.load_state_dict(cp['state_dict'])
    best_epoch, stop_recs = cp['epoch'], cp['stop_recordings']
    epochs = [json.loads(l) for l in (OUT / 'epochs.jsonl').read_text().splitlines()]
    best = cp.get('best_STOP_CRPS_seconds', min(e['STOP_CRPS_seconds'] for e in epochs))
    cp.setdefault('stop_clips', int(np.isin(fit['recording_id'], stop_recs).sum()))
    test = load('TEST'); rows = np.arange(len(test['scene_id']))
    test_crps = crps(model, test, rows, hscale, dscale, device)
    # Marginal baseline: the FIT training-clip empirical distribution, the same for every history.
    y_fit = np.sort(scaled(fit['ttc_seconds'][train_rows])); y_test = scaled(test['ttc_seconds'])
    grid = np.linspace(0, 4, 2001); F = np.searchsorted(y_fit, grid, side='right') / len(y_fit)
    marginal = np.array([np.trapezoid((F - (grid >= y)) ** 2, grid) for y in y_test]) * TTC_CAP / 4.
    pit = []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(rows), 256):
            r = rows[s:s + 256]
            params = model(features(test, r, hscale, dscale, device))
            y = torch.tensor(y_test[r], dtype=torch.float64, device=device)
            lo = cdf_from_params(params, y, side='left'); hi = cdf_from_params(params, y, side='right')
            pit.append(((lo + hi) / 2).cpu().numpy())
    pit = np.concatenate(pit)
    hist = np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit)
    result = dict(status='complete', best_epoch=best_epoch, STOP_CRPS_seconds=best, TTC_cap_seconds=TTC_CAP,
                  train_clips=int(len(train_rows)), stop_clips=cp['stop_clips'], stop_recordings=stop_recs,
                  TEST_clips=int(len(rows)), TEST_CRPS_seconds=float(test_crps.mean()),
                  TEST_marginal_CRPS_seconds=float(marginal.mean()),
                  TEST_CRPS_skill_vs_marginal=float(1 - test_crps.mean() / marginal.mean()),
                  TEST_PIT_histogram_mid=hist.tolist(), TEST_PIT_max_bin_deviation=float(np.abs(hist - .1).max()),
                  checkpoint=c.bind(OUT / 'checkpoint.pt'), code=c.bind(Path(__file__)),
                  wall_seconds=time.monotonic() - started)
    (OUT / 'training_result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('checkpoint', 'code')}), flush=True)


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'evaluate':
        evaluate(sys.argv[2] if len(sys.argv) > 2 else 'cuda:0')
    else:
        train(sys.argv[1] if len(sys.argv) > 1 else 'cuda:0')
