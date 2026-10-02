#!/usr/bin/env python3
"""Percentile requests on minimum TTC for pure car-following histories.

The generator is the final one (frozen, no retraining). Its trained null branch
(classifier-free scale 0) gives a natural prior, and the same road and background
geometry callbacks as the paper keep the scene feasible. TTC control adds a
sampling-time step in the last 20 DDIM steps: a Newton-type update of the predicted
clean coefficients toward the TTC target from the TTC reference, with at most two inner
steps, a 0.5 m position cap per step, and a stop once the realized percentile is within
0.01 of the request. It mirrors the paper's PET guidance with TTC in place of PET.

Arms: 'TTC guidance' (reference target plus TTC guidance) and 'No guidance' (the same
prior and geometry callbacks, no risk step). Evaluation: 96 TEST car-following histories
(32 per vehicle-count stratum), p in {0.1, 0.3, 0.5, 0.7, 0.9}, three noise draws each.
"""
import hashlib, json, math, os, sys, time
from multiprocessing import get_context
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.publication_pipeline.final_generation_adapter import FrozenFinalGenerator, initialize_runtime
from pcontrol.generation.direct_p_cfg import _CFGPrediction
from pcontrol.generation.diffusion import ddim_sample
from pcontrol.reference.mixed_cdf import rank_from_params, quantile_from_params
from pcontrol.research.p_path_ablation import GeometryOnlyGuidance, forbidden_rank
from pcontrol.research.ttc_reference import TTC_CAP, CAP_TAG, scaled, features as ref_features, make_model, normalizer
from pcontrol.research.ttc_reference import OUT as REF_OUT

NP = ROOT / 'outputs/natural_percentile'
DATA = NP / 'experiments_v1/ttc_data'
REF = REF_OUT
OUT = NP / f'experiments_v1/ttc_generation{CAP_TAG}'
QUEUE = NP / 'transformer_publication_v1_20260916/final_validation_v1/TEST/queue/queue.json'
GRID = (.1, .3, .5, .7, .9)
PER_STRATUM, SEED = 32, 20260929
LAST_STEPS, INNER, STRENGTH, POS_CAP, TOL = (int(os.environ.get('TTC_LAST', 20)), int(os.environ.get('TTC_INNER', 2)),
                                             float(os.environ.get('TTC_STRENGTH', .5)), float(os.environ.get('TTC_POSCAP', .5)), .01)
SIGNAL = os.environ.get('TTC_SIGNAL', 'ttc')  # 'ttc': gap/closing; 'inverse': closing/gap (1/TTC); 'inverse_soft': log-sum-exp of 1/TTC over time
BETA = float(os.environ.get('TTC_BETA', 200.))  # soft-max temperature for 'inverse_soft' (1/s)
ATOM_MARGIN = float(os.environ.get('TTC_ATOM_MARGIN', 0.))  # 1/s below the t0 approach rate for targets at the TTC(0) atom
SPLIT = os.environ.get('TTC_SPLIT', 'TEST')  # 'DEV': FIT recordings held out from the TTC reference (guidance design)
CONFIG = f"{SIGNAL}_L{LAST_STEPS}_I{INNER}_S{STRENGTH}_C{POS_CAP}" + (f"_B{BETA:g}" if SIGNAL == "inverse_soft" else "") + ("_bounded" if os.environ.get("TTC_BOUNDED", "0") == "1" else "") + (f"_M{ATOM_MARGIN:g}" if ATOM_MARGIN else "")
# The first TEST run (initial settings) stays in ttc_generation/. Settings were then chosen on DEV,
# and the chosen configuration is run once on TEST into its own folder.
if SPLIT != 'TEST':
    OUT = OUT.parent / f"{OUT.name}_dev_{CONFIG}"
elif CONFIG != 'ttc_L20_I2_S0.5_C0.5':
    OUT = OUT.parent / f"{OUT.name}_test_{CONFIG}"


BOUNDED = os.environ.get('TTC_BOUNDED', '0') == '1'
# Physical bound: the window includes t0, so the minimum TTC cannot exceed the TTC of the observed
# t0 state. The bounded reference moves the probability that the learned CDF places above TTC(0)
# into a point mass at TTC(0). It is used for targets, scores and history selection.


def ttc0_seconds(history, dims, ego, lead):
    last = history[-1]
    gap = last[lead, 0] - last[ego, 0] - .5 * (dims[ego, 0] + dims[lead, 0]); closing = last[ego, 2] - last[lead, 2]
    return float(gap / closing) if closing > 0 and gap > 0 else float('inf')


def bounded_functions(params, t0_seconds):
    """rank(y_scaled) -> (p_low, p_high) and target(p) -> y_scaled for the reference bounded at TTC(0)."""
    from pcontrol.reference.mixed_cdf import cdf_from_params
    y0 = float(scaled(t0_seconds)) if BOUNDED and np.isfinite(t0_seconds) else 4.
    def cdf(y, side):
        return float(cdf_from_params(params, torch.tensor([y], dtype=torch.float64), side=side)[0])
    def rank(y):
        if y >= y0 - 1e-12:
            return 0., 1. - cdf(y0, 'left')
        return 1. - cdf(y, 'right'), 1. - cdf(y, 'left')
    def target(p):
        return min(float(quantile_from_params(params, torch.tensor([1. - p], dtype=torch.float64))[0]), y0)
    return rank, target, y0


def stratum(n):
    return 'N3_5' if n <= 5 else ('N6_8' if n <= 8 else 'N9_plus')


def exact_ttc(future, dims, ego, lead):
    f = np.asarray(future, dtype=np.float64)
    gap = f[:, lead, 0] - f[:, ego, 0] - .5 * (dims[ego, 0] + dims[lead, 0])
    closing = f[:, ego, 2] - f[:, lead, 2]
    if np.any(gap <= 0): return 0.
    ttc = np.where(closing > 0, gap / np.where(closing > 0, closing, 1.), np.inf)
    return float(min(ttc.min(), TTC_CAP))


class TTCGuidance(GeometryOnlyGuidance):
    """Paper geometry callbacks plus a TTC Newton step in the late DDIM steps."""
    def setup(self, dims, lead, rank_fn, requested_p, target_scaled, y0=None):
        self.dims_np, self.lead, self.rank_fn2 = np.asarray(dims, dtype=np.float64), lead, rank_fn
        self.p, self.target = float(requested_p), float(target_scaled); self.ttc_trace = []
        self.y0 = y0  # bounded-reference atom at TTC(0), if any
        return self

    def __call__(self, x0, timesteps, x_t):
        result = super().__call__(x0, timesteps, x_t)
        step = self.calls - 1
        if step < 50 - LAST_STEPS: return result
        current = result[0].detach().double().clone(); ego, lead = self.ego_index, self.lead
        half = .5 * (self.dims_np[ego, 0] + self.dims_np[lead, 0])
        for inner in range(INNER):
            with torch.enable_grad():
                state = current.detach().requires_grad_(True); fut = self.decoder(state)
                ttc = exact_ttc(fut.detach().numpy(), self.dims_np, ego, lead)
                rank = self.rank_fn2(scaled(ttc)); err_p = max(rank[0] - self.p, self.p - rank[1], 0.)
                row = dict(step=step, inner=inner, TTC=ttc, p_low=rank[0], p_high=rank[1], interval_error=err_p)
                if err_p <= TOL:
                    row['kind'] = 'within_target_band'; self.ttc_trace.append(row); break
                gap = fut[:, lead, 0] - fut[:, ego, 0] - half
                closing = fut[:, ego, 2] - fut[:, lead, 2]
                if SIGNAL in ('inverse', 'inverse_soft'):
                    # 1/TTC is linear in the closing speed; the target 1/TTC is 1/q (0 at the cap). The soft version
                    # replaces the maximum over time by log-sum-exp, so one step lowers every near-maximal instant.
                    rate = closing / gap.clamp_min(.5)
                    signal = rate.max() if SIGNAL == 'inverse' else torch.logsumexp(BETA * rate, 0) / BETA
                    exact_rate = rate.max().detach()  # the error uses the exact maximum, the soft version only the direction
                    goal = 0. if self.target >= 4. else 1. / (self.target * TTC_CAP / 4.)
                    if self.y0 is not None and self.y0 < 4. and abs(self.target - self.y0) < 1e-12:
                        goal -= ATOM_MARGIN  # inward margin: every later instant must approach more slowly than at t0
                    error = exact_rate - goal
                else:
                    signal = 4. * (gap / closing.clamp_min(.1)).min() / TTC_CAP
                    error = signal.detach() - self.target
                grad = torch.autograd.grad(signal, state)[0]; norm2 = grad.square().sum()
                if not bool(torch.isfinite(grad).all()) or float(norm2) <= 1e-14:
                    row['kind'] = 'flat_or_nonfinite'; self.ttc_trace.append(row); break
                delta = -STRENGTH * error * grad / norm2
                change = self.decoder(state.detach() + delta)[..., :2] - fut.detach()[..., :2]
                factor = min(1., POS_CAP / max(float(change.norm(dim=-1).max()), 1e-15))
                current = state.detach() + delta * factor
                row.update(kind='TTC_step', signal=float(signal), position_step_m=float(change.norm(dim=-1).max()) * factor)
                self.ttc_trace.append(row)
        return current[None].to(x0.dtype)

    def report(self):
        return dict(ttc_steps=sum(r['kind'] == 'TTC_step' for r in self.ttc_trace), trace=self.ttc_trace,
                    geometry_callbacks=self.calls, observed_future_input=False, best_of_K=False)


CAP_MASS_MAX = .5  # histories where the reference expects an approach: at most half its mass at the 60 s cap


def subset(z, rows):
    """Row subset of a ragged clip archive (agents and road boundaries re-offset)."""
    out = {k: z[k][rows] for k in ('scene_id', 'recording_id', 'num_agents', 'ego', 'leader', 'ttc_seconds')}
    agents = [np.arange(z['offsets'][i], z['offsets'][i + 1]) for i in rows]
    roads = [np.arange(z['road_offsets'][i], z['road_offsets'][i + 1]) for i in rows]
    for k in ('history', 'future', 'dimensions'): out[k] = np.concatenate([z[k][a] for a in agents])
    out['road_boundaries'] = np.concatenate([z['road_boundaries'][r] for r in roads])
    out['offsets'] = np.concatenate([[0], np.cumsum([len(a) for a in agents])])
    out['road_offsets'] = np.concatenate([[0], np.cumsum([len(r) for r in roads])])
    return out


def select_cases():
    """History-only selection: TEST car-following clips whose TTC reference puts at most half of its
    mass on 'no approach within 60 s', then 32 per vehicle-count stratum in a fixed hash order."""
    cp = torch.load(REF / 'checkpoint.pt', map_location='cpu', weights_only=False)
    z = dict(np.load(DATA / ('TEST.npz' if SPLIT == 'TEST' else 'FIT.npz'), allow_pickle=False))
    if SPLIT != 'TEST':
        keep = np.isin(z['recording_id'], cp['stop_recordings'])
        z = subset(z, np.flatnonzero(keep))
    model = make_model(); model.load_state_dict(cp['state_dict']); model.eval()
    hs, ds, _ = normalizer(); n = len(z['scene_id']); cap = []
    with torch.no_grad():
        for s0 in range(0, n, 512):
            cap.append(model(ref_features(z, np.arange(s0, min(n, s0 + 512)), hs, ds, 'cpu')).cap_mass.numpy())
    cap = np.concatenate(cap)
    if BOUNDED:  # a history already closing below the cap at t0 has no mass at the cap
        for i in range(n):
            lo, hi = z['offsets'][i], z['offsets'][i + 1]
            h = np.transpose(z['history'][lo:hi], (1, 0, 2))
            if ttc0_seconds(h, z['dimensions'][lo:hi], int(z['ego'][i]), int(z['leader'][i])) < TTC_CAP: cap[i] = 0.
    order = sorted(range(n), key=lambda i: hashlib.sha256(f"{SEED}|{z['scene_id'][i]}".encode()).hexdigest())
    chosen = {s: [] for s in ('N3_5', 'N6_8', 'N9_plus')}
    for i in order:
        s = stratum(int(z['num_agents'][i]))
        if cap[i] <= CAP_MASS_MAX and len(chosen[s]) < PER_STRATUM: chosen[s].append(i)
    select_cases.eligible = int((cap <= CAP_MASS_MAX).sum())
    return z, [i for s in ('N3_5', 'N6_8', 'N9_plus') for i in chosen[s]]


def _load_split():
    z = dict(np.load(DATA / ('TEST.npz' if SPLIT == 'TEST' else 'FIT.npz'), allow_pickle=False))
    if SPLIT != 'TEST':
        cp = torch.load(REF / 'checkpoint.pt', map_location='cpu', weights_only=False)
        z = subset(z, np.flatnonzero(np.isin(z['recording_id'], cp['stop_recordings'])))
    return z


def init():
    global ENGINE, TTC_MODEL, HS, DS, Z
    initialize_runtime(); torch.set_num_threads(2)
    Z = _load_split()
    queue = c.json_file(c.bind(QUEUE))
    ENGINE = FrozenFinalGenerator(queue['contract'], 'canonical')
    cp = torch.load(REF / 'checkpoint.pt', map_location='cpu', weights_only=False)
    TTC_MODEL = make_model(); TTC_MODEL.load_state_dict(cp['state_dict']); TTC_MODEL.eval().requires_grad_(False)
    HS, DS, _ = normalizer()


def run_case(args):
    ci, i = args
    z = Z
    lo, hi = z['offsets'][i], z['offsets'][i + 1]; n = int(hi - lo)
    history = np.transpose(z['history'][lo:hi], (1, 0, 2)).copy(); dims = z['dimensions'][lo:hi].copy()
    bounds = z['road_boundaries'][z['road_offsets'][i]:z['road_offsets'][i + 1]].copy()
    ego, lead = int(z['ego'][i]), int(z['leader'][i])
    ego_mask = np.zeros(n, bool); ego_mask[ego] = True
    feats = dict(history=history, dimensions=dims, road_boundaries=bounds, ego_mask=ego_mask, agent_mask=np.ones(n, bool))
    prepared = ENGINE.prepare_case(feats); case = prepared['case']
    with torch.no_grad():
        params = TTC_MODEL(ref_features(z, [i], HS, DS, 'cpu'))
    rank_fn, target_fn, y0 = bounded_functions(params, ttc0_seconds(history, dims, ego, lead))
    rng = np.random.default_rng(int(hashlib.sha256(f'{SEED}|{z["scene_id"][i]}'.encode()).hexdigest()[:8], 16))
    noises = [rng.standard_normal((n, 8, 2)).astype(np.float32) for _ in range(3)]
    rows, futures = [], {}
    for zi, noise in enumerate(noises):
        for arm in ('TTC guidance', 'No guidance'):
            for p in (GRID if arm == 'TTC guidance' else (None,)):
                target = target_fn(p) if p is not None else None
                guide = GeometryOnlyGuidance(prepared['decoder'], case['dimensions'], case['history'][-1], forbidden_rank, .5, 2.,
                                             ego_index=prepared['ego'], road_boundaries=case['road_boundaries'], **ENGINE.profile)
                if p is not None:
                    guide = TTCGuidance(prepared['decoder'], case['dimensions'], case['history'][-1], forbidden_rank, .5, 2.,
                                        ego_index=prepared['ego'], road_boundaries=case['road_boundaries'], **ENGINE.profile
                                        ).setup(case['dimensions'], lead, rank_fn, p, target, y0)
                adapter = _CFGPrediction(ENGINE.model, prepared['features'], torch.tensor([.5], dtype=torch.float32), 0.)
                tick = time.monotonic()
                coef = ddim_sample(adapter, ENGINE.schedule, prepared['features'], torch.tensor(noise[None]), steps=50,
                                   x0_callback=guide, prediction_type='v')[0].numpy().astype(np.float64)
                future = ENGINE.basis.decode(coef * np.asarray(ENGINE.cn['scale']) + np.asarray(ENGINE.cn['mean']), case['history'][-1])
                assert np.isfinite(future).all() and np.array_equal(future[0], case['history'][-1])
                ttc = exact_ttc(future, dims, ego, lead)
                for q in (GRID if p is None else (p,)):
                    tq = target_fn(q)
                    lo_r, hi_r = rank_fn(scaled(ttc)); e = max(lo_r - q, q - hi_r, 0.)
                    rows.append(dict(arm=arm, case_index=ci, scene_id=str(z['scene_id'][i]), stratum=stratum(n), num_agents=n,
                        noise_index=zi, requested_p=q, TTC_seconds=ttc, target_TTC_seconds=tq * TTC_CAP / 4.,
                        target_atom=tq in (0., 4.) or tq == bounded_functions(params, ttc0_seconds(history, dims, ego, lead))[2], p_low=lo_r, p_high=hi_r, interval_error=e, interval_Fine=e <= .05,
                        TTC_target_abs_error_seconds=abs(ttc - tq * TTC_CAP / 4.), sampling_seconds=time.monotonic() - tick,
                        guidance=guide.report() if p is not None else None))
                futures[f'{arm.replace(" ", "_")}_z{zi}' + (f'_p{p}' if p is not None else '')] = future
    np.savez_compressed(OUT / f'case_{ci:03d}.npz', **futures)
    return rows


def summarize(rows):
    e = np.array([r['interval_error'] for r in rows]); t = np.array([r['TTC_target_abs_error_seconds'] for r in rows])
    atom = np.array([r['target_atom'] for r in rows])
    return dict(requests=len(rows), Fine=float((e <= .05).mean()), P_MAE=float(e.mean()), TTC_target_MAE_seconds=float(t.mean()),
                atom_target_requests=int(atom.sum()), Fine_atom=float((e[atom] <= .05).mean()) if atom.any() else None,
                Fine_continuous=float((e[~atom] <= .05).mean()) if (~atom).any() else None)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    z, chosen = select_cases()
    # spawn: the parent has already used torch, and forked workers can deadlock on its thread pools.
    with get_context('spawn').Pool(24, initializer=init) as pool:
        rows = [r for rr in pool.map(run_case, list(enumerate(chosen)), chunksize=1) for r in rr]
    report = dict(split=SPLIT, signal=SIGNAL, histories=len(chosen), eligible_TEST_car_following_clips=select_cases.eligible, cap_mass_max=CAP_MASS_MAX,
                  TTC_cap_seconds=TTC_CAP, settings=dict(last_steps=LAST_STEPS, inner=INNER, strength=STRENGTH,
                  position_cap_m=POS_CAP, stop_tolerance=TOL), arms={})
    for arm in ('TTC guidance', 'No guidance'):
        mine = [r for r in rows if r['arm'] == arm]
        report['arms'][arm] = dict(all=summarize(mine), by_p={str(p): summarize([r for r in mine if r['requested_p'] == p]) for p in GRID},
            by_N={s: summarize([r for r in mine if r['stratum'] == s]) for s in ('N3_5', 'N6_8', 'N9_plus')})
        print(arm, json.dumps(report['arms'][arm]['all']), flush=True)
        print('  by p', json.dumps({p: round(v['Fine'], 3) for p, v in report['arms'][arm]['by_p'].items()}), flush=True)
    report['code'] = c.bind(Path(__file__))
    (OUT / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    (OUT / 'rows.json').write_text(json.dumps(rows) + '\n')


if __name__ == '__main__':
    main()
