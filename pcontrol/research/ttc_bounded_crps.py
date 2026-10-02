#!/usr/bin/env python3
"""CRPS of the bounded TTC reference on the TEST car-following clips.

The future window starts at the last observed frame, so the minimum TTC over the future can
never exceed TTC_0, the TTC at that frame, which the history determines. The bounded reference
moves the mass that the learned reference assigns above TTC_0 to an atom at TTC_0. The realized
value always lies at or below TTC_0, so the bound can only lower CRPS, clip by clip (asserted).
CRPS is integrated on an 801-point grid of the scaled support [0, 4] and reported in seconds.
The unconditional baseline is the empirical CDF of the reference's training clips.
"""
import json, os, sys
from pathlib import Path
import numpy as np
import torch
os.environ.setdefault('TTC_CAP_S', '20')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.reference.mixed_cdf import cdf_from_params
from pcontrol.research.ttc_reference import OUT, TTC_CAP, load, normalizer, make_model, features, scaled
from pcontrol.research.ttc_generation import ttc0_seconds


def main():
    cp = torch.load(OUT / 'checkpoint.pt', map_location='cpu', weights_only=False)
    model = make_model(); model.load_state_dict(cp['state_dict']); model.eval()
    hs, ds, _ = normalizer()
    test, fit = load('TEST'), load('FIT'); n = len(test['scene_id'])
    grid = np.linspace(0., 4., 801); tgrid = torch.tensor(grid, dtype=torch.float64)
    y = scaled(test['ttc_seconds'])
    train = np.flatnonzero(~np.isin(fit['recording_id'], cp['stop_recordings']))
    yf = np.sort(scaled(fit['ttc_seconds'][train])); marginal = np.searchsorted(yf, grid, side='right') / len(yf)
    raw, bounded, uncond, t0s = [], [], [], []
    with torch.no_grad():
        for s in range(0, n, 256):
            rows = np.arange(s, min(n, s + 256))
            F = cdf_from_params(model(features(test, rows, hs, ds, 'cpu')), tgrid[None].expand(len(rows), -1), side='right').numpy()
            for b, i in enumerate(rows):
                lo, hi = test['offsets'][i], test['offsets'][i + 1]
                h = np.transpose(test['history'][lo:hi], (1, 0, 2))
                t0 = ttc0_seconds(h, test['dimensions'][lo:hi], int(test['ego'][i]), int(test['leader'][i])); t0s.append(t0)
                y0 = float(scaled(t0)) if np.isfinite(t0) else 4.
                ind = (grid >= y[i]).astype(float)
                assert y[i] <= y0 + 1e-9, (i, y[i], y0)  # the realized minimum never exceeds TTC_0
                Fb = np.where(grid >= y0, 1., F[b])
                raw.append(np.trapezoid((F[b] - ind) ** 2, grid)); bounded.append(np.trapezoid((Fb - ind) ** 2, grid))
                uncond.append(np.trapezoid((marginal - ind) ** 2, grid))
    raw, bounded, uncond, t0s = map(np.array, (raw, bounded, uncond, t0s))
    assert (bounded <= raw + 1e-12).all()
    k = TTC_CAP / 4.
    report = dict(status='complete', TTC_cap_seconds=TTC_CAP, TEST_clips=n, grid_points=len(grid),
                  CRPS_unconditional_seconds=float(uncond.mean() * k), CRPS_learned_seconds=float(raw.mean() * k),
                  CRPS_bounded_seconds=float(bounded.mean() * k),
                  clips_with_TTC0_below_cap=int((t0s < TTC_CAP).sum()), clips_changed_by_bound=int((bounded < raw - 1e-12).sum()),
                  checkpoint=c.bind(OUT / 'checkpoint.pt'), code=c.bind(Path(__file__)))
    (OUT / 'bounded_crps.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k2: v for k2, v in report.items() if k2 not in ('checkpoint', 'code')}))


if __name__ == '__main__':
    main()
