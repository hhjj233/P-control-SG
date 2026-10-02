#!/usr/bin/env python3
"""Score the added baselines with the exact rescoring used for Table 2.

Uses score_rows from pcontrol/research/score_canonical_evidence.py: atom-compatible
rank interval, Fine at 0.05, P-MAE, PET-target MAE, and geometry replayed from the saved
trajectories. Adds each RADE guidance scale as its own model.
"""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.research.score_canonical_evidence import score_rows

REV = ROOT / 'outputs/natural_percentile/experiments_v1'
OUT = REV / 'scored_baselines.json'
SOURCES = {f'rade_w{w}': REV / f'rade/TEST_w{w}/result.json' for w in ('2.5', '1', '4')}
# RADE v2 (longer training, pcontrol/research/rade_baseline_v2.py): every guidance scale evaluated on TEST.
SOURCES.update({'rade_v2_' + q.parent.name.replace('TEST_', ''): q for q in sorted((REV / 'rade_v2').glob('TEST_w*/result.json'))})


def main():
    models = {}
    for name, path in SOURCES.items():
        if not path.exists():
            print('missing', name); continue
        models[name] = score_rows(path, 'TEST', replay=True)
        s = models[name]['summary']
        print(name, json.dumps({k: s[k] for k in ('interval_Fine_rate', 'interval_P_MAE', 'PET_target_MAE_seconds')}), flush=True)
    OUT.write_text(json.dumps(dict(status='complete', models=models, code=c.bind(Path(__file__)),
                                   scorer='pcontrol/research/score_canonical_evidence.py::score_rows'), indent=2) + '\n')


if __name__ == '__main__':
    main()
