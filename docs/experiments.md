# Experiments of the paper

The scripts below produced the results of the paper. They live in `pcontrol/research/` and are kept as they were run (see [training.md](training.md) for how they read their inputs). The plotting and table-formatting scripts are not included.

| Result | What it measures | Scripts |
| --- | --- | --- |
| Final evaluation sets and generation | 11,649 primary-set scenes (16 test recordings), 9,277 additional-set scenes (14 recordings) and 603 single-recording scenes, 96 histories per set (70 for the single recording), five requests and three noise draws | `prepare_final_natural_validation.py`, `run_final_natural_validation.py`, `run_final_validation_thread_repair_v1.py`, `check_final_generator_replay.py`, `check_final_reference_cuda_v2.py`, `rehearse_final_validation.py` |
| Table 1 | CRPS, tail calibration and cap Brier score of the references | `pcontrol/publication_pipeline/final_execution_reference.py` (run by `run_final_natural_validation.py`), `classical_reference_baselines.py` (ECDF, kNN, quantile regression forest) |
| Table 2, Fig. 4 | Request realization under the interval criterion | `score_canonical_evidence.py` (our method), `baseline_models.py`, `run_baseline_generators.py`, `evaluate_baseline_generators.py` (P-CVAE, Standard P-diffusion), `rade_baseline.py`, `rade_baseline_v2.py`, `score_added_baselines.py` (RADE) |
| Section 5.3 | Cluster-bootstrap intervals of Fine over histories | `realization_statistics.py` |
| Table 3, Figs. 6–7 | P-path ablation: Condition only, Guidance only, No P path | `p_path_ablation.py`, `launch_p_path_ablation.py`, `audit_p_path_ablation.py` |
| Table 4 | Speed and acceleration of the generated futures against the observed ones | `kinematic_realism.py` |
| Table 5 | Percentile requests on time to collision in car following | `ttc_data.py`, `ttc_reference.py`, `ttc_bounded_crps.py`, `ttc_generation.py`, `ttc_method.py`, `ttc_behavior.py` |
| Table 6, data of Figs. 10–12 | IDM and MOBIL planner on the generated scenarios | `planner_test.py`, `planner_analysis.py`, `planner_percentile.py` |
| Appendix B | Reference encoders | `run_transformer_reference_ablation.py`, `train_transformer_capacity_control.py`, `validate_transformer_capacity_control.py`, `validate_time_attention_reference.py` |

The external priors of Table 2 (TrafficGen, CTG++ and STRIVE) were trained and sampled with their official implementations on the same training clips. Their code and outputs are not part of this repository. `kinematic_realism.py` and `realization_statistics.py` read their scored outputs from `external_baselines/`.

## Reproducing the main result with the released models
The tools reproduce the primary-set row of Table 2 without retraining:
```bash
python pcontrol/tools/prepare_highd.py --raw-root highD/data --split test --out data/scenes
python pcontrol/tools/generate.py --scenes data/scenes --paper-histories --out outputs/generation
python pcontrol/tools/evaluate.py --rows outputs/generation/rows.jsonl
```
`--paper-histories` selects the paper's 96 histories (32 per vehicle-count group) with the paper's selection rule, and `--noise paper` (the default) uses the paper's noise for each history. With the software versions listed in the README, the generated futures equal those of the paper.
