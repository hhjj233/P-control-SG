# Training

The released models were trained in the order below. Every step has a script in `pcontrol/research/` and a configuration in `configs/natural_percentile/`.

**How the scripts were run.** The scripts are kept as they were run for the paper.
- Each script reads its configuration, which names its input files with their SHA-256 and the folder it writes to.
- The scripts check their inputs against these digests and write each output only once.
- Most scripts take `--policy <configuration> --policy-sha256 <digest>` and a subcommand. The header of each script describes its commands.

**To retrain from scratch:**
1. Prepare your scenes.
2. Write configurations that point to your own data and output folders.
3. Run the steps in this order.

The configurations that ship here keep the paths and digests of the original runs, so they document what was run but cannot be reused as they are.

## 1. Scenes
| Step | Script | Configuration |
| --- | --- | --- |
| highD source manifest of the training recordings | `build_natural_highd_clips.py` | `build_highd_clips_v2.json` |
| History-selected ego scenes with their recorded futures and PET | `build_natural_ego_scenes.py` | `build_ego_scenes_v1.json` |
| Scenes whose vehicles are all observed over the future | `build_natural_complete_scene_view.py` | `complete_scene_view_v1.json` |
| More fitting scenes (up to 4,096 histories per recording) | `expand_natural_complete_scenes.py` | `expand_complete_scenes_v1.json` |

This gives 9,913 fitting, 538 early-stopping, 504 calibration and 487 development scenes from the training recordings. The evaluation scenes are built by `prepare_final_natural_validation.py` (see `docs/experiments.md`) with the same rules, which `pcontrol/tools/prepare_highd.py` reproduces.

## 2. Reference
| Step | Script | Configuration |
| --- | --- | --- |
| Train the static-query and range-query references with CRPS, early stopping on its own recordings | `pilot_time_attention_cdf.py` | `time_attention_cdf_pilot_v1.json` |
| Select the calibration map on the calibration scenes and validate on the development scenes | `run_transformer_reference_validation.py`, `validate_time_attention_reference.py` | `transformer_reference_validation_v1.json` |
| Validation of the reference used by the generator | `validate_guarded_reference.py` | `guarded_reference_validation_v1.json` |
| Leave-recording-out references, one per fold, for the generator's percentile labels | `crossfit_time_attention_reference.py`, `crossfit_guarded_reference.py` | `guarded_reference_crossfit_v1.json` |

The range-query reference with its calibration map is `checkpoints/reference.pt`. The encoders compared in the appendix of the paper were trained by `run_transformer_reference_ablation.py` and `train_transformer_capacity_control.py`, and validated by `validate_transformer_capacity_control.py`. The classical estimators of Table 1 come from `classical_reference_baselines.py`.

## 3. Generator data
| Step | Script |
| --- | --- |
| Trajectory coefficients of the scenes and their normalizers | `prepare_natural_diffusion.py` |
| Leave-recording-out percentile labels of the fitting scenes | `prepare_natural_direct_p_labels.py` |
| Risk tangents used by the training losses | `prepare_natural_risk_tangent.py` |

## 4. Generator
Each stage continues from the weights of the previous one. Only the first stage starts from random weights.

| Stage | Script | Configuration | Epochs |
| --- | --- | --- | --- |
| 1. Percentile-conditioned joint diffusion from scratch, on leave-recording-out labels | `train_natural_direct_p.py` | `natural_direct_p_policy_v1.json` | 80 |
| 2. Paired continuation with classifier-free guidance on p | `refine_natural_direct_p.py` | `natural_direct_p_refinement_policy_v1.json` | 120 |
| 3. Risk-tangent losses | `train_natural_risk_tangent.py` | `natural_risk_tangent_policy_v1.json` | 40 |
| 4. CDF-descriptor adapter | `pilot_natural_ruler_context.py` | `ruler_context_pilot_v1.json` | 3 |
| 5. Physical-target conditioning | `pilot_natural_target_ruler.py` | `target_ruler_pilot_v1.json` | 3 |
| 6. Vehicle-level risk adapter | `pilot_natural_dynamic_risk.py` | `dynamic_risk_pilot_v1.json` | 3 |
| 7. Percentile and PET-value losses | `pilot_dynamic_dual_value.py` | `dynamic_dual_value_v1.json` | 3 |
| 8. Gradients through the sampling chain | `pilot_dynamic_full_chain.py` | `dynamic_full_chain_v1.json` | 1 |
| 9. Fine-tuning of the whole generator | `finetune_joint_risk_generator_v2.py` | `joint_generator_finetune_v2.json` | 6 |
| 10. Precision-band continuation | `train_precision_band_generator.py` | `precision_band_continuation_v1.json` | 3 |
| 11. Continuation over a wider range of histories and requests | `train_wide_coupled_risk_generator.py` | `wide_coupled_risk_continuation_v1.json` | 3 |
| 12–13. Conditioning on the temporal-relational reference, two runs | `train_time_attention_generator.py` | `time_attention_full_pipeline_v1.json` | 16, 3 |
| 14. Adaptation to the final reference | `train_guarded_generator_adaptation.py` | `guarded_generator_adaptation_v1.json` | 15 |
| 15. Final training of the released generator | `train_guarded_terminal_pair.py` | `guarded_atom_terminal_pair_v1.json` | 3 |

The final stage trains two variants from the same weights. The variant used in the paper, `canonical`, is `checkpoints/generator.pt`. Other development runs of the project do not enter this weight lineage.

## Sampling
Generation uses 50 DDIM steps of a 100-step cosine schedule, classifier-free guidance of scale 2.5 on the request, and one noise path per request. Risk guidance acts during the final 15 steps and background separation during the final five, together with the road constraints. The settings are stored in `checkpoints/generator.pt` under `generation_profile`.
