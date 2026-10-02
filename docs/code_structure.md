# Code structure

`pcontrol/api.py` and `pcontrol/tools/` are the entry points of the released models. The rest of `pcontrol/` is the code of the paper, and this page shows where each part of the method lives.

## Request definition (Section 3)
| Part | Module |
| --- | --- |
| Minimum ego–SV PET from vehicle footprints, capped at 4 s | `pcontrol/data/scene_pet.py`, `pcontrol/data/scene_pet_broadphase.py` |
| Mixed CDF with point masses at 0 and at the cap | `pcontrol/reference/mixed_cdf.py` |
| Physical target q_H(p) through the generalized inverse | `pcontrol/reference/torch_frozen_inverse.py`, `pcontrol/generation/atom_aware_rank_target.py` |
| Compatible percentile interval and interval error | `rank` of the conditioned reference (`pcontrol/plugins/risk_plugin.py`, `pcontrol/time_attention_pipeline/common.py`) |

## Reference (Section 4.1)
| Part | Module |
| --- | --- |
| Temporal–relational encoder with range-dependent queries | `pcontrol/reference/time_attention_cdf.py` |
| Static-query reference and other encoders of the comparison | `pcontrol/reference/history_encoder_ablation.py`, `pcontrol/reference/capacity_matched_mlp.py` |
| CRPS and calibration diagnostics | `pcontrol/reference/scores.py`, `pcontrol/reference/publication_calibration.py` |
| Calibration map | `pcontrol/reference/scene_calibration.py`, `pcontrol/reference/scene_context_calibration.py` |
| Leave-recording-out labels | `pcontrol/reference/direct_p_crossfit.py`, `pcontrol/publication_pipeline/generator_data.py` |

## Generator (Section 4.2)
| Part | Module |
| --- | --- |
| Trajectory coefficients (cosine acceleration basis) | `pcontrol/generation/trajectory_basis.py` |
| Diffusion schedule and DDIM sampling | `pcontrol/generation/diffusion.py` |
| Joint denoiser over vehicles with masked attention | `pcontrol/generation/diffusion.py` (`JointHistoryDenoiser`) |
| Percentile conditioning and classifier-free guidance on p | `pcontrol/generation/direct_p.py`, `pcontrol/generation/direct_p_cfg.py` |
| CDF descriptor, physical-target and vehicle-level risk inputs | `pcontrol/generation/cdf_shape_context.py`, `pcontrol/generation/ruler_context_direct_p.py`, `pcontrol/generation/target_ruler_direct_p.py`, `pcontrol/generation/dynamic_risk_direct_p.py` |
| Released generator class | `pcontrol/generation/trainable_risk_generator.py` (`TrainableRiskGenerator`, which builds on all of the above) |
| Training losses | `pcontrol/generation/*_loss.py`, `pcontrol/generation/terminal_*.py` |

## Sampling-time guidance and constraints (Section 4.3)
| Part | Module |
| --- | --- |
| Guided DDIM with the request | `pcontrol/generation/percentile_sampling_guidance.py` |
| Risk guidance toward the physical target | `pcontrol/generation/risk_guidance.py`, `pcontrol/generation/directional_percentile_guidance.py`, `pcontrol/generation/sampling_geometry.py` |
| Road constraints | `pcontrol/generation/road_constrained_guidance.py`, `pcontrol/generation/road_envelope.py` |
| Background separation | `pcontrol/generation/background_constrained_guidance.py`, `pcontrol/generation/background_envelope.py` |

## Evaluation
| Part | Module |
| --- | --- |
| Footprint overlap and road checks | `pcontrol/research/audit_pair_overlap_intervals.py`, `pcontrol/generation/evaluation.py`, `pcontrol/generation/scene_quality_diagnostics.py` |
| Final evaluation adapters | `pcontrol/publication_pipeline/` |

## Other folders
- `pcontrol/research/`: the scripts of the training stages and experiments ([training.md](training.md), [experiments.md](experiments.md)).
- `configs/natural_percentile/`: their configurations.
- `configs/highd_split_v1.json`: the recording split.
- `checkpoints/`: the released reference and generator.
