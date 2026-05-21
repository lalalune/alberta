# Step 2 UPGD-Memory OPMNIST

Superseded on 2026-05-09: this note was generated before the RLS deployment
gate was made fully prequential. The RLS calibrator weights were causal, but
the current-step gate could still be turned on by the current label. Use the
`*_honest.md` rerun after the gate-causality fix for promotion decisions.

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `sklearn_digits_28x28`
- Steps: `2000`
- Seeds: `3`
- Permutations: `5`
- Task block size: `400`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.024236 +/- 0.002808 | 0.830833 +/- 0.023467 | 0.011934 +/- 0.000920 | 0.930000 +/- 0.007687 |
| `step2_hybrid_memory_trace_rls_cal` | 0.022672 +/- 0.002651 | 0.837500 +/- 0.022500 | 0.011339 +/- 0.001164 | 0.931000 +/- 0.007566 |
| `mlp_h64` | 0.060363 +/- 0.001010 | 0.660833 +/- 0.009610 | 0.067780 +/- 0.001490 | 0.581833 +/- 0.006547 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.043195 +/- 0.001660; wins/losses/ties 3/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.232667 +/- 0.016169; wins/losses/ties 3/0/0.
- `final_window_mse` vs `mlp_h64`: +0.036127 +/- 0.003798; wins/losses/ties 3/0/0.
- `final_window_accuracy` vs `mlp_h64`: +0.170000 +/- 0.023229; wins/losses/ties 3/0/0.
- `test_mse` vs `mlp_h64`: +0.055846 +/- 0.002155; wins/losses/ties 3/0/0.
- `test_accuracy` vs `mlp_h64`: +0.348167 +/- 0.014143; wins/losses/ties 3/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_rls_cal` `online_mean_mse` vs `mlp_h64`: +0.044908 +/- 0.001628; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `online_mean_accuracy` vs `mlp_h64`: +0.237000 +/- 0.015567; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `final_window_mse` vs `mlp_h64`: +0.037691 +/- 0.003640; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `final_window_accuracy` vs `mlp_h64`: +0.176667 +/- 0.020276; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `test_mse` vs `mlp_h64`: +0.056441 +/- 0.002387; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `test_accuracy` vs `mlp_h64`: +0.349167 +/- 0.014099; wins/losses/ties 3/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

## 2026-05-09 RLS Calibration Assessment

This run tests whether the Step 8 reward-calibration result transfers to the
Step 2 online classifier setting without adding pixel-space RLS cost. The
candidate uses a tiny prediction-space RLS calibrator over
`[1, base_prediction]` with identity initialization and a causal utility gate.
The covariance is only `(n_classes + 1) x (n_classes + 1)` per class.

Paired `step2_hybrid_memory_trace_rls_cal` minus
`step2_hybrid_memory_trace` deltas, where positive favors RLS calibration:

- Online mean MSE: `+0.001713`, `3/3` wins.
- Online mean accuracy: `+0.004333`, `3/3` wins.
- Final-window MSE: `+0.001564`, `3/3` wins.
- Final-window accuracy: `+0.006667`, `2/3` wins.
- Held-out test MSE: `+0.000595`, `3/3` wins.
- Held-out test accuracy: `+0.001000`, `2/3` wins.

The same calibrator around a bare single UPGD improves online/final MSE but
hurts held-out accuracy versus the bare UPGD in the smaller 8x8 base ablation.
The useful integration point is therefore the packaged hybrid-memory learner,
not a standalone calibrated single learner.

Status: promising candidate, not yet promoted default. It should be rerun on
larger OPMNIST fractions and the existing digit regimes before changing the
canonical Step 2 default.
