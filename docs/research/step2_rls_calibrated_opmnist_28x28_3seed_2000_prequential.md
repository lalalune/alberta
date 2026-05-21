# Step 2 UPGD-Memory OPMNIST

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
| `step2_hybrid_memory_trace_rls_cal` | 0.024265 +/- 0.002865 | 0.832500 +/- 0.021360 | 0.011379 +/- 0.001198 | 0.930000 +/- 0.008322 |
| `mlp_h64` | 0.060363 +/- 0.001010 | 0.660833 +/- 0.009610 | 0.067780 +/- 0.001490 | 0.581833 +/- 0.006547 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.043195 +/- 0.001660; wins/losses/ties 3/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.232667 +/- 0.016169; wins/losses/ties 3/0/0.
- `final_window_mse` vs `mlp_h64`: +0.036127 +/- 0.003798; wins/losses/ties 3/0/0.
- `final_window_accuracy` vs `mlp_h64`: +0.170000 +/- 0.023229; wins/losses/ties 3/0/0.
- `test_mse` vs `mlp_h64`: +0.055846 +/- 0.002155; wins/losses/ties 3/0/0.
- `test_accuracy` vs `mlp_h64`: +0.348167 +/- 0.014143; wins/losses/ties 3/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_rls_cal` `online_mean_mse` vs `mlp_h64`: +0.043549 +/- 0.001638; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `online_mean_accuracy` vs `mlp_h64`: +0.232167 +/- 0.016433; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `final_window_mse` vs `mlp_h64`: +0.036098 +/- 0.003852; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `final_window_accuracy` vs `mlp_h64`: +0.171667 +/- 0.020224; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `test_mse` vs `mlp_h64`: +0.056401 +/- 0.002427; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `test_accuracy` vs `mlp_h64`: +0.348167 +/- 0.014814; wins/losses/ties 3/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

## 2026-05-09 Fully Prequential RLS Assessment

This run uses the fully causal RLS calibrator: current predictions are computed
from the previous RLS weights and the previous deployment gate. The current
label updates the RLS weights and gate only for the next step.

Paired `step2_hybrid_memory_trace_rls_cal` minus
`step2_hybrid_memory_trace` deltas, where positive favors RLS calibration:

- Online mean MSE: `+0.000354 +/- 0.000024`, `3/3` wins.
- Online mean accuracy: `-0.000500 +/- 0.000500`, `0/1/2` wins/losses/ties.
- Final-window MSE: `-0.000029 +/- 0.000071`, `2/1/0` wins/losses/ties.
- Final-window accuracy: `+0.001667 +/- 0.003333`, `2/1/0` wins/losses/ties.
- Held-out test MSE: `+0.000555 +/- 0.000286`, `3/3` wins.
- Held-out test accuracy: `-0.000000 +/- 0.000764`, `1/2/0` wins/losses/ties.

Interpretation: prediction-space RLS calibration is safe and slightly useful
for online/held-out MSE on this compact OPMNIST-style stream, but it does not
produce a reliable accuracy gain. Keep it behind `--include-rls-calibrated`
instead of promoting it as the default.
