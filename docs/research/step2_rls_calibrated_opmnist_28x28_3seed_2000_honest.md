# Step 2 UPGD-Memory OPMNIST

Superseded on 2026-05-09: this rerun fixed post-target RLS weight scoring, but
the deployment gate could still turn on using the current label before returning
the current prediction. Use `step2_rls_calibrated_opmnist_28x28_3seed_2000_prequential.md`
for the fully causal result.

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
| `step2_hybrid_memory_trace_rls_cal` | 0.024221 +/- 0.002868 | 0.832500 +/- 0.021360 | 0.011379 +/- 0.001198 | 0.930000 +/- 0.008322 |
| `mlp_h64` | 0.060363 +/- 0.001010 | 0.660833 +/- 0.009610 | 0.067780 +/- 0.001490 | 0.581833 +/- 0.006547 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.043195 +/- 0.001660; wins/losses/ties 3/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.232667 +/- 0.016169; wins/losses/ties 3/0/0.
- `final_window_mse` vs `mlp_h64`: +0.036127 +/- 0.003798; wins/losses/ties 3/0/0.
- `final_window_accuracy` vs `mlp_h64`: +0.170000 +/- 0.023229; wins/losses/ties 3/0/0.
- `test_mse` vs `mlp_h64`: +0.055846 +/- 0.002155; wins/losses/ties 3/0/0.
- `test_accuracy` vs `mlp_h64`: +0.348167 +/- 0.014143; wins/losses/ties 3/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_rls_cal` `online_mean_mse` vs `mlp_h64`: +0.043579 +/- 0.001644; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `online_mean_accuracy` vs `mlp_h64`: +0.232167 +/- 0.016433; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `final_window_mse` vs `mlp_h64`: +0.036143 +/- 0.003855; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `final_window_accuracy` vs `mlp_h64`: +0.171667 +/- 0.020224; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `test_mse` vs `mlp_h64`: +0.056401 +/- 0.002427; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_rls_cal` `test_accuracy` vs `mlp_h64`: +0.348167 +/- 0.014814; wins/losses/ties 3/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
