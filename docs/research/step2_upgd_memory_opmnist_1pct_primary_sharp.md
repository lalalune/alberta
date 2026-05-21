# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `480000`
- Seeds: `1`
- Permutations: `800`
- Task block size: `60000`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.008546 +/- 0.000000 | 0.944200 +/- 0.000000 | 0.044990 +/- 0.000000 | 0.691950 +/- 0.000000 |
| `step2_hybrid_memory_trace_sharp` | 0.010566 +/- 0.000000 | 0.945000 +/- 0.000000 | 0.057244 +/- 0.000000 | 0.700187 +/- 0.000000 |
| `mlp_h64` | 0.017511 +/- 0.000000 | 0.931200 +/- 0.000000 | 0.075937 +/- 0.000000 | 0.425487 +/- 0.000000 |
| `mlp_h128` | 0.016469 +/- 0.000000 | 0.939200 +/- 0.000000 | 0.090950 +/- 0.000000 | 0.420037 +/- 0.000000 |
| `mlp_h64_sharp` | 0.012312 +/- 0.000000 | 0.927600 +/- 0.000000 | 0.089257 +/- 0.000000 | 0.401162 +/- 0.000000 |
| `mlp_h128_sharp` | 0.011130 +/- 0.000000 | 0.936400 +/- 0.000000 | 0.092561 +/- 0.000000 | 0.429425 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h128_sharp`: +0.003716 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h128`: +0.013215 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h128_sharp`: +0.002584 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h128`: +0.005000 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h64`: +0.030947 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h128_sharp`: +0.262525 +/- 0.000000; wins/losses/ties 1/0/0.

## Additional Candidate vs Best MLP

- `step2_hybrid_memory_trace_sharp` `online_mean_mse` vs `mlp_h128_sharp`: +0.000833 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_sharp` `online_mean_accuracy` vs `mlp_h128`: +0.013523 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_sharp` `final_window_mse` vs `mlp_h128_sharp`: +0.000565 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_sharp` `final_window_accuracy` vs `mlp_h128`: +0.005800 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_sharp` `test_mse` vs `mlp_h64`: +0.018694 +/- 0.000000; wins/losses/ties 1/0/0.
- `step2_hybrid_memory_trace_sharp` `test_accuracy` vs `mlp_h128_sharp`: +0.270762 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
