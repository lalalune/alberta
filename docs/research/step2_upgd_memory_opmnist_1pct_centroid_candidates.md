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
| `mlp_h64` | 0.017364 +/- 0.000000 | 0.929800 +/- 0.000000 | 0.082161 +/- 0.000000 | 0.401650 +/- 0.000000 |
| `mlp_h128` | 0.016728 +/- 0.000000 | 0.939200 +/- 0.000000 | 0.086684 +/- 0.000000 | 0.405100 +/- 0.000000 |
| `mlp_h64_sharp` | 0.011707 +/- 0.000000 | 0.931400 +/- 0.000000 | 0.084441 +/- 0.000000 | 0.457712 +/- 0.000000 |
| `mlp_h128_sharp` | 0.010612 +/- 0.000000 | 0.940600 +/- 0.000000 | 0.128652 +/- 0.000000 | 0.320537 +/- 0.000000 |
| `centroid_hysteretic64_center_c030` | 0.010790 +/- 0.000000 | 0.938800 +/- 0.000000 | 0.101122 +/- 0.000000 | 0.348725 +/- 0.000000 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h128_sharp`: +0.003764 +/- 0.000000; wins/losses/ties 1/0/0.
- `online_mean_accuracy` vs `mlp_h128`: +0.013090 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_mse` vs `mlp_h128_sharp`: +0.002066 +/- 0.000000; wins/losses/ties 1/0/0.
- `final_window_accuracy` vs `mlp_h128_sharp`: +0.003600 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_mse` vs `mlp_h64`: +0.037170 +/- 0.000000; wins/losses/ties 1/0/0.
- `test_accuracy` vs `mlp_h64_sharp`: +0.234237 +/- 0.000000; wins/losses/ties 1/0/0.

## Additional Candidate vs Best MLP

- `centroid_hysteretic64_center_c030` `online_mean_mse` vs `mlp_h128_sharp`: +0.001687 +/- 0.000000; wins/losses/ties 1/0/0.
- `centroid_hysteretic64_center_c030` `online_mean_accuracy` vs `mlp_h128`: +0.009496 +/- 0.000000; wins/losses/ties 1/0/0.
- `centroid_hysteretic64_center_c030` `final_window_mse` vs `mlp_h128_sharp`: -0.000178 +/- 0.000000; wins/losses/ties 0/1/0.
- `centroid_hysteretic64_center_c030` `final_window_accuracy` vs `mlp_h128_sharp`: -0.001800 +/- 0.000000; wins/losses/ties 0/1/0.
- `centroid_hysteretic64_center_c030` `test_mse` vs `mlp_h64`: -0.018961 +/- 0.000000; wins/losses/ties 0/1/0.
- `centroid_hysteretic64_center_c030` `test_accuracy` vs `mlp_h64_sharp`: -0.108987 +/- 0.000000; wins/losses/ties 0/1/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
