# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `openml`
- Steps: `48000000`
- Seeds: `1`
- Permutations: `800`
- Task block size: `60000`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `mlp_h64` | 0.026268 +/- 0.000000 | 0.892600 +/- 0.000000 | 0.103178 +/- 0.000000 | 0.127530 +/- 0.000000 |
| `mlp_h128` | 0.021230 +/- 0.000000 | 0.909800 +/- 0.000000 | 0.100632 +/- 0.000000 | 0.130353 +/- 0.000000 |
| `upgd_structure_linear_h128` | 0.021811 +/- 0.000000 | 0.906200 +/- 0.000000 | 0.100694 +/- 0.000000 | 0.130291 +/- 0.000000 |
| `upgd_structure_softmax_h128` | 0.014868 +/- 0.000000 | 0.908600 +/- 0.000000 | 0.148280 +/- 0.000000 | 0.134712 +/- 0.000000 |

## Primary vs Best MLP


## Additional Candidate vs Best MLP

- `upgd_structure_linear_h128` `online_mean_mse` vs `mlp_h128`: -0.000546 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h128` `online_mean_mse` vs `mlp_h128`: +0.010102 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h128` `online_mean_accuracy` vs `mlp_h128`: -0.007040 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h128` `online_mean_accuracy` vs `mlp_h128`: +0.018775 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h128` `final_window_mse` vs `mlp_h128`: -0.000581 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h128` `final_window_mse` vs `mlp_h128`: +0.006362 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_linear_h128` `final_window_accuracy` vs `mlp_h128`: -0.003600 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h128` `final_window_accuracy` vs `mlp_h128`: -0.001200 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_linear_h128` `test_mse` vs `mlp_h128`: -0.000062 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h128` `test_mse` vs `mlp_h128`: -0.047648 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_linear_h128` `test_accuracy` vs `mlp_h128`: -0.000062 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h128` `test_accuracy` vs `mlp_h128`: +0.004359 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

This run satisfies that protocol gate for one seed. The best candidate in this
run is `upgd_structure_softmax_h128`: versus `mlp_h128`, it improves online
MSE by `+0.010102`, online accuracy by `+0.018775`, final-window MSE by
`+0.006362`, and all-permutation held-out test accuracy by `+0.004359`. It
still trails on final-window accuracy by `-0.001200` and held-out test MSE by
`-0.047648`, so this is stronger evidence for retained accuracy but not an
all-metric OPMNIST closure.
