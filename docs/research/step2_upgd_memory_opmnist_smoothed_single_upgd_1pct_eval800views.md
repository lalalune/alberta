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
| `mlp_h64` | 0.018059 +/- 0.000000 | 0.927600 +/- 0.000000 | 0.098090 +/- 0.000000 | 0.108986 +/- 0.000000 |
| `mlp_h128` | 0.016609 +/- 0.000000 | 0.938800 +/- 0.000000 | 0.108615 +/- 0.000000 | 0.105385 +/- 0.000000 |
| `upgd_structure_softmax_h64_smooth30` | 0.015016 +/- 0.000000 | 0.945000 +/- 0.000000 | 0.121390 +/- 0.000000 | 0.110458 +/- 0.000000 |
| `upgd_structure_softmax_h64_smooth40` | 0.020447 +/- 0.000000 | 0.946200 +/- 0.000000 | 0.112425 +/- 0.000000 | 0.111101 +/- 0.000000 |
| `upgd_structure_softmax_h128_smooth30` | 0.014397 +/- 0.000000 | 0.949200 +/- 0.000000 | 0.121078 +/- 0.000000 | 0.111330 +/- 0.000000 |
| `upgd_structure_softmax_h128_smooth40` | 0.019916 +/- 0.000000 | 0.950400 +/- 0.000000 | 0.112098 +/- 0.000000 | 0.110958 +/- 0.000000 |

## Primary vs Best MLP


## Additional Candidate vs Best MLP

- `upgd_structure_softmax_h64_smooth30` `online_mean_mse` vs `mlp_h128`: +0.004167 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_smooth40` `online_mean_mse` vs `mlp_h128`: -0.001052 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h128_smooth30` `online_mean_mse` vs `mlp_h128`: +0.004642 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128_smooth40` `online_mean_mse` vs `mlp_h128`: -0.000666 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h64_smooth30` `online_mean_accuracy` vs `mlp_h128`: +0.011487 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_smooth40` `online_mean_accuracy` vs `mlp_h128`: +0.011508 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128_smooth30` `online_mean_accuracy` vs `mlp_h128`: +0.015167 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128_smooth40` `online_mean_accuracy` vs `mlp_h128`: +0.014517 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_smooth30` `final_window_mse` vs `mlp_h128`: +0.001592 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_smooth40` `final_window_mse` vs `mlp_h128`: -0.003839 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h128_smooth30` `final_window_mse` vs `mlp_h128`: +0.002211 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128_smooth40` `final_window_mse` vs `mlp_h128`: -0.003307 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h64_smooth30` `final_window_accuracy` vs `mlp_h128`: +0.006200 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_smooth40` `final_window_accuracy` vs `mlp_h128`: +0.007400 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128_smooth30` `final_window_accuracy` vs `mlp_h128`: +0.010400 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128_smooth40` `final_window_accuracy` vs `mlp_h128`: +0.011600 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_smooth30` `test_mse` vs `mlp_h64`: -0.023300 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h64_smooth40` `test_mse` vs `mlp_h64`: -0.014336 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h128_smooth30` `test_mse` vs `mlp_h64`: -0.022988 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h128_smooth40` `test_mse` vs `mlp_h64`: -0.014008 +/- 0.000000; wins/losses/ties 0/1/0.
- `upgd_structure_softmax_h64_smooth30` `test_accuracy` vs `mlp_h64`: +0.001472 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h64_smooth40` `test_accuracy` vs `mlp_h64`: +0.002115 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128_smooth30` `test_accuracy` vs `mlp_h64`: +0.002344 +/- 0.000000; wins/losses/ties 1/0/0.
- `upgd_structure_softmax_h128_smooth40` `test_accuracy` vs `mlp_h64`: +0.001972 +/- 0.000000; wins/losses/ties 1/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
