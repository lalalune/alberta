# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `sklearn_digits_8x8`
- Steps: `600`
- Seeds: `3`
- Permutations: `5`
- Task block size: `120`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.035045 +/- 0.001669 | 0.756667 +/- 0.012019 | 0.019039 +/- 0.000663 | 0.888333 +/- 0.008838 |
| `mlp_h64` | 0.078308 +/- 0.001864 | 0.465000 +/- 0.026458 | 0.084559 +/- 0.001770 | 0.343000 +/- 0.020841 |
| `mlp_h128` | 0.088547 +/- 0.000232 | 0.470000 +/- 0.025981 | 0.093923 +/- 0.000752 | 0.336333 +/- 0.025432 |
| `upgd_structure_softmax_h64_rls_cal` | 0.077870 +/- 0.001788 | 0.345000 +/- 0.012583 | 0.070945 +/- 0.000756 | 0.451667 +/- 0.021169 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.049514 +/- 0.000631; wins/losses/ties 3/0/0.
- `online_mean_accuracy` vs `mlp_h128`: +0.341111 +/- 0.009686; wins/losses/ties 3/0/0.
- `final_window_mse` vs `mlp_h64`: +0.043264 +/- 0.000727; wins/losses/ties 3/0/0.
- `final_window_accuracy` vs `mlp_h128`: +0.286667 +/- 0.019221; wins/losses/ties 3/0/0.
- `test_mse` vs `mlp_h64`: +0.065519 +/- 0.002164; wins/losses/ties 3/0/0.
- `test_accuracy` vs `mlp_h64`: +0.545333 +/- 0.020003; wins/losses/ties 3/0/0.

## Additional Candidate vs Best MLP

- `upgd_structure_softmax_h64_rls_cal` `online_mean_mse` vs `mlp_h64`: +0.008875 +/- 0.001737; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_rls_cal` `online_mean_accuracy` vs `mlp_h128`: -0.051667 +/- 0.029360; wins/losses/ties 0/2/1.
- `upgd_structure_softmax_h64_rls_cal` `final_window_mse` vs `mlp_h64`: +0.000438 +/- 0.002890; wins/losses/ties 2/1/0.
- `upgd_structure_softmax_h64_rls_cal` `final_window_accuracy` vs `mlp_h128`: -0.125000 +/- 0.027538; wins/losses/ties 0/3/0.
- `upgd_structure_softmax_h64_rls_cal` `test_mse` vs `mlp_h64`: +0.013613 +/- 0.001054; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_rls_cal` `test_accuracy` vs `mlp_h64`: +0.108667 +/- 0.017892; wins/losses/ties 3/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.
