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
| `upgd_structure_softmax_h64` | 0.055155 +/- 0.000367 | 0.655000 +/- 0.010104 | 0.038282 +/- 0.001158 | 0.809167 +/- 0.012414 |
| `upgd_structure_softmax_h64_dream_surprise` | 0.039995 +/- 0.000947 | 0.720833 +/- 0.014167 | 0.024407 +/- 0.001885 | 0.861167 +/- 0.017201 |
| `upgd_structure_softmax_h64_dream_hybrid` | 0.038214 +/- 0.000631 | 0.733333 +/- 0.010240 | 0.022703 +/- 0.000575 | 0.868500 +/- 0.005033 |
| `upgd_structure_softmax_h64_dream_surprise2` | 0.033616 +/- 0.000324 | 0.756667 +/- 0.009280 | 0.021547 +/- 0.001574 | 0.857500 +/- 0.011558 |
| `upgd_structure_softmax_h64_dream_surprise_c256` | 0.038369 +/- 0.000627 | 0.750833 +/- 0.012276 | 0.023217 +/- 0.001351 | 0.863167 +/- 0.013569 |
| `mlp_h64` | 0.060916 +/- 0.000638 | 0.648333 +/- 0.015023 | 0.066261 +/- 0.001686 | 0.588833 +/- 0.004667 |

## Primary vs Best MLP


## Additional Candidate vs Best MLP

- `upgd_structure_softmax_h64` `online_mean_mse` vs `mlp_h64`: +0.002037 +/- 0.000291; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise` `online_mean_mse` vs `mlp_h64`: +0.015620 +/- 0.000351; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_hybrid` `online_mean_mse` vs `mlp_h64`: +0.016097 +/- 0.001000; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise2` `online_mean_mse` vs `mlp_h64`: +0.023574 +/- 0.000537; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise_c256` `online_mean_mse` vs `mlp_h64`: +0.017338 +/- 0.000645; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `online_mean_accuracy` vs `mlp_h64`: -0.005000 +/- 0.005033; wins/losses/ties 1/2/0.
- `upgd_structure_softmax_h64_dream_surprise` `online_mean_accuracy` vs `mlp_h64`: +0.075000 +/- 0.004500; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_hybrid` `online_mean_accuracy` vs `mlp_h64`: +0.073333 +/- 0.007928; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise2` `online_mean_accuracy` vs `mlp_h64`: +0.109000 +/- 0.003279; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise_c256` `online_mean_accuracy` vs `mlp_h64`: +0.101333 +/- 0.009684; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `final_window_mse` vs `mlp_h64`: +0.005760 +/- 0.000349; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise` `final_window_mse` vs `mlp_h64`: +0.020921 +/- 0.001548; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_hybrid` `final_window_mse` vs `mlp_h64`: +0.022701 +/- 0.001069; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise2` `final_window_mse` vs `mlp_h64`: +0.027300 +/- 0.000864; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise_c256` `final_window_mse` vs `mlp_h64`: +0.022546 +/- 0.001177; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `final_window_accuracy` vs `mlp_h64`: +0.006667 +/- 0.023819; wins/losses/ties 2/1/0.
- `upgd_structure_softmax_h64_dream_surprise` `final_window_accuracy` vs `mlp_h64`: +0.072500 +/- 0.028976; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_hybrid` `final_window_accuracy` vs `mlp_h64`: +0.085000 +/- 0.017500; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise2` `final_window_accuracy` vs `mlp_h64`: +0.108333 +/- 0.018615; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise_c256` `final_window_accuracy` vs `mlp_h64`: +0.102500 +/- 0.025981; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `test_mse` vs `mlp_h64`: +0.027979 +/- 0.001660; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise` `test_mse` vs `mlp_h64`: +0.041854 +/- 0.002603; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_hybrid` `test_mse` vs `mlp_h64`: +0.043558 +/- 0.002009; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise2` `test_mse` vs `mlp_h64`: +0.044714 +/- 0.002369; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise_c256` `test_mse` vs `mlp_h64`: +0.043044 +/- 0.002386; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64` `test_accuracy` vs `mlp_h64`: +0.220333 +/- 0.016292; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise` `test_accuracy` vs `mlp_h64`: +0.272333 +/- 0.020580; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_hybrid` `test_accuracy` vs `mlp_h64`: +0.279667 +/- 0.009404; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise2` `test_accuracy` vs `mlp_h64`: +0.268667 +/- 0.015023; wins/losses/ties 3/0/0.
- `upgd_structure_softmax_h64_dream_surprise_c256` `test_accuracy` vs `mlp_h64`: +0.274333 +/- 0.016895; wins/losses/ties 3/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

## 2026-05-09 Extended Dream Ablation

All dream variants here improve `upgd_structure_softmax_h64` on every paired
metric in this compact 3-seed run.

| Candidate | Online MSE | Online Acc | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| `dream_surprise` | +0.013583 | +0.080000 | +0.015160 | +0.065833 | +0.013875 | +0.052000 |
| `dream_hybrid` | +0.014060 | +0.078333 | +0.016941 | +0.078333 | +0.015579 | +0.059333 |
| `dream_surprise2` | +0.021536 | +0.114000 | +0.021539 | +0.101667 | +0.016735 | +0.048333 |
| `dream_surprise_c256` | +0.015301 | +0.106333 | +0.016786 | +0.095833 | +0.015065 | +0.054000 |

Two surprise dreams per real step is best for online and final-window MSE.
Hybrid surprise+learning-progress and the larger buffer have the best retained
test accuracy in this 3-seed slice. This motivates a larger check with
`dream_hybrid`, `dream_surprise2`, and `dream_surprise_c256`.
