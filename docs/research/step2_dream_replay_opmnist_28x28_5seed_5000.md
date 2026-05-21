# Step 2 UPGD-Memory OPMNIST

This note records the resumable OPMNIST run for the packaged UPGD-memory trace learner, optional simple candidates, and fair MLP baselines.

- Primary method: `step2_hybrid_memory_trace`
- MNIST source: `sklearn_digits_28x28`
- Steps: `5000`
- Seeds: `5`
- Permutations: `10`
- Task block size: `500`

| Method | Final MSE | Final Acc | Test MSE | Test Acc |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.028279 +/- 0.000706 | 0.802800 +/- 0.004477 | 0.017302 +/- 0.000469 | 0.886040 +/- 0.002855 |
| `upgd_structure_softmax_h64` | 0.043758 +/- 0.000592 | 0.678800 +/- 0.007165 | 0.030514 +/- 0.000403 | 0.794120 +/- 0.003215 |
| `upgd_structure_softmax_h64_dream_hybrid` | 0.032618 +/- 0.000896 | 0.762800 +/- 0.007144 | 0.027520 +/- 0.001284 | 0.809760 +/- 0.008459 |
| `upgd_structure_softmax_h64_dream_surprise2` | 0.030268 +/- 0.000750 | 0.781200 +/- 0.006127 | 0.026906 +/- 0.000627 | 0.817040 +/- 0.004235 |
| `upgd_structure_softmax_h64_dream_surprise_c256` | 0.031546 +/- 0.001228 | 0.768800 +/- 0.009515 | 0.026044 +/- 0.000721 | 0.819880 +/- 0.006286 |
| `mlp_h64` | 0.056129 +/- 0.000554 | 0.688600 +/- 0.006161 | 0.073131 +/- 0.001506 | 0.515800 +/- 0.014838 |

## Primary vs Best MLP

- `online_mean_mse` vs `mlp_h64`: +0.033912 +/- 0.000428; wins/losses/ties 5/0/0.
- `online_mean_accuracy` vs `mlp_h64`: +0.164080 +/- 0.004313; wins/losses/ties 5/0/0.
- `final_window_mse` vs `mlp_h64`: +0.027851 +/- 0.000819; wins/losses/ties 5/0/0.
- `final_window_accuracy` vs `mlp_h64`: +0.114200 +/- 0.005826; wins/losses/ties 5/0/0.
- `test_mse` vs `mlp_h64`: +0.055829 +/- 0.001817; wins/losses/ties 5/0/0.
- `test_accuracy` vs `mlp_h64`: +0.370240 +/- 0.015838; wins/losses/ties 5/0/0.

## Additional Candidate vs Best MLP

- `upgd_structure_softmax_h64` `online_mean_mse` vs `mlp_h64`: +0.007341 +/- 0.000376; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_hybrid` `online_mean_mse` vs `mlp_h64`: +0.020462 +/- 0.000162; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_surprise2` `online_mean_mse` vs `mlp_h64`: +0.024694 +/- 0.000100; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_surprise_c256` `online_mean_mse` vs `mlp_h64`: +0.021111 +/- 0.000407; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64` `online_mean_accuracy` vs `mlp_h64`: -0.006720 +/- 0.003859; wins/losses/ties 1/4/0.
- `upgd_structure_softmax_h64_dream_hybrid` `online_mean_accuracy` vs `mlp_h64`: +0.074440 +/- 0.000842; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_surprise2` `online_mean_accuracy` vs `mlp_h64`: +0.098600 +/- 0.002642; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_surprise_c256` `online_mean_accuracy` vs `mlp_h64`: +0.085160 +/- 0.003785; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64` `final_window_mse` vs `mlp_h64`: +0.012371 +/- 0.000579; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_hybrid` `final_window_mse` vs `mlp_h64`: +0.023511 +/- 0.000760; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_surprise2` `final_window_mse` vs `mlp_h64`: +0.025862 +/- 0.000619; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_surprise_c256` `final_window_mse` vs `mlp_h64`: +0.024583 +/- 0.000898; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64` `final_window_accuracy` vs `mlp_h64`: -0.009800 +/- 0.008470; wins/losses/ties 1/4/0.
- `upgd_structure_softmax_h64_dream_hybrid` `final_window_accuracy` vs `mlp_h64`: +0.074200 +/- 0.007067; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_surprise2` `final_window_accuracy` vs `mlp_h64`: +0.092600 +/- 0.003614; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_surprise_c256` `final_window_accuracy` vs `mlp_h64`: +0.080200 +/- 0.006003; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64` `test_mse` vs `mlp_h64`: +0.042617 +/- 0.001326; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_hybrid` `test_mse` vs `mlp_h64`: +0.045611 +/- 0.002563; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_surprise2` `test_mse` vs `mlp_h64`: +0.046225 +/- 0.001837; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_surprise_c256` `test_mse` vs `mlp_h64`: +0.047088 +/- 0.001913; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64` `test_accuracy` vs `mlp_h64`: +0.278320 +/- 0.014235; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_hybrid` `test_accuracy` vs `mlp_h64`: +0.293960 +/- 0.020704; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_surprise2` `test_accuracy` vs `mlp_h64`: +0.301240 +/- 0.014987; wins/losses/ties 5/0/0.
- `upgd_structure_softmax_h64_dream_surprise_c256` `test_accuracy` vs `mlp_h64`: +0.304080 +/- 0.019895; wins/losses/ties 5/0/0.

## Scale Status

A full published-scale OPMNIST result requires 800 completed 60,000-example task blocks, or 48,000,000 online updates. This runner reports exact completed blocks and leaves a checkpoint/status sidecar for continuation rather than treating partial runs as full closure.

## 2026-05-09 Larger Dream-Replay Check

The larger 5-seed/5000-step run confirms the single-UPGD result. Paired deltas
versus `upgd_structure_softmax_h64`, where positive favors the dream variant:

| Candidate | Online MSE | Online Acc | Final MSE | Final Acc | Test MSE | Test Acc |
|---|---:|---:|---:|---:|---:|---:|
| `dream_hybrid` | +0.013120 +/- 0.000373 | +0.081160 +/- 0.003247 | +0.011140 +/- 0.000885 | +0.084000 +/- 0.009884 | +0.002994 +/- 0.001475 | +0.015640 +/- 0.009527 |
| `dream_surprise2` | +0.017353 +/- 0.000289 | +0.105320 +/- 0.002590 | +0.013490 +/- 0.000546 | +0.102400 +/- 0.006742 | +0.003607 +/- 0.001014 | +0.022920 +/- 0.007000 |
| `dream_surprise_c256` | +0.013770 +/- 0.000276 | +0.091880 +/- 0.004742 | +0.012212 +/- 0.001063 | +0.090000 +/- 0.008786 | +0.004470 +/- 0.000635 | +0.025760 +/- 0.005858 |

All three variants win `5/5` against the bare h64 softmax UPGD on every metric
except `dream_hybrid` test MSE/test accuracy, which are `4/1`.

Against `step2_hybrid_memory_trace`, however, the dream variants still lose
every metric (`0/5` wins). The closest branch is `dream_surprise2`, but it is
still behind primary by `-0.009218` online MSE, `-0.065480` online accuracy,
`-0.001989` final-window MSE, `-0.021600` final-window accuracy, `-0.009605`
held-out test MSE, and `-0.069000` held-out test accuracy.

Interpretation: surprise-gated dream replay is a strong improvement to the
bare single UPGD learner and supports the intuition that high-surprise examples
should be re-experienced until they become lower-surprise. It is not yet the
canonical Step 2 default because the existing UPGD-memory hybrid remains better
and because replay adds extra updates per real step.
