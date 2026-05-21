# Step 2 CIFAR Stream Benchmark

This note records the external CIFAR-style supervised stream probe for the current packaged single UPGD-memory trace learner.

Positive paired differences favor `step2_hybrid_memory_trace` in the primary comparison. For MSE, the difference is best MLP minus candidate; for accuracy, it is candidate minus best MLP.

## Dataset

- Source: `cifar10_python_archive`
- Real CIFAR-10 evidence: `True`
- Train shape: `[2000, 3072]`
- Test shape: `[800, 3072]`

## Results

### `iid`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.084227 +/- 0.002480 | 0.368889 +/- 0.029897 | 0.097469 +/- 0.001450 | 0.241250 +/- 0.012011 |
| `upgd_step2_default` | 0.096462 +/- 0.000099 | 0.275556 +/- 0.014572 | 0.097754 +/- 0.001319 | 0.235000 +/- 0.005907 |
| `mlp_h32` | 0.107020 +/- 0.001819 | 0.197778 +/- 0.025628 | 0.119414 +/- 0.006636 | 0.158333 +/- 0.012381 |
| `mlp_h64` | 0.154282 +/- 0.001308 | 0.195556 +/- 0.005879 | 0.164051 +/- 0.006397 | 0.157083 +/- 0.011048 |

- Primary candidate: `step2_hybrid_memory_trace`
- Best MLP for final-window MSE: `mlp_h32`
- Best MLP for held-out accuracy: `mlp_h32`
- `final_window_mse` primary-vs-best-MLP diff: +0.022792 +/- 0.004289; wins/losses/ties 3/0/0.
- `test_accuracy` primary-vs-best-MLP diff: +0.082917 +/- 0.019530; wins/losses/ties 3/0/0.

### `class_blocked`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.001945 +/- 0.000091 | 1.000000 +/- 0.000000 | 0.132952 +/- 0.001721 | 0.176667 +/- 0.006859 |
| `upgd_step2_default` | 0.014541 +/- 0.001260 | 0.964444 +/- 0.008012 | 0.161466 +/- 0.001785 | 0.128750 +/- 0.004018 |
| `mlp_h32` | 0.001046 +/- 0.000092 | 1.000000 +/- 0.000000 | 0.177255 +/- 0.001808 | 0.117500 +/- 0.000000 |
| `mlp_h64` | 0.001657 +/- 0.000059 | 1.000000 +/- 0.000000 | 0.176829 +/- 0.001202 | 0.117500 +/- 0.000000 |

- Primary candidate: `step2_hybrid_memory_trace`
- Best MLP for final-window MSE: `mlp_h32`
- Best MLP for held-out accuracy: `mlp_h32`
- `final_window_mse` primary-vs-best-MLP diff: -0.000898 +/- 0.000089; wins/losses/ties 0/3/0.
- `test_accuracy` primary-vs-best-MLP diff: +0.059167 +/- 0.006859; wins/losses/ties 3/0/0.

## Interpretation

Synthetic smoke results only validate wiring, reproducibility, optional dependency behavior, and metric accounting. A stronger CIFAR claim requires real CIFAR-10 runs with `--allow-download`, multiple seeds, larger streams, and no test-set tuning.
