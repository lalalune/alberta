# Step 2 CIFAR Stream Benchmark

This note records the external CIFAR-style supervised stream probe for the current packaged single UPGD-memory trace learner.

Positive paired differences favor `step2_hybrid_memory_trace` in the primary comparison. For MSE, the difference is best MLP minus candidate; for accuracy, it is candidate minus best MLP.

## Dataset

- Source: `cifar10_python_archive`
- Real CIFAR-10 evidence: `True`
- Train shape: `[3000, 3072]`
- Test shape: `[1000, 3072]`

## Results

### `iid`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.078847 +/- 0.001049 | 0.344167 +/- 0.015161 | 0.085905 +/- 0.001063 | 0.239667 +/- 0.011724 |
| `upgd_step2_default` | 0.086024 +/- 0.000804 | 0.318333 +/- 0.015899 | 0.090538 +/- 0.001057 | 0.261667 +/- 0.016707 |
| `mlp_h32` | 0.106235 +/- 0.001039 | 0.199167 +/- 0.014743 | 0.108111 +/- 0.002737 | 0.196333 +/- 0.014240 |
| `mlp_h64` | 0.157624 +/- 0.002017 | 0.156667 +/- 0.009825 | 0.156959 +/- 0.003710 | 0.169000 +/- 0.014000 |
| `step2_hybrid_memory_trace_sharp` | 0.119696 +/- 0.002836 | 0.344167 +/- 0.015161 | 0.136318 +/- 0.003611 | 0.239667 +/- 0.011724 |
| `mlp_h32_sharp` | 0.134701 +/- 0.000713 | 0.199167 +/- 0.014743 | 0.136210 +/- 0.005635 | 0.196333 +/- 0.014240 |
| `mlp_h64_sharp` | 0.159029 +/- 0.001727 | 0.156667 +/- 0.009825 | 0.159972 +/- 0.002032 | 0.169000 +/- 0.014000 |

- Primary candidate: `step2_hybrid_memory_trace`
- Best MLP for final-window MSE: `mlp_h32`
- Best MLP for held-out accuracy: `mlp_h32`
- `final_window_mse` primary-vs-best-MLP diff: +0.027387 +/- 0.000451; wins/losses/ties 3/0/0.
- `test_accuracy` primary-vs-best-MLP diff: +0.043333 +/- 0.010914; wins/losses/ties 3/0/0.

Additional candidate comparisons:

- `upgd_step2_default` final-window MSE diff vs best MLP: +0.020211 +/- 0.000706; wins/losses/ties 3/0/0.
- `upgd_step2_default` held-out accuracy diff vs best MLP: +0.065333 +/- 0.015191; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_sharp` final-window MSE diff vs best MLP: -0.013461 +/- 0.001996; wins/losses/ties 0/3/0.
- `step2_hybrid_memory_trace_sharp` held-out accuracy diff vs best MLP: +0.043333 +/- 0.010914; wins/losses/ties 3/0/0.

## Interpretation

Synthetic smoke results only validate wiring, reproducibility, optional dependency behavior, and metric accounting. A stronger CIFAR claim requires real CIFAR-10 runs with `--allow-download`, multiple seeds, larger streams, and no test-set tuning.
