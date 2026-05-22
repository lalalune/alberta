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
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.078847 +/- 0.001049 | 0.344167 +/- 0.015161 | 0.085905 +/- 0.001063 | 0.239667 +/- 0.011724 |
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
- `step2_hybrid_memory_trace_adaptive_sharp` final-window MSE diff vs best MLP: +0.027387 +/- 0.000451; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` held-out accuracy diff vs best MLP: +0.043333 +/- 0.010914; wins/losses/ties 3/0/0.

### `class_blocked`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.004404 +/- 0.000049 | 0.997500 +/- 0.000000 | 0.114472 +/- 0.001955 | 0.186333 +/- 0.010682 |
| `upgd_step2_default` | 0.015336 +/- 0.000788 | 0.919167 +/- 0.005833 | 0.171924 +/- 0.001362 | 0.095333 +/- 0.000333 |
| `mlp_h32` | 0.001905 +/- 0.000027 | 0.990833 +/- 0.000833 | 0.178844 +/- 0.000639 | 0.095000 +/- 0.000000 |
| `mlp_h64` | 0.001631 +/- 0.000017 | 0.993333 +/- 0.000833 | 0.175533 +/- 0.001683 | 0.095000 +/- 0.000000 |
| `step2_hybrid_memory_trace_sharp` | 0.000500 +/- 0.000000 | 0.997500 +/- 0.000000 | 0.157583 +/- 0.002096 | 0.186333 +/- 0.010682 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.000500 +/- 0.000000 | 0.997500 +/- 0.000000 | 0.157583 +/- 0.002096 | 0.186333 +/- 0.010682 |
| `mlp_h32_sharp` | 0.001871 +/- 0.000188 | 0.990833 +/- 0.000833 | 0.181000 +/- 0.000000 | 0.095000 +/- 0.000000 |
| `mlp_h64_sharp` | 0.001217 +/- 0.000148 | 0.993333 +/- 0.000833 | 0.181000 +/- 0.000000 | 0.095000 +/- 0.000000 |

- Primary candidate: `step2_hybrid_memory_trace`
- Best MLP for final-window MSE: `mlp_h64_sharp`
- Best MLP for held-out accuracy: `mlp_h32`
- `final_window_mse` primary-vs-best-MLP diff: -0.003188 +/- 0.000184; wins/losses/ties 0/3/0.
- `test_accuracy` primary-vs-best-MLP diff: +0.091333 +/- 0.010682; wins/losses/ties 3/0/0.

Additional candidate comparisons:

- `upgd_step2_default` final-window MSE diff vs best MLP: -0.014119 +/- 0.000648; wins/losses/ties 0/3/0.
- `upgd_step2_default` held-out accuracy diff vs best MLP: +0.000333 +/- 0.000333; wins/losses/ties 1/0/2.
- `step2_hybrid_memory_trace_sharp` final-window MSE diff vs best MLP: +0.000717 +/- 0.000148; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_sharp` held-out accuracy diff vs best MLP: +0.091333 +/- 0.010682; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` final-window MSE diff vs best MLP: +0.000717 +/- 0.000148; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` held-out accuracy diff vs best MLP: +0.091333 +/- 0.010682; wins/losses/ties 3/0/0.

## Interpretation

This is real CIFAR-10 evidence from the local public Python archive, not the
synthetic fallback.  The adaptive raw/sharp readout preserves the raw
`step2_hybrid_memory_trace` metrics on IID CIFAR and switches to the sharpened
readout on class-blocked CIFAR.  That closes the class-blocked final-window MSE
caveat while retaining the held-out accuracy win.  The result is still a
3-seed, 2,000-step image-scale probe, not a final replacement for broader
multi-source vision evidence.
