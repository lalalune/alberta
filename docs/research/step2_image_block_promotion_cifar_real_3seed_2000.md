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
| `step2_hybrid_memory_trace` | 0.078500 +/- 0.001239 | 0.344167 +/- 0.017220 | 0.085828 +/- 0.001162 | 0.242333 +/- 0.011851 |
| `upgd_step2_default` | 0.085169 +/- 0.000496 | 0.343333 +/- 0.007407 | 0.090412 +/- 0.000933 | 0.277000 +/- 0.013317 |
| `mlp_h32` | 0.106046 +/- 0.001162 | 0.212500 +/- 0.010000 | 0.107391 +/- 0.003801 | 0.198000 +/- 0.014295 |
| `mlp_h64` | 0.154054 +/- 0.002429 | 0.158333 +/- 0.014743 | 0.158157 +/- 0.009688 | 0.163667 +/- 0.010682 |
| `mlp_h128` | 0.640853 +/- 0.009665 | 0.120000 +/- 0.005774 | 0.618883 +/- 0.041059 | 0.125000 +/- 0.003512 |
| `mlp_h256` | 1.001080 +/- 0.009388 | 0.138333 +/- 0.013333 | 1.081975 +/- 0.100401 | 0.118000 +/- 0.010017 |
| `mlp_h128_128` | 0.245993 +/- 0.007946 | 0.150000 +/- 0.008036 | 0.240473 +/- 0.029529 | 0.138000 +/- 0.009292 |
| `proto_mem_s20` | 0.138149 +/- 0.003586 | 0.263333 +/- 0.019490 | 0.149970 +/- 0.000663 | 0.198333 +/- 0.004256 |
| `proto_mem_s32` | 0.131339 +/- 0.003725 | 0.290000 +/- 0.018764 | 0.147007 +/- 0.000196 | 0.206000 +/- 0.003215 |
| `step2_hybrid_memory_trace_sharp` | 0.119636 +/- 0.002951 | 0.344167 +/- 0.017220 | 0.136185 +/- 0.003901 | 0.242333 +/- 0.011851 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.078500 +/- 0.001239 | 0.344167 +/- 0.017220 | 0.085828 +/- 0.001162 | 0.242333 +/- 0.011851 |
| `mlp_h32_sharp` | 0.132654 +/- 0.000930 | 0.212500 +/- 0.010000 | 0.135276 +/- 0.005243 | 0.198000 +/- 0.014295 |
| `mlp_h64_sharp` | 0.158781 +/- 0.002745 | 0.158333 +/- 0.014743 | 0.158885 +/- 0.003112 | 0.163667 +/- 0.010682 |

- Primary candidate: `step2_hybrid_memory_trace`
- Best MLP for final-window MSE: `mlp_h32`
- Best MLP for held-out accuracy: `mlp_h32`
- `final_window_mse` primary-vs-best-MLP diff: +0.027546 +/- 0.000586; wins/losses/ties 3/0/0.
- `test_accuracy` primary-vs-best-MLP diff: +0.044333 +/- 0.007446; wins/losses/ties 3/0/0.

Additional candidate comparisons:

- `upgd_step2_default` final-window MSE diff vs best MLP: +0.020877 +/- 0.000997; wins/losses/ties 3/0/0.
- `upgd_step2_default` held-out accuracy diff vs best MLP: +0.079000 +/- 0.018248; wins/losses/ties 3/0/0.
- `proto_mem_s20` final-window MSE diff vs best MLP: -0.032103 +/- 0.003192; wins/losses/ties 0/3/0.
- `proto_mem_s20` held-out accuracy diff vs best MLP: +0.000333 +/- 0.010525; wins/losses/ties 1/2/0.
- `proto_mem_s32` final-window MSE diff vs best MLP: -0.025293 +/- 0.003171; wins/losses/ties 0/3/0.
- `proto_mem_s32` held-out accuracy diff vs best MLP: +0.008000 +/- 0.012503; wins/losses/ties 2/1/0.
- `step2_hybrid_memory_trace_sharp` final-window MSE diff vs best MLP: -0.013590 +/- 0.001830; wins/losses/ties 0/3/0.
- `step2_hybrid_memory_trace_sharp` held-out accuracy diff vs best MLP: +0.044333 +/- 0.007446; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` final-window MSE diff vs best MLP: +0.027546 +/- 0.000586; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` held-out accuracy diff vs best MLP: +0.044333 +/- 0.007446; wins/losses/ties 3/0/0.

### `class_blocked`

| Method | Final-window MSE | Final-window accuracy | Held-out MSE | Held-out accuracy |
| --- | ---: | ---: | ---: | ---: |
| `step2_hybrid_memory_trace` | 0.004535 +/- 0.000050 | 0.997500 +/- 0.000000 | 0.110673 +/- 0.002012 | 0.194667 +/- 0.013220 |
| `upgd_step2_default` | 0.015106 +/- 0.000644 | 0.918333 +/- 0.005465 | 0.168758 +/- 0.002366 | 0.096333 +/- 0.001333 |
| `mlp_h32` | 0.001875 +/- 0.000019 | 0.990833 +/- 0.001667 | 0.179155 +/- 0.001364 | 0.095000 +/- 0.000000 |
| `mlp_h64` | 0.001589 +/- 0.000031 | 0.993333 +/- 0.000833 | 0.178295 +/- 0.001360 | 0.095000 +/- 0.000000 |
| `mlp_h128` | 0.022735 +/- 0.003000 | 0.992500 +/- 0.002887 | 0.198566 +/- 0.009292 | 0.095000 +/- 0.000000 |
| `mlp_h256` | 0.705731 +/- 0.015520 | 0.405000 +/- 0.002887 | 0.963767 +/- 0.139005 | 0.110000 +/- 0.013868 |
| `mlp_h128_128` | 0.002137 +/- 0.000064 | 0.996667 +/- 0.000833 | 0.183369 +/- 0.003834 | 0.095000 +/- 0.000000 |
| `proto_mem_s20` | 0.153431 +/- 0.003848 | 0.185000 +/- 0.023229 | 0.153530 +/- 0.000344 | 0.183667 +/- 0.001453 |
| `proto_mem_s32` | 0.156646 +/- 0.005080 | 0.168333 +/- 0.024889 | 0.151726 +/- 0.001625 | 0.185000 +/- 0.008083 |
| `step2_hybrid_memory_trace_sharp` | 0.000500 +/- 0.000000 | 0.997500 +/- 0.000000 | 0.155462 +/- 0.002475 | 0.194667 +/- 0.013220 |
| `step2_hybrid_memory_trace_adaptive_sharp` | 0.000500 +/- 0.000000 | 0.997500 +/- 0.000000 | 0.155462 +/- 0.002475 | 0.194667 +/- 0.013220 |
| `mlp_h32_sharp` | 0.001909 +/- 0.000296 | 0.990833 +/- 0.001667 | 0.181000 +/- 0.000000 | 0.095000 +/- 0.000000 |
| `mlp_h64_sharp` | 0.001333 +/- 0.000167 | 0.993333 +/- 0.000833 | 0.181000 +/- 0.000000 | 0.095000 +/- 0.000000 |

- Primary candidate: `step2_hybrid_memory_trace`
- Best MLP for final-window MSE: `mlp_h64_sharp`
- Best MLP for held-out accuracy: `mlp_h256`
- `final_window_mse` primary-vs-best-MLP diff: -0.003201 +/- 0.000147; wins/losses/ties 0/3/0.
- `test_accuracy` primary-vs-best-MLP diff: +0.084667 +/- 0.026333; wins/losses/ties 3/0/0.

Additional candidate comparisons:

- `upgd_step2_default` final-window MSE diff vs best MLP: -0.013773 +/- 0.000631; wins/losses/ties 0/3/0.
- `upgd_step2_default` held-out accuracy diff vs best MLP: -0.013667 +/- 0.012574; wins/losses/ties 1/2/0.
- `proto_mem_s20` final-window MSE diff vs best MLP: -0.152098 +/- 0.003702; wins/losses/ties 0/3/0.
- `proto_mem_s20` held-out accuracy diff vs best MLP: +0.073667 +/- 0.014111; wins/losses/ties 3/0/0.
- `proto_mem_s32` final-window MSE diff vs best MLP: -0.155313 +/- 0.004925; wins/losses/ties 0/3/0.
- `proto_mem_s32` held-out accuracy diff vs best MLP: +0.075000 +/- 0.019655; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_sharp` final-window MSE diff vs best MLP: +0.000833 +/- 0.000167; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_sharp` held-out accuracy diff vs best MLP: +0.084667 +/- 0.026333; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` final-window MSE diff vs best MLP: +0.000833 +/- 0.000167; wins/losses/ties 3/0/0.
- `step2_hybrid_memory_trace_adaptive_sharp` held-out accuracy diff vs best MLP: +0.084667 +/- 0.026333; wins/losses/ties 3/0/0.

## Interpretation

Synthetic smoke results only validate wiring, reproducibility, optional dependency behavior, and metric accounting. A stronger CIFAR claim requires real CIFAR-10 runs with `--allow-download`, multiple seeds, larger streams, and no test-set tuning.
