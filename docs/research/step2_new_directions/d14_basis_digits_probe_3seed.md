# D14 Unified Basis LMS Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: basis_tanh_heavy, basis_structure_heavy, basis_exact_union.

This is one normalized LMS predictor over a concatenated basis bank. It is not a route selector and does not include an MLP baseline inside the candidate prediction.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  | 1.4879 +/- 0.4610 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  | 1.8204 +/- 0.4133 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  | 1.3709 +/- 0.2336 |
| `basis_tanh_heavy` | 0.0513 +/- 0.0006 | 0.0589 +/- 0.0008 | 0.7700 +/- 0.0135 | 0.8571 +/- 0.0148 | 541.0000 +/- 0.0000 | 0.2379 +/- 0.0641 |
| `basis_structure_heavy` | 0.0666 +/- 0.0013 | 0.0718 +/- 0.0004 | 0.6367 +/- 0.0150 | 0.7384 +/- 0.0243 | 541.0000 +/- 0.0000 | 0.2932 +/- 0.0273 |
| `basis_exact_union` | 0.0545 +/- 0.0005 | 0.0620 +/- 0.0005 | 0.7711 +/- 0.0011 | 0.8374 +/- 0.0069 | 541.0000 +/- 0.0000 | 0.5696 +/- 0.1384 |

`final_window_mse` best-basis-vs-best-MLP diff: -0.0132 +/- 0.0005; wins/losses/ties 0/3/0; best-basis counts {'basis_tanh_heavy': 3}.
`test_accuracy` best-basis-vs-best-MLP diff: -0.0501 +/- 0.0140; wins/losses/ties 0/3/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 1.4388 +/- 1.0719 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 1.2392 +/- 0.5810 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.8764 +/- 0.3842 |
| `basis_tanh_heavy` | 0.2884 +/- 0.0899 | 0.1808 +/- 0.0275 |  |  | 388.0000 +/- 0.0000 | 0.2538 +/- 0.1509 |
| `basis_structure_heavy` | 0.5439 +/- 0.2040 | 0.3176 +/- 0.0559 |  |  | 388.0000 +/- 0.0000 | 0.2758 +/- 0.0707 |
| `basis_exact_union` | 0.4087 +/- 0.1535 | 0.2417 +/- 0.0426 |  |  | 388.0000 +/- 0.0000 | 0.4870 +/- 0.1018 |

`final_window_mse` best-basis-vs-best-MLP diff: -0.0167 +/- 0.0128; wins/losses/ties 1/2/0; best-basis counts {'basis_tanh_heavy': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.1033 +/- 0.6733 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.9814 +/- 1.3309 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.8423 +/- 0.2173 |
| `basis_tanh_heavy` | 0.9387 +/- 0.6412 | 1.0137 +/- 0.2470 |  |  | 485.0000 +/- 0.0000 | 0.3737 +/- 0.1185 |
| `basis_structure_heavy` | 0.8573 +/- 0.5098 | 0.9202 +/- 0.2289 |  |  | 485.0000 +/- 0.0000 | 0.5819 +/- 0.1093 |
| `basis_exact_union` | 0.8571 +/- 0.4995 | 0.9117 +/- 0.2246 |  |  | 485.0000 +/- 0.0000 | 0.7240 +/- 0.1252 |

`final_window_mse` best-basis-vs-best-MLP diff: +0.1308 +/- 0.1252; wins/losses/ties 3/0/0; best-basis counts {'basis_exact_union': 1, 'basis_tanh_heavy': 2}.
