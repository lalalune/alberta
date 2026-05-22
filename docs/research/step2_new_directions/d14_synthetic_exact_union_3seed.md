# D14 Unified Basis LMS Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: basis_exact_union.

This is one normalized LMS predictor over a concatenated basis bank. It is not a route selector and does not include an MLP baseline inside the candidate prediction.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 0.6849 +/- 0.2798 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.7733 +/- 0.1652 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.5632 +/- 0.0872 |
| `basis_exact_union` | 0.4143 +/- 0.1550 | 0.2423 +/- 0.0420 |  |  | 388.0000 +/- 0.0000 | 0.2117 +/- 0.0315 |

`final_window_mse` best-basis-vs-best-MLP diff: -0.1426 +/- 0.0638; wins/losses/ties 0/3/0; best-basis counts {'basis_exact_union': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.6406 +/- 0.2548 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.5764 +/- 0.2826 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.4419 +/- 0.1009 |
| `basis_exact_union` | 1.8413 +/- 0.2957 | 1.9744 +/- 0.4023 |  |  | 323.0000 +/- 0.0000 | 0.1057 +/- 0.0163 |

`final_window_mse` best-basis-vs-best-MLP diff: -0.6929 +/- 0.0477; wins/losses/ties 0/3/0; best-basis counts {'basis_exact_union': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.5240 +/- 1.0748 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.2474 +/- 0.7317 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.7593 +/- 0.2017 |
| `basis_exact_union` | 0.8565 +/- 0.4997 | 0.9123 +/- 0.2247 |  |  | 485.0000 +/- 0.0000 | 0.3362 +/- 0.1503 |

`final_window_mse` best-basis-vs-best-MLP diff: +0.0909 +/- 0.1460; wins/losses/ties 1/2/0; best-basis counts {'basis_exact_union': 3}.
