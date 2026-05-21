# D14 Unified Basis LMS Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: basis_balanced, basis_tanh_heavy, basis_structure_heavy.

This is one normalized LMS predictor over a concatenated basis bank. It is not a route selector and does not include an MLP baseline inside the candidate prediction.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 0.9783 +/- 0.5125 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.8346 +/- 0.4373 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.4720 +/- 0.0280 |
| `basis_balanced` | 0.3806 +/- 0.1311 | 0.2300 +/- 0.0377 |  |  | 388.0000 +/- 0.0000 | 0.1794 +/- 0.0556 |
| `basis_tanh_heavy` | 0.2875 +/- 0.0876 | 0.1804 +/- 0.0281 |  |  | 388.0000 +/- 0.0000 | 0.2158 +/- 0.0561 |
| `basis_structure_heavy` | 0.5433 +/- 0.2040 | 0.3175 +/- 0.0560 |  |  | 388.0000 +/- 0.0000 | 0.3622 +/- 0.0446 |

`final_window_mse` best-basis-vs-best-MLP diff: -0.0158 +/- 0.0118; wins/losses/ties 0/3/0; best-basis counts {'basis_tanh_heavy': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.6313 +/- 0.2895 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.5133 +/- 0.1845 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.6422 +/- 0.0608 |
| `basis_balanced` | 1.4683 +/- 0.2835 | 1.5681 +/- 0.3226 |  |  | 323.0000 +/- 0.0000 | 0.1126 +/- 0.0060 |
| `basis_tanh_heavy` | 1.2584 +/- 0.2880 | 1.3740 +/- 0.2883 |  |  | 323.0000 +/- 0.0000 | 0.2417 +/- 0.0207 |
| `basis_structure_heavy` | 1.6768 +/- 0.2382 | 1.8267 +/- 0.3651 |  |  | 323.0000 +/- 0.0000 | 0.3167 +/- 0.0773 |

`final_window_mse` best-basis-vs-best-MLP diff: -0.1099 +/- 0.0429; wins/losses/ties 0/3/0; best-basis counts {'basis_tanh_heavy': 3}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 1.4323 +/- 0.9072 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 1.5386 +/- 0.9011 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.9351 +/- 0.1203 |
| `basis_balanced` | 0.8874 +/- 0.5719 | 0.9571 +/- 0.2355 |  |  | 485.0000 +/- 0.0000 | 0.3879 +/- 0.0887 |
| `basis_tanh_heavy` | 0.9440 +/- 0.6468 | 1.0144 +/- 0.2477 |  |  | 485.0000 +/- 0.0000 | 0.5625 +/- 0.1345 |
| `basis_structure_heavy` | 0.8573 +/- 0.5098 | 0.9202 +/- 0.2289 |  |  | 485.0000 +/- 0.0000 | 0.5606 +/- 0.1294 |

`final_window_mse` best-basis-vs-best-MLP diff: +0.1240 +/- 0.1181; wins/losses/ties 3/0/0; best-basis counts {'basis_structure_heavy': 1, 'basis_tanh_heavy': 2}.
