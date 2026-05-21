# D14 Unified Basis LMS Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: basis_tanh_exact, basis_fourier_exact, basis_poly_exact.

This is one normalized LMS predictor over a concatenated basis bank. It is not a route selector and does not include an MLP baseline inside the candidate prediction.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  | 0.6618 +/- 0.2611 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  | 0.7882 +/- 0.1170 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  | 0.7361 +/- 0.2586 |
| `basis_tanh_exact` | 0.2283 +/- 0.0701 | 0.1489 +/- 0.0224 |  |  | 388.0000 +/- 0.0000 | 0.2368 +/- 0.0774 |
| `basis_fourier_exact` | 0.7142 +/- 0.2457 | 0.4545 +/- 0.0743 |  |  | 388.0000 +/- 0.0000 | 0.3975 +/- 0.1613 |
| `basis_poly_exact` | 2.0642 +/- 0.9382 | 1.3004 +/- 0.2433 |  |  | 388.0000 +/- 0.0000 | 0.3718 +/- 0.0617 |

`final_window_mse` best-basis-vs-best-MLP diff: +0.0434 +/- 0.0263; wins/losses/ties 2/1/0; best-basis counts {'basis_tanh_exact': 3}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  | 0.3997 +/- 0.1704 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  | 0.5147 +/- 0.1504 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  | 0.4065 +/- 0.0451 |
| `basis_tanh_exact` | 1.1439 +/- 0.2690 | 1.3934 +/- 0.2904 |  |  | 323.0000 +/- 0.0000 | 0.1309 +/- 0.0326 |
| `basis_fourier_exact` | 0.6397 +/- 0.1150 | 0.6042 +/- 0.1217 |  |  | 323.0000 +/- 0.0000 | 0.1826 +/- 0.0522 |
| `basis_poly_exact` | 5.1373 +/- 0.9872 | 5.2737 +/- 1.4355 |  |  | 323.0000 +/- 0.0000 | 0.1857 +/- 0.0903 |

`final_window_mse` best-basis-vs-best-MLP diff: +0.5638 +/- 0.2959; wins/losses/ties 3/0/0; best-basis counts {'basis_fourier_exact': 2, 'basis_tanh_exact': 1}.

## synthetic_polynomial

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.9679 +/- 0.6494 | 1.0371 +/- 0.2385 |  |  |  | 0.9377 +/- 0.6443 |
| `mlp_h128` | 0.9727 +/- 0.6489 | 1.0458 +/- 0.2382 |  |  |  | 0.9640 +/- 0.5943 |
| `mlp_h64_64` | 0.9475 +/- 0.6444 | 1.0231 +/- 0.2310 |  |  |  | 0.5638 +/- 0.1256 |
| `basis_tanh_exact` | 1.0231 +/- 0.7078 | 1.0966 +/- 0.2652 |  |  | 485.0000 +/- 0.0000 | 0.2264 +/- 0.0614 |
| `basis_fourier_exact` | 1.0110 +/- 0.6757 | 1.0854 +/- 0.2704 |  |  | 485.0000 +/- 0.0000 | 0.2882 +/- 0.0937 |
| `basis_poly_exact` | 0.8507 +/- 0.3960 | 0.8710 +/- 0.2091 |  |  | 485.0000 +/- 0.0000 | 0.1890 +/- 0.0833 |

`final_window_mse` best-basis-vs-best-MLP diff: +0.1904 +/- 0.2027; wins/losses/ties 1/2/0; best-basis counts {'basis_poly_exact': 1, 'basis_tanh_exact': 2}.
