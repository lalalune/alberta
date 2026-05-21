# D14 Unified Basis LMS Results

Protocol: 1 paired seeds, 120 online steps, final window 40. Candidate configs: basis_balanced.

This is one normalized LMS predictor over a concatenated basis bank. It is not a route selector and does not include an MLP baseline inside the candidate prediction.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Basis dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2590 +/- 0.0000 | 0.2945 +/- 0.0000 |  |  |  | 2.7441 +/- 0.0000 |
| `mlp_h128` | 0.2785 +/- 0.0000 | 0.3061 +/- 0.0000 |  |  |  | 1.9310 +/- 0.0000 |
| `mlp_h64_64` | 0.2920 +/- 0.0000 | 0.3426 +/- 0.0000 |  |  |  | 1.1551 +/- 0.0000 |
| `basis_balanced` | 0.2535 +/- 0.0000 | 0.3262 +/- 0.0000 |  |  | 323.0000 +/- 0.0000 | 0.0072 +/- 0.0000 |

`final_window_mse` best-basis-vs-best-MLP diff: +0.0056 +/- 0.0000; wins/losses/ties 1/0/0; best-basis counts {'basis_balanced': 1}.
