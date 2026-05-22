# D15 Groupwise Basis LMS Results

Protocol: 1 paired seeds, 120 online steps, final window 40. Candidate configs: groupwise_canonical.

Each candidate is one additive predictor with blockwise normalized LMS. Every included basis family updates from the same residual at every step; there is no prediction router or expert selection.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Feature dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2590 +/- 0.0000 | 0.2945 +/- 0.0000 |  |  |  | 2.1797 +/- 0.0000 |
| `mlp_h128` | 0.2785 +/- 0.0000 | 0.3061 +/- 0.0000 |  |  |  | 1.8744 +/- 0.0000 |
| `mlp_h64_64` | 0.2920 +/- 0.0000 | 0.3426 +/- 0.0000 |  |  |  | 1.1515 +/- 0.0000 |
| `groupwise_canonical` | 0.6851 +/- 0.0000 | 0.5887 +/- 0.0000 |  |  | 329.0000 +/- 0.0000 | 0.0194 +/- 0.0000 |

`final_window_mse` best-groupwise-vs-best-MLP diff: -0.4261 +/- 0.0000; wins/losses/ties 0/1/0; best-groupwise counts {'groupwise_canonical': 1}.
