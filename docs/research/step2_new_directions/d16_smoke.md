# D16 Additive Universal Learner Results

Protocol: 1 paired seeds, 120 online steps, final window 40. Candidate configs: additive_group_memory.

Each candidate is one additive predictor updated from one global residual at every step. Positive candidate-vs-MLP differences favor the additive learner.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2590 +/- 0.0000 | 0.2945 +/- 0.0000 |  |  |  |  | 2.0661 +/- 0.0000 |
| `mlp_h128` | 0.2785 +/- 0.0000 | 0.3061 +/- 0.0000 |  |  |  |  | 2.0581 +/- 0.0000 |
| `mlp_h64_64` | 0.2920 +/- 0.0000 | 0.3426 +/- 0.0000 |  |  |  |  | 1.1159 +/- 0.0000 |
| `additive_group_memory` | 0.6110 +/- 0.0000 | 0.3959 +/- 0.0000 |  |  | 16.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0841 +/- 0.0000 |

`final_window_mse` best-additive-vs-best-MLP diff: -0.3520 +/- 0.0000; wins/losses/ties 0/1/0; best-additive counts {'additive_group_memory': 1}.
