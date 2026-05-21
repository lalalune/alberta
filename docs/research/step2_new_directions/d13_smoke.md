# D13 Koopman/Reservoir RLS Results

Protocol: 1 paired seeds, 120 online steps, final window 40. Configs: koopman_tanh256.

This is a single-predictor test. The candidate methods use fixed nonlinear observables and an online RLS readout; they do not route over MLP predictions or train an MLP trunk. Positive candidate-vs-MLP paired differences favor the reservoir learner.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Readout dim | Runtime s |
|---|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2590 +/- 0.0000 | 0.2945 +/- 0.0000 |  |  |  | 8.2916 +/- 0.0000 |
| `mlp_h128` | 0.2785 +/- 0.0000 | 0.3061 +/- 0.0000 |  |  |  | 8.5805 +/- 0.0000 |
| `mlp_h64_64` | 0.2920 +/- 0.0000 | 0.3426 +/- 0.0000 |  |  |  | 3.5640 +/- 0.0000 |
| `koopman_tanh256` | 0.1567 +/- 0.0000 | 0.1596 +/- 0.0000 |  |  | 261.0000 +/- 0.0000 | 0.0901 +/- 0.0000 |

`final_window_mse` best-candidate-vs-best-MLP diff: +0.1023 +/- 0.0000; wins/losses/ties 1/0/0; best-candidate counts {'koopman_tanh256': 1}.

## Interpretation Bar

A positive row here means the fixed-observable mechanism has headroom on that benchmark. A Step 2 closure claim still requires one canonical configuration to beat the best fair MLP across the full benchmark set without per-dataset selection.
