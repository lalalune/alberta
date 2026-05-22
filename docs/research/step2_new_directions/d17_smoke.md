# D17 Non-MLP Block Hedge Results

Protocol: 1 paired seeds, 120 online steps, final window 40. Candidate configs: blockhedge_group_memory.

The candidate gate only combines non-MLP mathematical blocks. All blocks update every timestep; weights are causal discounted-Hedge weights computed before each current target update.

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Group w | Memory w | Poly w | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2590 +/- 0.0000 | 0.2945 +/- 0.0000 |  |  |  |  |  | 3.8875 +/- 0.0000 |
| `mlp_h128` | 0.2785 +/- 0.0000 | 0.3061 +/- 0.0000 |  |  |  |  |  | 3.9790 +/- 0.0000 |
| `mlp_h64_64` | 0.2920 +/- 0.0000 | 0.3426 +/- 0.0000 |  |  |  |  |  | 2.9219 +/- 0.0000 |
| `blockhedge_group_memory` | 0.3379 +/- 0.0000 | 0.2982 +/- 0.0000 |  |  | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.1409 +/- 0.0000 |

`final_window_mse` best-block-vs-best-MLP diff: -0.0789 +/- 0.0000; wins/losses/ties 0/1/0; best-block counts {'blockhedge_group_memory': 1}.
