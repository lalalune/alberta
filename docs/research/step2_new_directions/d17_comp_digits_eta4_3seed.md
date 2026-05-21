# D17 Non-MLP Block Hedge Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: blockhedge_group_memory.

The candidate gate only combines non-MLP mathematical blocks. All blocks update every timestep; weights are causal discounted-Hedge weights computed before each current target update.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Group w | Memory w | Poly w | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  |  |  | 5.5519 +/- 1.1759 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  |  |  | 5.8621 +/- 2.4114 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  |  |  | 4.4982 +/- 0.7341 |
| `blockhedge_group_memory` | 0.0533 +/- 0.0007 | 0.0584 +/- 0.0007 | 0.8011 +/- 0.0128 | 0.8553 +/- 0.0113 | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 16.7195 +/- 1.0896 |

`final_window_mse` best-block-vs-best-MLP diff: -0.0151 +/- 0.0008; wins/losses/ties 0/3/0; best-block counts {'blockhedge_group_memory': 3}.
`test_accuracy` best-block-vs-best-MLP diff: -0.0519 +/- 0.0126; wins/losses/ties 0/3/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Group w | Memory w | Poly w | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  |  |  | 4.0467 +/- 1.7949 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  |  |  | 2.8066 +/- 1.1753 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  |  |  | 3.1974 +/- 1.0615 |
| `blockhedge_group_memory` | 0.2392 +/- 0.0740 | 0.1557 +/- 0.0254 |  |  | 1.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 10.1638 +/- 2.8885 |

`final_window_mse` best-block-vs-best-MLP diff: +0.0325 +/- 0.0227; wins/losses/ties 2/1/0; best-block counts {'blockhedge_group_memory': 3}.
