# D16 Additive Universal Learner Results

Protocol: 3 paired seeds, 1200 online steps, final window 300. Candidate configs: additive_group_only, additive_group_memory.

Each candidate is one additive predictor updated from one global residual at every step. Positive candidate-vs-MLP differences favor the additive learner.

## controlled_interaction

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.4283 +/- 0.0243 | 0.6403 +/- 0.0519 |  |  |  |  | 0.8650 +/- 0.4254 |
| `mlp_h128` | 0.5002 +/- 0.0329 | 0.6511 +/- 0.0492 |  |  |  |  | 0.8927 +/- 0.2140 |
| `mlp_h64_64` | 0.5736 +/- 0.0421 | 0.7389 +/- 0.0464 |  |  |  |  | 0.8491 +/- 0.2241 |
| `additive_group_only` | 0.4756 +/- 0.0610 | 0.8855 +/- 0.0703 |  |  | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.3308 +/- 0.0422 |
| `additive_group_memory` | 0.4271 +/- 0.0374 | 0.8254 +/- 0.0527 |  |  | 128.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 4.0529 +/- 0.0620 |

`final_window_mse` best-additive-vs-best-MLP diff: +0.0011 +/- 0.0133; wins/losses/ties 1/2/0; best-additive counts {'additive_group_memory': 3}.

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0383 +/- 0.0010 | 0.0555 +/- 0.0004 | 0.8600 +/- 0.0117 | 0.8998 +/- 0.0075 |  |  | 1.6107 +/- 0.2197 |
| `mlp_h128` | 0.0442 +/- 0.0008 | 0.0593 +/- 0.0008 | 0.8467 +/- 0.0145 | 0.9041 +/- 0.0043 |  |  | 1.4354 +/- 0.2476 |
| `mlp_h64_64` | 0.0401 +/- 0.0015 | 0.0597 +/- 0.0013 | 0.8356 +/- 0.0194 | 0.8837 +/- 0.0034 |  |  | 1.3180 +/- 0.4191 |
| `additive_group_only` | 0.0542 +/- 0.0002 | 0.0625 +/- 0.0006 | 0.7922 +/- 0.0068 | 0.8503 +/- 0.0104 | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.5841 +/- 0.1165 |
| `additive_group_memory` | 0.0536 +/- 0.0007 | 0.0622 +/- 0.0008 | 0.7900 +/- 0.0084 | 0.8639 +/- 0.0097 | 128.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 6.5639 +/- 2.6264 |

`final_window_mse` best-additive-vs-best-MLP diff: -0.0154 +/- 0.0005; wins/losses/ties 0/3/0; best-additive counts {'additive_group_memory': 2, 'additive_group_only': 1}.
`test_accuracy` best-additive-vs-best-MLP diff: -0.0433 +/- 0.0097; wins/losses/ties 0/3/0.

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  |  |  |  | 1.0245 +/- 0.6767 |
| `mlp_h128` | 0.2758 +/- 0.0936 | 0.1787 +/- 0.0311 |  |  |  |  | 1.1547 +/- 0.6448 |
| `mlp_h64_64` | 0.3289 +/- 0.1051 | 0.2184 +/- 0.0326 |  |  |  |  | 0.8615 +/- 0.2778 |
| `additive_group_only` | 0.2445 +/- 0.0796 | 0.1572 +/- 0.0262 |  |  | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.3662 +/- 0.0888 |
| `additive_group_memory` | 0.2390 +/- 0.0719 | 0.1556 +/- 0.0253 |  |  | 128.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 1.6171 +/- 0.1301 |

`final_window_mse` best-additive-vs-best-MLP diff: +0.0353 +/- 0.0233; wins/losses/ties 2/1/0; best-additive counts {'additive_group_memory': 2, 'additive_group_only': 1}.

## synthetic_frequency

| Method | Final MSE | Mean MSE | Final Acc | Test Acc | Memory centers | Poly centers | Runtime s |
|---|---:|---:|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.1619 +/- 0.2671 | 1.3319 +/- 0.2426 |  |  |  |  | 0.7661 +/- 0.1741 |
| `mlp_h128` | 1.1782 +/- 0.2588 | 1.3467 +/- 0.2402 |  |  |  |  | 0.5907 +/- 0.1374 |
| `mlp_h64_64` | 1.1493 +/- 0.2579 | 1.3167 +/- 0.2328 |  |  |  |  | 0.6616 +/- 0.2183 |
| `additive_group_only` | 0.7981 +/- 0.0953 | 0.9099 +/- 0.2037 |  |  | 0.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 0.3091 +/- 0.0853 |
| `additive_group_memory` | 0.8085 +/- 0.0976 | 0.9138 +/- 0.2036 |  |  | 128.0000 +/- 0.0000 | 0.0000 +/- 0.0000 | 2.4643 +/- 0.2829 |

`final_window_mse` best-additive-vs-best-MLP diff: +0.3504 +/- 0.1831; wins/losses/ties 3/0/0; best-additive counts {'additive_group_only': 3}.
