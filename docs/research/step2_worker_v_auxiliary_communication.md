# Worker V Auxiliary Communication Probe

Protocol: 5 seeds, 1200 steps, final window 300. Auxiliary config: random projection dim 8, aux loss weight 0.04.

Methods are single learners only: fair `mlp64`, `upgd_low_noise`, and `aux_comm_mlp64`. Positive paired deltas favor `aux_comm_mlp64`; W/L/T is wins/losses/ties for the auxiliary learner.

## controlled_rare

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mlp64` | 0.0914 +/- 0.0150 | 0.1129 +/- 0.0101 |  |  |
| `upgd_low_noise` | 0.0960 +/- 0.0144 | 0.1167 +/- 0.0099 |  |  |
| `aux_comm_mlp64` | 0.0945 +/- 0.0162 | 0.1157 +/- 0.0111 |  |  |

Paired primary deltas (`final_window_mse`, positive favors aux):

- vs `mlp64`: -0.0031 +/- 0.0023; 1/4/0
- vs `upgd_low_noise`: +0.0015 +/- 0.0038; 3/2/0

## digits_label_drift

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mlp64` | 0.0394 +/- 0.0012 | 0.0555 +/- 0.0005 | 0.8620 +/- 0.0150 | 0.8779 +/- 0.0108 |
| `upgd_low_noise` | 0.0548 +/- 0.0007 | 0.0723 +/- 0.0007 | 0.6860 +/- 0.0081 | 0.8148 +/- 0.0098 |
| `aux_comm_mlp64` | 0.0411 +/- 0.0010 | 0.0579 +/- 0.0005 | 0.8467 +/- 0.0148 | 0.8790 +/- 0.0082 |

Paired primary deltas (`final_window_mse`, positive favors aux):

- vs `mlp64`: -0.0016 +/- 0.0006; 0/5/0
- vs `upgd_low_noise`: +0.0137 +/- 0.0009; 5/0/0
Paired held-out accuracy deltas (positive favors aux):

- vs `mlp64`: +0.0011 +/- 0.0157; 2/3/0
- vs `upgd_low_noise`: +0.0642 +/- 0.0164; 5/0/0

## digits_mask_noise

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mlp64` | 0.0493 +/- 0.0010 | 0.0601 +/- 0.0007 | 0.7860 +/- 0.0116 | 0.7985 +/- 0.0190 |
| `upgd_low_noise` | 0.0457 +/- 0.0017 | 0.0621 +/- 0.0008 | 0.7873 +/- 0.0133 | 0.8219 +/- 0.0099 |
| `aux_comm_mlp64` | 0.0479 +/- 0.0007 | 0.0605 +/- 0.0003 | 0.7953 +/- 0.0069 | 0.8152 +/- 0.0111 |

Paired primary deltas (`final_window_mse`, positive favors aux):

- vs `mlp64`: +0.0014 +/- 0.0003; 5/0/0
- vs `upgd_low_noise`: -0.0022 +/- 0.0011; 1/4/0
Paired held-out accuracy deltas (positive favors aux):

- vs `mlp64`: +0.0167 +/- 0.0085; 4/1/0
- vs `upgd_low_noise`: -0.0067 +/- 0.0140; 3/2/0

## digits_permuted_pixels

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mlp64` | 0.0473 +/- 0.0010 | 0.0598 +/- 0.0007 | 0.8207 +/- 0.0059 | 0.8471 +/- 0.0107 |
| `upgd_low_noise` | 0.0560 +/- 0.0010 | 0.0744 +/- 0.0007 | 0.7093 +/- 0.0171 | 0.7985 +/- 0.0027 |
| `aux_comm_mlp64` | 0.0495 +/- 0.0014 | 0.0615 +/- 0.0007 | 0.7953 +/- 0.0171 | 0.8438 +/- 0.0042 |

Paired primary deltas (`final_window_mse`, positive favors aux):

- vs `mlp64`: -0.0022 +/- 0.0008; 1/4/0
- vs `upgd_low_noise`: +0.0065 +/- 0.0009; 5/0/0
Paired held-out accuracy deltas (positive favors aux):

- vs `mlp64`: -0.0033 +/- 0.0082; 3/2/0
- vs `upgd_low_noise`: +0.0453 +/- 0.0050; 5/0/0

## synthetic_compositional

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mlp64` | 0.2660 +/- 0.0591 | 0.1881 +/- 0.0186 |  |  |
| `upgd_low_noise` | 0.4046 +/- 0.0868 | 0.2778 +/- 0.0300 |  |  |
| `aux_comm_mlp64` | 0.2590 +/- 0.0532 | 0.1876 +/- 0.0188 |  |  |

Paired primary deltas (`final_window_mse`, positive favors aux):

- vs `mlp64`: +0.0070 +/- 0.0075; 3/2/0
- vs `upgd_low_noise`: +0.1456 +/- 0.0343; 5/0/0

## Interpretation

`aux_comm_mlp64` has positive mean final-window MSE deltas against `mlp64` on 2/5 datasets and against `upgd_low_noise` on 4/5 datasets.

Auxiliary communication does not improve universality in this run; it behaves mostly like extra noisy loss on the shared trunk.
