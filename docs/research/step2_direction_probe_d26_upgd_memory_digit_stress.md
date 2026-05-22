# D26 UPGD-Memory Digit Stress

Protocol: 5 paired seed(s), 600 online steps, final window 150.

| Dataset | Method | Final MSE | Final Acc | Test MSE | Test Acc | Protos | Gate | Runtime s |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `digits_class_blocked` | `mlp_h64` | 0.006586 +/- 0.000546 | 0.986667 +/- 0.002108 | 0.137827 +/- 0.006046 | 0.129128 +/- 0.018419 |  |  | 0.413622 +/- 0.027670 |
| `digits_class_blocked` | `mlp_h128` | 0.009272 +/- 0.000285 | 0.985333 +/- 0.002494 | 0.140046 +/- 0.005767 | 0.125046 +/- 0.018835 |  |  | 0.393043 +/- 0.016583 |
| `digits_class_blocked` | `mlp_h64_64` | 0.004264 +/- 0.000093 | 0.988000 +/- 0.001333 | 0.166289 +/- 0.002873 | 0.099814 +/- 0.000909 |  |  | 0.454756 +/- 0.024236 |
| `digits_class_blocked` | `upgdmem_s20_alloc18` | 0.012261 +/- 0.001560 | 0.912000 +/- 0.012543 | 0.085987 +/- 0.001644 | 0.431540 +/- 0.015760 | 84.000000 +/- 1.414214 | 0.172845 +/- 0.002894 | 0.501810 +/- 0.017161 |
| `digits_class_blocked` | `upgdmem_s10_alloc18` | 0.011306 +/- 0.001757 | 0.926667 +/- 0.012824 | 0.086431 +/- 0.001613 | 0.426345 +/- 0.015593 | 45.800000 +/- 1.200000 | 0.174014 +/- 0.002108 | 0.463645 +/- 0.018669 |
| `digits_class_blocked` | `upgdmem_s20_alloc18_mem0` | 0.009233 +/- 0.001393 | 0.946667 +/- 0.008692 | 0.089056 +/- 0.000983 | 0.486456 +/- 0.004708 | 84.000000 +/- 1.414214 | 0.566338 +/- 0.002136 | 0.476263 +/- 0.013378 |

## digits_class_blocked Comparisons

- `final_window_mse`: diff=-0.004949, wins=0/5/0, best={'upgdmem_s10_alloc18': 1, 'upgdmem_s20_alloc18_mem0': 4}
- `final_window_accuracy`: diff=-0.041333, wins=0/5/0, best={'upgdmem_s10_alloc18': 2, 'upgdmem_s20_alloc18_mem0': 3}
- `test_mse`: diff=+0.048372, wins=5/0/0, best={'upgdmem_s10_alloc18': 1, 'upgdmem_s20_alloc18': 4}
- `test_accuracy`: diff=+0.351391, wins=5/0/0, best={'upgdmem_s20_alloc18_mem0': 5}

| `digits_mask_noise` | `mlp_h64` | 0.057344 +/- 0.001412 | 0.717333 +/- 0.017075 | 0.051908 +/- 0.003138 | 0.764750 +/- 0.021861 |  |  | 0.575917 +/- 0.192092 |
| `digits_mask_noise` | `mlp_h128` | 0.058928 +/- 0.002310 | 0.726667 +/- 0.020763 | 0.054480 +/- 0.003435 | 0.758070 +/- 0.029705 |  |  | 0.538152 +/- 0.152958 |
| `digits_mask_noise` | `mlp_h64_64` | 0.065561 +/- 0.001446 | 0.644000 +/- 0.018571 | 0.058031 +/- 0.002921 | 0.701670 +/- 0.029390 |  |  | 0.449016 +/- 0.036162 |
| `digits_mask_noise` | `upgdmem_s20_alloc18` | 0.025773 +/- 0.001422 | 0.825333 +/- 0.010625 | 0.023315 +/- 0.001163 | 0.848609 +/- 0.007708 | 170.800000 +/- 1.685230 | 0.274239 +/- 0.003211 | 0.537443 +/- 0.072250 |
| `digits_mask_noise` | `upgdmem_s10_alloc18` | 0.027446 +/- 0.001799 | 0.810667 +/- 0.011470 | 0.023550 +/- 0.001276 | 0.845640 +/- 0.009265 | 99.000000 +/- 0.316228 | 0.272971 +/- 0.004289 | 0.473953 +/- 0.042503 |
| `digits_mask_noise` | `upgdmem_s20_alloc18_mem0` | 0.027047 +/- 0.002403 | 0.817333 +/- 0.019619 | 0.023316 +/- 0.001405 | 0.837848 +/- 0.012315 | 170.800000 +/- 1.685230 | 0.649780 +/- 0.002698 | 0.463278 +/- 0.011170 |

## digits_mask_noise Comparisons

- `final_window_mse`: diff=+0.031604, wins=5/0/0, best={'upgdmem_s20_alloc18': 3, 'upgdmem_s20_alloc18_mem0': 2}
- `final_window_accuracy`: diff=+0.097333, wins=5/0/0, best={'upgdmem_s20_alloc18': 3, 'upgdmem_s20_alloc18_mem0': 2}
- `test_mse`: diff=+0.029241, wins=5/0/0, best={'upgdmem_s10_alloc18': 1, 'upgdmem_s20_alloc18': 3, 'upgdmem_s20_alloc18_mem0': 1}
- `test_accuracy`: diff=+0.080891, wins=5/0/0, best={'upgdmem_s10_alloc18': 2, 'upgdmem_s20_alloc18': 2, 'upgdmem_s20_alloc18_mem0': 1}

