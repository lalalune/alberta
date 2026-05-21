# D18 OPMNIST Bridge

Protocol: 3 paired seed(s), 1000 online steps, final window 200, source `sklearn_digits_28x28`, 5 permutation tasks, block size 200.
Deployment transforms: D18 `softmax`, MLP `raw`. Raw held-out metrics are reported separately from deployment-normalized held-out metrics.

Learners: D18 configs d18_step2_canonical versus fair MLP baselines mlp_h64, mlp_h128, mlp_h64_64.

## Protocol Gates

| Gate | Value |
| --- | --- |
| `source_kind` | `local_sklearn_digits_28x28` |
| `is_true_mnist` | `False` |
| `protocol` | `compact_online_permuted_pixels` |
| `steps` | `1000` |
| `n_permutations` | `5` |
| `task_block_size` | `200` |
| `sample_with_replacement` | `False` |
| `task_sampling` | `sequential_epoch` |
| `include_identity_permutation` | `False` |
| `permutations_are_random_pixel_orders` | `True` |
| `task_id_provided_to_learner` | `False` |
| `prediction_before_update_every_step` | `True` |
| `all_experts_update_every_step` | `True` |
| `single_pass_examples_within_task` | `True` |
| `test_permutation_views` | `5` |
| `test_views_cover_observed_permutations` | `True` |
| `test_views_cover_all_permutations` | `True` |
| `full_mnist_task_blocks` | `False` |
| `matches_dohare_opmnist_core_protocol` | `False` |
| `matches_dohare_opmnist_published_task_count` | `False` |

## Aggregate Metrics

| Method | Final MSE | Final Acc | Raw Test MSE | Raw Test Acc | Deploy Test MSE | Deploy Test Acc | Runtime s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d18_step2_canonical | 0.069065 +/- 0.000351 | 0.580000 +/- 0.005000 | 0.203917 +/- 0.006567 | 0.475667 +/- 0.039303 | 0.085167 +/- 0.000758 | 0.475667 +/- 0.039303 | 62.901070 +/- 4.460724 |
| mlp_h128 | 0.082511 +/- 0.002480 | 0.481667 +/- 0.010138 | 0.081689 +/- 0.002736 | 0.550500 +/- 0.036750 | 0.081689 +/- 0.002736 | 0.550500 +/- 0.036750 | 4.912606 +/- 1.314854 |
| mlp_h64 | 0.077333 +/- 0.001427 | 0.505000 +/- 0.012583 | 0.083670 +/- 0.001610 | 0.468000 +/- 0.021789 | 0.083670 +/- 0.001610 | 0.468000 +/- 0.021789 | 5.713681 +/- 1.418562 |
| mlp_h64_64 | 0.083585 +/- 0.001673 | 0.443333 +/- 0.016915 | 0.104175 +/- 0.007594 | 0.345333 +/- 0.029127 | 0.104175 +/- 0.007594 | 0.345333 +/- 0.029127 | 4.194233 +/- 1.148172 |

## D18 vs Best MLP

- `final_window_mse`: diff=+0.008268, wins=3/0/0
- `final_window_accuracy`: diff=+0.073333, wins=3/0/0
- `test_mse`: diff=-0.122405, wins=0/3/0
- `test_accuracy`: diff=-0.074833, wins=0/3/0
- `deployment_test_mse`: diff=-0.003654, wins=1/2/0
- `deployment_test_accuracy`: diff=-0.074833, wins=0/3/0

## Blockers

- This run uses the local sklearn-digits 28x28 fallback; it is an OPMNIST analogue, not true MNIST.
- This run does not match the Dohare OPMNIST core protocol gates.
- This run does not match the 800-task/48M-step published OPMNIST task-count gate.
- This bridge is a materialized research runner; it is not yet the fused JAX/core production learner.
