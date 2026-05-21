# D18 OPMNIST Bridge

Protocol: 1 paired seed(s), 120 online steps, final window 40, source `sklearn_digits_28x28`, 2 permutation tasks, block size 40.
Deployment transforms: D18 `softmax`, MLP `raw`. Raw held-out metrics are reported separately from deployment-normalized held-out metrics.

Learners: D18 configs d18_step2_canonical versus fair MLP baselines mlp_h64, mlp_h128, mlp_h64_64.

## Protocol Gates

| Gate | Value |
| --- | --- |
| `source_kind` | `local_sklearn_digits_28x28` |
| `is_true_mnist` | `False` |
| `protocol` | `compact_online_permuted_pixels` |
| `steps` | `120` |
| `n_permutations` | `2` |
| `task_block_size` | `40` |
| `sample_with_replacement` | `False` |
| `task_sampling` | `sequential_epoch` |
| `include_identity_permutation` | `False` |
| `permutations_are_random_pixel_orders` | `True` |
| `task_id_provided_to_learner` | `False` |
| `prediction_before_update_every_step` | `True` |
| `all_experts_update_every_step` | `True` |
| `single_pass_examples_within_task` | `True` |
| `test_permutation_views` | `2` |
| `test_views_cover_observed_permutations` | `True` |
| `test_views_cover_all_permutations` | `True` |
| `full_mnist_task_blocks` | `False` |
| `matches_dohare_opmnist_core_protocol` | `False` |
| `matches_dohare_opmnist_published_task_count` | `False` |

## Aggregate Metrics

| Method | Final MSE | Final Acc | Raw Test MSE | Raw Test Acc | Deploy Test MSE | Deploy Test Acc | Runtime s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mlp_h64 | 0.097750 +/- 0.000000 | 0.300000 +/- 0.000000 | 0.079647 +/- 0.000000 | 0.443750 +/- 0.000000 | 0.079647 +/- 0.000000 | 0.443750 +/- 0.000000 | 5.464026 +/- 0.000000 |
| mlp_h128 | 0.093339 +/- 0.000000 | 0.375000 +/- 0.000000 | 0.078334 +/- 0.000000 | 0.543750 +/- 0.000000 | 0.078334 +/- 0.000000 | 0.543750 +/- 0.000000 | 4.271077 +/- 0.000000 |
| mlp_h64_64 | 0.102781 +/- 0.000000 | 0.200000 +/- 0.000000 | 0.090251 +/- 0.000000 | 0.350000 +/- 0.000000 | 0.090251 +/- 0.000000 | 0.350000 +/- 0.000000 | 3.156040 +/- 0.000000 |
| d18_step2_canonical | 0.090392 +/- 0.000000 | 0.275000 +/- 0.000000 | 0.073274 +/- 0.000000 | 0.381250 +/- 0.000000 | 0.085790 +/- 0.000000 | 0.381250 +/- 0.000000 | 2.145093 +/- 0.000000 |

## D18 vs Best MLP

- `final_window_mse`: diff=+0.002947, wins=1/0/0
- `final_window_accuracy`: diff=-0.100000, wins=0/1/0
- `test_mse`: diff=+0.005060, wins=1/0/0
- `test_accuracy`: diff=-0.162500, wins=0/1/0
- `deployment_test_mse`: diff=-0.007456, wins=0/1/0
- `deployment_test_accuracy`: diff=-0.162500, wins=0/1/0

## Blockers

- This run uses the local sklearn-digits 28x28 fallback; it is an OPMNIST analogue, not true MNIST.
- This run does not match the Dohare OPMNIST core protocol gates.
- This run does not match the 800-task/48M-step published OPMNIST task-count gate.
- This bridge is a materialized research runner; it is not yet the fused JAX/core production learner.
