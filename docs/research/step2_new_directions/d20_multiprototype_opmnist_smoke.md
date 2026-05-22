# D20 Multi-Prototype OPMNIST

Protocol: 1 paired seed(s), 120 online steps, final window 40, source `sklearn_digits_28x28`, 2 permutation tasks, block size 40.

D20 is a single online memory learner: multiple novelty-allocated prototypes per class, softmax over nearest-prototype class logits, no task id, no MLP expert, and no prediction router.

## Aggregate Metrics

| Method | Final MSE | Final Acc | Test MSE | Test Acc | Prototypes | Runtime s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp_h64` | 0.095234 +/- 0.000000 | 0.325000 +/- 0.000000 | 0.070428 +/- 0.000000 | 0.581250 +/- 0.000000 |  | 4.573825 +/- 0.000000 |
| `mlp_h128` | 0.094819 +/- 0.000000 | 0.300000 +/- 0.000000 | 0.084869 +/- 0.000000 | 0.550000 +/- 0.000000 |  | 4.036662 +/- 0.000000 |
| `mlp_h64_64` | 0.107575 +/- 0.000000 | 0.200000 +/- 0.000000 | 0.095459 +/- 0.000000 | 0.406250 +/- 0.000000 |  | 3.779301 +/- 0.000000 |
| `d20_s20_n0p08_bw0p01_eta0p3` | 0.037877 +/- 0.000000 | 0.725000 +/- 0.000000 | 0.017890 +/- 0.000000 | 0.887500 +/- 0.000000 | 56.000000 +/- 0.000000 | 0.492145 +/- 0.000000 |

## D20 vs Best MLP

- `final_window_mse`: diff=+0.056942, wins=1/0/0
- `final_window_accuracy`: diff=+0.400000, wins=1/0/0
- `test_mse`: diff=+0.052538, wins=1/0/0
- `test_accuracy`: diff=+0.306250, wins=1/0/0
- `deployment_test_mse`: diff=+0.052538, wins=1/0/0
- `deployment_test_accuracy`: diff=+0.306250, wins=1/0/0

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

## Interpretation

The result shows that OPMNIST retention needs task-view memory geometry: multiple prototypes per class, not a single averaged class prototype. This is a candidate component for folding into D18 or the core fast/slow learner, not yet a full all14 Step 2 replacement.
