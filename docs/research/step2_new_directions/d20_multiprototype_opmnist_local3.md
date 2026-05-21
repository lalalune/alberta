# D20 Multi-Prototype OPMNIST

Protocol: 3 paired seed(s), 1000 online steps, final window 200, source `sklearn_digits_28x28`, 5 permutation tasks, block size 200.

D20 is a single online memory learner: multiple novelty-allocated prototypes per class, softmax over nearest-prototype class logits, no task id, no MLP expert, and no prediction router.

## Aggregate Metrics

| Method | Final MSE | Final Acc | Test MSE | Test Acc | Prototypes | Runtime s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp_h64` | 0.076341 +/- 0.001639 | 0.498333 +/- 0.014814 | 0.092098 +/- 0.004249 | 0.433167 +/- 0.023467 |  | 2.541095 +/- 1.050941 |
| `mlp_h128` | 0.080721 +/- 0.002698 | 0.520000 +/- 0.013229 | 0.085800 +/- 0.002722 | 0.526333 +/- 0.014917 |  | 2.493634 +/- 0.573024 |
| `mlp_h64_64` | 0.084254 +/- 0.001921 | 0.423333 +/- 0.023511 | 0.093019 +/- 0.002019 | 0.360667 +/- 0.030852 |  | 2.142445 +/- 0.453199 |
| `d20_s20_n0p08_bw0p01_eta0p3` | 0.026361 +/- 0.000711 | 0.825000 +/- 0.010000 | 0.013433 +/- 0.000855 | 0.907333 +/- 0.005364 | 178.333333 +/- 2.403701 | 5.057055 +/- 0.221301 |

## D20 vs Best MLP

- `final_window_mse`: diff=+0.049980, wins=3/0/0
- `final_window_accuracy`: diff=+0.305000, wins=3/0/0
- `test_mse`: diff=+0.070844, wins=3/0/0
- `test_accuracy`: diff=+0.381000, wins=3/0/0
- `deployment_test_mse`: diff=+0.070844, wins=3/0/0
- `deployment_test_accuracy`: diff=+0.381000, wins=3/0/0

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

## Interpretation

The result shows that OPMNIST retention needs task-view memory geometry: multiple prototypes per class, not a single averaged class prototype. This is a candidate component for folding into D18 or the core fast/slow learner, not yet a full all14 Step 2 replacement.
