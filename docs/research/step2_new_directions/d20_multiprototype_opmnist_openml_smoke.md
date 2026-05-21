# D20 Multi-Prototype OPMNIST

Protocol: 1 paired seed(s), 1000 online steps, final window 200, source `openml`, 5 permutation tasks, block size 200.

D20 is a single online memory learner: multiple novelty-allocated prototypes per class, softmax over nearest-prototype class logits, no task id, no MLP expert, and no prediction router.

## Aggregate Metrics

| Method | Final MSE | Final Acc | Test MSE | Test Acc | Prototypes | Runtime s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp_h64` | 0.092701 +/- 0.000000 | 0.400000 +/- 0.000000 | 0.103770 +/- 0.000000 | 0.294800 +/- 0.000000 |  | 2.784414 +/- 0.000000 |
| `mlp_h128` | 0.100829 +/- 0.000000 | 0.325000 +/- 0.000000 | 0.097477 +/- 0.000000 | 0.346400 +/- 0.000000 |  | 2.039450 +/- 0.000000 |
| `mlp_h64_64` | 0.100479 +/- 0.000000 | 0.280000 +/- 0.000000 | 0.098299 +/- 0.000000 | 0.276000 +/- 0.000000 |  | 1.618862 +/- 0.000000 |
| `d20_s20_n0p08_bw0p01_eta0p3` | 0.055277 +/- 0.000000 | 0.595000 +/- 0.000000 | 0.043641 +/- 0.000000 | 0.684000 +/- 0.000000 | 187.000000 +/- 0.000000 | 1.656063 +/- 0.000000 |

## D20 vs Best MLP

- `final_window_mse`: diff=+0.037424, wins=1/0/0
- `final_window_accuracy`: diff=+0.195000, wins=1/0/0
- `test_mse`: diff=+0.053836, wins=1/0/0
- `test_accuracy`: diff=+0.337600, wins=1/0/0
- `deployment_test_mse`: diff=+0.053836, wins=1/0/0
- `deployment_test_accuracy`: diff=+0.337600, wins=1/0/0

## Protocol Gates

| Gate | Value |
| --- | --- |
| `source_kind` | `openml_mnist_784` |
| `is_true_mnist` | `True` |
| `is_full_mnist_split` | `False` |
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
