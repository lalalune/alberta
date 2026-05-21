# Step 2 Learned Resource Manager On Stateful External Benchmarks

All streams use the bundled sklearn digits dataset.  The manager is causal: each prediction uses allocation weights learned before seeing the current label.

```json
{
  "steps": 1200,
  "n_seeds": 10,
  "seed": 0,
  "train_fraction": 0.7,
  "final_window": 300,
  "hidden_size": 64,
  "step_size": 0.03,
  "sparsity": 0.5,
  "low_sigma": 0.0001,
  "high_sigma": 0.001,
  "cbp_decay": 0.99,
  "cbp_replacement_rate": 0.0005,
  "cbp_maturity": 100,
  "manager_learning_rate": 2.0,
  "manager_discount": 0.995,
  "retention_manager_learning_rate": 4.0,
  "retention_manager_discount": 0.999,
  "manager_exploration": 0.01,
  "manager_loss_decay": 0.99,
  "resource_cost_weight": 0.0,
  "resource_policy_names": [
    "mlp_static",
    "upgd_low",
    "upgd_high",
    "cbp_replace"
  ],
  "resource_policy_costs": {
    "mlp_static": 0.0,
    "upgd_low": 0.1,
    "upgd_high": 1.0,
    "cbp_replace": 0.5
  },
  "n_states": 5,
  "block_size": 240,
  "mask_keep_fraction": 0.5,
  "noise_std": 0.05,
  "benchmarks": [
    "digits_recurrent_permutation",
    "digits_recurrent_mask_noise",
    "digits_class_blocked_retention"
  ]
}
```

## digits_recurrent_permutation

Recurring pixel-permutation states; held-out evaluation averages over all recurrent permutations.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mlp_static` | 0.0656 +/- 0.0005 | 0.6647 +/- 0.0046 | 0.1022 +/- 0.0013 | 0.3425 +/- 0.0055 |
| `upgd_low` | 0.0742 +/- 0.0008 | 0.5433 +/- 0.0092 | 0.0736 +/- 0.0009 | 0.5364 +/- 0.0137 |
| `upgd_high` | 0.0741 +/- 0.0007 | 0.5317 +/- 0.0113 | 0.0768 +/- 0.0007 | 0.4941 +/- 0.0096 |
| `cbp_replace` | 0.0646 +/- 0.0009 | 0.6750 +/- 0.0095 | 0.1035 +/- 0.0014 | 0.3250 +/- 0.0084 |
| `resource_manager` | 0.0548 +/- 0.0006 | 0.7070 +/- 0.0058 | 0.0907 +/- 0.0012 | 0.3624 +/- 0.0079 |
| `resource_manager_retention` | 0.0578 +/- 0.0008 | 0.7040 +/- 0.0070 | 0.0948 +/- 0.0015 | 0.3490 +/- 0.0085 |

Paired resource-manager vs `mlp_static`:

- `final_window_mse`: diff +0.0108 +/- 0.0003; wins/losses/ties 10/0/0.
- `test_accuracy`: diff +0.0199 +/- 0.0049; wins/losses/ties 9/1/0.
- `resource_manager_retention` held-out `test_accuracy`: diff +0.0065 +/- 0.0045; wins/losses/ties 8/2/0.

## digits_recurrent_mask_noise

Recurring feature-mask states with online noise; held-out evaluation averages over the recurring masks.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mlp_static` | 0.0504 +/- 0.0013 | 0.7730 +/- 0.0120 | 0.0553 +/- 0.0015 | 0.7486 +/- 0.0110 |
| `upgd_low` | 0.0493 +/- 0.0008 | 0.7820 +/- 0.0098 | 0.0498 +/- 0.0009 | 0.7836 +/- 0.0088 |
| `upgd_high` | 0.0517 +/- 0.0009 | 0.7723 +/- 0.0141 | 0.0539 +/- 0.0009 | 0.7576 +/- 0.0098 |
| `cbp_replace` | 0.0499 +/- 0.0010 | 0.7760 +/- 0.0081 | 0.0559 +/- 0.0016 | 0.7446 +/- 0.0152 |
| `resource_manager` | 0.0395 +/- 0.0009 | 0.8327 +/- 0.0085 | 0.0443 +/- 0.0011 | 0.7942 +/- 0.0119 |
| `resource_manager_retention` | 0.0420 +/- 0.0011 | 0.8193 +/- 0.0088 | 0.0479 +/- 0.0012 | 0.7809 +/- 0.0121 |

Paired resource-manager vs `mlp_static`:

- `final_window_mse`: diff +0.0109 +/- 0.0005; wins/losses/ties 10/0/0.
- `test_accuracy`: diff +0.0456 +/- 0.0044; wins/losses/ties 10/0/0.
- `resource_manager_retention` held-out `test_accuracy`: diff +0.0323 +/- 0.0044; wins/losses/ties 10/0/0.

## digits_class_blocked_retention

Digit-class blocks create current-block specialization pressure; held-out evaluation is balanced over all classes.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mlp_static` | 0.0051 +/- 0.0001 | 0.9840 +/- 0.0016 | 0.1324 +/- 0.0020 | 0.1265 +/- 0.0104 |
| `upgd_low` | 0.0205 +/- 0.0005 | 0.9340 +/- 0.0037 | 0.0982 +/- 0.0021 | 0.2421 +/- 0.0140 |
| `upgd_high` | 0.0217 +/- 0.0007 | 0.9290 +/- 0.0044 | 0.1059 +/- 0.0018 | 0.2050 +/- 0.0149 |
| `cbp_replace` | 0.0048 +/- 0.0001 | 0.9850 +/- 0.0015 | 0.1341 +/- 0.0023 | 0.1263 +/- 0.0098 |
| `resource_manager` | 0.0041 +/- 0.0001 | 0.9853 +/- 0.0014 | 0.1316 +/- 0.0020 | 0.1258 +/- 0.0098 |
| `resource_manager_retention` | 0.0202 +/- 0.0008 | 0.9327 +/- 0.0042 | 0.0986 +/- 0.0023 | 0.2386 +/- 0.0131 |

Paired resource-manager vs `mlp_static`:

- `final_window_mse`: diff +0.0010 +/- 0.0000; wins/losses/ties 10/0/0.
- `test_accuracy`: diff -0.0007 +/- 0.0027; wins/losses/ties 2/3/5.
- `resource_manager_retention` held-out `test_accuracy`: diff +0.1121 +/- 0.0176; wins/losses/ties 10/0/0.

## Assessment Rule

Positive paired differences favor the learned manager.  For MSE, the difference is baseline minus manager; for accuracy, it is manager minus baseline.
