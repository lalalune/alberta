# Step 2 Learned Resource Manager On Stateful External Benchmarks

All streams use the bundled sklearn digits dataset.  The manager is causal: each prediction uses allocation weights learned before seeing the current label.

```json
{
  "steps": 1200,
  "n_seeds": 5,
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
| `mlp_static` | 0.0665 +/- 0.0006 | 0.6620 +/- 0.0053 | 0.1031 +/- 0.0015 | 0.3363 +/- 0.0025 |
| `upgd_low` | 0.0753 +/- 0.0015 | 0.5253 +/- 0.0143 | 0.0729 +/- 0.0013 | 0.5434 +/- 0.0192 |
| `upgd_high` | 0.0748 +/- 0.0012 | 0.5193 +/- 0.0189 | 0.0761 +/- 0.0009 | 0.5057 +/- 0.0143 |
| `cbp_replace` | 0.0667 +/- 0.0009 | 0.6660 +/- 0.0135 | 0.1040 +/- 0.0010 | 0.3128 +/- 0.0088 |
| `resource_manager` | 0.0559 +/- 0.0009 | 0.7000 +/- 0.0082 | 0.0907 +/- 0.0007 | 0.3537 +/- 0.0105 |
| `resource_manager_retention` | 0.0591 +/- 0.0010 | 0.6993 +/- 0.0071 | 0.0947 +/- 0.0011 | 0.3381 +/- 0.0095 |

Paired resource-manager vs `mlp_static`:

- `final_window_mse`: diff +0.0106 +/- 0.0005; wins/losses/ties 5/0/0.
- `test_accuracy`: diff +0.0174 +/- 0.0098; wins/losses/ties 4/1/0.

## digits_recurrent_mask_noise

Recurring feature-mask states with online noise; held-out evaluation averages over the recurring masks.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mlp_static` | 0.0504 +/- 0.0015 | 0.7693 +/- 0.0153 | 0.0533 +/- 0.0016 | 0.7581 +/- 0.0144 |
| `upgd_low` | 0.0494 +/- 0.0008 | 0.7840 +/- 0.0103 | 0.0495 +/- 0.0011 | 0.7837 +/- 0.0087 |
| `upgd_high` | 0.0515 +/- 0.0010 | 0.7787 +/- 0.0123 | 0.0536 +/- 0.0012 | 0.7688 +/- 0.0110 |
| `cbp_replace` | 0.0498 +/- 0.0008 | 0.7760 +/- 0.0107 | 0.0558 +/- 0.0024 | 0.7492 +/- 0.0236 |
| `resource_manager` | 0.0394 +/- 0.0007 | 0.8360 +/- 0.0092 | 0.0436 +/- 0.0016 | 0.8008 +/- 0.0170 |
| `resource_manager_retention` | 0.0419 +/- 0.0009 | 0.8233 +/- 0.0018 | 0.0475 +/- 0.0020 | 0.7875 +/- 0.0192 |

Paired resource-manager vs `mlp_static`:

- `final_window_mse`: diff +0.0110 +/- 0.0010; wins/losses/ties 5/0/0.
- `test_accuracy`: diff +0.0427 +/- 0.0070; wins/losses/ties 5/0/0.

## digits_class_blocked_retention

Digit-class blocks create current-block specialization pressure; held-out evaluation is balanced over all classes.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mlp_static` | 0.0052 +/- 0.0001 | 0.9847 +/- 0.0017 | 0.1325 +/- 0.0024 | 0.1447 +/- 0.0176 |
| `upgd_low` | 0.0208 +/- 0.0007 | 0.9340 +/- 0.0037 | 0.1019 +/- 0.0020 | 0.2252 +/- 0.0098 |
| `upgd_high` | 0.0229 +/- 0.0007 | 0.9253 +/- 0.0045 | 0.1075 +/- 0.0019 | 0.1978 +/- 0.0165 |
| `cbp_replace` | 0.0050 +/- 0.0001 | 0.9847 +/- 0.0013 | 0.1327 +/- 0.0011 | 0.1362 +/- 0.0161 |
| `resource_manager` | 0.0042 +/- 0.0001 | 0.9860 +/- 0.0012 | 0.1309 +/- 0.0012 | 0.1410 +/- 0.0169 |
| `resource_manager_retention` | 0.0211 +/- 0.0011 | 0.9300 +/- 0.0047 | 0.1027 +/- 0.0025 | 0.2237 +/- 0.0107 |

Paired resource-manager vs `mlp_static`:

- `final_window_mse`: diff +0.0010 +/- 0.0000; wins/losses/ties 5/0/0.
- `test_accuracy`: diff -0.0037 +/- 0.0026; wins/losses/ties 0/2/3.

## Assessment Rule

Positive paired differences favor the learned manager.  For MSE, the difference is baseline minus manager; for accuracy, it is manager minus baseline.
