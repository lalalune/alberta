# Step 2 Learned Resource Manager On Stateful External Benchmarks

All streams use the bundled sklearn digits dataset.  The manager is causal: each prediction uses allocation weights learned before seeing the current label.

```json
{
  "steps": 160,
  "n_seeds": 1,
  "seed": 0,
  "train_fraction": 0.7,
  "final_window": 40,
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
  "block_size": 40,
  "mask_keep_fraction": 0.5,
  "noise_std": 0.05,
  "benchmarks": [
    "digits_recurrent_permutation"
  ]
}
```

## digits_recurrent_permutation

Recurring pixel-permutation states; held-out evaluation averages over all recurrent permutations.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mlp_static` | 0.0955 +/- 0.0000 | 0.4750 +/- 0.0000 | 0.1042 +/- 0.0000 | 0.3065 +/- 0.0000 |
| `upgd_low` | 0.1125 +/- 0.0000 | 0.3000 +/- 0.0000 | 0.1092 +/- 0.0000 | 0.2223 +/- 0.0000 |
| `upgd_high` | 0.1061 +/- 0.0000 | 0.3250 +/- 0.0000 | 0.1196 +/- 0.0000 | 0.2033 +/- 0.0000 |
| `cbp_replace` | 0.0915 +/- 0.0000 | 0.4000 +/- 0.0000 | 0.1079 +/- 0.0000 | 0.3169 +/- 0.0000 |
| `resource_manager` | 0.0751 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.0848 +/- 0.0000 | 0.3659 +/- 0.0000 |

Paired resource-manager vs `mlp_static`:

- `final_window_mse`: diff +0.0204 +/- 0.0000; wins/losses/ties 1/0/0.
- `test_accuracy`: diff +0.0594 +/- 0.0000; wins/losses/ties 1/0/0.

## Assessment Rule

Positive paired differences favor the learned manager.  For MSE, the difference is baseline minus manager; for accuracy, it is manager minus baseline.
