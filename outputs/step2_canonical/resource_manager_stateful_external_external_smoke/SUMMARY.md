# Step 2 Learned Resource Manager On Stateful External Benchmarks

Default streams use the bundled sklearn digits dataset.  The optional external image stream uses OpenML Fashion-MNIST only when explicitly enabled, otherwise a local 28x28 sklearn-digits fallback.  The manager is causal: each prediction uses allocation weights learned before seeing the current label.

```json
{
  "steps": 80,
  "n_seeds": 1,
  "seed": 0,
  "train_fraction": 0.7,
  "final_window": 20,
  "hidden_size": 16,
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
  "n_states": 3,
  "block_size": 20,
  "context_delay_blocks": 1,
  "mask_keep_fraction": 0.5,
  "noise_std": 0.05,
  "external_image_source": "digits_28x28_fallback",
  "allow_openml_download": false,
  "external_sample_limit": 3000,
  "benchmarks": [
    "external_delayed_contextual_permutation"
  ]
}
```

## external_delayed_contextual_permutation

Fashion-MNIST-style 28x28 image stream when OpenML is enabled, otherwise expanded sklearn digits fallback; recurring pixel permutations use true hidden states while manager context ids are delayed by whole blocks.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mlp_static` | 0.0877 +/- 0.0000 | 0.3000 +/- 0.0000 | 0.0944 +/- 0.0000 | 0.2931 +/- 0.0000 |
| `upgd_low` | 0.1380 +/- 0.0000 | 0.1000 +/- 0.0000 | 0.1489 +/- 0.0000 | 0.1596 +/- 0.0000 |
| `upgd_high` | 0.1159 +/- 0.0000 | 0.6000 +/- 0.0000 | 0.1436 +/- 0.0000 | 0.1602 +/- 0.0000 |
| `cbp_replace` | 0.0864 +/- 0.0000 | 0.5000 +/- 0.0000 | 0.0984 +/- 0.0000 | 0.2968 +/- 0.0000 |
| `resource_manager` | 0.0756 +/- 0.0000 | 0.4500 +/- 0.0000 | 0.0830 +/- 0.0000 | 0.3500 +/- 0.0000 |
| `resource_manager_retention` | 0.0800 +/- 0.0000 | 0.3500 +/- 0.0000 | 0.0868 +/- 0.0000 | 0.3414 +/- 0.0000 |

Paired resource-manager vs `mlp_static`:

- `final_window_mse`: diff +0.0120 +/- 0.0000; wins/losses/ties 1/0/0.
- `test_accuracy`: diff +0.0569 +/- 0.0000; wins/losses/ties 1/0/0.
- `resource_manager_retention` held-out `test_accuracy`: diff +0.0482 +/- 0.0000; wins/losses/ties 1/0/0.

## Assessment Rule

Positive paired differences favor the learned manager.  For MSE, the difference is baseline minus manager; for accuracy, it is manager minus baseline.
