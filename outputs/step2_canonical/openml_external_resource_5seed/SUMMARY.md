# Step 2 Learned Resource Manager On Stateful External Benchmarks

Default streams use the bundled sklearn digits dataset.  The optional external image stream uses OpenML Fashion-MNIST only when explicitly enabled, otherwise a local 28x28 sklearn-digits fallback.  The manager is causal: each prediction uses allocation weights learned before seeing the current label.

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
  "context_delay_blocks": 1,
  "mask_keep_fraction": 0.5,
  "noise_std": 0.05,
  "external_image_source": "openml_fashion_mnist",
  "allow_openml_download": true,
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
| `mlp_static` | 0.0829 +/- 0.0022 | 0.4873 +/- 0.0190 | 0.1001 +/- 0.0009 | 0.3137 +/- 0.0117 |
| `upgd_low` | 0.0968 +/- 0.0010 | 0.3360 +/- 0.0132 | 0.0837 +/- 0.0009 | 0.4519 +/- 0.0101 |
| `upgd_high` | 0.1034 +/- 0.0021 | 0.2767 +/- 0.0177 | 0.0990 +/- 0.0006 | 0.3190 +/- 0.0082 |
| `cbp_replace` | 0.0831 +/- 0.0009 | 0.4780 +/- 0.0126 | 0.1017 +/- 0.0025 | 0.3020 +/- 0.0085 |
| `resource_manager` | 0.0727 +/- 0.0010 | 0.5053 +/- 0.0182 | 0.0888 +/- 0.0009 | 0.3383 +/- 0.0080 |
| `resource_manager_retention` | 0.0761 +/- 0.0013 | 0.5053 +/- 0.0189 | 0.0924 +/- 0.0014 | 0.3369 +/- 0.0108 |

Paired resource-manager vs `mlp_static`:

- `final_window_mse`: diff +0.0102 +/- 0.0015; wins/losses/ties 5/0/0.
- `test_accuracy`: diff +0.0246 +/- 0.0073; wins/losses/ties 4/1/0.
- `resource_manager_retention` held-out `test_accuracy`: diff +0.0232 +/- 0.0058; wins/losses/ties 5/0/0.

## Assessment Rule

Positive paired differences favor the learned manager.  For MSE, the difference is baseline minus manager; for accuracy, it is manager minus baseline.
