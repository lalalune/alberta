# Step 2 Learned Resource Manager On Stateful External Benchmarks

Default streams use the bundled sklearn digits dataset.  The optional external image stream uses OpenML Fashion-MNIST only when explicitly enabled, otherwise a local 28x28 sklearn-digits fallback.  The manager is causal: each prediction uses allocation weights learned before seeing the current label.

```json
{
  "steps": 300,
  "n_seeds": 3,
  "seed": 0,
  "train_fraction": 0.7,
  "final_window": 100,
  "hidden_size": 32,
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
  "block_size": 60,
  "context_delay_blocks": 1,
  "mask_keep_fraction": 0.5,
  "noise_std": 0.05,
  "external_image_source": "openml_fashion_mnist",
  "allow_openml_download": true,
  "external_sample_limit": 1000,
  "benchmarks": [
    "external_delayed_contextual_permutation"
  ]
}
```

## external_delayed_contextual_permutation

Fashion-MNIST-style 28x28 image stream when OpenML is enabled, otherwise expanded sklearn digits fallback; recurring pixel permutations use true hidden states while manager context ids are delayed by whole blocks.

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mlp_static` | 0.1005 +/- 0.0035 | 0.3200 +/- 0.0058 | 0.1010 +/- 0.0020 | 0.2704 +/- 0.0200 |
| `upgd_low` | 0.1256 +/- 0.0078 | 0.1800 +/- 0.0115 | 0.1136 +/- 0.0045 | 0.2513 +/- 0.0155 |
| `upgd_high` | 0.1296 +/- 0.0045 | 0.1800 +/- 0.0321 | 0.1195 +/- 0.0011 | 0.2269 +/- 0.0119 |
| `cbp_replace` | 0.0979 +/- 0.0019 | 0.2967 +/- 0.0318 | 0.0969 +/- 0.0011 | 0.2798 +/- 0.0121 |
| `resource_manager` | 0.0852 +/- 0.0027 | 0.3533 +/- 0.0291 | 0.0866 +/- 0.0012 | 0.3113 +/- 0.0140 |
| `resource_manager_retention` | 0.0887 +/- 0.0027 | 0.3333 +/- 0.0176 | 0.0881 +/- 0.0005 | 0.3044 +/- 0.0090 |

Paired resource-manager vs `mlp_static`:

- `final_window_mse`: diff +0.0153 +/- 0.0010; wins/losses/ties 3/0/0.
- `test_accuracy`: diff +0.0409 +/- 0.0153; wins/losses/ties 3/0/0.
- `resource_manager_retention` held-out `test_accuracy`: diff +0.0340 +/- 0.0184; wins/losses/ties 3/0/0.

## Assessment Rule

Positive paired differences favor the learned manager.  For MSE, the difference is baseline minus manager; for accuracy, it is manager minus baseline.
