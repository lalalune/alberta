# Step 2 Published-Style External Stressors

Protocol: 3 seeds, 20000 online steps, final window 5000, benchmarks=slowly_changing_regression.

The learner side is the existing Step 2 strict prediction-space portfolio over fair MLP widths, UPGD, and dynamic sparse rewiring. Positive paired differences favor the portfolio; for MSE the value is best MLP minus portfolio, and for accuracy it is portfolio minus best MLP.

Reference protocols: Dohare et al. 2024 Nature Online Permuted MNIST and Slowly-Changing Regression. This runner is compact unless real OpenML MNIST is explicitly enabled.

## Configuration

```json
{
  "benchmarks": [
    "slowly_changing_regression"
  ],
  "steps": 20000,
  "n_seeds": 3,
  "seed": 0,
  "final_window": 5000,
  "mnist_source": "auto",
  "allow_openml_download": false,
  "mnist_split": "stratified",
  "openml_data_home": null,
  "openml_n_retries": 2,
  "openml_retry_delay": 1.0,
  "train_fraction": 0.7,
  "max_train_examples": 4000,
  "max_test_examples": 1000,
  "n_permutations": 5,
  "task_block_size": 300,
  "sample_with_replacement": false,
  "task_sampling": "random",
  "mnist_published_scale": false,
  "scr_preset": "dohare_small",
  "long_scr": true,
  "regression_bits": 20,
  "regression_slow_bits": 15,
  "regression_flip_interval": 1000,
  "regression_target_hidden": 100,
  "regression_beta": 0.7,
  "regression_noise_std": 0.01,
  "expert_names": [
    "mlp_h64",
    "mlp_h128",
    "mlp_h64_64",
    "upgd_low_noise",
    "dynamic_sparse"
  ],
  "mlp_comparator_methods": [
    "mlp_h64",
    "mlp_h128",
    "mlp_h64_64"
  ],
  "step_size": 0.03,
  "sparsity": 0.5,
  "perturbation_sigma": 0.0001,
  "perturbation_warmup_steps": 0,
  "perturbation_ramp_steps": 0,
  "dynamic_hidden_size": 64,
  "dynamic_utility_decay": 0.99,
  "dynamic_rewire_interval": 500,
  "dynamic_unit_replacement_rate": 0.05,
  "hedge_eta": 1.0,
  "hedge_discount": 0.995,
  "router_policy": "convex",
  "router_decay": 0.02,
  "guard_tolerance": 0.0,
  "digits_deployment_objective": "mse",
  "online_retention_mse_guard": true,
  "online_retention_min_lifetime_class_fraction": 0.7,
  "online_retention_max_recent_class_fraction": 0.5,
  "retention_router": "none",
  "retention_upgd_deployment_weight": 1.0,
  "retention_min_lifetime_class_fraction": 0.8,
  "retention_max_recent_class_fraction": 0.4,
  "output_dir": "outputs/step2_canonical",
  "result_prefix": "published_stressors_long_scr"
}
```

## slowly_changing_regression

Source/protocol: `lightweight_slowly_changing_binary_regression`.

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0054 +/- 0.0004 | 0.0088 +/- 0.0009 |  |  |
| `mlp_h64` | 0.0065 +/- 0.0005 | 0.0101 +/- 0.0010 |  |  |
| `mlp_h128` | 0.0075 +/- 0.0007 | 0.0107 +/- 0.0010 |  |  |
| `mlp_h64_64` | 0.0043 +/- 0.0003 | 0.0077 +/- 0.0010 |  |  |
| `upgd_low_noise` | 0.0065 +/- 0.0003 | 0.0102 +/- 0.0013 |  |  |
| `dynamic_sparse` | 0.0081 +/- 0.0006 | 0.0127 +/- 0.0012 |  |  |

`final_window_mse` portfolio-vs-best-MLP diff: -0.0011 +/- 0.0003; wins/losses/ties 0/3/0.

Limits:

- Matches the main structure described by Dohare et al.: binary inputs with slow-changing bits, iid random bits, a constant bias, and a fixed LTU target network. This local run is far shorter than the paper/reproduction repository's million-plus-example runs and uses the Step 2 portfolio architecture rather than the paper's learner architecture.

## Assessment

Primary comparator status: `all_primary_nonnegative_vs_best_mlp=False`.
Published-scale status: `published_scale_external_claim_supported=False` (true OpenML MNIST=False, full MNIST split=False, full MNIST task blocks=False, Dohare SCR config=False, SCR steps=20000).

This should be reported as published-scale external evidence only when `published_scale_external_claim_supported=true`. With the default sklearn-digits fallback or shorter SCR settings, the result narrows the gap but is not a full published-scale reproduction.

## References

- [Dohare et al. 2024 Nature: Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7): Online Permuted MNIST and Slowly-Changing Regression protocols.
- [shibhansh/loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity): Public reproduction repository for the loss-of-plasticity paper.
