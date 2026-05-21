# Step 2 Published-Style External Stressors

Protocol: 3 seeds, 600 online steps, final window 150, benchmarks=permuted_mnist_like.

The learner side is the existing Step 2 strict prediction-space portfolio over fair MLP widths, UPGD, and dynamic sparse rewiring. Positive paired differences favor the portfolio; for MSE the value is best MLP minus portfolio, and for accuracy it is portfolio minus best MLP.

Reference protocols: Dohare et al. 2024 Nature Online Permuted MNIST and Slowly-Changing Regression. This runner is compact unless real OpenML MNIST is explicitly enabled.

## Configuration

```json
{
  "benchmarks": [
    "permuted_mnist_like"
  ],
  "steps": 600,
  "n_seeds": 3,
  "seed": 0,
  "final_window": 150,
  "mnist_source": "openml",
  "allow_openml_download": true,
  "allow_torchvision_download": false,
  "mnist_split": "stratified",
  "openml_data_home": "outputs/step2_canonical/openml_external_cache",
  "torchvision_data_home": null,
  "openml_n_retries": 2,
  "openml_retry_delay": 1.0,
  "train_fraction": 0.7,
  "max_train_examples": 1000,
  "max_test_examples": 300,
  "n_permutations": 3,
  "task_block_size": 200,
  "sample_with_replacement": false,
  "task_sampling": "random",
  "include_identity_permutation": false,
  "max_test_permutation_views": null,
  "mnist_published_scale": false,
  "scr_preset": "compact",
  "long_scr": false,
  "regression_bits": 20,
  "regression_slow_bits": 5,
  "regression_flip_interval": 50,
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
  "dynamic_rewire_interval": 240,
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
  "output_dir": "outputs/step2_canonical/openml_external_published_3seed_small",
  "result_prefix": "published_stressors_openml_3seed_small"
}
```

## permuted_mnist_like

Source/protocol: `openml_mnist_784`.

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0717 +/- 0.0003 | 0.0773 +/- 0.0003 | 0.4889 +/- 0.0044 | 0.5507 +/- 0.0091 |
| `mlp_h64` | 0.0883 +/- 0.0014 | 0.0972 +/- 0.0008 | 0.4178 +/- 0.0089 | 0.3759 +/- 0.0203 |
| `mlp_h128` | 0.0869 +/- 0.0016 | 0.0965 +/- 0.0007 | 0.4400 +/- 0.0200 | 0.4637 +/- 0.0124 |
| `mlp_h64_64` | 0.0933 +/- 0.0013 | 0.1018 +/- 0.0008 | 0.3689 +/- 0.0097 | 0.2993 +/- 0.0324 |
| `upgd_low_noise` | 0.0865 +/- 0.0009 | 0.0951 +/- 0.0012 | 0.3622 +/- 0.0182 | 0.4478 +/- 0.0241 |
| `dynamic_sparse` | 0.0931 +/- 0.0036 | 0.1103 +/- 0.0023 | 0.3867 +/- 0.0115 | 0.4396 +/- 0.0161 |

`final_window_mse` portfolio-vs-best-MLP diff: +0.0151 +/- 0.0012; wins/losses/ties 3/0/0.
`test_accuracy` portfolio-vs-best-MLP diff: +0.0870 +/- 0.0081; wins/losses/ties 3/0/0.

Limits:

- True OpenML MNIST. This is a full source split only when is_full_mnist_split=true; the online protocol still depends on task_block_size, task_sampling, and n_permutations.
- Dohare et al. used true MNIST, randomized pixel permutations, 60,000 examples per task, 800 tasks in the main OPMNIST protocol, a single pass through each task in random order, no mini-batches, and no task-switch indication. This runner is only a published-scale OPMNIST run when matches_dohare_opmnist_core_protocol and matches_dohare_opmnist_published_task_count are both true.

## Assessment

Primary comparator status: `all_primary_nonnegative_vs_best_mlp=True`.
Published-scale status: `published_scale_external_claim_supported=False` (true MNIST=True, OpenML MNIST=True, torchvision MNIST=False, full MNIST split=False, full MNIST task blocks=False, single-pass task order=False, random permutations=True, no task id=True, OPMNIST tasks=3, OPMNIST steps=600).

This should be reported as published-scale external evidence only when `published_scale_external_claim_supported=true`. With the default sklearn-digits fallback or shorter SCR settings, the result narrows the gap but is not a full published-scale reproduction.

## References

- [Dohare et al. 2024 Nature: Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7): Online Permuted MNIST and Slowly-Changing Regression protocols.
- [shibhansh/loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity): Public reproduction repository for the loss-of-plasticity paper.
