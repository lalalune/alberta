# Step 2 Published-Style External Stressors

Protocol: 5 seeds, 1500 online steps, final window 300, benchmarks=permuted_mnist_like, slowly_changing_regression.

The learner side is the existing Step 2 strict prediction-space portfolio over fair MLP widths, UPGD, and dynamic sparse rewiring. Positive paired differences favor the portfolio; for MSE the value is best MLP minus portfolio, and for accuracy it is portfolio minus best MLP.

Reference protocols: Dohare et al. 2024 Nature Online Permuted MNIST and Slowly-Changing Regression. This runner is compact unless real OpenML MNIST is explicitly enabled.

## Configuration

```json
{
  "benchmarks": [
    "permuted_mnist_like",
    "slowly_changing_regression"
  ],
  "steps": 1500,
  "n_seeds": 5,
  "seed": 0,
  "final_window": 300,
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
  "output_dir": "outputs/step2_canonical",
  "result_prefix": "published_stressors"
}
```

## permuted_mnist_like

Source/protocol: `local_sklearn_digits_28x28`.

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0621 +/- 0.0015 | 0.0628 +/- 0.0005 | 0.6167 +/- 0.0064 | 0.6089 +/- 0.0122 |
| `mlp_h64` | 0.0692 +/- 0.0021 | 0.0718 +/- 0.0008 | 0.5840 +/- 0.0198 | 0.4364 +/- 0.0087 |
| `mlp_h128` | 0.0724 +/- 0.0021 | 0.0724 +/- 0.0008 | 0.6040 +/- 0.0165 | 0.5573 +/- 0.0129 |
| `mlp_h64_64` | 0.0759 +/- 0.0011 | 0.0773 +/- 0.0008 | 0.5327 +/- 0.0145 | 0.3645 +/- 0.0189 |
| `upgd_low_noise` | 0.0782 +/- 0.0016 | 0.0802 +/- 0.0006 | 0.4600 +/- 0.0263 | 0.5852 +/- 0.0105 |
| `dynamic_sparse` | 0.0799 +/- 0.0022 | 0.0842 +/- 0.0011 | 0.4473 +/- 0.0297 | 0.5722 +/- 0.0126 |

`final_window_mse` portfolio-vs-best-MLP diff: +0.0068 +/- 0.0004; wins/losses/ties 5/0/0.
`test_accuracy` portfolio-vs-best-MLP diff: +0.0517 +/- 0.0077; wins/losses/ties 5/0/0.

Limits:

- This is not MNIST. It preserves the 10-class handwritten-digit classification form and, when expanded, the 784-dimensional pixel permutation stressor, but it has only 1,797 source examples and simpler images than MNIST.
- Dohare et al. used true MNIST, 60,000 examples per task, many tasks, a single pass, and no task-switch indication. This compact runner uses shorter task blocks and optional subsampling so it can run locally in minutes.

## slowly_changing_regression

Source/protocol: `lightweight_slowly_changing_binary_regression`.

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0334 +/- 0.0028 | 0.0360 +/- 0.0017 |  |  |
| `mlp_h64` | 0.0343 +/- 0.0029 | 0.0374 +/- 0.0016 |  |  |
| `mlp_h128` | 0.0361 +/- 0.0034 | 0.0393 +/- 0.0021 |  |  |
| `mlp_h64_64` | 0.0333 +/- 0.0027 | 0.0385 +/- 0.0023 |  |  |
| `upgd_low_noise` | 0.0398 +/- 0.0035 | 0.0437 +/- 0.0021 |  |  |
| `dynamic_sparse` | 0.0426 +/- 0.0031 | 0.0472 +/- 0.0020 |  |  |

`final_window_mse` portfolio-vs-best-MLP diff: -0.0003 +/- 0.0004; wins/losses/ties 3/2/0.

Limits:

- Matches the main structure described by Dohare et al.: binary inputs with slow-changing bits, iid random bits, a constant bias, and a fixed LTU target network. This local run is far shorter than the paper/reproduction repository's million-plus-example runs and uses the Step 2 portfolio architecture rather than the paper's learner architecture.

## Assessment

Primary comparator status: `all_primary_nonnegative_vs_best_mlp=False`.
Published-scale status: `published_scale_external_claim_supported=False` (true OpenML MNIST=False, full MNIST split=False, full MNIST task blocks=False, Dohare SCR config=False, SCR steps=1500).

This should be reported as published-scale external evidence only when `published_scale_external_claim_supported=true`. With the default sklearn-digits fallback or shorter SCR settings, the result narrows the gap but is not a full published-scale reproduction.

## References

- [Dohare et al. 2024 Nature: Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7): Online Permuted MNIST and Slowly-Changing Regression protocols.
- [shibhansh/loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity): Public reproduction repository for the loss-of-plasticity paper.
