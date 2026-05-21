# Step 2 Published-Style External Stressors

Protocol: 1 seeds, 120 online steps, final window 40, benchmarks=permuted_mnist_like, slowly_changing_regression.

The learner side is the existing Step 2 strict prediction-space portfolio over fair MLP widths, UPGD, and dynamic sparse rewiring. Positive paired differences favor the portfolio; for MSE the value is best MLP minus portfolio, and for accuracy it is portfolio minus best MLP.

Reference protocols: Dohare et al. 2024 Nature Online Permuted MNIST and Slowly-Changing Regression. This runner is compact unless real OpenML MNIST is explicitly enabled.

## Configuration

```json
{
  "benchmarks": [
    "permuted_mnist_like",
    "slowly_changing_regression"
  ],
  "steps": 120,
  "n_seeds": 1,
  "seed": 0,
  "final_window": 40,
  "mnist_source": "auto",
  "allow_openml_download": false,
  "mnist_split": "stratified",
  "openml_data_home": null,
  "openml_n_retries": 2,
  "openml_retry_delay": 1.0,
  "train_fraction": 0.7,
  "max_train_examples": 200,
  "max_test_examples": 80,
  "n_permutations": 2,
  "task_block_size": 40,
  "sample_with_replacement": false,
  "task_sampling": "random",
  "mnist_published_scale": false,
  "scr_preset": "compact",
  "long_scr": false,
  "regression_bits": 8,
  "regression_slow_bits": 3,
  "regression_flip_interval": 20,
  "regression_target_hidden": 16,
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
  "dynamic_rewire_interval": 60,
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
  "result_prefix": "published_stressors_smoke"
}
```

## permuted_mnist_like

Source/protocol: `local_sklearn_digits_28x28`.

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0780 +/- 0.0000 | 0.0913 +/- 0.0000 | 0.4250 +/- 0.0000 | 0.3625 +/- 0.0000 |
| `mlp_h64` | 0.0897 +/- 0.0000 | 0.1102 +/- 0.0000 | 0.4000 +/- 0.0000 | 0.2125 +/- 0.0000 |
| `mlp_h128` | 0.0974 +/- 0.0000 | 0.1191 +/- 0.0000 | 0.3750 +/- 0.0000 | 0.3187 +/- 0.0000 |
| `mlp_h64_64` | 0.1111 +/- 0.0000 | 0.1300 +/- 0.0000 | 0.1750 +/- 0.0000 | 0.1937 +/- 0.0000 |
| `upgd_low_noise` | 0.0917 +/- 0.0000 | 0.1029 +/- 0.0000 | 0.2750 +/- 0.0000 | 0.3000 +/- 0.0000 |
| `dynamic_sparse` | 0.1026 +/- 0.0000 | 0.1223 +/- 0.0000 | 0.2250 +/- 0.0000 | 0.2875 +/- 0.0000 |

`final_window_mse` portfolio-vs-best-MLP diff: +0.0117 +/- 0.0000; wins/losses/ties 1/0/0.
`test_accuracy` portfolio-vs-best-MLP diff: +0.0438 +/- 0.0000; wins/losses/ties 1/0/0.

Limits:

- This is not MNIST. It preserves the 10-class handwritten-digit classification form and, when expanded, the 784-dimensional pixel permutation stressor, but it has only 1,797 source examples and simpler images than MNIST.
- Dohare et al. used true MNIST, 60,000 examples per task, many tasks, a single pass, and no task-switch indication. This compact runner uses shorter task blocks and optional subsampling so it can run locally in minutes.

## slowly_changing_regression

Source/protocol: `lightweight_slowly_changing_binary_regression`.

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0657 +/- 0.0000 | 0.0680 +/- 0.0000 |  |  |
| `mlp_h64` | 0.0744 +/- 0.0000 | 0.0874 +/- 0.0000 |  |  |
| `mlp_h128` | 0.0718 +/- 0.0000 | 0.0662 +/- 0.0000 |  |  |
| `mlp_h64_64` | 0.0761 +/- 0.0000 | 0.0753 +/- 0.0000 |  |  |
| `upgd_low_noise` | 0.0547 +/- 0.0000 | 0.0741 +/- 0.0000 |  |  |
| `dynamic_sparse` | 0.0781 +/- 0.0000 | 0.0861 +/- 0.0000 |  |  |

`final_window_mse` portfolio-vs-best-MLP diff: +0.0061 +/- 0.0000; wins/losses/ties 1/0/0.

Limits:

- Matches the main structure described by Dohare et al.: binary inputs with slow-changing bits, iid random bits, a constant bias, and a fixed LTU target network. This local run is far shorter than the paper/reproduction repository's million-plus-example runs and uses the Step 2 portfolio architecture rather than the paper's learner architecture.

## Assessment

Primary comparator status: `all_primary_nonnegative_vs_best_mlp=True`.
Published-scale status: `published_scale_external_claim_supported=False` (true OpenML MNIST=False, full MNIST split=False, full MNIST task blocks=False, Dohare SCR config=False, SCR steps=120).

This should be reported as published-scale external evidence only when `published_scale_external_claim_supported=true`. With the default sklearn-digits fallback or shorter SCR settings, the result narrows the gap but is not a full published-scale reproduction.

## References

- [Dohare et al. 2024 Nature: Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7): Online Permuted MNIST and Slowly-Changing Regression protocols.
- [shibhansh/loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity): Public reproduction repository for the loss-of-plasticity paper.
