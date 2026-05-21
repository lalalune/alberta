# Step 2 Published-Style External Stressors

Protocol: 1 seeds, 1860000 online steps, final window 60000, benchmarks=permuted_mnist_like.

The learner side is the existing Step 2 strict prediction-space portfolio over fair MLP widths, UPGD, and dynamic sparse rewiring. Positive paired differences favor the portfolio; for MSE the value is best MLP minus portfolio, and for accuracy it is portfolio minus best MLP.

Reference protocols: Dohare et al. 2024 Nature Online Permuted MNIST and Slowly-Changing Regression. This runner is compact unless real OpenML MNIST is explicitly enabled.

## Configuration

```json
{
  "benchmarks": [
    "permuted_mnist_like"
  ],
  "steps": 1860000,
  "n_seeds": 1,
  "seed": 0,
  "final_window": 60000,
  "mnist_source": "openml",
  "allow_openml_download": true,
  "allow_torchvision_download": false,
  "mnist_split": "canonical",
  "openml_data_home": null,
  "torchvision_data_home": null,
  "openml_n_retries": 2,
  "openml_retry_delay": 1.0,
  "train_fraction": 0.7,
  "max_train_examples": null,
  "max_test_examples": null,
  "n_permutations": 800,
  "task_block_size": 60000,
  "sample_with_replacement": false,
  "task_sampling": "sequential_epoch",
  "include_identity_permutation": false,
  "max_test_permutation_views": 31,
  "evaluate_all_permutation_views": false,
  "mnist_published_scale": true,
  "opmnist_streaming": true,
  "opmnist_chunk_size": 20000,
  "opmnist_resume": true,
  "opmnist_resume_path": "outputs/step2_worker_m_opmnist_30block/step2_worker_m_opmnist_30block_seed0_opmnist_resume.pkl",
  "opmnist_force_restart": false,
  "opmnist_status_target_steps": 48000000,
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
  "output_dir": "outputs/step2_opmnist_scale_31block",
  "result_prefix": "opmnist_true_mnist_31block_mse"
}
```

## permuted_mnist_like

Source/protocol: `openml_mnist_784`.

| Method | Final MSE | Mean MSE | Final Acc | Test Acc |
|---|---:|---:|---:|---:|
| `mixture` | 0.0248 +/- 0.0000 | 0.0222 +/- 0.0000 | 0.8904 +/- 0.0000 | 0.2600 +/- 0.0000 |
| `mlp_h64` | 0.0305 +/- 0.0000 | 0.0284 +/- 0.0000 | 0.8697 +/- 0.0000 | 0.2125 +/- 0.0000 |
| `mlp_h128` | 0.0278 +/- 0.0000 | 0.0265 +/- 0.0000 | 0.8819 +/- 0.0000 | 0.2079 +/- 0.0000 |
| `mlp_h64_64` | 0.0345 +/- 0.0000 | 0.0290 +/- 0.0000 | 0.8423 +/- 0.0000 | 0.1817 +/- 0.0000 |
| `upgd_low_noise` | 0.0400 +/- 0.0000 | 0.0305 +/- 0.0000 | 0.8006 +/- 0.0000 | 0.2737 +/- 0.0000 |
| `dynamic_sparse` | 0.0301 +/- 0.0000 | 0.0251 +/- 0.0000 | 0.8539 +/- 0.0000 | 0.3203 +/- 0.0000 |

`final_window_mse` portfolio-vs-best-MLP diff: +0.0030 +/- 0.0000; wins/losses/ties 1/0/0.
`test_accuracy` portfolio-vs-best-MLP diff: +0.0475 +/- 0.0000; wins/losses/ties 1/0/0.

Limits:

- True OpenML MNIST. This is a full source split only when is_full_mnist_split=true; the online protocol still depends on task_block_size, task_sampling, and n_permutations.
- Dohare et al. used true MNIST, randomized pixel permutations, 60,000 examples per task, 800 tasks in the main OPMNIST protocol, a single pass through each task in random order, no mini-batches, and no task-switch indication. This runner is only a published-scale OPMNIST run when matches_dohare_opmnist_core_protocol and matches_dohare_opmnist_published_task_count are both true.

## Assessment

Primary comparator status: `all_primary_nonnegative_vs_best_mlp=True`.
Published-scale status: `published_scale_external_claim_supported=False` (true MNIST=True, OpenML MNIST=True, torchvision MNIST=False, full MNIST split=True, full MNIST task blocks=True, single-pass task order=True, random permutations=True, no task id=True, prediction-before-update=True, all experts update=True, OPMNIST tasks=800, full blocks=31, 60k blocks=31, OPMNIST steps=1860000, SCR public config=False, SCR steps=0/1000000, SCR protocol=False).
Published-scale SCR status: `published_scale_scr_claim_supported=False`.

This should be reported as published-scale external evidence only when `published_scale_external_claim_supported=true`. With the default sklearn-digits fallback or shorter SCR settings, the result narrows the gap but is not a full published-scale reproduction.

## References

- [Dohare et al. 2024 Nature: Loss of plasticity in deep continual learning](https://www.nature.com/articles/s41586-024-07711-7): Online Permuted MNIST and Slowly-Changing Regression protocols.
- [shibhansh/loss-of-plasticity](https://github.com/shibhansh/loss-of-plasticity): Public reproduction repository for the loss-of-plasticity paper.
