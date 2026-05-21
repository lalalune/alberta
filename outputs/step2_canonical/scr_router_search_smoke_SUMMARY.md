# Step 2 SCR Router Search

Protocol: 1 seeds, 120 online steps, final window 40, SCR preset `compact`.

All variants reuse the same SCR stream family, the same fair MLP comparator grid, and the same non-MLP experts. Positive paired differences mean best fair MLP final-window MSE minus router final-window MSE.

## Configuration

```json
{
  "variant_names": [
    "convex_reference",
    "stable_mlp_selector",
    "guarded_best_mlp",
    "slow_meta"
  ],
  "steps": 120,
  "n_seeds": 1,
  "seed": 0,
  "final_window": 40,
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
  "base_hedge_eta": 1.0,
  "base_hedge_discount": 0.995,
  "base_router_policy": "convex",
  "base_router_decay": 0.02,
  "base_guard_tolerance": 0.0,
  "online_retention_mse_guard": true,
  "online_retention_min_lifetime_class_fraction": 0.7,
  "online_retention_max_recent_class_fraction": 0.5,
  "output_dir": "outputs/step2_canonical",
  "result_prefix": "scr_router_search_smoke"
}
```

## Router Results

| Variant | Router Final MSE | Best Fair MLP Final MSE | Diff vs Best MLP | Closed? |
|---|---:|---:|---:|---:|
| `convex_reference` | 0.065691 +/- 0.000000 | 0.071825 +/- 0.000000 | +0.006135 +/- 0.000000; 1/0/0 | `True` |
| `stable_mlp_selector` | 0.071825 +/- 0.000000 | 0.071825 +/- 0.000000 | +0.000000 +/- 0.000000; 0/0/1 | `True` |
| `guarded_best_mlp` | 0.069833 +/- 0.000000 | 0.071825 +/- 0.000000 | +0.001992 +/- 0.000000; 1/0/0 | `True` |
| `slow_meta` | 0.071825 +/- 0.000000 | 0.071825 +/- 0.000000 | +0.000000 +/- 0.000000; 0/0/1 | `True` |

## Best Variant

Best router: `convex_reference`.

Current all-expert convex Hedge reference from the published stressor runner.

Feasible SCR comparator closed vs best fair MLP: `True`.
Published-scale SCR reproduction closed: `False`.

The published-scale flag remains false unless the run uses the Dohare public SCR configuration for at least 1,000,000 online examples. A shorter `--long-scr` run should be reported as a feasible local SCR closure only.

## Variant Details

### convex_reference

Current all-expert convex Hedge reference from the published stressor runner.

Overrides:

```json
{
  "router_policy": "convex",
  "hedge_eta": 1.0,
  "hedge_discount": 0.995,
  "router_decay": 0.02,
  "guard_tolerance": 0.0
}
```

### stable_mlp_selector

Causal EMA selector restricted to the fair MLP widths; this is a stable MLP fallback route for SCR.

Overrides:

```json
{
  "router_policy": "mlp_selector",
  "hedge_eta": 1.0,
  "hedge_discount": 0.995,
  "router_decay": 0.02,
  "guard_tolerance": 0.0
}
```

### guarded_best_mlp

Use the all-expert convex route only when its causal EMA is within a small tolerance of the better fair-MLP route.

Overrides:

```json
{
  "router_policy": "guarded_best_mlp",
  "hedge_eta": 1.0,
  "hedge_discount": 0.995,
  "router_decay": 0.02,
  "guard_tolerance": 0.0001
}
```

### slow_meta

Lower-variance meta route with slower router EMA and slower Hedge forgetting.

Overrides:

```json
{
  "router_policy": "meta",
  "hedge_eta": 1.0,
  "hedge_discount": 0.999,
  "router_decay": 0.005,
  "guard_tolerance": 0.0
}
```
