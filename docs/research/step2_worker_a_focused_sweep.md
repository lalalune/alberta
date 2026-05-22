# Step 2 Worker A Focused Sweep

Worker A scope: no source edits. Outputs use `outputs/step2_worker_a_*`; notes use
`docs/research/step2_worker_a_*`.

## Baseline

Comparison anchor was the existing current-best all-suite run:

`outputs/step2_conclusive_pruned_hedge_safe_poly_selector100_all_10seed/results.json`

Baseline command:

```bash
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 1.0 \
  --hedge-discount 0.995 \
  --selector-window 100 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse
```

Focused baseline on the three weak datasets: `26/4/0` total seed W/L/T vs best
same-run MLP.

## Focused Sweep

Each variant used:

```bash
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks controlled_rare,synthetic_compositional,synthetic_polynomial \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 1.0 \
  --hedge-discount 0.995 \
  --selector-window 100 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse \
  <variant flag> \
  --output-dir outputs/step2_worker_a_focus_<variant> \
  --note-path docs/research/step2_worker_a_focus_<variant>.md
```

Positive delta means `best MLP final_window_mse - conclusive final_window_mse`.

| Variant | Extra flag | controlled_rare | synthetic_compositional | synthetic_polynomial | Total W/L/T | Mean delta |
|---|---|---:|---:|---:|---:|---:|
| current_best_baseline | none | +0.014419 8/2/0 | +0.043445 9/1/0 | +0.061279 9/1/0 | 26/4/0 | +0.039714 |
| guard_margin_5e4 | `--guard-margin 5e-4` | +0.014459 8/2/0 | +0.043190 9/1/0 | +0.061261 9/1/0 | 26/4/0 | +0.039637 |
| hedge_discount_0p99 | `--hedge-discount 0.99` | +0.013881 8/2/0 | +0.046465 10/0/0 | +0.061252 9/1/0 | 27/3/0 | +0.040533 |
| hedge_discount_0p999 | `--hedge-discount 0.999` | +0.016041 9/1/0 | +0.041728 9/1/0 | +0.059704 7/3/0 | 25/5/0 | +0.039158 |
| hedge_eta_0p5 | `--hedge-eta 0.5` | +0.014423 8/2/0 | +0.046028 10/0/0 | +0.061333 9/1/0 | 27/3/0 | +0.040595 |
| hedge_eta_2p0 | `--hedge-eta 2.0` | +0.014517 8/2/0 | +0.041746 9/1/0 | +0.060819 8/2/0 | 25/5/0 | +0.039027 |
| route_switch_margin_5e4 | `--route-switch-margin 5e-4` | +0.014388 8/2/0 | +0.043480 9/1/0 | +0.062244 8/2/0 | 25/5/0 | +0.040037 |
| safe_gate_step_size_0p02 | `--safe-gate-step-size 0.02` | +0.013703 8/2/0 | +0.043283 9/1/0 | +0.057677 6/4/0 | 23/7/0 | +0.038221 |
| safe_gate_step_size_0p1 | `--safe-gate-step-size 0.1` | +0.014635 8/2/0 | +0.042647 9/1/0 | +0.058669 8/2/0 | 25/5/0 | +0.038650 |
| selector_window_60 | `--selector-window 60` | +0.013631 8/2/0 | +0.036985 9/1/0 | +0.058329 7/3/0 | 24/6/0 | +0.036315 |
| selector_window_80 | `--selector-window 80` | +0.014362 8/2/0 | +0.043313 9/1/0 | +0.056239 6/4/0 | 23/7/0 | +0.037971 |

Best focused variant: `hedge_eta_0p5`. It improved total focused seed W/L/T
from `26/4/0` to `27/3/0`, fixing the synthetic-compositional seed loss while
preserving controlled-rare and synthetic-polynomial behavior.

`hedge_discount_0p99` also reached `27/3/0`, but `hedge_eta_0p5` had the
slightly higher three-dataset mean delta.

## Full Validation

Promoted command:

```bash
PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 0.5 \
  --hedge-discount 0.995 \
  --selector-window 100 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse \
  --output-dir outputs/step2_worker_a_full_hedge_eta_0p5_all_10seed_rerun \
  --note-path docs/research/step2_worker_a_full_hedge_eta_0p5_all_10seed_rerun.md
```

All-suite validation result: positive mean final-window MSE delta on `13/14`
datasets and exact MSE tie on `digits_class_blocked`; seed W/L/T improved from
baseline `126/4/10` to `127/3/10`.

| Dataset | Baseline delta W/L/T | `hedge_eta_0p5` delta W/L/T |
|---|---:|---:|
| controlled_nonlinear | +0.020404 10/0/0 | +0.020609 10/0/0 |
| controlled_interaction | +0.147745 10/0/0 | +0.147744 10/0/0 |
| controlled_triple | +0.579143 10/0/0 | +0.579204 10/0/0 |
| controlled_rare | +0.014419 8/2/0 | +0.014423 8/2/0 |
| controlled_polynomial | +0.532656 10/0/0 | +0.532656 10/0/0 |
| controlled_frequency | +0.155974 10/0/0 | +0.156008 10/0/0 |
| synthetic_polynomial | +0.061279 9/1/0 | +0.061333 9/1/0 |
| synthetic_frequency | +0.804476 10/0/0 | +0.804476 10/0/0 |
| synthetic_compositional | +0.043445 9/1/0 | +0.046028 10/0/0 |
| digits_iid | +0.006427 10/0/0 | +0.006486 10/0/0 |
| digits_class_blocked | +0.000000 0/0/10 | +0.000000 0/0/10 |
| digits_permuted_pixels | +0.007925 10/0/0 | +0.008483 10/0/0 |
| digits_mask_noise | +0.006157 10/0/0 | +0.006320 10/0/0 |
| digits_label_drift | +0.006958 10/0/0 | +0.006999 10/0/0 |

Digits held-out accuracy remained positive vs best MLP on every digits regime:

| Dataset | Test accuracy delta W/L/T |
|---|---:|
| digits_iid | +0.011132 10/0/0 |
| digits_class_blocked | +0.349165 10/0/0 |
| digits_permuted_pixels | +0.013173 9/0/1 |
| digits_mask_noise | +0.010946 9/1/0 |
| digits_label_drift | +0.006308 8/2/0 |

## Generated Files

Completed focused outputs:

- `outputs/step2_worker_a_focus_selector_window_60`
- `outputs/step2_worker_a_focus_selector_window_80`
- `outputs/step2_worker_a_focus_hedge_eta_0p5`
- `outputs/step2_worker_a_focus_hedge_eta_2p0`
- `outputs/step2_worker_a_focus_hedge_discount_0p99`
- `outputs/step2_worker_a_focus_hedge_discount_0p999`
- `outputs/step2_worker_a_focus_route_switch_margin_5e4`
- `outputs/step2_worker_a_focus_guard_margin_5e4`
- `outputs/step2_worker_a_focus_safe_gate_step_size_0p02`
- `outputs/step2_worker_a_focus_safe_gate_step_size_0p1`

Completed full validation:

- `outputs/step2_worker_a_full_hedge_eta_0p5_all_10seed_rerun`
- `docs/research/step2_worker_a_full_hedge_eta_0p5_all_10seed_rerun.md`

Diagnostic and incomplete artifacts, excluded from metrics:

- `outputs/step2_worker_a_focus_selector_window_120`: interrupted after one curve file.
- `outputs/step2_worker_a_full_hedge_eta_0p5_all_10seed`: first all-suite attempt exited without `results.json`.
- `outputs/step2_worker_a_probe_hedge_eta_0p5_digits_iid`: 2-seed digits probe used to confirm digits did not fail deterministically.

Conclusion: promote `--hedge-eta 0.5` over the current `--hedge-eta 1.0`.
This is a small but clean route-robustness improvement: it removes the
synthetic-compositional seed-level loss and does not introduce any all-suite
aggregate MSE regressions.
