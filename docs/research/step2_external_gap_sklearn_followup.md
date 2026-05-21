# Step 2 External Gap Sklearn Follow-up

Date: 2026-05-06.

## Question

Can a bounded, simple, single UPGD learner fix the two external sklearn gaps -
`diabetes_regression` loss and `digits_class_blocked` final-window tracking -
without portfolios, replay, retained MLP heads, MLP fallback, or core UPGD
edits?

Short answer: no. The sweep found useful one-row knobs, but no variant
materially improves both weak rows while preserving the shuffled-digits
positive control.

## Run

Artifacts:

- `output/subagents/external_gap_sklearn/run_external_gap_sweep.py`
- `output/subagents/external_gap_sklearn/smoke/`
- `output/subagents/external_gap_sklearn/main_3seed_2000/`

The runner imports and reuses the existing external-suite helpers from
`examples/The Alberta Plan/Step2/step2_external_suite.py`: dataset loading,
online stream construction, prequential scan, held-out evaluation, and curve
summary. It does not edit the benchmark or core UPGD files.

Main command:

```bash
.venv/bin/python output/subagents/external_gap_sklearn/run_external_gap_sweep.py \
  --steps 2000 \
  --n-seeds 3 \
  --final-window 500 \
  --output-dir output/subagents/external_gap_sklearn/main_3seed_2000
```

Smoke and syntax checks:

```bash
.venv/bin/python -m py_compile output/subagents/external_gap_sklearn/run_external_gap_sweep.py
.venv/bin/python output/subagents/external_gap_sklearn/run_external_gap_sweep.py --smoke
```

Protocol: seeds `0..2`, 2000 online prequential steps, final window 500,
train fraction 0.7. Benchmarks were `diabetes_regression`,
`digits_class_blocked`, and positive-control `digits_shuffled`. Positive
paired differences below favor the UPGD variant: for MSE this is
`MLP64 - UPGD`; for accuracy this is `UPGD - MLP64`.

## Ranked Variants

| Rank | Variant | Mechanism | Diabetes d final MSE | Class-blocked d final MSE | Class-blocked d test acc | Shuffled d final MSE | Shuffled d test acc | Read |
|---:|---|---|---:|---:|---:|---:|---:|---|
| 1 | `upgd_headnorm_h64` | Hidden-norm readout update normalization. | +0.1289 (3/3) | -0.0131 (0/3) | +0.4886 (3/3) | +0.0028 (3/3) | +0.0043 (1/3) | Best diabetes result, but worst class-blocked tracking and much weaker positive-control margin. |
| 2 | `upgd_lr01_h64` | Lower base step size, 0.01. | +0.0754 (3/3) | -0.0036 (0/3) | +0.4985 (3/3) | +0.0188 (3/3) | +0.0254 (3/3) | Best balanced diabetes knob; preserves shuffled digits, but does not fix class-blocked tracking. |
| 3 | `upgd_h32` | Width 32, otherwise Step 2 defaults. | +0.0184 (3/3) | -0.0033 (0/3) | +0.4521 (3/3) | +0.0181 (3/3) | +0.0192 (3/3) | Small diabetes improvement with intact positive control; class-blocked tracking still red. |
| 4 | `upgd_class_mse_h64` | Linear MSE readout for classification instead of softmax CE. | -0.1011 (0/3) | +0.0004 (3/3) | -0.0223 (1/3) | +0.0013 (3/3) | -0.0006 (1/3) | Only current-block tracking fix, but it loses diabetes and collapses class-blocked retention. |
| 5 | `upgd_default_h64` | Promoted external default. | -0.0937 (0/3) | -0.0040 (0/3) | +0.4910 (3/3) | +0.0190 (3/3) | +0.0235 (3/3) | Reproduces the known shape: strong shuffled and retained class-blocked accuracy, weak diabetes and current tracking. |
| 6 | `upgd_kappa1_h64` | More conservative ObGD bound, kappa=1.0. | -0.0194 (0/3) | -0.0092 (0/3) | +0.5448 (3/3) | +0.0175 (3/3) | +0.0142 (3/3) | Best class-blocked held-out accuracy, but tracking and diabetes both worsen. |
| 7 | `upgd_sigma0_h64` | Disable utility perturbation. | -0.0913 (0/3) | -0.0042 (0/3) | +0.4774 (3/3) | +0.0189 (3/3) | +0.0229 (3/3) | Noise is not the diabetes cause; removing it does not fix tracking. |
| 8 | `upgd_margin_tiny_h64` | Tiny multiclass margin correction. | -0.0911 (0/3) | -0.0042 (0/3) | +0.4811 (3/3) | +0.0189 (3/3) | +0.0229 (3/3) | Margin is effectively neutral here. |
| 9 | `upgd_reg_hpi_h64` | Regression hidden-plus-input readout; classification default. | -0.2187 (0/3) | -0.0043 (0/3) | +0.5102 (3/3) | +0.0190 (3/3) | +0.0216 (3/3) | Direct input skip hurts diabetes. |
| 10 | `upgd_headx2_h64` | Output-head learning-rate multiplier x2. | -0.3360 (0/3) | -0.0037 (0/3) | +0.4651 (3/3) | +0.0191 (3/3) | +0.0223 (3/3) | Too much readout rate for diabetes; no tracking fix. |

## Key Metrics

Diabetes regression:

| Method | Final-window MSE | Test MSE | d final MSE vs MLP64 | d test MSE vs MLP64 |
|---|---:|---:|---:|---:|
| `mlp64` | 0.5250 +/- 0.0145 | 0.6416 +/- 0.0394 | 0.0000 | 0.0000 |
| `upgd_default_h64` | 0.6187 +/- 0.0184 | 0.6903 +/- 0.0623 | -0.0937 | -0.0488 |
| `upgd_headnorm_h64` | 0.3960 +/- 0.0106 | 0.5208 +/- 0.0107 | +0.1289 | +0.1208 |
| `upgd_lr01_h64` | 0.4495 +/- 0.0087 | 0.5620 +/- 0.0211 | +0.0754 | +0.0796 |
| `upgd_h32` | 0.5065 +/- 0.0084 | 0.5934 +/- 0.0400 | +0.0184 | +0.0482 |

Digits class-blocked:

| Method | Final-window MSE | Final-window acc | Test MSE | Test acc | d final MSE vs MLP64 | d test acc vs MLP64 |
|---|---:|---:|---:|---:|---:|---:|
| `mlp64` | 0.0041 +/- 0.0002 | 0.9860 +/- 0.0012 | 0.1375 +/- 0.0046 | 0.1336 +/- 0.0307 | 0.0000 | 0.0000 |
| `upgd_default_h64` | 0.0081 +/- 0.0012 | 0.9440 +/- 0.0081 | 0.0527 +/- 0.0033 | 0.6246 +/- 0.0226 | -0.0040 | +0.4910 |
| `upgd_lr01_h64` | 0.0078 +/- 0.0015 | 0.9487 +/- 0.0093 | 0.0494 +/- 0.0014 | 0.6320 +/- 0.0077 | -0.0036 | +0.4985 |
| `upgd_class_mse_h64` | 0.0038 +/- 0.0002 | 0.9860 +/- 0.0000 | 0.1446 +/- 0.0056 | 0.1113 +/- 0.0102 | +0.0004 | -0.0223 |
| `upgd_headnorm_h64` | 0.0172 +/- 0.0019 | 0.9227 +/- 0.0146 | 0.0502 +/- 0.0011 | 0.6221 +/- 0.0038 | -0.0131 | +0.4886 |

Positive-control digits shuffled:

| Method | Final-window MSE | Test acc | d final MSE vs MLP64 | d test acc vs MLP64 |
|---|---:|---:|---:|---:|
| `mlp64` | 0.0250 +/- 0.0005 | 0.9351 +/- 0.0043 | 0.0000 | 0.0000 |
| `upgd_default_h64` | 0.0060 +/- 0.0003 | 0.9586 +/- 0.0059 | +0.0190 | +0.0235 |
| `upgd_lr01_h64` | 0.0062 +/- 0.0001 | 0.9604 +/- 0.0054 | +0.0188 | +0.0254 |
| `upgd_headnorm_h64` | 0.0222 +/- 0.0003 | 0.9394 +/- 0.0114 | +0.0028 | +0.0043 |
| `upgd_class_mse_h64` | 0.0237 +/- 0.0009 | 0.9344 +/- 0.0022 | +0.0013 | -0.0006 |

## Mechanism Read

Diabetes looks like an output-update scaling problem, not a feature-discovery
or perturbation problem. Hidden-norm readout normalization and lower learning
rate both materially improve final-window and held-out MSE. Hidden-plus-input,
sigma zero, kappa 1.0, tiny margin, and head x2 do not.

Class-blocked tracking and retention are still split. Softmax CE variants
retain old classes well on held-out test accuracy, but lose current-block MSE
and accuracy. Linear MSE classification tracks the active class block and beats
MLP64 final-window MSE by a small amount, but it gives back the entire retained
classifier advantage and lands near the MLP64 chance-retention regime.

The shuffled positive control rules out several tempting knobs. Lower learning
rate preserves the promoted UPGD win. Hidden-norm and linear-MSE classification
do not fail catastrophically versus MLP64, but they mostly destroy the large
positive-control margin that the promoted softmax UPGD had.

## Recommendation

Do not promote a new single UPGD default from this sweep.

For a diabetes-specific external row, `upgd_headnorm_h64` is the strongest
candidate and `upgd_lr01_h64` is the safer candidate because it preserves
shuffled-digits behavior. For class-blocked final-window tracking, the only
simple fix found here is `upgd_class_mse_h64`, but it is not acceptable as a
general replacement because it loses retention and diabetes.

The next useful experiment should combine only the two non-conflicting
directions in a task-aware way: lower regression step/readout scaling for
diabetes, while leaving multiclass softmax CE intact. The class-blocked
tracking gap still appears to require the separate output-head anti-drift work;
simple CE/MSE, kappa, width, perturbation, margin, and fixed head-rate knobs do
not close it cleanly.
