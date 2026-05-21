# Step 1 Normalization Ablation Summary

- Seeds: 30, steps per run: 20000, final-window: last 5000 steps.
- Wall-clock: 26.6 s.

## EMA vs no normalization (% MSE improvement, paired by seed)

Cells show `pct_improvement (95% CI on per-seed difference)`. Positive = EMANormalizer beats None.

| Optimizer | XDistShift | DynamicScaleShift |
| --- | --- | --- |
| LMS | unstable (left 30/30 NaN, right 0/30 NaN) | unstable (left 30/30 NaN, right 0/30 NaN) |
| IDBD | unstable (left 30/30 NaN, right 22/30 NaN) | unstable (left 30/30 NaN, right 30/30 NaN) |
| Autostep | <-1000% (Δ CI [-4, -2.3], d=-1.33, p=3.7e-13) | +35.3% (Δ CI [+5.23, +7.15], d=+2.30, p=2.3e-36) |
| Adam | <-1000% (Δ CI [-120, -58.1], d=-1.03, p=1.5e-08) | +7.8% (Δ CI [-2.99, +6.13], d=+0.12, p=0.5) |

## Welford vs no normalization (% MSE improvement, paired by seed)

| Optimizer | XDistShift | DynamicScaleShift |
| --- | --- | --- |
| LMS | unstable (left 30/30 NaN, right 0/30 NaN) | unstable (left 30/30 NaN, right 29/30 NaN) |
| IDBD | unstable (left 30/30 NaN, right 27/30 NaN) | unstable (left 30/30 NaN, right 30/30 NaN) |
| Autostep | -4.3% (Δ CI [-0.0268, +0.00278], d=-0.29, p=0.11) | -10.5% (Δ CI [-2.59, -1.09], d=-0.88, p=1.6e-06) |
| Adam | <-1000% (Δ CI [-56.8, -18.7], d=-0.71, p=0.0001) | +2.9% (Δ CI [-3.82, +5], d=+0.05, p=0.79) |

## StreamingBatch vs no normalization (% MSE improvement, paired by seed)

| Optimizer | XDistShift | DynamicScaleShift |
| --- | --- | --- |
| LMS | unstable (left 30/30 NaN, right 0/30 NaN) | unstable (left 30/30 NaN, right 0/30 NaN) |
| IDBD | unstable (left 30/30 NaN, right 30/30 NaN) | unstable (left 30/30 NaN, right 30/30 NaN) |
| Autostep | <-1000% (Δ CI [-3.89, -2.24], d=-1.33, p=3.5e-13) | +34.5% (Δ CI [+5.04, +7.06], d=+2.14, p=1.2e-31) |
| Adam | <-1000% (Δ CI [-121, -58.9], d=-1.03, p=1.5e-08) | +7.7% (Δ CI [-3, +6.12], d=+0.12, p=0.5) |

## Per-cell mean (final-window MSE)

| Stream | Optimizer | None | EMA | Welford | StreamingBatch |
| --- | --- | --- | --- | --- | --- |
| XDistShift | LMS | unstable (30/30 NaN) | 2.889 | 0.2895 | 2.908 |
| XDistShift | IDBD | unstable (30/30 NaN) | 4.378 (22/30 NaN) | 0.3633 (27/30 NaN) | unstable (30/30 NaN) |
| XDistShift | Autostep | 0.2827 | 3.435 | 0.2947 | 3.343 |
| XDistShift | Adam | 0.2802 | 89.18 | 38.01 | 90.27 |
| DynamicScaleShift | LMS | unstable (30/30 NaN) | 6.544 | 5.507e+34 (29/30 NaN) | 6.481 |
| DynamicScaleShift | IDBD | unstable (30/30 NaN) | unstable (30/30 NaN) | unstable (30/30 NaN) | unstable (30/30 NaN) |
| DynamicScaleShift | Autostep | 17.56 | 11.37 | 19.4 | 11.51 |
| DynamicScaleShift | Adam | 20.21 | 18.64 | 19.62 | 18.65 |

## Headline findings

- `LMS` on `XDistShift`: 100% of seeds NaN without normalization; with `EMA` MSE = 2.889 (normalization is REQUIRED, not just helpful).
- `IDBD` on `XDistShift`: 100% of seeds NaN without normalization; with `EMA` MSE = 4.378 (normalization is REQUIRED, not just helpful).
- `LMS` on `DynamicScaleShift`: 100% of seeds NaN without normalization; with `EMA` MSE = 6.544 (normalization is REQUIRED, not just helpful).

## Notes

- `pct_improvement_of_right_over_left` is computed from per-cell means, while the CI/Cohen's d/p-value characterise the per-seed paired distribution `MSE(no-norm) - MSE(normalizer)`.
- Paired t-tests use the normal approximation. With 30 seeds the approximation is good but we also report the sign-test p-value in the JSON (`paired_sign_p_value`).