# Step 1 Robustness Study Summary

- Stream: AlbertaPlanStep1Stream(feature_dim=20, num_relevant=5, drift_rate_w=0.001, drift_rate_b=0.001, noise_std=1.0)
- Seeds: 30, steps per run: 15000, final-window: last 5000 steps.
- Hyperparameter grid: 11 log-spaced points in [1e-4, 10**-0.5].
- Wall-clock: 73.5 s.

## Robustness comparison

Lower `robustness_ratio` and wider `working_range_decades` mean the optimizer is less sensitive to its hyperparameter.
 `working_range_decades` is the span of the grid that yields mean MSE within 1.5x of the best mean MSE.

| Optimizer | Best HP | Best MSE | Min MSE | Max MSE | Grid-mean MSE | Robustness ratio | Working range (decades) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LMS | 0.0005012 | 1.009 | 1.009 | 3.381 | 1.393 | 1.38 | 2.45 |
| IDBD | 0.0631 | 1.044 | 1.044 | 2.122 | 1.462 | 1.4 | 2.10 |
| Autostep | 0.3162 | 1.018 | 1.018 | 1.234 | 1.153 | 1.13 | 3.50 |
| Adam | 0.001122 | 1.012 | 1.012 | 12.82 | 2.708 | 2.67 | 1.75 |

## Per-grid-point mean MSE

| HP | LMS | IDBD | Autostep | Adam |
| --- | --- | --- | --- | --- |
| 0.0001 | 1.428 | 2.122 | 1.234 | 2.999 |
| 0.0002239 | 1.034 | 2.024 | 1.233 | 1.527 |
| 0.0005012 | 1.009 | 1.867 | 1.231 | 1.023 |
| 0.001122 | 1.012 | 1.663 | 1.226 | 1.012 |
| 0.002512 | 1.026 | 1.455 | 1.216 | 1.026 |
| 0.005623 | 1.061 | 1.284 | 1.192 | 1.059 |
| 0.01259 | 1.152 | 1.163 | 1.149 | 1.139 |
| 0.02818 | 1.433 | 1.087 | 1.099 | 1.34 |
| 0.0631 | 3.381 | 1.044 | 1.057 | 1.913 |
| 0.1413 | nan | 1.16 | 1.03 | 3.924 |
| 0.3162 | nan | 1.208 | 1.018 | 12.82 |

## Notes

- `Best HP` is the grid point that minimised the mean MSE across seeds; the working range is computed in log10 between the leftmost and rightmost grid points whose mean MSE is within 1.5x of the best.
- Cells where every seed blew up are stored as `null` in the JSON and `nan` in this table.