# Worker S Math/Geometric Candidates

Status: **no universal replacement for fair MLP**. Two routes are worth keeping
as portfolio candidates, but only with a guarded MLP fallback:

- `orthogonal_projection`: best fixed candidate on rare and digits IID; overall
  `8/10/0` W/L/T over 18 paired dataset-seed cells, mean final-MSE delta
  `-0.0433` versus the per-seed best fair MLP.
- `cheb_tensor`: best fixed candidate on frequency; overall `4/14/0`,
  mean final-MSE delta `-0.0380`.

The sweep oracle that picks the best math/geometric candidate per seed/dataset
only reaches `10/8/0`, mean final-MSE delta `+0.0167`. That is useful headroom,
not a single learner claim.

## Exact commands

Smoke:

```bash
source .venv/bin/activate && python outputs/step2_worker_s_math_geometric_sweep/worker_s_math_geometric_sweep.py --smoke --output-dir outputs/step2_worker_s_smoke --note-path outputs/step2_worker_s_smoke/SMOKE.md
```

Full sweep used for this note:

```bash
source .venv/bin/activate && python outputs/step2_worker_s_math_geometric_sweep/worker_s_math_geometric_sweep.py --datasets controlled_nonlinear,controlled_polynomial,synthetic_compositional,controlled_frequency,controlled_rare,digits_iid --n-seeds 3 --steps 1200 --final-window 300 --output-dir outputs/step2_worker_s_math_geometric_sweep --note-path docs/research/step2_worker_s_math_geometric_candidates.md
```

Verification commands run after the script edit:

```bash
source .venv/bin/activate && ruff check outputs/step2_worker_s_math_geometric_sweep/worker_s_math_geometric_sweep.py
```

## Output files

- `outputs/step2_worker_s_math_geometric_sweep/worker_s_math_geometric_sweep.py`
  - standalone runner, 1,070 lines; no `src/` edits.
- `outputs/step2_worker_s_math_geometric_sweep/results.json`
  - full machine-readable results, 3 seeds x 6 datasets x 13 methods.
- `outputs/step2_worker_s_math_geometric_sweep/SUMMARY.md`
  - generated raw aggregate tables.
- `outputs/step2_worker_s_smoke/results.json`,
  `outputs/step2_worker_s_smoke/SUMMARY.md`, and
  `outputs/step2_worker_s_smoke/SMOKE.md`
  - smoke harness outputs.
- `docs/research/step2_worker_s_math_geometric_candidates.md`
  - this curated report.

## Protocol

Datasets: `controlled_nonlinear`, `controlled_polynomial`,
`synthetic_compositional`, `controlled_frequency`, `controlled_rare`,
`digits_iid`.

Comparator: per-seed best fair MLP among `mlp_h64`, `mlp_h128`,
`mlp_h64_64`, `mlp_32x32_no_ln`, and `mlp_64x64_no_ln`. The no-LayerNorm
controls use the established stronger controlled-suite setting
(`step_size=0.1`, `sparsity=0.0`); the LayerNorm controls use `step_size=0.03`,
`sparsity=0.5`.

Candidate implementation: all candidates use the same masked multi-head
Autostep readout with online input whitening and per-feature RMS normalization.
Only the feature map changes: whitened linear, orthogonal projection, random
Fourier, Chebyshev/tensor, TensorSketch, Nystrom-style RBF, residual RBF
expansion, and geometry-normalized RFF.

## Fixed Candidate Summary

Positive final-MSE delta favors the candidate over the per-seed best fair MLP.

| Candidate | Mean delta over 18 cells | W/L/T | Decision |
|---|---:|---:|---|
| `whiten_linear` | `-0.1142` | `2/16/0` | reject as standalone |
| `orthogonal_projection` | `-0.0433` | `8/10/0` | keep as guarded route |
| `rff` | `-0.0795` | `4/14/0` | reject as standalone |
| `cheb_tensor` | `-0.0380` | `4/14/0` | keep as frequency route |
| `tensor_sketch` | `-0.3201` | `3/15/0` | reject |
| `nystrom_rbf` | `-0.1804` | `0/18/0` | reject |
| `residual_rbf` | `-0.1706` | `0/18/0` | reject |
| `geom_norm_rff` | `-0.1612` | `0/18/0` | reject |

## Dataset Winners

| Dataset | Best fixed candidate | Mean delta | W/L/T | Interpretation |
|---|---|---:|---:|---|
| `controlled_frequency` | `cheb_tensor` | `+0.1578 +/- 0.0918` | `2/1/0` | plausible because Chebyshev terms directly expose low-degree trigonometric polynomial structure such as `sin(3 theta)`. |
| `controlled_nonlinear` | `cheb_tensor` | `-0.0036 +/- 0.0110` | `1/2/0` | statistical tie-ish, not a win; MLP still handles tanh mixtures better. |
| `controlled_polynomial` | `orthogonal_projection` | `-0.0719 +/- 0.0972` | `1/2/0` | no fixed math feature beats the stronger no-LN MLP. |
| `controlled_rare` | `orthogonal_projection` | `+0.0032 +/- 0.0012` | `3/0/0` | plausible because whitened orthogonal coordinates plus squared projection energy give a stable low-drift readout for sparse-head updates. |
| `digits_iid` | `orthogonal_projection` | `+0.0037 +/- 0.0008` | `3/0/0` | also improves held-out test accuracy by `+0.0204 +/- 0.0103`; plausible as a well-conditioned random kitchen-sink classifier for small standardized images. |
| `synthetic_compositional` | `whiten_linear` | `-0.0387 +/- 0.0134` | `0/3/0` | compositional tanh oracle still needs learned nonlinear features; fixed bases underfit. |

## Mechanistic Notes

`orthogonal_projection` is the only candidate with a plausible portfolio role:
it preserves input geometry, adds squared projection energies, and keeps the
readout convex/linear after the feature lift. That fits digits and rare-head
tracking, but it does not generate compositional features.

`cheb_tensor` is mechanistically credible for frequency because the feature
bank contains explicit low-degree Chebyshev products. It is not universal once
the no-LayerNorm MLP controls are included.

RBF/Nystrom/residual-center methods failed because local interpolation centers
do not match the algebraic and compositional structure of the controlled tasks.
The residual expander added target-aware centers causally after high loss, but
those centers were not reusable enough to beat MLP.

Geometry-normalized RFF likely discarded useful radial information on digits and
controlled streams; it lost every paired cell.

## Detailed Raw Tables

Protocol: 3 paired seeds, 1200 online steps, final window 300.
Comparator: per-seed best fair MLP among `mlp_h64`, `mlp_h128`, `mlp_h64_64`, `mlp_32x32_no_ln`, `mlp_64x64_no_ln`.
Paired diffs are positive when the candidate is better.

## controlled_frequency

| Method | Final MSE | Mean MSE | Test Acc | Runtime s | Feature dim |
|---|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.6747 +/- 0.0094 | 0.6982 +/- 0.0026 |  | 0.2303 +/- 0.0322 |  |
| `mlp_h128` | 0.6869 +/- 0.0153 | 0.7061 +/- 0.0049 |  | 0.3132 +/- 0.0936 |  |
| `mlp_h64_64` | 0.6658 +/- 0.0032 | 0.6902 +/- 0.0004 |  | 0.2760 +/- 0.0735 |  |
| `mlp_32x32_no_ln` | 0.5274 +/- 0.0023 | 0.5362 +/- 0.0007 |  | 0.2017 +/- 0.0137 |  |
| `mlp_64x64_no_ln` | 0.5393 +/- 0.0018 | 0.5492 +/- 0.0029 |  | 0.2802 +/- 0.0893 |  |
| `whiten_linear` | 0.5708 +/- 0.0039 | 0.5735 +/- 0.0020 |  | 0.1404 +/- 0.0360 | 9.0000 +/- 0.0000 |
| `orthogonal_projection` | 0.5655 +/- 0.0066 | 0.5738 +/- 0.0048 |  | 0.2200 +/- 0.0847 | 393.0000 +/- 0.0000 |
| `rff` | 0.5986 +/- 0.0347 | 0.6346 +/- 0.0194 |  | 0.2413 +/- 0.1373 | 201.0000 +/- 0.0000 |
| `cheb_tensor` | 0.3696 +/- 0.0938 | 0.4873 +/- 0.0811 |  | 0.5157 +/- 0.0683 | 217.0000 +/- 0.0000 |
| `tensor_sketch` | 0.5486 +/- 0.0107 | 0.5612 +/- 0.0013 |  | 0.6504 +/- 0.2085 | 201.0000 +/- 0.0000 |
| `nystrom_rbf` | 0.6662 +/- 0.0087 | 0.6889 +/- 0.0077 |  | 0.1699 +/- 0.0433 | 201.0000 +/- 0.0000 |
| `residual_rbf` | 0.6594 +/- 0.0148 | 0.6923 +/- 0.0095 |  | 0.1461 +/- 0.0063 | 153.0000 +/- 0.0000 |
| `geom_norm_rff` | 0.7459 +/- 0.0142 | 0.7392 +/- 0.0113 |  | 0.1295 +/- 0.0380 | 211.0000 +/- 0.0000 |

Best candidate oracle vs best fair MLP final MSE: +0.1578 +/- 0.0918; W/L/T 2/1/0; best candidate counts [('cheb_tensor', 3)].

| Candidate | Final-MSE diff vs best MLP | W/L/T |
|---|---:|---:|
| `whiten_linear` | -0.0434 +/- 0.0057 | 0/3/0 |
| `orthogonal_projection` | -0.0381 +/- 0.0088 | 0/3/0 |
| `rff` | -0.0712 +/- 0.0360 | 0/3/0 |
| `cheb_tensor` | +0.1578 +/- 0.0918 | 2/1/0 |
| `tensor_sketch` | -0.0212 +/- 0.0095 | 0/3/0 |
| `nystrom_rbf` | -0.1388 +/- 0.0110 | 0/3/0 |
| `residual_rbf` | -0.1320 +/- 0.0171 | 0/3/0 |
| `geom_norm_rff` | -0.2185 +/- 0.0157 | 0/3/0 |

## controlled_nonlinear

| Method | Final MSE | Mean MSE | Test Acc | Runtime s | Feature dim |
|---|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0935 +/- 0.0048 | 0.1323 +/- 0.0050 |  | 0.2280 +/- 0.0444 |  |
| `mlp_h128` | 0.1099 +/- 0.0060 | 0.1417 +/- 0.0023 |  | 0.2793 +/- 0.0610 |  |
| `mlp_h64_64` | 0.1364 +/- 0.0127 | 0.1869 +/- 0.0144 |  | 0.2988 +/- 0.0334 |  |
| `mlp_32x32_no_ln` | 0.1084 +/- 0.0186 | 0.1773 +/- 0.0211 |  | 0.2586 +/- 0.0393 |  |
| `mlp_64x64_no_ln` | 0.1091 +/- 0.0125 | 0.1797 +/- 0.0182 |  | 0.2202 +/- 0.0289 |  |
| `whiten_linear` | 0.2234 +/- 0.0065 | 0.2231 +/- 0.0052 |  | 0.1116 +/- 0.0338 | 9.0000 +/- 0.0000 |
| `orthogonal_projection` | 0.0989 +/- 0.0108 | 0.1230 +/- 0.0050 |  | 0.1557 +/- 0.0426 | 393.0000 +/- 0.0000 |
| `rff` | 0.1270 +/- 0.0038 | 0.1516 +/- 0.0053 |  | 0.1166 +/- 0.0280 | 201.0000 +/- 0.0000 |
| `cheb_tensor` | 0.0961 +/- 0.0067 | 0.1316 +/- 0.0095 |  | 0.5266 +/- 0.0433 | 217.0000 +/- 0.0000 |
| `tensor_sketch` | 0.2984 +/- 0.0254 | 0.4090 +/- 0.0045 |  | 0.4343 +/- 0.1075 | 201.0000 +/- 0.0000 |
| `nystrom_rbf` | 0.1877 +/- 0.0084 | 0.2061 +/- 0.0041 |  | 0.1869 +/- 0.0402 | 201.0000 +/- 0.0000 |
| `residual_rbf` | 0.1816 +/- 0.0017 | 0.2000 +/- 0.0024 |  | 0.2424 +/- 0.0432 | 153.0000 +/- 0.0000 |
| `geom_norm_rff` | 0.2123 +/- 0.0066 | 0.2119 +/- 0.0026 |  | 0.1395 +/- 0.0586 | 211.0000 +/- 0.0000 |

Best candidate oracle vs best fair MLP final MSE: -0.0007 +/- 0.0115; W/L/T 1/2/0; best candidate counts [('cheb_tensor', 1), ('orthogonal_projection', 2)].

| Candidate | Final-MSE diff vs best MLP | W/L/T |
|---|---:|---:|
| `whiten_linear` | -0.1309 +/- 0.0023 | 0/3/0 |
| `orthogonal_projection` | -0.0063 +/- 0.0136 | 1/2/0 |
| `rff` | -0.0345 +/- 0.0063 | 0/3/0 |
| `cheb_tensor` | -0.0036 +/- 0.0110 | 1/2/0 |
| `tensor_sketch` | -0.2059 +/- 0.0216 | 0/3/0 |
| `nystrom_rbf` | -0.0952 +/- 0.0050 | 0/3/0 |
| `residual_rbf` | -0.0890 +/- 0.0038 | 0/3/0 |
| `geom_norm_rff` | -0.1198 +/- 0.0028 | 0/3/0 |

## controlled_polynomial

| Method | Final MSE | Mean MSE | Test Acc | Runtime s | Feature dim |
|---|---:|---:|---:|---:|---:|
| `mlp_h64` | 1.4174 +/- 0.1927 | 1.2191 +/- 0.0744 |  | 0.2667 +/- 0.0425 |  |
| `mlp_h128` | 1.4208 +/- 0.2044 | 1.2138 +/- 0.0743 |  | 0.2871 +/- 0.0563 |  |
| `mlp_h64_64` | 1.5074 +/- 0.2088 | 1.3063 +/- 0.0710 |  | 0.3436 +/- 0.0279 |  |
| `mlp_32x32_no_ln` | 1.1658 +/- 0.1896 | 1.1189 +/- 0.0674 |  | 0.2539 +/- 0.0434 |  |
| `mlp_64x64_no_ln` | 1.2331 +/- 0.1440 | 1.1362 +/- 0.0425 |  | 0.2701 +/- 0.0333 |  |
| `whiten_linear` | 1.6160 +/- 0.1497 | 1.2754 +/- 0.0577 |  | 0.0838 +/- 0.0145 | 9.0000 +/- 0.0000 |
| `orthogonal_projection` | 1.2235 +/- 0.2022 | 1.0252 +/- 0.0902 |  | 0.1257 +/- 0.0175 | 393.0000 +/- 0.0000 |
| `rff` | 1.2395 +/- 0.2112 | 1.0979 +/- 0.0814 |  | 0.1693 +/- 0.0596 | 201.0000 +/- 0.0000 |
| `cheb_tensor` | 1.3519 +/- 0.1056 | 1.2073 +/- 0.0682 |  | 0.4394 +/- 0.0410 | 217.0000 +/- 0.0000 |
| `tensor_sketch` | 1.7692 +/- 0.2371 | 1.5916 +/- 0.1097 |  | 0.4407 +/- 0.0979 | 201.0000 +/- 0.0000 |
| `nystrom_rbf` | 1.6731 +/- 0.1942 | 1.3825 +/- 0.0789 |  | 0.1641 +/- 0.0595 | 201.0000 +/- 0.0000 |
| `residual_rbf` | 1.6149 +/- 0.1949 | 1.3216 +/- 0.0724 |  | 0.1827 +/- 0.0423 | 153.0000 +/- 0.0000 |
| `geom_norm_rff` | 1.6507 +/- 0.1796 | 1.3500 +/- 0.0683 |  | 0.1746 +/- 0.0295 | 211.0000 +/- 0.0000 |

Best candidate oracle vs best fair MLP final MSE: -0.0260 +/- 0.0746; W/L/T 1/2/0; best candidate counts [('cheb_tensor', 1), ('orthogonal_projection', 1), ('rff', 1)].

| Candidate | Final-MSE diff vs best MLP | W/L/T |
|---|---:|---:|
| `whiten_linear` | -0.4644 +/- 0.1140 | 0/3/0 |
| `orthogonal_projection` | -0.0719 +/- 0.0972 | 1/2/0 |
| `rff` | -0.0879 +/- 0.0497 | 0/3/0 |
| `cheb_tensor` | -0.2003 +/- 0.1049 | 0/3/0 |
| `tensor_sketch` | -0.6175 +/- 0.1403 | 0/3/0 |
| `nystrom_rbf` | -0.5215 +/- 0.1270 | 0/3/0 |
| `residual_rbf` | -0.4633 +/- 0.1084 | 0/3/0 |
| `geom_norm_rff` | -0.4991 +/- 0.1278 | 0/3/0 |

## controlled_rare

| Method | Final MSE | Mean MSE | Test Acc | Runtime s | Feature dim |
|---|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.1110 +/- 0.0196 | 0.1298 +/- 0.0141 |  | 0.2098 +/- 0.0063 |  |
| `mlp_h128` | 0.1112 +/- 0.0188 | 0.1290 +/- 0.0132 |  | 0.2293 +/- 0.0247 |  |
| `mlp_h64_64` | 0.1227 +/- 0.0189 | 0.1467 +/- 0.0125 |  | 0.3290 +/- 0.0357 |  |
| `mlp_32x32_no_ln` | 0.0839 +/- 0.0217 | 0.1112 +/- 0.0138 |  | 0.3277 +/- 0.0209 |  |
| `mlp_64x64_no_ln` | 0.0870 +/- 0.0213 | 0.1123 +/- 0.0143 |  | 0.2915 +/- 0.0061 |  |
| `whiten_linear` | 0.0822 +/- 0.0210 | 0.0939 +/- 0.0141 |  | 0.0733 +/- 0.0103 | 9.0000 +/- 0.0000 |
| `orthogonal_projection` | 0.0806 +/- 0.0209 | 0.0957 +/- 0.0149 |  | 0.1928 +/- 0.1068 | 393.0000 +/- 0.0000 |
| `rff` | 0.0877 +/- 0.0231 | 0.0977 +/- 0.0132 |  | 0.1533 +/- 0.0495 | 201.0000 +/- 0.0000 |
| `cheb_tensor` | 0.0983 +/- 0.0169 | 0.1249 +/- 0.0131 |  | 0.5189 +/- 0.0583 | 217.0000 +/- 0.0000 |
| `tensor_sketch` | 0.1513 +/- 0.0251 | 0.2116 +/- 0.0131 |  | 0.3583 +/- 0.0610 | 201.0000 +/- 0.0000 |
| `nystrom_rbf` | 0.0935 +/- 0.0189 | 0.1078 +/- 0.0144 |  | 0.1274 +/- 0.0235 | 201.0000 +/- 0.0000 |
| `residual_rbf` | 0.0880 +/- 0.0196 | 0.0985 +/- 0.0144 |  | 0.2190 +/- 0.0717 | 153.0000 +/- 0.0000 |
| `geom_norm_rff` | 0.0907 +/- 0.0188 | 0.1013 +/- 0.0126 |  | 0.1547 +/- 0.0370 | 211.0000 +/- 0.0000 |

Best candidate oracle vs best fair MLP final MSE: +0.0041 +/- 0.0009; W/L/T 3/0/0; best candidate counts [('orthogonal_projection', 2), ('whiten_linear', 1)].

| Candidate | Final-MSE diff vs best MLP | W/L/T |
|---|---:|---:|
| `whiten_linear` | +0.0017 +/- 0.0014 | 2/1/0 |
| `orthogonal_projection` | +0.0032 +/- 0.0012 | 3/0/0 |
| `rff` | -0.0038 +/- 0.0025 | 1/2/0 |
| `cheb_tensor` | -0.0144 +/- 0.0050 | 0/3/0 |
| `tensor_sketch` | -0.0674 +/- 0.0071 | 0/3/0 |
| `nystrom_rbf` | -0.0096 +/- 0.0028 | 0/3/0 |
| `residual_rbf` | -0.0041 +/- 0.0022 | 0/3/0 |
| `geom_norm_rff` | -0.0069 +/- 0.0038 | 0/3/0 |

## digits_iid

| Method | Final MSE | Mean MSE | Test Acc | Runtime s | Feature dim |
|---|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.0310 +/- 0.0011 | 0.0440 +/- 0.0001 | 0.9264 +/- 0.0070 | 0.8996 +/- 0.2758 |  |
| `mlp_h128` | 0.0305 +/- 0.0005 | 0.0440 +/- 0.0005 | 0.9239 +/- 0.0077 | 0.6466 +/- 0.0582 |  |
| `mlp_h64_64` | 0.0306 +/- 0.0012 | 0.0470 +/- 0.0013 | 0.9041 +/- 0.0128 | 0.6568 +/- 0.1438 |  |
| `mlp_32x32_no_ln` | 0.0275 +/- 0.0009 | 0.0457 +/- 0.0006 | 0.9258 +/- 0.0113 | 0.5232 +/- 0.0208 |  |
| `mlp_64x64_no_ln` | 0.0301 +/- 0.0007 | 0.0454 +/- 0.0012 | 0.9338 +/- 0.0143 | 0.5331 +/- 0.0690 |  |
| `whiten_linear` | 0.0373 +/- 0.0007 | 0.0412 +/- 0.0003 | 0.9221 +/- 0.0019 | 0.1335 +/- 0.0394 | 65.0000 +/- 0.0000 |
| `orthogonal_projection` | 0.0238 +/- 0.0011 | 0.0303 +/- 0.0007 | 0.9573 +/- 0.0011 | 0.3457 +/- 0.0348 | 449.0000 +/- 0.0000 |
| `rff` | 0.0263 +/- 0.0012 | 0.0325 +/- 0.0007 | 0.9505 +/- 0.0027 | 0.3911 +/- 0.1189 | 257.0000 +/- 0.0000 |
| `cheb_tensor` | 0.0278 +/- 0.0005 | 0.0347 +/- 0.0002 | 0.9456 +/- 0.0053 | 0.6614 +/- 0.0703 | 385.0000 +/- 0.0000 |
| `tensor_sketch` | 0.0251 +/- 0.0009 | 0.0307 +/- 0.0006 | 0.9555 +/- 0.0056 | 0.4962 +/- 0.0469 | 257.0000 +/- 0.0000 |
| `nystrom_rbf` | 0.0506 +/- 0.0011 | 0.0547 +/- 0.0002 | 0.8095 +/- 0.0374 | 0.3756 +/- 0.0820 | 257.0000 +/- 0.0000 |
| `residual_rbf` | 0.0477 +/- 0.0015 | 0.0494 +/- 0.0007 | 0.8411 +/- 0.0326 | 0.2270 +/- 0.0297 | 209.0000 +/- 0.0000 |
| `geom_norm_rff` | 0.0426 +/- 0.0011 | 0.0465 +/- 0.0005 | 0.8837 +/- 0.0202 | 0.2459 +/- 0.0214 | 323.0000 +/- 0.0000 |

Best candidate oracle vs best fair MLP final MSE: +0.0037 +/- 0.0008; W/L/T 3/0/0; best candidate counts [('orthogonal_projection', 3)].

| Candidate | Final-MSE diff vs best MLP | W/L/T |
|---|---:|---:|
| `whiten_linear` | -0.0098 +/- 0.0005 | 0/3/0 |
| `orthogonal_projection` | +0.0037 +/- 0.0008 | 3/0/0 |
| `rff` | +0.0011 +/- 0.0005 | 3/0/0 |
| `cheb_tensor` | -0.0003 +/- 0.0005 | 1/2/0 |
| `tensor_sketch` | +0.0023 +/- 0.0004 | 3/0/0 |
| `nystrom_rbf` | -0.0231 +/- 0.0006 | 0/3/0 |
| `residual_rbf` | -0.0202 +/- 0.0006 | 0/3/0 |
| `geom_norm_rff` | -0.0152 +/- 0.0007 | 0/3/0 |

| Candidate | Test-accuracy diff vs best MLP | W/L/T |
|---|---:|---:|
| `whiten_linear` | -0.0148 +/- 0.0130 | 1/2/0 |
| `orthogonal_projection` | +0.0204 +/- 0.0103 | 3/0/0 |
| `rff` | +0.0136 +/- 0.0136 | 1/2/0 |
| `cheb_tensor` | +0.0087 +/- 0.0059 | 3/0/0 |
| `tensor_sketch` | +0.0186 +/- 0.0102 | 2/1/0 |
| `nystrom_rbf` | -0.1274 +/- 0.0434 | 0/3/0 |
| `residual_rbf` | -0.0959 +/- 0.0366 | 0/3/0 |
| `geom_norm_rff` | -0.0532 +/- 0.0200 | 0/3/0 |

## synthetic_compositional

| Method | Final MSE | Mean MSE | Test Acc | Runtime s | Feature dim |
|---|---:|---:|---:|---:|---:|
| `mlp_h64` | 0.2781 +/- 0.1017 | 0.1827 +/- 0.0328 |  | 0.4101 +/- 0.0891 |  |
| `mlp_h128` | 0.2741 +/- 0.0962 | 0.1799 +/- 0.0297 |  | 0.2821 +/- 0.0449 |  |
| `mlp_h64_64` | 0.3524 +/- 0.1314 | 0.2316 +/- 0.0434 |  | 0.4011 +/- 0.0460 |  |
| `mlp_32x32_no_ln` | 0.5318 +/- 0.2720 | 0.3229 +/- 0.0809 |  | 0.4255 +/- 0.0161 |  |
| `mlp_64x64_no_ln` | 0.5310 +/- 0.2175 | 0.3221 +/- 0.0672 |  | 0.2720 +/- 0.0258 |  |
| `whiten_linear` | 0.3117 +/- 0.1087 | 0.2071 +/- 0.0359 |  | 0.0851 +/- 0.0094 | 7.0000 +/- 0.0000 |
| `orthogonal_projection` | 0.4234 +/- 0.1582 | 0.2853 +/- 0.0534 |  | 0.2182 +/- 0.0720 | 391.0000 +/- 0.0000 |
| `rff` | 0.5537 +/- 0.1772 | 0.3568 +/- 0.0689 |  | 0.1479 +/- 0.0223 | 199.0000 +/- 0.0000 |
| `cheb_tensor` | 0.4399 +/- 0.1578 | 0.2930 +/- 0.0544 |  | 0.7549 +/- 0.1006 | 211.0000 +/- 0.0000 |
| `tensor_sketch` | 1.2837 +/- 0.5313 | 0.8906 +/- 0.1427 |  | 0.3773 +/- 0.0358 | 199.0000 +/- 0.0000 |
| `nystrom_rbf` | 0.5671 +/- 0.2400 | 0.3175 +/- 0.0778 |  | 0.1871 +/- 0.0283 | 199.0000 +/- 0.0000 |
| `residual_rbf` | 0.5879 +/- 0.2576 | 0.3232 +/- 0.0833 |  | 0.1675 +/- 0.0386 | 151.0000 +/- 0.0000 |
| `geom_norm_rff` | 0.3806 +/- 0.1386 | 0.2346 +/- 0.0446 |  | 0.1854 +/- 0.0764 | 207.0000 +/- 0.0000 |

Best candidate oracle vs best fair MLP final MSE: -0.0387 +/- 0.0134; W/L/T 0/3/0; best candidate counts [('whiten_linear', 3)].

| Candidate | Final-MSE diff vs best MLP | W/L/T |
|---|---:|---:|
| `whiten_linear` | -0.0387 +/- 0.0134 | 0/3/0 |
| `orthogonal_projection` | -0.1504 +/- 0.0641 | 0/3/0 |
| `rff` | -0.2807 +/- 0.0817 | 0/3/0 |
| `cheb_tensor` | -0.1669 +/- 0.0613 | 0/3/0 |
| `tensor_sketch` | -1.0107 +/- 0.4352 | 0/3/0 |
| `nystrom_rbf` | -0.2941 +/- 0.1434 | 0/3/0 |
| `residual_rbf` | -0.3149 +/- 0.1614 | 0/3/0 |
| `geom_norm_rff` | -0.1075 +/- 0.0421 | 0/3/0 |
