# Step 2 Deep Feature Lifecycle

Seeds: 3. Steps: 800. Final window: 200.

Positive paired differences mean the method beat the best fair MLP.

## `nonlinear`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `upgd` | 0.0312 +/- 0.0077 | 0.0479 +/- 0.0040 |
| `mlp_64` | 0.0622 +/- 0.0139 | 0.0782 +/- 0.0076 |
| `mlp_64_64` | 0.0825 +/- 0.0184 | 0.1075 +/- 0.0070 |
| `deep_protected` | 0.0844 +/- 0.0223 | 0.1034 +/- 0.0058 |
| `deep_imprint` | 0.0854 +/- 0.0181 | 0.1033 +/- 0.0070 |
| `deep_hybrid` | 0.0870 +/- 0.0202 | 0.1033 +/- 0.0105 |
| `deep_budgeted` | 0.0887 +/- 0.0211 | 0.1042 +/- 0.0071 |
| `deep_feature_lifecycle` | 0.0892 +/- 0.0204 | 0.1085 +/- 0.0120 |
| `deep_orthogonal` | 0.0988 +/- 0.0252 | 0.1088 +/- 0.0145 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0203 +/- 0.0053 | 0/3 |
| `upgd` | +0.0311 +/- 0.0062 | 3/3 |
| `deep_feature_lifecycle` | -0.0270 +/- 0.0099 | 0/3 |
| `deep_imprint` | -0.0232 +/- 0.0043 | 0/3 |
| `deep_orthogonal` | -0.0366 +/- 0.0121 | 0/3 |
| `deep_protected` | -0.0222 +/- 0.0084 | 0/3 |
| `deep_budgeted` | -0.0265 +/- 0.0074 | 0/3 |
| `deep_hybrid` | -0.0247 +/- 0.0065 | 0/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=6.33, `deep_imprint`=4.00, `deep_orthogonal`=7.00, `deep_protected`=2.00, `deep_budgeted`=4.33, `deep_hybrid`=4.33.

## `interaction`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `upgd` | 0.5606 +/- 0.1939 | 0.5962 +/- 0.1246 |
| `deep_protected` | 1.0285 +/- 0.2760 | 1.1859 +/- 0.2522 |
| `deep_imprint` | 1.0346 +/- 0.2621 | 1.1790 +/- 0.2696 |
| `deep_orthogonal` | 1.0651 +/- 0.3280 | 1.1922 +/- 0.2939 |
| `mlp_64` | 1.1011 +/- 0.3050 | 1.1163 +/- 0.2441 |
| `mlp_64_64` | 1.1210 +/- 0.3639 | 1.1920 +/- 0.2433 |
| `deep_hybrid` | 1.1355 +/- 0.4149 | 1.2067 +/- 0.2462 |
| `deep_feature_lifecycle` | 1.1419 +/- 0.3289 | 1.1811 +/- 0.2388 |
| `deep_budgeted` | 1.1422 +/- 0.3893 | 1.2094 +/- 0.2201 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0199 +/- 0.0606 | 2/3 |
| `upgd` | +0.5405 +/- 0.1166 | 3/3 |
| `deep_feature_lifecycle` | -0.0408 +/- 0.0325 | 1/3 |
| `deep_imprint` | +0.0665 +/- 0.0433 | 2/3 |
| `deep_orthogonal` | +0.0360 +/- 0.0258 | 2/3 |
| `deep_protected` | +0.0726 +/- 0.0339 | 3/3 |
| `deep_budgeted` | -0.0411 +/- 0.1040 | 2/3 |
| `deep_hybrid` | -0.0344 +/- 0.1167 | 2/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=6.67, `deep_imprint`=9.67, `deep_orthogonal`=6.00, `deep_protected`=6.33, `deep_budgeted`=8.00, `deep_hybrid`=8.00.

## `out_of_class_polynomial`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `upgd` | 0.5870 +/- 0.1799 | 0.4848 +/- 0.1402 |
| `deep_budgeted` | 1.2017 +/- 0.3680 | 1.0153 +/- 0.2946 |
| `mlp_64_64` | 1.2060 +/- 0.3637 | 1.0152 +/- 0.2944 |
| `deep_feature_lifecycle` | 1.2185 +/- 0.3704 | 1.0334 +/- 0.2985 |
| `deep_imprint` | 1.2295 +/- 0.3733 | 1.0340 +/- 0.2978 |
| `deep_orthogonal` | 1.2299 +/- 0.3748 | 1.0327 +/- 0.3002 |
| `deep_hybrid` | 1.2356 +/- 0.3784 | 1.0220 +/- 0.3007 |
| `mlp_64` | 1.2497 +/- 0.3850 | 1.0383 +/- 0.2978 |
| `deep_protected` | 1.2744 +/- 0.4058 | 1.0390 +/- 0.3030 |

Best fair MLP: `mlp_64_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64` | -0.0437 +/- 0.0214 | 0/3 |
| `upgd` | +0.6191 +/- 0.1838 | 3/3 |
| `deep_feature_lifecycle` | -0.0124 +/- 0.0143 | 1/3 |
| `deep_imprint` | -0.0235 +/- 0.0118 | 0/3 |
| `deep_orthogonal` | -0.0239 +/- 0.0112 | 0/3 |
| `deep_protected` | -0.0684 +/- 0.0453 | 1/3 |
| `deep_budgeted` | +0.0043 +/- 0.0079 | 2/3 |
| `deep_hybrid` | -0.0296 +/- 0.0216 | 0/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=12.67, `deep_imprint`=12.00, `deep_orthogonal`=8.33, `deep_protected`=10.67, `deep_budgeted`=8.00, `deep_hybrid`=8.00.

## `frequency_mismatch`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `upgd` | 0.9805 +/- 0.5891 | 0.9539 +/- 0.5986 |
| `deep_imprint` | 1.8689 +/- 1.0351 | 1.9320 +/- 1.1808 |
| `mlp_64` | 1.9081 +/- 1.0914 | 1.9133 +/- 1.2083 |
| `deep_hybrid` | 1.9145 +/- 1.0906 | 1.9436 +/- 1.1804 |
| `mlp_64_64` | 1.9178 +/- 1.1013 | 1.9449 +/- 1.1729 |
| `deep_budgeted` | 1.9181 +/- 1.0939 | 1.9436 +/- 1.1697 |
| `deep_protected` | 1.9254 +/- 1.0940 | 1.9551 +/- 1.2056 |
| `deep_feature_lifecycle` | 1.9277 +/- 1.0823 | 1.9916 +/- 1.2317 |
| `deep_orthogonal` | 1.9562 +/- 1.1340 | 1.9657 +/- 1.2107 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0097 +/- 0.0148 | 1/3 |
| `upgd` | +0.9276 +/- 0.5026 | 3/3 |
| `deep_feature_lifecycle` | -0.0196 +/- 0.0202 | 1/3 |
| `deep_imprint` | +0.0393 +/- 0.0587 | 2/3 |
| `deep_orthogonal` | -0.0481 +/- 0.0516 | 1/3 |
| `deep_protected` | -0.0173 +/- 0.0189 | 1/3 |
| `deep_budgeted` | -0.0100 +/- 0.0234 | 1/3 |
| `deep_hybrid` | -0.0063 +/- 0.0253 | 1/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=14.33, `deep_imprint`=13.33, `deep_orthogonal`=13.67, `deep_protected`=10.33, `deep_budgeted`=8.00, `deep_hybrid`=8.00.

## `compositional`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `upgd` | 0.0773 +/- 0.0055 | 0.1165 +/- 0.0034 |
| `mlp_64` | 0.1476 +/- 0.0098 | 0.1678 +/- 0.0123 |
| `deep_budgeted` | 0.1684 +/- 0.0164 | 0.1982 +/- 0.0119 |
| `deep_hybrid` | 0.1781 +/- 0.0125 | 0.2025 +/- 0.0104 |
| `deep_imprint` | 0.1872 +/- 0.0168 | 0.2121 +/- 0.0100 |
| `mlp_64_64` | 0.1924 +/- 0.0178 | 0.2156 +/- 0.0060 |
| `deep_orthogonal` | 0.1965 +/- 0.0082 | 0.2134 +/- 0.0148 |
| `deep_feature_lifecycle` | 0.2050 +/- 0.0161 | 0.2177 +/- 0.0102 |
| `deep_protected` | 0.2068 +/- 0.0184 | 0.2200 +/- 0.0089 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0448 +/- 0.0087 | 0/3 |
| `upgd` | +0.0702 +/- 0.0045 | 3/3 |
| `deep_feature_lifecycle` | -0.0574 +/- 0.0090 | 0/3 |
| `deep_imprint` | -0.0396 +/- 0.0071 | 0/3 |
| `deep_orthogonal` | -0.0489 +/- 0.0021 | 0/3 |
| `deep_protected` | -0.0592 +/- 0.0124 | 0/3 |
| `deep_budgeted` | -0.0208 +/- 0.0112 | 0/3 |
| `deep_hybrid` | -0.0306 +/- 0.0033 | 0/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=7.00, `deep_imprint`=9.33, `deep_orthogonal`=4.00, `deep_protected`=3.00, `deep_budgeted`=3.67, `deep_hybrid`=5.00.

## `digits_iid`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `upgd` | 0.0199 +/- 0.0004 | 0.0306 +/- 0.0005 |
| `mlp_64` | 0.0371 +/- 0.0009 | 0.0506 +/- 0.0004 |
| `deep_imprint` | 0.0397 +/- 0.0009 | 0.0553 +/- 0.0018 |
| `deep_feature_lifecycle` | 0.0402 +/- 0.0014 | 0.0549 +/- 0.0015 |
| `mlp_64_64` | 0.0402 +/- 0.0007 | 0.0565 +/- 0.0010 |
| `deep_budgeted` | 0.0411 +/- 0.0017 | 0.0568 +/- 0.0009 |
| `deep_protected` | 0.0415 +/- 0.0004 | 0.0584 +/- 0.0002 |
| `deep_orthogonal` | 0.0420 +/- 0.0014 | 0.0567 +/- 0.0005 |
| `deep_hybrid` | 0.0426 +/- 0.0008 | 0.0575 +/- 0.0014 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0031 +/- 0.0011 | 0/3 |
| `upgd` | +0.0172 +/- 0.0010 | 3/3 |
| `deep_feature_lifecycle` | -0.0031 +/- 0.0007 | 0/3 |
| `deep_imprint` | -0.0026 +/- 0.0018 | 0/3 |
| `deep_orthogonal` | -0.0049 +/- 0.0007 | 0/3 |
| `deep_protected` | -0.0044 +/- 0.0006 | 0/3 |
| `deep_budgeted` | -0.0040 +/- 0.0014 | 0/3 |
| `deep_hybrid` | -0.0055 +/- 0.0007 | 0/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=0.00, `deep_imprint`=0.00, `deep_orthogonal`=0.00, `deep_protected`=0.00, `deep_budgeted`=0.00, `deep_hybrid`=0.00.

## Verdict

The best native deep feature lifecycle variant was `deep_imprint`, which beat the best fair MLP on 2/6 streams. A single general deep feature-construction algorithm should be treated as a partial or negative Step 2 result unless a native variant wins robustly across the full matrix.

## Audit And Claim Boundary

The native deep lifecycle path performs candidate hidden-unit testing inside
the online learner rather than as an offline post-processing step. The variants
tested here cover the simple mechanisms most likely to matter: residual
imprinting, orthogonalized candidate initialization, age/utility protection,
layer-wise replacement budgeting, and perturbation/plasticity hybridization.

The result is negative for canonical promotion. Deep replacement machinery is
wired and produces finite metrics, but it does not beat the best fair MLP
robustly and is dominated by UPGD on every stream in this compact closure run.
This closes the implementation gap, not the research gap: native deep feature
generation/testing remains an opt-in Step 2 research path rather than the
single general feature-construction answer.

Reproduction command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_deep_feature_lifecycle.py" \
  --seeds 3 \
  --num-steps 800 \
  --final-window 200 \
  --output-dir outputs/step2_deep_feature_lifecycle_closure \
  --note-path docs/research/step2_deep_feature_lifecycle_closure.md
```
