# Step 2 Deep Feature Lifecycle

Seeds: 5. Steps: 800. Final window: 200.

Positive paired differences mean the method beat the best fair MLP.

## `nonlinear`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `mlp_64` | 0.0612 +/- 0.0093 | 0.0815 +/- 0.0091 |
| `mlp_64_64` | 0.0855 +/- 0.0121 | 0.1138 +/- 0.0107 |
| `deep_active_perturb_low` | 0.0860 +/- 0.0138 | 0.1128 +/- 0.0139 |
| `deep_lr_low` | 0.0861 +/- 0.0151 | 0.1110 +/- 0.0163 |
| `deep_preserve_outgoing` | 0.0913 +/- 0.0136 | 0.1121 +/- 0.0150 |
| `deep_upgd_hybrid` | 0.0958 +/- 0.0161 | 0.1166 +/- 0.0164 |
| `upgd` | 0.0972 +/- 0.0131 | 0.1533 +/- 0.0193 |
| `deep_no_layernorm` | 0.1452 +/- 0.0193 | 0.1818 +/- 0.0202 |
| `deep_preserve_no_layernorm` | 0.1647 +/- 0.0286 | 0.1858 +/- 0.0232 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0243 +/- 0.0040 | 0/5 |
| `upgd` | -0.0360 +/- 0.0046 | 0/5 |
| `deep_no_layernorm` | -0.0840 +/- 0.0117 | 0/5 |
| `deep_preserve_outgoing` | -0.0301 +/- 0.0047 | 0/5 |
| `deep_preserve_no_layernorm` | -0.1036 +/- 0.0195 | 0/5 |
| `deep_active_perturb_low` | -0.0248 +/- 0.0049 | 0/5 |
| `deep_upgd_hybrid` | -0.0347 +/- 0.0074 | 0/5 |
| `deep_lr_low` | -0.0249 +/- 0.0061 | 0/5 |

Mean deep-feature promotions per run: `deep_no_layernorm`=11.60, `deep_preserve_outgoing`=4.40, `deep_preserve_no_layernorm`=11.20, `deep_active_perturb_low`=6.00, `deep_upgd_hybrid`=4.80, `deep_lr_low`=0.60.

## `interaction`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_active_perturb_low` | 0.9166 +/- 0.1939 | 1.0736 +/- 0.1591 |
| `deep_preserve_outgoing` | 0.9251 +/- 0.1798 | 1.0661 +/- 0.1643 |
| `mlp_64` | 0.9502 +/- 0.1978 | 0.9909 +/- 0.1562 |
| `mlp_64_64` | 0.9697 +/- 0.2254 | 1.0817 +/- 0.1502 |
| `deep_lr_low` | 0.9881 +/- 0.2461 | 1.0766 +/- 0.1560 |
| `deep_preserve_no_layernorm` | 0.9954 +/- 0.2503 | 1.1358 +/- 0.1705 |
| `deep_upgd_hybrid` | 0.9969 +/- 0.2369 | 1.0878 +/- 0.1432 |
| `deep_no_layernorm` | 1.0082 +/- 0.2562 | 1.1313 +/- 0.1673 |
| `upgd` | 1.4799 +/- 0.3614 | 1.6119 +/- 0.2337 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0194 +/- 0.0333 | 2/5 |
| `upgd` | -0.5297 +/- 0.1699 | 0/5 |
| `deep_no_layernorm` | -0.0579 +/- 0.0732 | 3/5 |
| `deep_preserve_outgoing` | +0.0252 +/- 0.0367 | 4/5 |
| `deep_preserve_no_layernorm` | -0.0451 +/- 0.0670 | 3/5 |
| `deep_active_perturb_low` | +0.0336 +/- 0.0118 | 5/5 |
| `deep_upgd_hybrid` | -0.0467 +/- 0.0555 | 2/5 |
| `deep_lr_low` | -0.0378 +/- 0.0589 | 2/5 |

Mean deep-feature promotions per run: `deep_no_layernorm`=12.20, `deep_preserve_outgoing`=7.80, `deep_preserve_no_layernorm`=10.80, `deep_active_perturb_low`=7.40, `deep_upgd_hybrid`=7.60, `deep_lr_low`=1.40.

## `out_of_class_polynomial`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_preserve_no_layernorm` | 0.8939 +/- 0.2478 | 0.8366 +/- 0.1800 |
| `deep_no_layernorm` | 0.8961 +/- 0.2496 | 0.8370 +/- 0.1798 |
| `mlp_64_64` | 0.9517 +/- 0.2554 | 0.8938 +/- 0.1885 |
| `deep_upgd_hybrid` | 0.9555 +/- 0.2554 | 0.9013 +/- 0.1883 |
| `deep_active_perturb_low` | 0.9622 +/- 0.2614 | 0.9019 +/- 0.1896 |
| `deep_lr_low` | 0.9699 +/- 0.2654 | 0.9037 +/- 0.1929 |
| `deep_preserve_outgoing` | 0.9748 +/- 0.2631 | 0.9211 +/- 0.1941 |
| `mlp_64` | 0.9904 +/- 0.2664 | 0.9185 +/- 0.1914 |
| `upgd` | 1.3859 +/- 0.3725 | 1.2803 +/- 0.2694 |

Best fair MLP: `mlp_64_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64` | -0.0387 +/- 0.0121 | 0/5 |
| `upgd` | -0.4342 +/- 0.1172 | 0/5 |
| `deep_no_layernorm` | +0.0556 +/- 0.0061 | 5/5 |
| `deep_preserve_outgoing` | -0.0231 +/- 0.0137 | 1/5 |
| `deep_preserve_no_layernorm` | +0.0579 +/- 0.0082 | 5/5 |
| `deep_active_perturb_low` | -0.0105 +/- 0.0098 | 1/5 |
| `deep_upgd_hybrid` | -0.0038 +/- 0.0071 | 2/5 |
| `deep_lr_low` | -0.0181 +/- 0.0152 | 0/5 |

Mean deep-feature promotions per run: `deep_no_layernorm`=11.80, `deep_preserve_outgoing`=10.60, `deep_preserve_no_layernorm`=11.20, `deep_active_perturb_low`=7.20, `deep_upgd_hybrid`=7.60, `deep_lr_low`=7.20.

## `frequency_mismatch`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_no_layernorm` | 1.5914 +/- 0.5982 | 1.4342 +/- 0.6158 |
| `deep_preserve_no_layernorm` | 1.6245 +/- 0.6466 | 1.4493 +/- 0.6308 |
| `deep_preserve_outgoing` | 1.6991 +/- 0.6201 | 1.5887 +/- 0.7015 |
| `mlp_64` | 1.6993 +/- 0.6462 | 1.5567 +/- 0.7104 |
| `deep_lr_low` | 1.7069 +/- 0.6341 | 1.5895 +/- 0.6960 |
| `deep_upgd_hybrid` | 1.7209 +/- 0.6423 | 1.5893 +/- 0.6918 |
| `deep_active_perturb_low` | 1.7274 +/- 0.6619 | 1.5840 +/- 0.7081 |
| `mlp_64_64` | 1.7432 +/- 0.6474 | 1.5884 +/- 0.6933 |
| `upgd` | 1.7872 +/- 0.6821 | 1.5798 +/- 0.7051 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0439 +/- 0.0225 | 1/5 |
| `upgd` | -0.0879 +/- 0.0598 | 1/5 |
| `deep_no_layernorm` | +0.1079 +/- 0.0749 | 3/5 |
| `deep_preserve_outgoing` | +0.0002 +/- 0.0330 | 2/5 |
| `deep_preserve_no_layernorm` | +0.0748 +/- 0.0607 | 3/5 |
| `deep_active_perturb_low` | -0.0281 +/- 0.0166 | 1/5 |
| `deep_upgd_hybrid` | -0.0217 +/- 0.0167 | 2/5 |
| `deep_lr_low` | -0.0076 +/- 0.0218 | 3/5 |

Mean deep-feature promotions per run: `deep_no_layernorm`=14.40, `deep_preserve_outgoing`=13.00, `deep_preserve_no_layernorm`=13.60, `deep_active_perturb_low`=8.00, `deep_upgd_hybrid`=8.00, `deep_lr_low`=11.20.

## `compositional`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `mlp_64` | 0.1395 +/- 0.0078 | 0.1677 +/- 0.0137 |
| `deep_upgd_hybrid` | 0.1597 +/- 0.0119 | 0.1953 +/- 0.0166 |
| `deep_lr_low` | 0.1631 +/- 0.0119 | 0.2028 +/- 0.0161 |
| `deep_active_perturb_low` | 0.1727 +/- 0.0177 | 0.2074 +/- 0.0143 |
| `mlp_64_64` | 0.1740 +/- 0.0158 | 0.2055 +/- 0.0159 |
| `deep_preserve_outgoing` | 0.1803 +/- 0.0151 | 0.2053 +/- 0.0157 |
| `upgd` | 0.2332 +/- 0.0114 | 0.3678 +/- 0.0377 |
| `deep_no_layernorm` | 0.3292 +/- 0.0509 | 0.3600 +/- 0.0234 |
| `deep_preserve_no_layernorm` | 0.3458 +/- 0.0501 | 0.3611 +/- 0.0267 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0345 +/- 0.0083 | 0/5 |
| `upgd` | -0.0937 +/- 0.0113 | 0/5 |
| `deep_no_layernorm` | -0.1897 +/- 0.0439 | 0/5 |
| `deep_preserve_outgoing` | -0.0408 +/- 0.0080 | 0/5 |
| `deep_preserve_no_layernorm` | -0.2063 +/- 0.0442 | 0/5 |
| `deep_active_perturb_low` | -0.0332 +/- 0.0118 | 0/5 |
| `deep_upgd_hybrid` | -0.0201 +/- 0.0055 | 0/5 |
| `deep_lr_low` | -0.0236 +/- 0.0043 | 0/5 |

Mean deep-feature promotions per run: `deep_no_layernorm`=12.20, `deep_preserve_outgoing`=4.00, `deep_preserve_no_layernorm`=11.00, `deep_active_perturb_low`=6.20, `deep_upgd_hybrid`=5.20, `deep_lr_low`=0.40.

## `digits_iid`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `mlp_64` | 0.0363 +/- 0.0007 | 0.0509 +/- 0.0005 |
| `mlp_64_64` | 0.0392 +/- 0.0007 | 0.0554 +/- 0.0011 |
| `deep_preserve_outgoing` | 0.0393 +/- 0.0006 | 0.0554 +/- 0.0010 |
| `deep_upgd_hybrid` | 0.0397 +/- 0.0014 | 0.0554 +/- 0.0012 |
| `deep_no_layernorm` | 0.0398 +/- 0.0006 | 0.0594 +/- 0.0004 |
| `deep_preserve_no_layernorm` | 0.0402 +/- 0.0003 | 0.0599 +/- 0.0004 |
| `deep_lr_low` | 0.0407 +/- 0.0012 | 0.0563 +/- 0.0012 |
| `deep_active_perturb_low` | 0.0413 +/- 0.0003 | 0.0580 +/- 0.0003 |
| `upgd` | 0.1903 +/- 0.0030 | 0.3030 +/- 0.0086 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0029 +/- 0.0007 | 0/5 |
| `upgd` | -0.1540 +/- 0.0030 | 0/5 |
| `deep_no_layernorm` | -0.0035 +/- 0.0007 | 0/5 |
| `deep_preserve_outgoing` | -0.0030 +/- 0.0010 | 0/5 |
| `deep_preserve_no_layernorm` | -0.0039 +/- 0.0009 | 0/5 |
| `deep_active_perturb_low` | -0.0050 +/- 0.0005 | 0/5 |
| `deep_upgd_hybrid` | -0.0034 +/- 0.0009 | 0/5 |
| `deep_lr_low` | -0.0044 +/- 0.0008 | 0/5 |

Mean deep-feature promotions per run: `deep_no_layernorm`=4.60, `deep_preserve_outgoing`=0.00, `deep_preserve_no_layernorm`=1.60, `deep_active_perturb_low`=0.00, `deep_upgd_hybrid`=0.00, `deep_lr_low`=0.00.

## Verdict

The best native deep feature lifecycle variant was `deep_no_layernorm`, which beat the best fair MLP on 2/6 streams. A single general deep feature-construction algorithm should be treated as a partial or negative Step 2 result unless a native variant wins robustly across the full matrix.

## Audit

This pass attacked the hard blocker with two additional native mechanisms:

- `early_promotion_outgoing_mode="preserve"` keeps the replaced early hidden unit's downstream column instead of zeroing it, while still clearing the outgoing trace column. This is the smallest native analogue of UPGD's continuity bias: new structure is inserted without disconnecting the downstream pathway.
- `active_perturbation_std` adds optional UPGD-style low-utility perturbation to active trunk rows inside the lifecycle wrapper. The tested closure variant used a conservative `1e-4` scale after the `1e-3` exploratory run damaged several probes.

The mechanisms did not change the decision. `deep_preserve_outgoing` produced a small interaction win by paired mean but did not transfer. `deep_preserve_no_layernorm` preserved the known no-LayerNorm polynomial/frequency wins but hurt nonlinear, compositional, and digits. `deep_active_perturb_low` stayed close to fair MLP on some probes but did not beat it robustly.

The initial reproduction run (`outputs/step2_deep_hard_repro_2seed_600`) reproduced the current failure mode in miniature: UPGD beat the fair MLP on all six probes, while native lifecycle variants only showed the known polynomial/frequency behavior. The final closure table above is the 5-seed native mechanism audit; it includes UPGD for visibility, but the native decision is made against the fair MLP because the UPGD comparison is sensitive to the example runner's method-list-dependent stream/key splitting.

## Commands

```bash
source .venv/bin/activate
pytest tests/test_deep_feature_lifecycle.py -q
python "examples/The Alberta Plan/Step2/step2_deep_feature_lifecycle.py" \
  --seeds 2 \
  --num-steps 600 \
  --final-window 200 \
  --methods mlp_64 mlp_64_64 upgd deep_fast_cadence deep_no_layernorm \
    deep_upgd_hybrid \
  --output-dir outputs/step2_deep_hard_repro_2seed_600 \
  --note-path docs/research/step2_deep_hard_closure.md
python "examples/The Alberta Plan/Step2/step2_deep_feature_lifecycle.py" \
  --seeds 5 \
  --num-steps 800 \
  --final-window 200 \
  --methods mlp_64 mlp_64_64 upgd deep_no_layernorm deep_preserve_outgoing \
    deep_preserve_no_layernorm deep_active_perturb_low deep_upgd_hybrid \
    deep_lr_low \
  --output-dir outputs/step2_deep_lifecycle_native_patch_final_5seed_800 \
  --note-path docs/research/step2_deep_hard_closure.md
ruff check .
mypy src/alberta_framework/core/deep_feature_lifecycle.py tests/test_deep_feature_lifecycle.py
pytest tests/ -q
```

## Conclusion

Native deep lifecycle is still not ready for canonical Step 2 promotion. The best simple additions are partially useful diagnostics, not a generally competitive MLP feature lifecycle mechanism.
