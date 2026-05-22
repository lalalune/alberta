# Step 2 Deep Feature Lifecycle

Seeds: 3. Steps: 1500. Final window: 500.

Positive paired differences mean the method beat the best fair MLP.

## Decision Summary

Decision: **rejected** for promotion.  The soft-gated native candidates are
implemented and temporally uniform, but no single native variant clears the
promotion bar.  The best representative native variant in this run,
`deep_active_perturb_preserve`, beat the best fair MLP on 2/6 probes.  The
best soft-gated variants, `deep_soft_gate_fast` and `deep_soft_gate_l1`, also
only reached 2/6 by mean paired difference and did not approach the desired
5/6 threshold.

Mechanism tested: bounded shadow candidates can opt into live scalar gates
into fixed hidden-unit slots.  The live prediction is used to shift the
underlying MLP target so active weights train against the real soft-gated
downstream computation.  Candidate gates, candidate utilities, candidate
incoming weights, active utilities, and active MLP weights all update every
time step.  Hardened soft candidates replace the active unit they were already
feeding, preserving downstream head weights to reduce promotion shock.

Commands:

```bash
source .venv/bin/activate && pytest tests/test_deep_feature_lifecycle.py -q
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_deep_feature_lifecycle.py" --seeds 3 --num-steps 1500 --final-window 500 --methods mlp_64 mlp_64_64 deep_feature_lifecycle deep_imprint deep_preserve_outgoing deep_active_perturb_preserve deep_soft_gate_fast deep_soft_gate_l1 deep_soft_gate_bank8
source .venv/bin/activate && pytest tests/ -v
source .venv/bin/activate && ruff check .
source .venv/bin/activate && mypy src/alberta_framework
```

The bare `mypy` command is not configured in this checkout and exits with
`Missing target module, package, files, or command.`  `mypy src tests` also
runs but currently fails on pre-existing untyped test files; package-source
mypy passes.

| Probe | Best fair MLP | Hard baseline | Best existing native | Best soft-gated | Soft promotions/run |
|---|---|---:|---:|---:|---:|
| nonlinear | `mlp_64` | -0.0274 | -0.0237 (`deep_imprint`) | -0.0213 (`deep_soft_gate_bank8`) | 0.33 |
| interaction | `mlp_64` | -0.0216 | -0.0120 (`deep_imprint`) | -0.0138 (`deep_soft_gate_fast`) | 0.33 |
| out_of_class_polynomial | `mlp_64_64` | -0.0286 | +0.0102 (`deep_active_perturb_preserve`) | +0.0079 (`deep_soft_gate_fast`) | 0.67 |
| frequency_mismatch | `mlp_64_64` | +0.0050 | +0.0180 (`deep_active_perturb_preserve`) | +0.0172 (`deep_soft_gate_l1`) | 0.67 |
| compositional | `mlp_64` | -0.0466 | -0.0478 (`deep_active_perturb_preserve`) | -0.0401 (`deep_soft_gate_l1`) | 0.33 |
| digits_iid | `mlp_64_64` | -0.0003 | +0.0007 (`deep_imprint`) | -0.0013 (`deep_soft_gate_l1`) | 0.00 |

## `nonlinear`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `mlp_64` | 0.0739 +/- 0.0080 | 0.0725 +/- 0.0072 |
| `deep_soft_gate_bank8` | 0.0952 +/- 0.0129 | 0.0946 +/- 0.0107 |
| `deep_imprint` | 0.0976 +/- 0.0095 | 0.0990 +/- 0.0111 |
| `deep_soft_gate_l1` | 0.0978 +/- 0.0143 | 0.0975 +/- 0.0102 |
| `mlp_64_64` | 0.1000 +/- 0.0156 | 0.0990 +/- 0.0095 |
| `deep_feature_lifecycle` | 0.1012 +/- 0.0136 | 0.0992 +/- 0.0128 |
| `deep_soft_gate_fast` | 0.1021 +/- 0.0157 | 0.0969 +/- 0.0097 |
| `deep_active_perturb_preserve` | 0.1026 +/- 0.0169 | 0.1011 +/- 0.0152 |
| `deep_preserve_outgoing` | 0.1032 +/- 0.0160 | 0.0983 +/- 0.0104 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0262 +/- 0.0077 | 0/3 |
| `deep_feature_lifecycle` | -0.0274 +/- 0.0066 | 0/3 |
| `deep_imprint` | -0.0237 +/- 0.0032 | 0/3 |
| `deep_preserve_outgoing` | -0.0293 +/- 0.0082 | 0/3 |
| `deep_active_perturb_preserve` | -0.0287 +/- 0.0090 | 0/3 |
| `deep_soft_gate_fast` | -0.0282 +/- 0.0085 | 0/3 |
| `deep_soft_gate_l1` | -0.0240 +/- 0.0067 | 0/3 |
| `deep_soft_gate_bank8` | -0.0213 +/- 0.0049 | 0/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=12.00, `deep_imprint`=8.67, `deep_preserve_outgoing`=4.33, `deep_active_perturb_preserve`=4.67, `deep_soft_gate_fast`=0.00, `deep_soft_gate_l1`=0.33, `deep_soft_gate_bank8`=0.33.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `mlp_64` | 771 | 0 | active learner updates every step |
| `deep_soft_gate_bank8` | 4931 | 656 | active + candidates update every step |
| `deep_imprint` | 4931 | 328 | active + candidates update every step |
| `deep_soft_gate_l1` | 4931 | 328 | active + candidates update every step |
| `mlp_64_64` | 4931 | 0 | active learner updates every step |
| `deep_feature_lifecycle` | 4931 | 328 | active + candidates update every step |
| `deep_soft_gate_fast` | 4931 | 328 | active + candidates update every step |
| `deep_active_perturb_preserve` | 4931 | 328 | active + candidates update every step |
| `deep_preserve_outgoing` | 4931 | 328 | active + candidates update every step |

## `interaction`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `mlp_64` | 0.6672 +/- 0.2646 | 0.9079 +/- 0.1541 |
| `deep_imprint` | 0.6792 +/- 0.2708 | 0.9651 +/- 0.1496 |
| `deep_soft_gate_fast` | 0.6810 +/- 0.2646 | 0.9602 +/- 0.1497 |
| `deep_active_perturb_preserve` | 0.6844 +/- 0.2870 | 0.9844 +/- 0.1711 |
| `mlp_64_64` | 0.6881 +/- 0.2677 | 0.9701 +/- 0.1597 |
| `deep_feature_lifecycle` | 0.6888 +/- 0.2637 | 0.9594 +/- 0.1498 |
| `deep_soft_gate_bank8` | 0.6950 +/- 0.2617 | 0.9935 +/- 0.1633 |
| `deep_preserve_outgoing` | 0.7181 +/- 0.2654 | 0.9696 +/- 0.1498 |
| `deep_soft_gate_l1` | 0.7201 +/- 0.2884 | 0.9976 +/- 0.1577 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0209 +/- 0.0103 | 0/3 |
| `deep_feature_lifecycle` | -0.0216 +/- 0.0109 | 0/3 |
| `deep_imprint` | -0.0120 +/- 0.0099 | 1/3 |
| `deep_preserve_outgoing` | -0.0509 +/- 0.0066 | 0/3 |
| `deep_active_perturb_preserve` | -0.0172 +/- 0.0319 | 1/3 |
| `deep_soft_gate_fast` | -0.0138 +/- 0.0053 | 0/3 |
| `deep_soft_gate_l1` | -0.0529 +/- 0.0242 | 0/3 |
| `deep_soft_gate_bank8` | -0.0278 +/- 0.0103 | 0/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=17.00, `deep_imprint`=11.67, `deep_preserve_outgoing`=14.33, `deep_active_perturb_preserve`=8.67, `deep_soft_gate_fast`=0.00, `deep_soft_gate_l1`=0.33, `deep_soft_gate_bank8`=0.33.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `mlp_64` | 771 | 0 | active learner updates every step |
| `deep_imprint` | 4931 | 328 | active + candidates update every step |
| `deep_soft_gate_fast` | 4931 | 328 | active + candidates update every step |
| `deep_active_perturb_preserve` | 4931 | 328 | active + candidates update every step |
| `mlp_64_64` | 4931 | 0 | active learner updates every step |
| `deep_feature_lifecycle` | 4931 | 328 | active + candidates update every step |
| `deep_soft_gate_bank8` | 4931 | 656 | active + candidates update every step |
| `deep_preserve_outgoing` | 4931 | 328 | active + candidates update every step |
| `deep_soft_gate_l1` | 4931 | 328 | active + candidates update every step |

## `out_of_class_polynomial`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_active_perturb_preserve` | 1.1522 +/- 0.4620 | 1.1268 +/- 0.3188 |
| `deep_soft_gate_fast` | 1.1546 +/- 0.4703 | 1.1289 +/- 0.3203 |
| `deep_soft_gate_l1` | 1.1611 +/- 0.4658 | 1.1263 +/- 0.3181 |
| `mlp_64_64` | 1.1624 +/- 0.4737 | 1.1273 +/- 0.3191 |
| `deep_imprint` | 1.1628 +/- 0.4686 | 1.1426 +/- 0.3197 |
| `deep_soft_gate_bank8` | 1.1660 +/- 0.4760 | 1.1300 +/- 0.3193 |
| `deep_feature_lifecycle` | 1.1910 +/- 0.4805 | 1.1501 +/- 0.3258 |
| `deep_preserve_outgoing` | 1.1912 +/- 0.4896 | 1.1574 +/- 0.3328 |
| `mlp_64` | 1.1933 +/- 0.4729 | 1.1566 +/- 0.3233 |

Best fair MLP: `mlp_64_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64` | -0.0308 +/- 0.0076 | 0/3 |
| `deep_feature_lifecycle` | -0.0286 +/- 0.0091 | 0/3 |
| `deep_imprint` | -0.0004 +/- 0.0054 | 1/3 |
| `deep_preserve_outgoing` | -0.0287 +/- 0.0180 | 1/3 |
| `deep_active_perturb_preserve` | +0.0102 +/- 0.0123 | 2/3 |
| `deep_soft_gate_fast` | +0.0079 +/- 0.0035 | 3/3 |
| `deep_soft_gate_l1` | +0.0014 +/- 0.0107 | 2/3 |
| `deep_soft_gate_bank8` | -0.0036 +/- 0.0037 | 1/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=25.67, `deep_imprint`=26.00, `deep_preserve_outgoing`=20.67, `deep_active_perturb_preserve`=15.00, `deep_soft_gate_fast`=0.67, `deep_soft_gate_l1`=0.67, `deep_soft_gate_bank8`=0.67.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_active_perturb_preserve` | 4931 | 328 | active + candidates update every step |
| `deep_soft_gate_fast` | 4931 | 328 | active + candidates update every step |
| `deep_soft_gate_l1` | 4931 | 328 | active + candidates update every step |
| `mlp_64_64` | 4931 | 0 | active learner updates every step |
| `deep_imprint` | 4931 | 328 | active + candidates update every step |
| `deep_soft_gate_bank8` | 4931 | 656 | active + candidates update every step |
| `deep_feature_lifecycle` | 4931 | 328 | active + candidates update every step |
| `deep_preserve_outgoing` | 4931 | 328 | active + candidates update every step |
| `mlp_64` | 771 | 0 | active learner updates every step |

## `frequency_mismatch`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_active_perturb_preserve` | 1.1259 +/- 0.4128 | 1.6774 +/- 0.8904 |
| `deep_soft_gate_l1` | 1.1267 +/- 0.4134 | 1.6774 +/- 0.8696 |
| `deep_soft_gate_fast` | 1.1302 +/- 0.4110 | 1.6748 +/- 0.8873 |
| `deep_feature_lifecycle` | 1.1389 +/- 0.4125 | 1.6946 +/- 0.8928 |
| `mlp_64_64` | 1.1439 +/- 0.4211 | 1.6903 +/- 0.8827 |
| `deep_soft_gate_bank8` | 1.1462 +/- 0.4252 | 1.6795 +/- 0.8804 |
| `mlp_64` | 1.1628 +/- 0.4285 | 1.6645 +/- 0.8891 |
| `deep_preserve_outgoing` | 1.1809 +/- 0.4396 | 1.6819 +/- 0.8838 |
| `deep_imprint` | 1.1826 +/- 0.4417 | 1.7571 +/- 0.9501 |

Best fair MLP: `mlp_64_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64` | -0.0188 +/- 0.0075 | 0/3 |
| `deep_feature_lifecycle` | +0.0050 +/- 0.0098 | 2/3 |
| `deep_imprint` | -0.0387 +/- 0.0206 | 1/3 |
| `deep_preserve_outgoing` | -0.0370 +/- 0.0384 | 1/3 |
| `deep_active_perturb_preserve` | +0.0180 +/- 0.0087 | 3/3 |
| `deep_soft_gate_fast` | +0.0137 +/- 0.0145 | 2/3 |
| `deep_soft_gate_l1` | +0.0172 +/- 0.0094 | 3/3 |
| `deep_soft_gate_bank8` | -0.0023 +/- 0.0042 | 1/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=27.00, `deep_imprint`=24.00, `deep_preserve_outgoing`=25.67, `deep_active_perturb_preserve`=14.33, `deep_soft_gate_fast`=0.67, `deep_soft_gate_l1`=0.67, `deep_soft_gate_bank8`=1.00.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_active_perturb_preserve` | 4610 | 304 | active + candidates update every step |
| `deep_soft_gate_l1` | 4610 | 304 | active + candidates update every step |
| `deep_soft_gate_fast` | 4610 | 304 | active + candidates update every step |
| `deep_feature_lifecycle` | 4610 | 304 | active + candidates update every step |
| `mlp_64_64` | 4610 | 0 | active learner updates every step |
| `deep_soft_gate_bank8` | 4610 | 608 | active + candidates update every step |
| `mlp_64` | 450 | 0 | active learner updates every step |
| `deep_preserve_outgoing` | 4610 | 304 | active + candidates update every step |
| `deep_imprint` | 4610 | 304 | active + candidates update every step |

## `compositional`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `mlp_64` | 0.2471 +/- 0.0312 | 0.1892 +/- 0.0081 |
| `deep_soft_gate_l1` | 0.2872 +/- 0.0424 | 0.2213 +/- 0.0182 |
| `deep_soft_gate_fast` | 0.2931 +/- 0.0488 | 0.2352 +/- 0.0207 |
| `deep_feature_lifecycle` | 0.2937 +/- 0.0521 | 0.2298 +/- 0.0213 |
| `deep_soft_gate_bank8` | 0.2940 +/- 0.0422 | 0.2261 +/- 0.0142 |
| `deep_active_perturb_preserve` | 0.2948 +/- 0.0519 | 0.2331 +/- 0.0210 |
| `deep_preserve_outgoing` | 0.2977 +/- 0.0481 | 0.2353 +/- 0.0175 |
| `mlp_64_64` | 0.3052 +/- 0.0575 | 0.2373 +/- 0.0228 |
| `deep_imprint` | 0.3090 +/- 0.0605 | 0.2439 +/- 0.0230 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0581 +/- 0.0276 | 0/3 |
| `deep_feature_lifecycle` | -0.0466 +/- 0.0220 | 0/3 |
| `deep_imprint` | -0.0619 +/- 0.0303 | 0/3 |
| `deep_preserve_outgoing` | -0.0506 +/- 0.0198 | 0/3 |
| `deep_active_perturb_preserve` | -0.0478 +/- 0.0213 | 0/3 |
| `deep_soft_gate_fast` | -0.0460 +/- 0.0196 | 0/3 |
| `deep_soft_gate_l1` | -0.0401 +/- 0.0130 | 0/3 |
| `deep_soft_gate_bank8` | -0.0469 +/- 0.0111 | 0/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=11.33, `deep_imprint`=10.33, `deep_preserve_outgoing`=6.00, `deep_active_perturb_preserve`=5.67, `deep_soft_gate_fast`=0.33, `deep_soft_gate_l1`=0.00, `deep_soft_gate_bank8`=0.00.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `mlp_64` | 643 | 0 | active learner updates every step |
| `deep_soft_gate_l1` | 4803 | 320 | active + candidates update every step |
| `deep_soft_gate_fast` | 4803 | 320 | active + candidates update every step |
| `deep_feature_lifecycle` | 4803 | 320 | active + candidates update every step |
| `deep_soft_gate_bank8` | 4803 | 640 | active + candidates update every step |
| `deep_active_perturb_preserve` | 4803 | 320 | active + candidates update every step |
| `deep_preserve_outgoing` | 4803 | 320 | active + candidates update every step |
| `mlp_64_64` | 4803 | 0 | active learner updates every step |
| `deep_imprint` | 4803 | 320 | active + candidates update every step |

## `digits_iid`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_imprint` | 0.0261 +/- 0.0008 | 0.0425 +/- 0.0009 |
| `mlp_64_64` | 0.0268 +/- 0.0011 | 0.0437 +/- 0.0007 |
| `deep_preserve_outgoing` | 0.0269 +/- 0.0017 | 0.0430 +/- 0.0014 |
| `deep_feature_lifecycle` | 0.0271 +/- 0.0014 | 0.0429 +/- 0.0009 |
| `deep_active_perturb_preserve` | 0.0272 +/- 0.0013 | 0.0439 +/- 0.0007 |
| `mlp_64` | 0.0275 +/- 0.0006 | 0.0406 +/- 0.0003 |
| `deep_soft_gate_l1` | 0.0281 +/- 0.0008 | 0.0444 +/- 0.0007 |
| `deep_soft_gate_fast` | 0.0282 +/- 0.0009 | 0.0452 +/- 0.0002 |
| `deep_soft_gate_bank8` | 0.0282 +/- 0.0008 | 0.0448 +/- 0.0009 |

Best fair MLP: `mlp_64_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64` | -0.0007 +/- 0.0005 | 1/3 |
| `deep_feature_lifecycle` | -0.0003 +/- 0.0005 | 1/3 |
| `deep_imprint` | +0.0007 +/- 0.0003 | 3/3 |
| `deep_preserve_outgoing` | -0.0001 +/- 0.0006 | 2/3 |
| `deep_active_perturb_preserve` | -0.0003 +/- 0.0005 | 1/3 |
| `deep_soft_gate_fast` | -0.0014 +/- 0.0003 | 0/3 |
| `deep_soft_gate_l1` | -0.0013 +/- 0.0006 | 0/3 |
| `deep_soft_gate_bank8` | -0.0014 +/- 0.0003 | 0/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=0.00, `deep_imprint`=0.00, `deep_preserve_outgoing`=0.00, `deep_active_perturb_preserve`=0.00, `deep_soft_gate_fast`=0.00, `deep_soft_gate_l1`=0.00, `deep_soft_gate_bank8`=0.00.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_imprint` | 8970 | 608 | active + candidates update every step |
| `mlp_64_64` | 8970 | 0 | active learner updates every step |
| `deep_preserve_outgoing` | 8970 | 608 | active + candidates update every step |
| `deep_feature_lifecycle` | 8970 | 608 | active + candidates update every step |
| `deep_active_perturb_preserve` | 8970 | 608 | active + candidates update every step |
| `mlp_64` | 4810 | 0 | active learner updates every step |
| `deep_soft_gate_l1` | 8970 | 608 | active + candidates update every step |
| `deep_soft_gate_fast` | 8970 | 608 | active + candidates update every step |
| `deep_soft_gate_bank8` | 8970 | 1216 | active + candidates update every step |

## Verdict

The best native deep feature lifecycle variant was `deep_active_perturb_preserve`, which beat the best fair MLP on 2/6 streams. A single general deep feature-construction algorithm should be treated as a partial or negative Step 2 result unless a native variant wins robustly across the full matrix.
