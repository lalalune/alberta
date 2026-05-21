# Step 2 Deep Feature Lifecycle

Decision: **reject for Step 2 promotion**.  The Net2Net/function-preserving
pivot improved over the prior 2/6 ceiling only to 3/6 streams, with one very
small interaction win and persistent losses on nonlinear, compositional, and
digits.  This is not a substantial enough gain to justify more native deep
lifecycle work before Step 2 closure.

Recommendation: stop native deep lifecycle work for Step 2 closure.  The
function-preserving mechanism is useful evidence that replacement disruption
was part of the problem, but the remaining performance pattern is still not a
robust feature-construction result.

Command:

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_deep_feature_lifecycle.py" --seeds 3 --num-steps 1500 --final-window 500 --output-dir outputs/step2_deep_net2net_attempt --note-path docs/research/step2_deep_net2net_attempt.md --methods mlp_64 mlp_64_64 upgd deep_feature_lifecycle deep_imprint deep_preserve_outgoing deep_active_perturb_preserve deep_net2net deep_net2net_guarded deep_net2net_final deep_net2net_fast
```

## Net2Net Summary

Positive paired differences mean the method beat the best fair MLP.

| Stream | Best fair MLP | Best Net2Net variant | Best MLP - Net2Net | Seed wins | Best existing native | Best MLP - existing |
|---|---|---|---:|---:|---|---:|
| nonlinear | `mlp_64` | `deep_net2net_final` | -0.0183 | 0/3 | `deep_feature_lifecycle` | -0.0264 |
| interaction | `mlp_64` | `deep_net2net_final` | +0.0012 | 1/3 | `deep_active_perturb_preserve` | -0.0101 |
| out_of_class_polynomial | `mlp_64_64` | `deep_net2net_final` | +0.0096 | 2/3 | `deep_active_perturb_preserve` | +0.0055 |
| frequency_mismatch | `mlp_64_64` | `deep_net2net_final` | +0.0190 | 2/3 | `deep_active_perturb_preserve` | +0.0009 |
| compositional | `mlp_64` | `deep_net2net` | -0.0426 | 0/3 | `deep_preserve_outgoing` | -0.0457 |
| digits_iid | `mlp_64_64` | `deep_net2net_final` | -0.0004 | 1/3 | `deep_feature_lifecycle` | +0.0007 |

Best Net2Net stream wins: `deep_net2net_final` won 3/6 by paired mean.
Existing native variants in this comparison also topped out at 2/6.

## Full Generated Results

Seeds: 3. Steps: 1500. Final window: 500.

Positive paired differences mean the method beat the best fair MLP.

## `nonlinear`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `mlp_64` | 0.0739 +/- 0.0080 | 0.0725 +/- 0.0072 |
| `deep_net2net_final` | 0.0921 +/- 0.0135 | 0.0951 +/- 0.0103 |
| `deep_net2net_guarded` | 0.0964 +/- 0.0135 | 0.0950 +/- 0.0110 |
| `deep_net2net` | 0.0986 +/- 0.0140 | 0.0979 +/- 0.0100 |
| `mlp_64_64` | 0.1000 +/- 0.0156 | 0.0990 +/- 0.0095 |
| `deep_feature_lifecycle` | 0.1002 +/- 0.0111 | 0.0999 +/- 0.0117 |
| `deep_preserve_outgoing` | 0.1018 +/- 0.0173 | 0.1008 +/- 0.0149 |
| `deep_net2net_fast` | 0.1020 +/- 0.0151 | 0.1006 +/- 0.0121 |
| `deep_imprint` | 0.1034 +/- 0.0155 | 0.0982 +/- 0.0104 |
| `deep_active_perturb_preserve` | 0.1065 +/- 0.0158 | 0.0988 +/- 0.0100 |
| `upgd` | 0.1745 +/- 0.0298 | 0.1465 +/- 0.0178 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0262 +/- 0.0077 | 0/3 |
| `upgd` | -0.1006 +/- 0.0266 | 0/3 |
| `deep_feature_lifecycle` | -0.0264 +/- 0.0047 | 0/3 |
| `deep_imprint` | -0.0295 +/- 0.0078 | 0/3 |
| `deep_preserve_outgoing` | -0.0280 +/- 0.0094 | 0/3 |
| `deep_active_perturb_preserve` | -0.0326 +/- 0.0098 | 0/3 |
| `deep_net2net` | -0.0247 +/- 0.0068 | 0/3 |
| `deep_net2net_guarded` | -0.0225 +/- 0.0055 | 0/3 |
| `deep_net2net_final` | -0.0183 +/- 0.0058 | 0/3 |
| `deep_net2net_fast` | -0.0282 +/- 0.0071 | 0/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=9.67, `deep_imprint`=9.33, `deep_preserve_outgoing`=4.00, `deep_active_perturb_preserve`=3.33, `deep_net2net`=7.33, `deep_net2net_guarded`=4.33, `deep_net2net_final`=2.00, `deep_net2net_fast`=15.33.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `mlp_64` | 771 | 0 | active learner updates every step |
| `deep_net2net_final` | 4931 | 328 | active + candidates update every step |
| `deep_net2net_guarded` | 4931 | 328 | active + candidates update every step |
| `deep_net2net` | 4931 | 328 | active + candidates update every step |
| `mlp_64_64` | 4931 | 0 | active learner updates every step |
| `deep_feature_lifecycle` | 4931 | 328 | active + candidates update every step |
| `deep_preserve_outgoing` | 4931 | 328 | active + candidates update every step |
| `deep_net2net_fast` | 4931 | 328 | active + candidates update every step |
| `deep_imprint` | 4931 | 328 | active + candidates update every step |
| `deep_active_perturb_preserve` | 4931 | 328 | active + candidates update every step |
| `upgd` | 771 | 0 | active learner updates every step |

## `interaction`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_net2net_final` | 0.6660 +/- 0.2541 | 0.9473 +/- 0.1487 |
| `mlp_64` | 0.6672 +/- 0.2646 | 0.9079 +/- 0.1541 |
| `deep_active_perturb_preserve` | 0.6773 +/- 0.2458 | 0.9683 +/- 0.1507 |
| `mlp_64_64` | 0.6881 +/- 0.2677 | 0.9701 +/- 0.1597 |
| `deep_net2net_guarded` | 0.6905 +/- 0.2657 | 0.9870 +/- 0.1663 |
| `deep_feature_lifecycle` | 0.6914 +/- 0.2700 | 0.9733 +/- 0.1515 |
| `deep_preserve_outgoing` | 0.6938 +/- 0.2781 | 0.9869 +/- 0.1679 |
| `deep_imprint` | 0.7059 +/- 0.2698 | 0.9654 +/- 0.1522 |
| `deep_net2net` | 0.7254 +/- 0.2871 | 0.9997 +/- 0.1582 |
| `deep_net2net_fast` | 0.7361 +/- 0.3000 | 0.9895 +/- 0.1591 |
| `upgd` | 1.0017 +/- 0.4103 | 1.4847 +/- 0.2515 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0209 +/- 0.0103 | 0/3 |
| `upgd` | -0.3345 +/- 0.1459 | 0/3 |
| `deep_feature_lifecycle` | -0.0243 +/- 0.0061 | 0/3 |
| `deep_imprint` | -0.0387 +/- 0.0141 | 0/3 |
| `deep_preserve_outgoing` | -0.0266 +/- 0.0311 | 1/3 |
| `deep_active_perturb_preserve` | -0.0101 +/- 0.0206 | 1/3 |
| `deep_net2net` | -0.0582 +/- 0.0226 | 0/3 |
| `deep_net2net_guarded` | -0.0233 +/- 0.0039 | 0/3 |
| `deep_net2net_final` | +0.0012 +/- 0.0114 | 1/3 |
| `deep_net2net_fast` | -0.0689 +/- 0.0355 | 0/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=14.33, `deep_imprint`=18.67, `deep_preserve_outgoing`=10.33, `deep_active_perturb_preserve`=7.33, `deep_net2net`=14.67, `deep_net2net_guarded`=10.00, `deep_net2net_final`=7.33, `deep_net2net_fast`=26.67.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_net2net_final` | 4931 | 328 | active + candidates update every step |
| `mlp_64` | 771 | 0 | active learner updates every step |
| `deep_active_perturb_preserve` | 4931 | 328 | active + candidates update every step |
| `mlp_64_64` | 4931 | 0 | active learner updates every step |
| `deep_net2net_guarded` | 4931 | 328 | active + candidates update every step |
| `deep_feature_lifecycle` | 4931 | 328 | active + candidates update every step |
| `deep_preserve_outgoing` | 4931 | 328 | active + candidates update every step |
| `deep_imprint` | 4931 | 328 | active + candidates update every step |
| `deep_net2net` | 4931 | 328 | active + candidates update every step |
| `deep_net2net_fast` | 4931 | 328 | active + candidates update every step |
| `upgd` | 771 | 0 | active learner updates every step |

## `out_of_class_polynomial`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_net2net_final` | 1.1528 +/- 0.4709 | 1.1252 +/- 0.3215 |
| `deep_active_perturb_preserve` | 1.1569 +/- 0.4692 | 1.1335 +/- 0.3211 |
| `deep_net2net_fast` | 1.1574 +/- 0.4693 | 1.1318 +/- 0.3225 |
| `deep_net2net` | 1.1582 +/- 0.4645 | 1.1277 +/- 0.3191 |
| `mlp_64_64` | 1.1624 +/- 0.4737 | 1.1273 +/- 0.3191 |
| `deep_feature_lifecycle` | 1.1642 +/- 0.4712 | 1.1434 +/- 0.3212 |
| `deep_preserve_outgoing` | 1.1660 +/- 0.4600 | 1.1434 +/- 0.3237 |
| `deep_net2net_guarded` | 1.1688 +/- 0.4775 | 1.1294 +/- 0.3198 |
| `mlp_64` | 1.1933 +/- 0.4729 | 1.1566 +/- 0.3233 |
| `deep_imprint` | 1.1941 +/- 0.4918 | 1.1588 +/- 0.3318 |
| `upgd` | 1.6862 +/- 0.6922 | 1.6235 +/- 0.4628 |

Best fair MLP: `mlp_64_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64` | -0.0308 +/- 0.0076 | 0/3 |
| `upgd` | -0.5238 +/- 0.2186 | 0/3 |
| `deep_feature_lifecycle` | -0.0018 +/- 0.0033 | 1/3 |
| `deep_imprint` | -0.0317 +/- 0.0184 | 0/3 |
| `deep_preserve_outgoing` | -0.0036 +/- 0.0176 | 1/3 |
| `deep_active_perturb_preserve` | +0.0055 +/- 0.0057 | 2/3 |
| `deep_net2net` | +0.0043 +/- 0.0114 | 2/3 |
| `deep_net2net_guarded` | -0.0064 +/- 0.0068 | 1/3 |
| `deep_net2net_final` | +0.0096 +/- 0.0101 | 2/3 |
| `deep_net2net_fast` | +0.0051 +/- 0.0093 | 2/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=26.00, `deep_imprint`=24.00, `deep_preserve_outgoing`=23.67, `deep_active_perturb_preserve`=14.67, `deep_net2net`=15.00, `deep_net2net_guarded`=14.00, `deep_net2net_final`=13.33, `deep_net2net_fast`=29.33.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_net2net_final` | 4931 | 328 | active + candidates update every step |
| `deep_active_perturb_preserve` | 4931 | 328 | active + candidates update every step |
| `deep_net2net_fast` | 4931 | 328 | active + candidates update every step |
| `deep_net2net` | 4931 | 328 | active + candidates update every step |
| `mlp_64_64` | 4931 | 0 | active learner updates every step |
| `deep_feature_lifecycle` | 4931 | 328 | active + candidates update every step |
| `deep_preserve_outgoing` | 4931 | 328 | active + candidates update every step |
| `deep_net2net_guarded` | 4931 | 328 | active + candidates update every step |
| `mlp_64` | 771 | 0 | active learner updates every step |
| `deep_imprint` | 4931 | 328 | active + candidates update every step |
| `upgd` | 771 | 0 | active learner updates every step |

## `frequency_mismatch`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_net2net_final` | 1.1250 +/- 0.4097 | 1.6701 +/- 0.8801 |
| `deep_net2net` | 1.1378 +/- 0.4154 | 1.6823 +/- 0.8702 |
| `deep_net2net_guarded` | 1.1415 +/- 0.4209 | 1.6767 +/- 0.8781 |
| `deep_active_perturb_preserve` | 1.1430 +/- 0.4155 | 1.6796 +/- 0.8842 |
| `mlp_64_64` | 1.1439 +/- 0.4211 | 1.6903 +/- 0.8827 |
| `upgd` | 1.1469 +/- 0.4173 | 1.6635 +/- 0.8857 |
| `deep_net2net_fast` | 1.1544 +/- 0.4276 | 1.6810 +/- 0.8872 |
| `deep_preserve_outgoing` | 1.1616 +/- 0.4330 | 1.6973 +/- 0.9006 |
| `mlp_64` | 1.1628 +/- 0.4285 | 1.6645 +/- 0.8891 |
| `deep_imprint` | 1.1669 +/- 0.4302 | 1.6839 +/- 0.8873 |
| `deep_feature_lifecycle` | 1.1754 +/- 0.4363 | 1.7255 +/- 0.9145 |

Best fair MLP: `mlp_64_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64` | -0.0188 +/- 0.0075 | 0/3 |
| `upgd` | -0.0029 +/- 0.0126 | 1/3 |
| `deep_feature_lifecycle` | -0.0315 +/- 0.0202 | 1/3 |
| `deep_imprint` | -0.0230 +/- 0.0326 | 1/3 |
| `deep_preserve_outgoing` | -0.0177 +/- 0.0214 | 2/3 |
| `deep_active_perturb_preserve` | +0.0009 +/- 0.0056 | 2/3 |
| `deep_net2net` | +0.0061 +/- 0.0093 | 1/3 |
| `deep_net2net_guarded` | +0.0024 +/- 0.0017 | 2/3 |
| `deep_net2net_final` | +0.0190 +/- 0.0134 | 2/3 |
| `deep_net2net_fast` | -0.0104 +/- 0.0068 | 1/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=26.33, `deep_imprint`=25.67, `deep_preserve_outgoing`=21.33, `deep_active_perturb_preserve`=15.00, `deep_net2net`=14.67, `deep_net2net_guarded`=14.67, `deep_net2net_final`=13.00, `deep_net2net_fast`=27.67.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_net2net_final` | 4610 | 304 | active + candidates update every step |
| `deep_net2net` | 4610 | 304 | active + candidates update every step |
| `deep_net2net_guarded` | 4610 | 304 | active + candidates update every step |
| `deep_active_perturb_preserve` | 4610 | 304 | active + candidates update every step |
| `mlp_64_64` | 4610 | 0 | active learner updates every step |
| `upgd` | 450 | 0 | active learner updates every step |
| `deep_net2net_fast` | 4610 | 304 | active + candidates update every step |
| `deep_preserve_outgoing` | 4610 | 304 | active + candidates update every step |
| `mlp_64` | 450 | 0 | active learner updates every step |
| `deep_imprint` | 4610 | 304 | active + candidates update every step |
| `deep_feature_lifecycle` | 4610 | 304 | active + candidates update every step |

## `compositional`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `mlp_64` | 0.2471 +/- 0.0312 | 0.1892 +/- 0.0081 |
| `deep_net2net` | 0.2897 +/- 0.0439 | 0.2219 +/- 0.0189 |
| `deep_net2net_fast` | 0.2911 +/- 0.0333 | 0.2328 +/- 0.0133 |
| `deep_preserve_outgoing` | 0.2927 +/- 0.0542 | 0.2336 +/- 0.0228 |
| `deep_imprint` | 0.2946 +/- 0.0485 | 0.2333 +/- 0.0175 |
| `deep_net2net_final` | 0.2972 +/- 0.0583 | 0.2305 +/- 0.0186 |
| `deep_net2net_guarded` | 0.2990 +/- 0.0452 | 0.2272 +/- 0.0154 |
| `deep_feature_lifecycle` | 0.3043 +/- 0.0608 | 0.2407 +/- 0.0237 |
| `mlp_64_64` | 0.3052 +/- 0.0575 | 0.2373 +/- 0.0228 |
| `deep_active_perturb_preserve` | 0.3057 +/- 0.0466 | 0.2398 +/- 0.0197 |
| `upgd` | 0.7123 +/- 0.1482 | 0.4510 +/- 0.0527 |

Best fair MLP: `mlp_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64_64` | -0.0581 +/- 0.0276 | 0/3 |
| `upgd` | -0.4652 +/- 0.1179 | 0/3 |
| `deep_feature_lifecycle` | -0.0572 +/- 0.0317 | 0/3 |
| `deep_imprint` | -0.0475 +/- 0.0200 | 0/3 |
| `deep_preserve_outgoing` | -0.0457 +/- 0.0244 | 0/3 |
| `deep_active_perturb_preserve` | -0.0586 +/- 0.0166 | 0/3 |
| `deep_net2net` | -0.0426 +/- 0.0140 | 0/3 |
| `deep_net2net_guarded` | -0.0519 +/- 0.0140 | 0/3 |
| `deep_net2net_final` | -0.0501 +/- 0.0296 | 0/3 |
| `deep_net2net_fast` | -0.0440 +/- 0.0046 | 0/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=9.67, `deep_imprint`=14.67, `deep_preserve_outgoing`=7.00, `deep_active_perturb_preserve`=5.67, `deep_net2net`=6.00, `deep_net2net_guarded`=4.00, `deep_net2net_final`=4.33, `deep_net2net_fast`=12.00.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `mlp_64` | 643 | 0 | active learner updates every step |
| `deep_net2net` | 4803 | 320 | active + candidates update every step |
| `deep_net2net_fast` | 4803 | 320 | active + candidates update every step |
| `deep_preserve_outgoing` | 4803 | 320 | active + candidates update every step |
| `deep_imprint` | 4803 | 320 | active + candidates update every step |
| `deep_net2net_final` | 4803 | 320 | active + candidates update every step |
| `deep_net2net_guarded` | 4803 | 320 | active + candidates update every step |
| `deep_feature_lifecycle` | 4803 | 320 | active + candidates update every step |
| `mlp_64_64` | 4803 | 0 | active learner updates every step |
| `deep_active_perturb_preserve` | 4803 | 320 | active + candidates update every step |
| `upgd` | 643 | 0 | active learner updates every step |

## `digits_iid`

| Method | Final-window MSE | Total MSE |
|---|---:|---:|
| `deep_feature_lifecycle` | 0.0261 +/- 0.0008 | 0.0425 +/- 0.0009 |
| `deep_preserve_outgoing` | 0.0267 +/- 0.0010 | 0.0438 +/- 0.0006 |
| `mlp_64_64` | 0.0268 +/- 0.0011 | 0.0437 +/- 0.0007 |
| `deep_imprint` | 0.0269 +/- 0.0017 | 0.0430 +/- 0.0014 |
| `deep_net2net_final` | 0.0272 +/- 0.0005 | 0.0437 +/- 0.0007 |
| `mlp_64` | 0.0275 +/- 0.0006 | 0.0406 +/- 0.0003 |
| `deep_net2net_fast` | 0.0278 +/- 0.0022 | 0.0447 +/- 0.0013 |
| `deep_net2net` | 0.0281 +/- 0.0008 | 0.0444 +/- 0.0007 |
| `deep_net2net_guarded` | 0.0282 +/- 0.0008 | 0.0448 +/- 0.0009 |
| `deep_active_perturb_preserve` | 0.0286 +/- 0.0010 | 0.0454 +/- 0.0003 |
| `upgd` | 0.1416 +/- 0.0029 | 0.2290 +/- 0.0021 |

Best fair MLP: `mlp_64_64`.

| Method | best_mlp - method | Wins |
|---|---:|---:|
| `mlp_64` | -0.0007 +/- 0.0005 | 1/3 |
| `upgd` | -0.1148 +/- 0.0019 | 0/3 |
| `deep_feature_lifecycle` | +0.0007 +/- 0.0003 | 3/3 |
| `deep_imprint` | -0.0001 +/- 0.0006 | 2/3 |
| `deep_preserve_outgoing` | +0.0001 +/- 0.0004 | 2/3 |
| `deep_active_perturb_preserve` | -0.0018 +/- 0.0001 | 0/3 |
| `deep_net2net` | -0.0013 +/- 0.0006 | 0/3 |
| `deep_net2net_guarded` | -0.0014 +/- 0.0002 | 0/3 |
| `deep_net2net_final` | -0.0004 +/- 0.0006 | 1/3 |
| `deep_net2net_fast` | -0.0010 +/- 0.0015 | 1/3 |

Mean deep-feature promotions per run: `deep_feature_lifecycle`=0.00, `deep_imprint`=0.00, `deep_preserve_outgoing`=0.00, `deep_active_perturb_preserve`=0.00, `deep_net2net`=0.00, `deep_net2net_guarded`=0.00, `deep_net2net_final`=0.00, `deep_net2net_fast`=0.00.

| Method | Active params | Candidate params | Temporal uniformity |
|---|---:|---:|---|
| `deep_feature_lifecycle` | 8970 | 608 | active + candidates update every step |
| `deep_preserve_outgoing` | 8970 | 608 | active + candidates update every step |
| `mlp_64_64` | 8970 | 0 | active learner updates every step |
| `deep_imprint` | 8970 | 608 | active + candidates update every step |
| `deep_net2net_final` | 8970 | 608 | active + candidates update every step |
| `mlp_64` | 4810 | 0 | active learner updates every step |
| `deep_net2net_fast` | 8970 | 608 | active + candidates update every step |
| `deep_net2net` | 8970 | 608 | active + candidates update every step |
| `deep_net2net_guarded` | 8970 | 608 | active + candidates update every step |
| `deep_active_perturb_preserve` | 8970 | 608 | active + candidates update every step |
| `upgd` | 4810 | 0 | active learner updates every step |

## Verdict

The best native deep feature lifecycle variant was `deep_net2net_final`, which beat the best fair MLP on 3/6 streams. A single general deep feature-construction algorithm should be treated as a partial or negative Step 2 result unless a native variant wins robustly across the full matrix.
