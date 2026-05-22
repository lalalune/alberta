# Step 2 Lead Follow-up: Learned Scalars, Minimal Telemetry, Regression Prototypes

Date: 2026-05-06.

This follow-up tests the three best next leads after the centroid-only
hysteretic Step 2 result:

1. learn the centroid temperature and/or mixture scalar;
2. promote minimal telemetry for compute/resource efficiency;
3. build a regression analogue of class centroids.

## 1. Learned Centroid Scalars

Artifacts:

- `output/subagents/centroid_learnable_worker_b/initial_5seed_1200/`
- `output/subagents/centroid_learnable_worker_b/refined_5seed_1200/`
- `output/subagents/centroid_learnable_worker_b/combo_only_5seed_1200/`

Result: useful diagnostic, not a promotion.

The best class-blocked learned scalar is capped mix learning:

| Method | Class-blocked final MSE | Diff vs fixed | Label-drift final MSE | Diff vs fixed |
|---|---:|---:|---:|---:|
| `learnmix_eta001_cap035_center_c030` | `0.001365 +/- 0.000032` | `+0.000147`, `2/5` wins | `0.035502 +/- 0.002648` | `-0.001001`, `1/5` wins |
| `learnmix_eta002_cap035_center_c030` | `0.001414 +/- 0.000050` | `+0.000098`, `2/5` wins | `0.036135 +/- 0.002672` | `-0.001634`, `2/5` wins |
| fixed `centhystbest64_center_c030_off005` | `0.001512 +/- 0.000130` | baseline | `0.034501 +/- 0.001768` | baseline |

The best label-drift learned scalar is the combined mix+temperature rule, but
it gives back class-blocked tracking:

| Method | Class-blocked diff vs fixed | Label-drift diff vs fixed | Decision |
|---|---:|---:|---|
| `learnmix_temp_eta001_cap035_center_c030` | `-0.000070 +/- 0.000043`, `0/5` wins | `+0.000278 +/- 0.001892`, `3/5` wins | Reject as default. |
| `learntemp_eta001_center_c030` | `-0.000031 +/- 0.000175`, `1/5` wins | `+0.000101 +/- 0.001600`, `3/5` wins | Too small/noisy. |

Interpretation: the scalar utility signal is pointed in the right direction,
but it is not strong enough to replace the fixed scalar defaults. Capped mix is
a class-blocked-only lead; learned temperature or mix+temperature is a
label-drift-only lead. None beats the fixed scalar gate on both blocker regimes.

## 2. Minimal Telemetry

Artifacts:

- `output/subagents/classblocked_prototype_features/centroid_hysteretic64_telemetry_compare_class_keyed_5seed_1200/`
- `output/subagents/classblocked_prototype_features/centroid_hysteretic64_telemetry_compare_label_keyed_5seed_1200/`

Implementation:

- Added `telemetry` to `ProtoConfig`.
- Added `key_name` to `ProtoConfig` so full and minimal variants can share the
  exact same learner initialization.
- Added preset `centroid_hysteretic64_telemetry_compare`.

Keyed full-vs-minimal comparison:

| Regime | Full final MSE | Minimal final MSE | Full elapsed | Minimal elapsed |
|---|---:|---:|---:|---:|
| `class_blocked` | `0.001512 +/- 0.000130` | `0.001512 +/- 0.000130` | `2.638s` | `2.713s` |
| `label_drift` | `0.034501 +/- 0.001768` | `0.034501 +/- 0.001768` | `2.972s` | `2.804s` |

Decision: promote minimal telemetry for production/evidence runs when gate and
advantage traces are not needed. It is behavior-identical under shared
initialization. The timing read is mixed and too noisy to claim a robust speed
win; the solid win is lower metric bandwidth and smaller artifacts.

## 3. Regression Prototype Analogue

Artifact:

- `output/subagents/regression_prototype_features/run_regression_prototypes.py`
- `output/subagents/regression_prototype_features/regression_proto_keyed_5seed_1200/`

Mechanism:

- keep a fixed-size bank of input prototypes and associated target-vector
  prototypes;
- predict from RBF similarities to slots as a local target chart;
- mix the local chart with the base UPGD prediction through a scalar
  hysteretic utility gate;
- update only after seeing the current target;
- allocate empty slots first and recycle the lowest-utility slot on novelty.

This is the regression analogue of class centroids: target-conditioned local
charts instead of one class prototype per target head.

Keyed 5-seed results:

| Dataset | Best prototype | Final MSE | Diff vs same-key UPGD | Test MSE | Test diff vs same-key UPGD |
|---|---|---:|---:|---:|---:|
| `diabetes` | `proto_reg_k32_m020_bw08_n12` | `0.477603 +/- 0.010552` | `+0.028963 +/- 0.008221`, `5/5` wins | `0.648629` | `+0.067903 +/- 0.020881`, `5/5` wins |
| `friedman1` | `proto_reg_k16_m030_bw10_n15` | `0.213531 +/- 0.011010` | `+0.000000`, `0/5` wins | `0.195169` | `+0.000000`, `0/5` wins |

Decision: this is a real positive result for diabetes regression and closes the
"classification-only prototype" hole partially. It is not universal yet:
on Friedman1 the utility gate turns the prototype branch off and the method
falls back to UPGD. That is acceptable as a first regression analogue because
it avoids harm, but the next target is making local charts useful on synthetic
nonlinear regression without overfitting tabular noise.

## Net Decision

Promote:

- fixed scalar `centhystbest64_center_c030_off005` remains the main
  classification candidate;
- minimal telemetry is safe when diagnostics are not needed;
- regression local target prototypes are a new positive branch for non-class
  Step 2 tasks.

Do not promote:

- learned scalar temperature/mix as default;
- per-class gates;
- regression prototypes as a universal replacement.

Next high-value work:

- learn the regression prototype bandwidth/mixture from utility;
- add a no-harm gate margin so regression prototypes shut off faster on
  Friedman-like streams;
- move the regression prototype into `src/alberta_framework/core/` with tests
  only after one more dataset sweep confirms the diabetes result.
