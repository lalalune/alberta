# Step 2 Future-Utility Closure

Date: May 5, 2026.

## Scope

This note audits and stress-tests the Step 2 future-utility estimator used by
fixed-budget and compositional feature discovery. The standard is not "can we
find one positive seed," but whether the signal is causal, temporally uniform,
and strong enough to support an unqualified Step 2 feature-discovery claim.

## Estimator Audit

- Causality: predictions are computed before the current update; utility uses
  only the current observation, current observed targets, previous weights, and
  previous traces. No future targets, future observations, heldout labels, or
  post-update predictions enter the utility signal.
- Temporal uniformity: active features, candidate features, utility traces,
  second moments, ages, and task activity estimates are updated every time step.
  Replacement/promotion remains event-driven, but the statistics that decide it
  are maintained uniformly.
- Leakage: the estimator is not a hindsight ablation. Candidate output weights
  and optional candidate residual imprinting use the current residual only.
  This is causal but can still be optimistic on the current sample, so heldout
  metrics are reported separately.
- Delayed usefulness: the prior traced estimator multiplied separate residual
  and feature traces. That is causal, but it can credit marginal correlation
  rather than actual delayed contribution. The new `contribution` trace directly
  tracks `error_i * feature_j`, making the delayed credit a TD(lambda)-style
  eligibility trace of the output-weight contribution.
- Rare tasks: rare-task credit is implemented as causal inverse-frequency
  weighting from an online task-activity EMA. It helps characterize masked-head
  streams but is not yet reliable enough to be a default.

## Variants Added

- `contribution_trace_output_loss_reduction`: TD(lambda)-style trace of
  `error * feature`, exactly matching the one-step LMS counterfactual when
  trace decay is zero.
- `trace_decay_from_half_life`: half-life sweep helper.
- `normalize_future_utility_signal`: optional `age`, `uncertainty`, and
  `uncertainty_age` normalization.
- Integration knobs in `FixedBudgetFeatureLearner` and
  `CompositionalFeatureLearner`: trace decay, trace mode, normalization,
  normalization decay, rare-task power, and task-activity decay where needed.
- The older marginal trace remains available as `future_utility_trace_mode="marginal"`
  for ablation and compatibility.

## Commands

```bash
source .venv/bin/activate
pytest tests/test_future_utility.py tests/test_feature_discovery.py::test_trace_output_loss_reduction_matches_one_step_at_zero_decay tests/test_compositional_features.py::TestCompositionalFeatureLearner::test_trace_future_utility_credits_recursive_candidate_alignment -q
python "examples/The Alberta Plan/Step2/step2_recursive_feature_utility_probe.py" --smoke --output-dir outputs/step2_future_utility_smoke
python "examples/The Alberta Plan/Step2/step2_recursive_feature_utility_probe.py" --seeds 5 --num-steps 2000 --final-window 400 --methods current,one_step,contrib_h4,contrib_h16,contrib_h64,marginal_h16,two_timescale_h16,uncertainty_age_h16 --output-dir outputs/step2_future_utility_triple_5seed
python "examples/The Alberta Plan/Step2/step2_recursive_feature_utility_probe.py" --seeds 5 --num-steps 2000 --final-window 400 --task-mode rare --methods current,one_step,contrib_h16,uncertainty_age_h16,rare_credit_h16 --output-dir outputs/step2_future_utility_rare_5seed
pytest tests/test_future_utility.py tests/test_feature_discovery.py tests/test_compositional_features.py -q
ruff check src/alberta_framework/core/future_utility.py src/alberta_framework/core/feature_discovery.py src/alberta_framework/core/compositional_features.py tests/test_future_utility.py "examples/The Alberta Plan/Step2/step2_recursive_feature_utility_probe.py"
```

## Triple-Product Probe

Output: `outputs/step2_future_utility_triple_5seed/recursive_feature_utility_results.json`

Five seeds, 2,000 online steps, final window 400. Lower is better.

| Method | Final-window MSE | Heldout MSE | Depth>=2 active | Paired note |
|---|---:|---:|---:|---|
| `current` | 1.1433 +/- 0.3151 | 1.2291 +/- 0.1622 | 11.0 | baseline |
| `one_step` | 1.1707 +/- 0.0610 | 1.0290 +/- 0.1803 | 11.4 | baseline future utility |
| `contrib_h4` | 1.1639 +/- 0.2222 | 1.0217 +/- 0.2848 | 10.4 | weak |
| `contrib_h16` | 0.6118 +/- 0.1850 | 0.7305 +/- 0.3199 | 11.0 | beats one-step 4/5 |
| `contrib_h64` | 0.6600 +/- 0.3622 | 0.5675 +/- 0.2906 | 12.0 | strong but high variance |
| `marginal_h16` | 1.3663 +/- 0.1880 | 1.6104 +/- 0.2367 | 12.0 | negative |
| `two_timescale_h16` | 0.7866 +/- 0.3501 | 0.9573 +/- 0.2505 | 11.0 | partial |
| `uncertainty_age_h16` | 1.0355 +/- 0.2627 | 1.1198 +/- 0.1813 | 13.2 | weak |

The direct contribution trace is the only clearly positive family. The older
marginal trace is worse than both current utility and one-step utility on this
probe, which argues against using it as evidence for delayed usefulness.

## Rare-Head Probe

Output: `outputs/step2_future_utility_rare_5seed/recursive_feature_utility_results.json`

Two heads: a frequent linear head and a rare masked triple-product head
available every 8th sample. Lower is better.

| Method | Final-window MSE | Heldout mean MSE | Rare-head heldout MSE | Depth>=2 active |
|---|---:|---:|---:|---:|
| `current` | 0.1521 +/- 0.0607 | 0.6951 +/- 0.1276 | 1.2273 | 10.2 |
| `one_step` | 0.0869 +/- 0.0206 | 0.5899 +/- 0.2576 | 1.1322 | 11.0 |
| `contrib_h16` | 0.0512 +/- 0.0052 | 0.4813 +/- 0.0885 | 0.9413 | 10.8 |
| `uncertainty_age_h16` | 0.0548 +/- 0.0144 | 0.5951 +/- 0.1472 | 1.1810 | 12.8 |
| `rare_credit_h16` | 0.0645 +/- 0.0223 | 0.4657 +/- 0.1258 | 0.9132 | 9.6 |

Rare-task-aware credit improves heldout rare-head MSE relative to current and
one-step, but it does not dominate the simpler contribution trace on online
final-window loss. This is useful evidence for a knob, not a default.

## Conclusion

Status: partial, not solved.

The future-utility gap is now sharply characterized. A direct
TD(lambda)-style contribution trace is causal, temporally uniform, and gives
positive 5-seed evidence on recursive compositional and rare-head probes. It
also exposes that the older marginal trace was not a convincing delayed-credit
estimator despite being causal.

This does not support an unqualified Step 2 claim. The best variants are still
high variance, the evidence is from focused probes rather than broad external
benchmarks, and no variant cleanly dominates across online and heldout metrics.
For canonical Step 2 integration, keep future utility as an experimental
configuration path. Use `future_utility_trace_mode="contribution"` with a
half-life sweep around 16-64 steps when studying recursive feature discovery,
and keep `rare_credit_h16` as a rare-head ablation rather than a default.
