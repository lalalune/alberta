# Worker P: TD/GVF Feature Discovery Stress

Date: 2026-05-05

## Scope

This pass stress-tested the current TD/GVF feature-discovery candidates on
observable Markov interaction prediction, coupled hidden-state prediction, and
behavior-mismatch off-policy TD. I used the existing Step 3 bridge harness rather
than adding new source code. The harness already includes:

- raw linear and raw MLP GVF baselines;
- fixed all-interaction and fixed history/trace positive controls;
- Step 2 tanh and Step 2 interaction feature learners frozen after discovery;
- TD-error-surprise interaction feature gating;
- predictive-state/history feature utility scoring;
- surprise-driven, random, and residual-proxy next-observation auxiliary cumulants;
- GVF-feedback features;
- off-policy raw TD, history features, TD-surprise interactions, and clipped-IS
  MSPBE predictive-state utility.

The residual-proxy auxiliary cumulant row is the available prediction-error
decorrelation proxy. Full whitening is not implemented in the current harness.

## Commands

```bash
source .venv/bin/activate
pytest tests/test_step3_feature_discovery_eval.py -q
```

Output: `19 passed in 4.93s`.

```bash
source .venv/bin/activate
python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --seeds 5 --discovery-steps 300 --eval-steps 700 --burn-in 75 --burn-tail 25 --n-features 16 --candidate-count 16 --n-aux-cumulants 4 --context-length 80 --replacement-interval 30 --min-feature-age 30 --candidate-min-age 15 --cumulant-maturity 50 --observation-dynamics ar1 --hide-last-channels 0 --include-squares --trace-sweep-lambdas 0.5 0.8 --output-dir outputs/step2_worker_p_tdgvf_ar1_5seed
```

```bash
source .venv/bin/activate
python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --seeds 5 --discovery-steps 300 --eval-steps 700 --burn-in 75 --burn-tail 25 --n-features 16 --candidate-count 16 --n-aux-cumulants 4 --context-length 80 --replacement-interval 30 --min-feature-age 30 --candidate-min-age 15 --cumulant-maturity 50 --observation-dynamics coupled_hidden_ar1 --hide-last-channels 2 --include-squares --trace-sweep-lambdas 0.5 0.8 --output-dir outputs/step2_worker_p_tdgvf_hidden_5seed
```

```bash
source .venv/bin/activate
python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --seeds 5 --discovery-steps 300 --eval-steps 700 --burn-in 75 --burn-tail 25 --n-features 16 --candidate-count 16 --n-aux-cumulants 4 --context-length 80 --replacement-interval 30 --min-feature-age 30 --candidate-min-age 15 --cumulant-maturity 50 --observation-dynamics ar1 --hide-last-channels 0 --include-squares --off-policy-scale 3.0 --trace-sweep-lambdas 0.5 0.8 --output-dir outputs/step2_worker_p_tdgvf_offpolicy_scale3_5seed
```

## Observable AR(1)

Output: `outputs/step2_worker_p_tdgvf_ar1_5seed`.

Mean target GVF RMSE over 5 seeds:

| Method | RMSE | Paired vs raw linear | Paired vs raw MLP |
|---|---:|---:|---:|
| fixed_interaction_linear_gvf | 2.890973 | +0.200870, 5/0/0 | +0.208960, 5/0/0 |
| td_surprise_interaction_features_linear_gvf | 2.929046 | +0.162797, 5/0/0 | +0.170887, 4/1/0 |
| step2_interaction_features_linear_gvf | 3.000575 | +0.091267, 5/0/0 | +0.099357, 4/1/0 |
| step2_tanh_features_linear_gvf | 3.047636 | +0.044207, 4/1/0 | +0.052297, 2/3/0 |
| given_linear_gvf | 3.091843 | baseline | -0.008090, 3/2/0 |
| given_mlp_gvf | 3.099933 | +0.008090, 2/3/0 | baseline |
| predictive_state_features_linear_gvf | 3.098305 | -0.006462, 1/4/0 | +0.001628, 2/3/0 |
| meta_proxy_aux_cumulants_mlp_gvf | 3.106889 | -0.015047, 2/3/0 | -0.006956, 2/3/0 |
| discovered_aux_cumulants_mlp_gvf | 3.142987 | -0.051144, 2/3/0 | -0.043054, 1/4/0 |
| gvf_feedback_features_linear_gvf | 3.228704 | -0.136861, 0/5/0 | -0.128771, 0/5/0 |

Interpretation: TD-surprise interactions remain the best learned discovery row
and beat raw linear and raw MLP on the observable AR(1) positive control. They
do not beat the fixed all-interaction control. That supports a narrow statement:
when the right interaction family is already in the candidate set and the state
is observable, TD-surprise can select useful GVF features.

## Coupled Hidden AR(1)

Output: `outputs/step2_worker_p_tdgvf_hidden_5seed`.

Mean target GVF RMSE over 5 seeds:

| Method | RMSE | Paired vs raw linear | Paired vs raw MLP |
|---|---:|---:|---:|
| gvf_feedback_features_linear_gvf | 5.387628 | +0.485883, 5/0/0 | +0.090277, 4/1/0 |
| meta_proxy_aux_cumulants_mlp_gvf | 5.463767 | +0.409743, 5/0/0 | +0.014138, 3/2/0 |
| given_mlp_gvf | 5.477905 | +0.395605, 5/0/0 | baseline |
| random_aux_cumulants_mlp_gvf | 5.525525 | +0.347986, 5/0/0 | -0.047620, 2/3/0 |
| discovered_aux_cumulants_mlp_gvf | 5.552309 | +0.321201, 5/0/0 | -0.074404, 1/4/0 |
| fixed_history_trace_linear_gvf | 5.691933 | +0.181578, 5/0/0 | -0.214028, 1/4/0 |
| td_surprise_interaction_features_linear_gvf | 5.801145 | +0.072365, 3/2/0 | -0.323240, 0/5/0 |
| fixed_interaction_linear_gvf | 5.818076 | +0.055435, 4/1/0 | -0.340170, 0/5/0 |
| given_linear_gvf | 5.873511 | baseline | -0.395605, 0/5/0 |
| predictive_state_features_linear_gvf | 5.899296 | -0.025785, 3/2/0 | -0.421391, 0/5/0 |
| step2_tanh_features_linear_gvf | 5.886955 | -0.013444, 3/2/0 | -0.409050, 0/5/0 |
| step2_interaction_features_linear_gvf | 5.950281 | -0.076770, 1/4/0 | -0.472376, 0/5/0 |

Interpretation: the best mean row is GVF-feedback, not a Step 2 feature learner,
and the run has very high seed variance. Step 2 tanh, Step 2 interactions,
TD-surprise interactions, fixed interactions, and predictive-state utility all
lose to raw MLP on every seed. The residual-proxy auxiliary row barely beats raw
MLP on mean and wins 3/5 seeds, so it is a weak Step 3 auxiliary-question signal,
not a hidden-state Step 2 solution.

## Off-Policy Probes

The off-policy probe is reported separately by the harness because it uses a
behavior-mismatch scalar TD target, not the main multi-head GVF target. Current
tooling has no raw MLP off-policy TD baseline; it uses `OffPolicyTDLinearLearner`.

Default behavior mismatch from the observable AR(1) run:

| Method | RMSE | Paired vs raw clipped-IS TD |
|---|---:|---:|
| off_policy_raw_linear_td_is | 0.964116 | baseline |
| off_policy_mspbe_predictive_state_linear_td_is | 0.980544 | -0.016428, 3/2/0 |
| off_policy_raw_linear_td_no_is | 1.434480 | -0.470364, 0/5/0 |
| off_policy_td_surprise_interaction_linear_td_is | 1.807059 | -0.842943, 0/5/0 |
| off_policy_history_trace_linear_td_is | 1.841027 | -0.876911, 0/5/0 |

Coupled-hidden behavior mismatch:

| Method | RMSE | Paired vs raw clipped-IS TD |
|---|---:|---:|
| off_policy_raw_linear_td_is | 1.302081 | baseline |
| off_policy_mspbe_predictive_state_linear_td_is | 1.379003 | -0.076922, 3/2/0 |
| off_policy_raw_linear_td_no_is | 1.997231 | -0.695150, 0/5/0 |
| off_policy_history_trace_linear_td_is | 2.023726 | -0.721645, 0/5/0 |
| off_policy_td_surprise_interaction_linear_td_is | 2.500706 | -1.198625, 0/5/0 |

Stronger behavior mismatch, `--off-policy-scale 3.0`:

| Method | RMSE | Paired vs raw clipped-IS TD |
|---|---:|---:|
| off_policy_raw_linear_td_is | 1.200612 | baseline |
| off_policy_mspbe_predictive_state_linear_td_is | 1.249900 | -0.049288, 1/4/0 |
| off_policy_td_surprise_interaction_linear_td_is | 1.673381 | -0.472769, 0/5/0 |
| off_policy_raw_linear_td_no_is | 1.851366 | -0.650754, 0/5/0 |
| off_policy_history_trace_linear_td_is | 1.855182 | -0.654570, 0/5/0 |

Interpretation: importance sampling is the core off-policy improvement. The
MSPBE predictive-state utility row is close in the default observable probe, but
it loses on mean in every off-policy stress and loses 4/5 seeds under stronger
behavior mismatch. TD-surprise interactions and fixed history features are
consistently worse than raw clipped-IS TD.

## Mechanism Assessment

- Eligibility-weighted feature utility: predictive-state utility does not help
  the main GVF rows, and clipped-IS MSPBE utility is close but not better than
  raw clipped-IS TD.
- TD-surprise gating: useful only for observable AR(1) interactions; it fails
  hidden-state and off-policy stress.
- Prediction-error decorrelation/whitening: the residual-proxy auxiliary row is
  weakly positive on hidden AR(1), but full whitening is not available and the
  result is too small to promote.
- Auxiliary next-observation predictions: surprise-discovered and random
  projections do not consistently beat raw MLP. Random auxiliaries being close
  also weakens a discovery-specific interpretation.
- Route-safe inclusion: current Step 2 portfolio tooling exposes safe routes for
  `recursive_features`, `polynomial_features`, `fourier_features`, and
  `tanh_random_features`. It does not support TD/GVF discovery candidates or the
  off-policy TD probe as route sources without source work and a common target
  interface. I did not add route-safe portfolio inclusion.

## Conclusion

Do not claim Step 2 solves TD/GVF feature discovery. The observable AR(1)
positive control is still real: TD-surprise interactions beat raw linear and raw
MLP, and Step 2 interactions also help there. Hidden-state stress shifts the
best signal to GVF-feedback or residual auxiliary questions, while native Step 2
feature learners lose to raw MLP. Off-policy stress is dominated by raw clipped
importance-sampling TD; feature additions do not improve it consistently.

The defensible placement is Step 3: TD/GVF feature discovery has a narrow
observable-state positive control, but hidden-state recovery, off-policy feature
finding, and robust auxiliary-question discovery remain open.

## Files Changed

- Added `docs/research/step2_worker_p_tdgfv_feature_discovery.md`.
- Added output directories:
  - `outputs/step2_worker_p_tdgvf_ar1_5seed/`
  - `outputs/step2_worker_p_tdgvf_hidden_5seed/`
  - `outputs/step2_worker_p_tdgvf_offpolicy_scale3_5seed/`

No source files were edited.
