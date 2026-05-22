# Step 3 Predictive-State Feature Attempt

Date: 2026-05-05

## Mechanisms Added

- `predictive_state_features_linear_gvf`: online causal lag/trace candidates,
  optional current-observation by predictive-state cross-products, selected by
  future-cumulant correlation utility on the discovery prefix, then frozen.
- `off_policy_mspbe_predictive_state_linear_td_is`: predictive-state candidates
  selected by a clipped-IS Bellman-residual gradient proxy
  `delta_t * (phi_t - gamma * phi_tp1)`, with a smaller TD step-size for the
  expanded feature set.
- `meta_gradient_proxy_interaction_features_linear_gvf`: observed interaction
  candidates selected by the same causal TD-loss-gradient proxy idea in the
  on-policy GVF harness. This fills the missing Veeriah-style comparison slot
  for DoD-7 as a scoped proxy, not as the full auxiliary-question method.

Both mechanisms are causal. Candidate rows at time `t` use only observations
through `t`; selections and feature scales are frozen after discovery warmup.
The off-policy evaluation rows reuse one shared sampled behavior-action,
reward, ratio, and target-return sequence across methods.

## Compact Probes

Commands:

```bash
source .venv/bin/activate && pytest tests/test_step3_feature_discovery_eval.py -q
source .venv/bin/activate && ruff check 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' tests/test_step3_feature_discovery_eval.py

source .venv/bin/activate && python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --output-dir outputs/step3_predictive_td_observable_ar1_3seed --seeds 3 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 --observation-dynamics ar1 --include-squares

source .venv/bin/activate && python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --output-dir outputs/step3_predictive_td_hidden_ar1_3seed --seeds 3 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 --observation-dynamics coupled_hidden_ar1 --include-squares
```

## Results

Target GVF RMSE, mean +/- stderr over 3 seeds:

| Probe | Method | RMSE |
| --- | --- | ---: |
| Observable AR(1) | step2_interaction_features_linear_gvf | 3.500076 +/- 0.475261 |
| Observable AR(1) | fixed_interaction_linear_gvf | 3.505439 +/- 0.515251 |
| Observable AR(1) | td_surprise_interaction_features_linear_gvf | 3.511584 +/- 0.504753 |
| Observable AR(1) | given_mlp_gvf | 3.588144 +/- 0.512277 |
| Observable AR(1) | predictive_state_features_linear_gvf | 3.802477 +/- 0.580926 |
| Coupled-hidden AR(1) | gvf_feedback_features_linear_gvf | 4.450676 +/- 0.931134 |
| Coupled-hidden AR(1) | predictive_state_features_linear_gvf | 4.479588 +/- 1.032519 |
| Coupled-hidden AR(1) | discovered_aux_cumulants_mlp_gvf | 4.502299 +/- 0.964522 |
| Coupled-hidden AR(1) | given_mlp_gvf | 4.503311 +/- 0.995058 |
| Coupled-hidden AR(1) | given_linear_gvf | 4.946301 +/- 1.208848 |

Off-policy behavior-mismatch RMSE from the coupled-hidden run:

| Method | RMSE |
| --- | ---: |
| off_policy_mspbe_predictive_state_linear_td_is | 1.191080 +/- 0.231761 |
| off_policy_raw_linear_td_is | 1.241050 +/- 0.215086 |
| off_policy_history_trace_linear_td_is | 2.016252 +/- 0.282430 |
| off_policy_raw_linear_td_no_is | 2.061659 +/- 0.594675 |
| off_policy_td_surprise_interaction_linear_td_is | 3.093539 +/- 1.029148 |

## Decision

Partial close.

- Closed on compact coupled-hidden AR(1): best discovery beats raw MLP
  (`4.450676` vs `4.503311`). The new predictive-state row also narrowly beats
  raw MLP (`4.479588` vs `4.503311`).
- Closed on compact off-policy behavior mismatch: the clipped-IS MSPBE
  predictive-state row beats raw clipped-IS TD (`1.191080` vs `1.241050`).
- Still open on robustness: both wins are small relative to stderr and need
  5-10 seed confirmation with longer discovery/eval horizons.
- Rejected as currently configured: TD-surprise interaction scoring remains bad
  for hidden-state and off-policy settings.
- Observable AR(1) remains best served by interaction discovery, not
  predictive-state features.

## Remaining Gaps

- Add a fair shared behavior-action sequence for all off-policy rows; current
  rows still sample each method separately.
- Replace the MSPBE proxy with true gradient-TD/GTD-style secondary weights if
  this direction survives longer runs.
- Tune predictive-state selection budget and trace timescales; fixed
  `n_features=24` may over-select noisy cross-products on observable AR(1).
- Confirm that the smaller off-policy feature step-size is not only delaying
  divergence by running longer behavior-mismatch horizons.
