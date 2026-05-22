# Step 3 TD/GVF Hard Feature Discovery Closure

Date: 2026-05-04

Update: 2026-05-05

## Scope

This pass audited and extended `examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py` for the TD/GVF hard blocker: hidden-state, off-policy, and general cumulant feature discovery. The harness remains a research harness, not a closure proof for Step 3 feature discovery.

## Harness Changes

- Added a coupled hidden AR(1) collector. The returned observation masks the hidden channels, while targets are computed from the unmasked state. Hidden variables are partly driven by visible history, so causal history/trace features have a positive-control signal.
- Added fixed causal history/trace features: observation lags plus EWMA traces. Row `t` only depends on observations through `t`.
- Added a separate behavior-mismatch off-policy TD probe using `OffPolicyTDLinearLearner` with Retrace clipping. These rows are reported separately from the main GVF aggregate because the target is a different expected target-policy return.
- Kept existing learned Step 2 tanh and interaction features, surprise-driven auxiliary cumulants, meta-proxy auxiliary projections, GVF feedback features, and trace-decay sweep rows.
- Added tests for target causality, hidden feature shape/masking, off-policy reporting separation, and no leakage in history features.
- Added an online TD-error surprise scorer over interaction candidates. A raw linear source Horde updates each discovery step; before each update, one-step TD errors weight causal candidate eligibility traces. The top-scored interactions are frozen for downstream GVF evaluation. The same scorer also accepts clipped importance ratios for an off-policy-aware behavior-mismatch feature row.
- Added `meta_gradient_proxy_interaction_features_linear_gvf`, a scoped Veeriah-style proxy over the same interaction candidate track. It scores each candidate before the source-Horde update by the correlation between TD error and Bellman feature difference `phi_t - gamma * phi_{t+1}`, freezes the selected columns after warmup, and reports it separately from the existing residual-projection `meta_proxy_aux_cumulants_mlp_gvf`.

## Candidate-Track Inventory

The current DoD-7 harness covers these candidate tracks:

- Random auxiliary cumulants: frozen random projections of `obs_{t+1}`.
- Surprise-driven auxiliary cumulants: `CumulantDiscovery` replacement over projection cumulants.
- Residual projection proxy: `meta_proxy_aux_cumulants_mlp_gvf`, a linear residual heuristic over `obs_{t+1}` projections.
- Fixed interaction positive control: all observed pair/square products.
- Step 2 learned features: tanh feature learner and interaction feature learner warmed on the discovery prefix.
- TD-surprise interaction selection: causal TD-error weighted interaction candidates.
- Veeriah-style meta-gradient proxy: causal TD-loss-gradient interaction candidates.
- Predictive-state selection: causal lag/trace candidates and optional current-by-history products.
- GVF feedback features: source-Horde predictions computed before the current transition update.
- Off-policy rows: shared evaluation behavior-action sequence with clipped IS; TD-surprise interaction and MSPBE predictive-state feature proxies.

## Audit Notes

- The original iid stream is not a valid downstream nexting feature-discovery positive control: `target_{t+1}` is independent of `obs_t`, so the downstream GVF cannot fairly benefit from constructed state features.
- Observable AR(1) fixes that narrow issue, but it is still a synthetic positive control.
- Masking hidden AR(1) channels does not by itself make hidden state recoverable. If hidden channels are independent of visible channels, history features cannot infer them. The new coupled-hidden stream is explicitly a POMDP-style positive control, not evidence that hidden-state discovery is solved generally.
- The Horde path is on-policy prediction. The new off-policy probe is intentionally separate and linear; it tests behavior/target mismatch plumbing, not nonlinear off-policy GVF feature discovery.
- Surprise/meta auxiliary cumulants are simple projection mechanisms over `obs_{t+1}`. The new meta-gradient proxy is closer to a feature-selection loss-gradient arm, but it still is not Veeriah et al.'s full auxiliary-question meta-gradient algorithm. It does not establish general cumulant discovery.

## Commands

```bash
source .venv/bin/activate && pytest tests/test_step3_feature_discovery_eval.py -q
source .venv/bin/activate && python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --quick --seeds 2 --observation-dynamics ar1 --hide-last-channels 0 --include-squares --trace-sweep-lambdas 0.5 0.8 --output-dir outputs/step3_tdgvf_hard_ar1_2seed
source .venv/bin/activate && python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --quick --seeds 2 --observation-dynamics coupled_hidden_ar1 --include-squares --trace-sweep-lambdas 0.5 0.8 --output-dir outputs/step3_tdgvf_hard_hidden_2seed
source .venv/bin/activate && python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --seeds 3 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 --n-features 12 --candidate-count 12 --n-aux-cumulants 4 --context-length 80 --replacement-interval 25 --min-feature-age 25 --candidate-min-age 10 --cumulant-maturity 40 --observation-dynamics ar1 --hide-last-channels 0 --include-squares --trace-sweep-lambdas 0.5 0.8 --output-dir outputs/step3_tdgvf_hard_ar1_3seed_td_surprise
source .venv/bin/activate && python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --seeds 3 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 --n-features 12 --candidate-count 12 --n-aux-cumulants 4 --context-length 80 --replacement-interval 25 --min-feature-age 25 --candidate-min-age 10 --cumulant-maturity 40 --observation-dynamics coupled_hidden_ar1 --include-squares --trace-sweep-lambdas 0.5 0.8 --output-dir outputs/step3_tdgvf_hard_hidden_3seed_td_surprise
source .venv/bin/activate && pytest tests/ -v
source .venv/bin/activate && ruff check .
source .venv/bin/activate && ruff check 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' tests/test_step3_feature_discovery_eval.py
source .venv/bin/activate && mypy
source .venv/bin/activate && mypy 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' tests/test_step3_feature_discovery_eval.py
```

`pytest tests/ -v` passed: 969 passed, 36 warnings. The bare `ruff check .` failed on an out-of-scope import-order issue in `src/alberta_framework/core/optimizers.py`. Scoped Ruff passed for the touched files. The bare `mypy` command has no configured target and exits with a usage error; scoped mypy passed for the touched files.

The 2026-05-05 checks:

```bash
source .venv/bin/activate && pytest tests/test_step3_feature_discovery_eval.py -q
source .venv/bin/activate && ruff check 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' tests/test_step3_feature_discovery_eval.py
source .venv/bin/activate && mypy 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' tests/test_step3_feature_discovery_eval.py
source .venv/bin/activate && pytest tests/ -v
source .venv/bin/activate && ruff check .
source .venv/bin/activate && mypy
```

Result: scoped pytest 15 passed; scoped Ruff passed; scoped mypy passed; full pytest 987 passed with 36 warnings; full Ruff passed. Bare `mypy` exits with a usage error because no target is configured.

## TD-Surprise Follow-Up: Observable AR(1)

Output: `outputs/step3_tdgvf_hard_ar1_3seed_td_surprise`

Mean target GVF RMSE over 3 seeds:

- `td_surprise_interaction_features_linear_gvf`: 3.920609
- `fixed_interaction_linear_gvf`: 3.939980
- `random_aux_cumulants_mlp_gvf`: 4.085782
- `meta_proxy_aux_cumulants_mlp_gvf`: 4.112212
- `given_mlp_gvf`: 4.142325
- `step2_interaction_features_linear_gvf`: 4.182506
- `discovered_aux_cumulants_mlp_gvf`: 4.183993
- `fixed_history_trace_linear_gvf`: 4.195837
- `given_linear_gvf`: 4.226761
- `step2_tanh_features_linear_gvf`: 4.271374

Summary: the TD-error surprise interaction scorer is a narrow positive-control win on observable AR(1). It beats raw linear and raw MLP and slightly beats the fixed all-interactions baseline in this compact run. The margin over fixed interactions is small, so the defensible claim is only that TD/GVF surprise can find useful interaction features when the relevant interaction family is already in the candidate set and the state is fully observable.

Off-policy behavior-mismatch probe:

- `off_policy_raw_linear_td_is`: 1.348565
- `off_policy_raw_linear_td_no_is`: 2.184216
- `off_policy_td_surprise_interaction_linear_td_is`: 2.630802
- `off_policy_history_trace_linear_td_is`: 3.305883

Importance sampling helps the raw off-policy probe. IS-weighted TD-surprise features do not help the off-policy probe.

## TD-Surprise Follow-Up: Coupled Hidden AR(1)

Output: `outputs/step3_tdgvf_hard_hidden_3seed_td_surprise`

Mean target GVF RMSE over 3 seeds:

- `given_mlp_gvf`: 5.040781
- `gvf_feedback_features_linear_gvf`: 5.083463
- `meta_proxy_aux_cumulants_mlp_gvf`: 5.084105
- `discovered_aux_cumulants_mlp_gvf`: 5.089088
- `random_aux_cumulants_mlp_gvf`: 5.122952
- `fixed_history_trace_linear_gvf`: 5.164499
- `fixed_interaction_linear_gvf`: 5.497355
- `td_surprise_interaction_features_linear_gvf`: 5.499763
- `given_linear_gvf`: 5.527150
- `step2_tanh_features_linear_gvf`: 5.532811
- `step2_interaction_features_linear_gvf`: 5.613613

Summary: raw MLP remains best. GVF feedback is the best discovery method and beats raw linear but not raw MLP. TD-surprise interactions do not solve hidden-state recovery.

Off-policy behavior-mismatch probe:

- `off_policy_raw_linear_td_is`: 1.582402
- `off_policy_td_surprise_interaction_linear_td_is`: 2.221343
- `off_policy_history_trace_linear_td_is`: 2.797690
- `off_policy_raw_linear_td_no_is`: 2.863259

Importance sampling again helps. IS-weighted TD-surprise features are better than no-IS and history features here, but still worse than raw IS.

## Observable AR(1) Positive Control

Output: `outputs/step3_tdgvf_hard_ar1_2seed`

Mean target GVF RMSE over 2 seeds:

- `fixed_interaction_linear_gvf`: 4.124688
- `random_aux_cumulants_mlp_gvf`: 4.278691
- `given_mlp_gvf`: 4.330915
- `discovered_aux_cumulants_mlp_gvf`: 4.361381
- `meta_proxy_aux_cumulants_mlp_gvf`: 4.373641
- `gvf_feedback_features_linear_gvf`: 4.427645
- `fixed_history_trace_linear_gvf`: 4.451042
- `given_linear_gvf`: 4.484863

Summary: fixed interactions remain the clean positive control. The best learned discovery method beats raw linear but not raw MLP. Random auxiliary cumulants also did well, which weakens any claim that the surprise-driven projections discovered task-relevant cumulants.

Off-policy behavior-mismatch probe:

- `off_policy_raw_linear_td_is`: 1.146506
- `off_policy_raw_linear_td_no_is`: 2.231538
- `off_policy_history_trace_linear_td_is`: 3.126888

Importance sampling helps the raw off-policy probe. History features did not help this off-policy probe.

## Coupled Hidden AR(1) / POMDP Probe

Output: `outputs/step3_tdgvf_hard_hidden_2seed`

Mean target GVF RMSE over 2 seeds:

- `gvf_feedback_features_linear_gvf`: 4.171705
- `fixed_history_trace_linear_gvf`: 4.290813
- `given_mlp_gvf`: 4.304076
- `meta_proxy_aux_cumulants_mlp_gvf`: 4.343584
- `discovered_aux_cumulants_mlp_gvf`: 4.354474
- `random_aux_cumulants_mlp_gvf`: 4.427686
- `fixed_interaction_linear_gvf`: 4.662113
- `given_linear_gvf`: 4.734984

Summary: GVF feedback and fixed history/trace features show small positive-control movement on the synthetic coupled-hidden stream. Surprise/meta auxiliary cumulants still do not beat raw MLP. This is a narrow two-seed signal, not a hidden-state discovery solution.

Off-policy behavior-mismatch probe:

- `off_policy_raw_linear_td_is`: 1.323257
- `off_policy_raw_linear_td_no_is`: 2.041910
- `off_policy_history_trace_linear_td_is`: 2.676756

Importance sampling again helps the raw off-policy probe. History features again hurt in this simple off-policy setup.

## Direct Conclusion

TD/GVF feature discovery remains Step 3 research. A narrow subset is defensibly closed: on the fully observable AR(1) interaction positive control, an online TD-error surprise scorer can find useful candidate interactions and beat raw linear/raw MLP in the compact 3-seed run. That does not close hidden-state discovery, off-policy feature finding, or general cumulant discovery. Coupled hidden AR(1) still favors raw MLP, and the off-policy behavior-mismatch probe still favors raw linear TD with clipped importance sampling over feature-augmented variants.

## Production Readiness Note

`src/alberta_framework/steps/step3.py` now packages only the safe part of this
evidence: given-feature Horde construction, the Step 2 constructed-feature array
handoff, and a finite smoke run. This helper makes downstream integration less
dependent on experiment scripts, but it does not change the research conclusion
above. General TD/GVF feature discovery, hidden-state construction, and
nonlinear off-policy discovery remain candidate research tracks rather than
closed Step 3 capabilities.
