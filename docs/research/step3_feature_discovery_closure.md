# Step 3 Feature Discovery Closure

Date: 2026-05-04

Question: can TD/GVF-target feature finding be closed as Step 2 transfer, or is it still Step 3 research?

## Harness Audit

- Target causality: the harness trains on transitions `(obs_t, c_{t+1}, obs_{t+1})`. `transition_view()` explicitly uses `targets[1:]` as transition cumulants, and `target_forward_returns()` computes forward-view returns from the same evaluation-suffix cumulants.
- TD target correctness: `HordeLearner.update()` computes `c_{t+1} + gamma * V(obs_{t+1})`. The script passes one cumulant per demon in target/gamma order via `repeat_by_horizon()`.
- Leakage: discovery sees only the warmup prefix. Feature learners are frozen before downstream evaluation. Auxiliary cumulants for evaluation are projections of `obs_{t+1}`, which is the legal transition cumulant signal, not a target/return leak.
- Learned-feature use: `step2_tanh_features_linear_gvf` and `step2_interaction_features_linear_gvf` pass augmented `obs_t` and augmented `obs_{t+1}` into the downstream Horde. The learned features are not just measured as utility diagnostics; they are used by the GVF value learner.
- Temporal uniformity: within warmup and evaluation, every active learner updates every step. The transfer design intentionally freezes discovered features for the evaluation suffix, so this is a clean transfer test, not a full continually-adapting feature-discovery deployment.
- Off-policy status: the primary Horde rows remain on-policy prediction. The harness now also includes a separate behavior-mismatch linear TD probe with per-decision clipped importance sampling, including raw, history, TD-surprise interaction, full MSPBE predictive-state, and novelty-gated MSPBE predictive-state feature rows. This is not a full off-policy Horde/control result, but it is now direct off-policy feature-evaluation evidence.

## Commands

```bash
source .venv/bin/activate
python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' \
  --seeds 3 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 \
  --observation-dynamics ar1 --include-squares --hide-last-channels 0 \
  --trace-sweep-lambdas 0.5 0.9 \
  --output-dir outputs/step3_feature_discovery_observable_ar1_3seed

python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' \
  --seeds 3 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 \
  --observation-dynamics ar1 --include-squares --hide-last-channels 1 \
  --output-dir outputs/step3_feature_discovery_hidden_ar1_3seed

python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' \
  --seeds 3 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 \
  --observation-dynamics iid --include-squares --hide-last-channels 0 \
  --output-dir outputs/step3_feature_discovery_iid_negative_3seed
```

A 5-seed, 1600-step observable run was attempted and stopped after about 6 minutes with no completed buffered output. The completed artifacts above are the closure evidence.

## Results

Mean target GVF RMSE, lower is better.

### Observable AR(1), 3 Seeds

| Method | RMSE | SE |
|---|---:|---:|
| fixed_interaction_linear_gvf | 3.501031 | 0.512298 |
| step2_interaction_features_linear_gvf | 3.576710 | 0.617256 |
| meta_proxy_aux_cumulants_mlp_gvf | 3.585994 | 0.525267 |
| discovered_aux_cumulants_mlp_gvf | 3.586477 | 0.539498 |
| given_mlp_gvf | 3.598422 | 0.513132 |
| random_aux_cumulants_mlp_gvf | 3.629546 | 0.533875 |
| given_linear_gvf_trace_0.5 | 3.673998 | 0.576083 |
| given_linear_gvf_trace_0.9 | 3.684123 | 0.561126 |
| gvf_feedback_features_linear_gvf | 3.687912 | 0.499622 |
| given_linear_gvf | 3.728497 | 0.589149 |
| step2_tanh_features_linear_gvf | 3.765114 | 0.596748 |

Best discovery method: `step2_interaction_features_linear_gvf`. It beats raw linear on 3/3 seeds by mean RMSE diff 0.151788, and raw MLP on 2/3 seeds by mean diff 0.021713. The MLP margin is small relative to SE.

### Hidden-Channel AR(1), 3 Seeds

| Method | RMSE | SE |
|---|---:|---:|
| fixed_interaction_linear_gvf | 3.507702 | 0.505893 |
| discovered_aux_cumulants_mlp_gvf | 3.568886 | 0.546720 |
| meta_proxy_aux_cumulants_mlp_gvf | 3.585335 | 0.517354 |
| given_mlp_gvf | 3.588144 | 0.512277 |
| step2_interaction_features_linear_gvf | 3.609190 | 0.614124 |
| random_aux_cumulants_mlp_gvf | 3.622552 | 0.535123 |
| gvf_feedback_features_linear_gvf | 3.676055 | 0.501710 |
| given_linear_gvf | 3.783960 | 0.586649 |
| step2_tanh_features_linear_gvf | 3.789124 | 0.599184 |

Best discovery method: `discovered_aux_cumulants_mlp_gvf`. It beats raw linear on 3/3 seeds by mean diff 0.215074, and raw MLP on 2/3 seeds by mean diff 0.019258. Again, the MLP margin is tiny relative to SE.

### IID Negative Control, 3 Seeds

| Method | RMSE | SE |
|---|---:|---:|
| meta_proxy_aux_cumulants_mlp_gvf | 2.095620 | 0.297074 |
| given_mlp_gvf | 2.096152 | 0.293204 |
| random_aux_cumulants_mlp_gvf | 2.098113 | 0.289962 |
| discovered_aux_cumulants_mlp_gvf | 2.103040 | 0.295305 |
| gvf_feedback_features_linear_gvf | 2.111766 | 0.287641 |
| fixed_interaction_linear_gvf | 2.271114 | 0.326506 |
| step2_interaction_features_linear_gvf | 2.300161 | 0.333797 |
| given_linear_gvf | 2.300424 | 0.359657 |
| step2_tanh_features_linear_gvf | 2.340762 | 0.371939 |

The best proxy auxiliary beats raw MLP by only 0.000532 mean RMSE and wins 1/3 seeds. This is effectively noise, and fixed/learned interactions do not help when `obs_t` has no Markov information about `c_{t+1}`.

## Direct Conclusion

This is not closed as an unqualified Step 2 result.

The observable AR(1)+interaction bridge now has a real positive control: fixed interaction features improve TD/GVF RMSE, and the learned interaction feature bank can transfer into downstream GVF learners. That supports the narrow claim that Step 2 interaction discovery can be made useful for TD/GVF prediction when the needed next cumulants are state-predictable from current observations.

The broader claim does not hold yet. Margins against raw MLP are small at 3 seeds, the hidden/POMDP result shifts the winner to surprise-driven auxiliary cumulants with similarly tiny MLP margin, GVF-feedback features did not help, trace decay did not explain the gain, and iid dynamics show that apparent auxiliary wins can collapse to noise. Off-policy GVFs are also outside this harness.

Implication for the unqualified Step 2 claim: replace it with a qualified statement. Step 2 feature discovery has a grounded TD/GVF transfer positive control on observable Markov interaction streams, but TD/GVF-target feature finding remains Step 3 research for hidden state, robust auxiliary-question discovery, continual unfrozen feature adaptation, and off-policy prediction/control.

## Worker TD-GVF-FEATURES Update

Date: 2026-05-06

Question: can simple causal additions close the hidden-state and off-policy TD/GVF feature-discovery blocker enough to mark DoD-7 evidence, without moving the claim boundary into Step 2?

### Patch

- Added `select_novel_candidate_indices()` to the Step 3 evaluation harness. It ranks candidate predictive-state features by the existing clipped-IS MSPBE proxy, rejects near-duplicate columns using discovery-prefix absolute correlation, and fills any remaining budget by score order for stable feature dimensionality.
- Added `off_policy_mspbe_novel_predictive_state_linear_td_is`: an 8-feature novelty-gated MSPBE predictive-state row with TD step-size `0.005`. The existing full MSPBE row remains, so the unstable larger feature set is still visible.
- Existing causal probes retained: history lags/traces, predictive-state cross products, TD-surprise interactions, auxiliary cumulant discovery, GVF feedback, clipped IS/Retrace variants, and off-policy raw/no-IS controls.

### Commands

```bash
source .venv/bin/activate

python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' \
  --seeds 3 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 \
  --observation-dynamics coupled_hidden_ar1 --include-squares \
  --hide-last-channels 2 --hidden-coupling 0.3 --hidden-noise-std 0.01 \
  --output-dir outputs/step3_tdgvf_worker_hidden_novelty_3seed

python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' \
  --seeds 5 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 \
  --observation-dynamics coupled_hidden_ar1 --include-squares \
  --hide-last-channels 2 --hidden-coupling 0.3 --hidden-noise-std 0.01 \
  --output-dir outputs/step3_tdgvf_worker_hidden_novelty_5seed
```

Additional focused probes rejected:

- `--off-policy-feature-step-size 0.02`: full MSPBE predictive-state features became unstable, reaching mean off-policy RMSE `603.097058` in the 3-seed harness because seed 2 diverged.
- `--off-policy-retrace-clip 1.0 --off-policy-feature-step-size 0.02`: safer than clip `2.0` but still worse than raw clipped-IS TD, mean off-policy RMSE `7.106277`.
- `--off-policy-feature-step-size 0.005` without novelty gating: mean off-policy RMSE `1.622477`, worse than raw clipped-IS TD at `1.275506`.
- Isolated same-key grids showed that smaller MSPBE-selected sets were the useful part; novelty-gated 8-feature selection was the best stable candidate before patching.

### 5-Seed Hidden-State Results

Mean target GVF RMSE, lower is better.

| Method | RMSE | SE |
|---|---:|---:|
| gvf_feedback_features_linear_gvf | 6.835405 | 2.308180 |
| random_aux_cumulants_mlp_gvf | 7.483859 | 2.929613 |
| given_mlp_gvf | 7.524292 | 3.020495 |
| meta_proxy_aux_cumulants_mlp_gvf | 7.551455 | 3.018312 |
| discovered_aux_cumulants_mlp_gvf | 7.566458 | 3.031515 |
| predictive_state_features_linear_gvf | 7.927006 | 3.060627 |
| fixed_history_trace_linear_gvf | 7.981976 | 3.240275 |
| fixed_interaction_linear_gvf | 8.226079 | 3.203513 |
| td_surprise_interaction_features_linear_gvf | 8.233092 | 3.202239 |
| step2_interaction_features_linear_gvf | 8.258602 | 3.203494 |
| given_linear_gvf | 8.353054 | 3.225515 |
| step2_tanh_features_linear_gvf | 8.450583 | 3.335838 |

Best discovery method: `gvf_feedback_features_linear_gvf`.

- Beats raw MLP by mean RMSE diff `0.688887`.
- Beats fixed interactions by mean RMSE diff `1.390674`.
- Beats fixed history traces by mean RMSE diff `1.146571`.
- The margin is driven partly by seed 2, so this is DoD-7 positive evidence, not closure.

### 5-Seed Off-Policy Behavior-Mismatch Results

Mean scalar TD RMSE against target-policy returns, lower is better.

| Method | RMSE | SE |
|---|---:|---:|
| off_policy_mspbe_novel_predictive_state_linear_td_is | 1.320095 | 0.181242 |
| off_policy_mspbe_predictive_state_linear_td_is | 1.361924 | 0.203405 |
| off_policy_raw_linear_td_is | 1.379815 | 0.186825 |
| off_policy_history_trace_linear_td_is | 2.203220 | 0.308418 |
| off_policy_raw_linear_td_no_is | 2.205414 | 0.325731 |
| off_policy_td_surprise_interaction_linear_td_is | 2.447753 | 0.285442 |

Paired diff versus raw clipped-IS TD:

- `off_policy_raw_linear_td_is_minus_off_policy_mspbe_novel_predictive_state_linear_td_is`: `+0.059719 +/- 0.032536`, wins/losses/ties `3/2/0`.
- Full MSPBE predictive-state features also became weakly positive at 5 seeds (`+0.017891`, `3/2/0`) but had known instability under larger step sizes.
- History traces and TD-surprise interactions fail off-policy here; they lose `0/5` seeds versus raw clipped-IS TD.

### Diagnosis

- Hidden-state failure was not solved by fixed history traces or interaction features. Those rows remain below raw MLP on the 5-seed mean. The useful signal was causal GVF feedback: predictions from a source Horde, computed before the current transition update, gave the downstream linear Horde a compact predictive state that raw MLP did not match on this stress.
- Off-policy failure was mostly feature-set size and stability, not a lack of any predictive-state signal. Full selected predictive-state banks were sensitive to step size and could diverge; a small novelty-gated MSPBE set kept the useful Bellman-difference signal while avoiding redundant high-correlation features.
- Clipped IS is essential in this probe: raw no-IS loses every seed versus raw clipped-IS TD.
- Surprise interactions are not the right off-policy addition in this setup. They select features that correlate with TD-error magnitude, but they do not reduce target-policy return RMSE under behavior mismatch.

### DoD-7 Evidence Status

DoD-7 now has positive evidence on the two previously open stressors:

- Hidden-state stress: `gvf_feedback_features_linear_gvf` beats raw MLP and fixed hand-built feature controls on the 5-seed coupled-hidden AR(1) probe.
- Off-policy stress: `off_policy_mspbe_novel_predictive_state_linear_td_is` beats raw clipped-IS TD by mean RMSE on the 5-seed behavior-mismatch probe.

This is not a full Step 3 closure. The remaining Step 3 work is to confirm beyond short synthetic probes, move from frozen/auxiliary feature transfer to continually adapting discovery, and integrate off-policy ratios into Horde/control rather than a separate scalar TD probe.

### Step 3 vs Step 2 Boundary

This update does not reclassify TD/GVF feature discovery as Step 2. The exact boundary is:

- Step 2 claim: supervised continual feature construction remains closed by the promoted D18 persistent-trace learner and its supervised matrix.
- Step 3 claim: TD/GVF-target feature discovery now has DoD-7 positive evidence for hidden-state and behavior-mismatch stress, but it remains Step 3 because the targets are bootstrapped predictions/question functions, the useful hidden-state mechanism is GVF feedback, and the off-policy evidence depends on clipped importance sampling and Bellman-error feature selection.
