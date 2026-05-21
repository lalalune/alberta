# Step 3 Predictive-State TD Scale-Up

Date: 2026-05-05

## Scope

This follow-up tested whether the partial predictive-state TD result survives
paired evaluation with shared off-policy behavior samples and more seeds. The
success criterion was strict: predictive-state features must beat raw MLP on
coupled-hidden prediction and beat raw clipped-IS TD on off-policy evaluation
with stable means and paired wins.

## Hardening Change

The off-policy evaluation now samples one behavior-policy rollout per seed and
reuses the sampled rewards, importance ratios, and target returns for every
off-policy row. This makes paired off-policy comparisons meaningful. Discovery
scoring still uses only the discovery prefix.

`aggregate_rows()` also reports off-policy paired wins/losses/ties versus
`off_policy_raw_linear_td_is`.

## Commands

```bash
source .venv/bin/activate && pytest tests/test_step3_feature_discovery_eval.py -q
source .venv/bin/activate && ruff check 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' tests/test_step3_feature_discovery_eval.py

source .venv/bin/activate && python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --output-dir outputs/step3_predictive_state_compact_shared_3seed --seeds 3 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 --observation-dynamics coupled_hidden_ar1 --include-squares

source .venv/bin/activate && python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --output-dir outputs/step3_predictive_state_default_shared_10seed --seeds 10 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 --observation-dynamics coupled_hidden_ar1 --include-squares

source .venv/bin/activate && python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --output-dir outputs/step3_predictive_state_no_cross_5seed --seeds 5 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 --observation-dynamics coupled_hidden_ar1 --include-squares --no-predictive-state-cross-products

source .venv/bin/activate && python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --output-dir outputs/step3_predictive_state_harder_hidden_5seed --seeds 5 --discovery-steps 400 --eval-steps 900 --burn-in 100 --burn-tail 50 --observation-dynamics coupled_hidden_ar1 --include-squares --feature-dim 10 --hide-last-channels 3 --active-pairs 7 --hidden-coupling 0.45 --hidden-noise-std 0.01 --history-lags 1 2 4 8 16 --history-trace-rhos 0.8 0.95 0.98 --n-features 32 --candidate-count 32 --off-policy-scale 2.0

source .venv/bin/activate && python 'examples/The Alberta Plan/Step3/step3_feature_discovery_eval.py' --output-dir outputs/step3_predictive_state_no_cross_n8_decay09_5seed --seeds 5 --discovery-steps 250 --eval-steps 500 --burn-in 50 --burn-tail 25 --observation-dynamics coupled_hidden_ar1 --include-squares --no-predictive-state-cross-products --n-features 8 --candidate-count 8 --predictive-state-score-decay 0.9 --off-policy-retrace-clip 1.0
```

## Results

Target GVF RMSE and off-policy RMSE are mean +/- stderr. Paired diffs are
baseline minus predictive-state, so positive is good for predictive-state.

| Run | Seeds | Variant | MLP RMSE | Predictive-state RMSE | MLP - predictive paired | Raw clipped-IS RMSE | MSPBE predictive-state RMSE | Raw IS - predictive paired |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| compact_shared | 3 | cross on, 24 features, clip 2 | 4.503311 +/- 0.995058 | 4.479588 +/- 1.032519 | +0.023722 +/- 0.043927, 2/1/0 | 1.241050 +/- 0.215086 | 1.235541 +/- 0.290742 | +0.005509 +/- 0.100473, 2/1/0 |
| default_shared | 10 | cross on, 24 features, clip 2 | 3.772114 +/- 0.382257 | 3.967441 +/- 0.375942 | -0.195327 +/- 0.062338, 2/8/0 | 1.236899 +/- 0.093027 | 1.211113 +/- 0.113322 | +0.025787 +/- 0.043294, 7/3/0 |
| no_cross | 5 | cross off, 24 features, clip 2 | 3.932254 +/- 0.680828 | 3.992342 +/- 0.705122 | -0.060088 +/- 0.111426, 2/3/0 | 1.225875 +/- 0.130861 | 1.196767 +/- 0.127114 | +0.029107 +/- 0.021124, 4/1/0 |
| harder_hidden | 5 | 400/900 horizon, 10 dims, 3 hidden, stronger coupling, scale 2 | 8.585786 +/- 1.311950 | 9.853770 +/- 1.636114 | -1.267984 +/- 0.364775, 0/5/0 | 3.228350 +/- 0.510898 | 3.309732 +/- 0.554060 | -0.081382 +/- 0.050876, 1/4/0 |
| no_cross_n8_decay09 | 5 | cross off, 8 features, score decay 0.9, clip 1 | 3.932254 +/- 0.680828 | 4.102273 +/- 0.686715 | -0.170019 +/- 0.059229, 1/4/0 | 1.389814 +/- 0.195298 | 1.294605 +/- 0.183048 | +0.095209 +/- 0.036934, 4/1/0 |

## Interpretation

The 3-seed result reproduced the narrow original mean win on coupled-hidden and
off-policy, but the 10-seed default run reversed the coupled-hidden conclusion:
predictive-state features lost to raw MLP on mean and in paired seeds
(`2/8/0`). Cross-products off reduced the on-policy loss but did not reverse it.
The harder hidden-state variant made the predictive-state linear row worse
(`0/5/0` against MLP), although GVF feedback was strong there. Reducing selected
features and shortening the predictive utility horizon improved off-policy under
clip 1, but it further hurt the on-policy comparison.

The off-policy result is partial. Under shared behavior rollouts, MSPBE-scored
predictive-state features beat raw clipped-IS TD in the 10-seed default run
(`7/3/0`) and in two 5-seed sensitivities (`4/1/0`), but lost in the harder
behavior-mismatch variant (`1/4/0`). The mean margins remain small relative to
stderr except in the tighter-clip sensitivity.

## Decision

Reject robust closure for predictive-state TD scale-up.

The required joint success criterion is not met. Predictive-state features do
not stably beat raw MLP on coupled-hidden prediction once scaled to 10 seeds,
and the longer/harder hidden variant also rejects the claim. Off-policy
predictive-state MSPBE scoring is still promising but only partial: it has a
small default 10-seed paired edge and improves under tighter clipping, yet it
does not survive the harder off-policy setting.

## Next Work

- Keep the shared off-policy rollout plumbing; it removes a real comparison
  confound.
- Do not treat `predictive_state_features_linear_gvf` as closed evidence for
  Step 3 feature discovery.
- If revisiting this direction, replace the proxy MSPBE scorer with a real
  gradient-TD/GTD-style secondary-weight criterion and test it first on the
  shared-rollout off-policy probe.
