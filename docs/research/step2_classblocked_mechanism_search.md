# Step 2 Class-Blocked Mechanism Search

Date: 2026-05-06.

## Scope

This pass tested simple class-blocked retention/tracking mechanisms that require
no core edits.  The runner is isolated at:

`output/subagents/classblocked_mechanism_search/run_mechanism_search.py`

The constraints were: causal online updates, one learner, no replay buffer, no
portfolio, no MLP fallback, no task-id input, and temporally uniform wrapper
logic.  Held-out evaluation uses the final causal wrapper state, not future
test labels.

## Protocol

Dataset: `sklearn.datasets.load_digits`, per-class train/test split with
`train_fraction=0.7`, train-only standardization, 10 one-hot heads.

Stream: repeated train epochs ordered as digit-class blocks.  Each step records
pre-update prediction loss/accuracy, then updates on the current one-hot label.

Main settings:

- Broad screen: 3 seeds, seeds `1020..1022`, `steps=1200`,
  `final_window=300`.
- Selected follow-up: 10 seeds, seeds `1020..1029`, `steps=1200`,
  `final_window=300`.
- Baselines: `mlp64` and `mlp64_64`.
- Positive paired diffs favor the candidate.  For MSE, diff is
  `baseline - method`; for accuracy, diff is `method - baseline`.

## Mechanisms Tested

- Existing head-plasticity controls:
  `upgd_density_repx025_meta`, `upgd_density_repx075_meta`,
  `upgd_density_meta003_retention`.
- Target centering wrappers:
  fixed uniform centering, cumulative class-prior centering, and EMA prior
  centering.  Priors are updated only after observing each online label.
- Recent-target input context:
  append a causal EMA class-prior vector to the observation.
- Readout knobs:
  `readout_input_mode=hidden_plus_input`, softmax CE readout, and tiny fixed
  margin updates.
- Calibration-only scoring:
  fixed softmax transforms applied to the emitted logits for scoring.
- Simple capacity/update sweeps:
  width 128 and conservative ObGD kappa/step-size softmax CE.

## Broad 3-Seed Screen

The full screen artifacts are:

- `output/subagents/classblocked_mechanism_search/screen_3seed_1200/mechanism_search_results.json`
- `output/subagents/classblocked_mechanism_search/screen_3seed_1200/SUMMARY.md`

Key rows:

| Method | Final MSE | Final acc | Test acc | d final MSE vs `mlp64` | d test acc vs `mlp64` |
|---|---:|---:|---:|---:|---:|
| `mlp64_64` | 0.002730 | 0.992222 | 0.099567 | +0.001742 | -0.002474 |
| `upgd_density_repx075_meta` | 0.003921 | 0.990000 | 0.103278 | +0.000551 | +0.001237 |
| `upgd_density_repx025_meta` | 0.004390 | 0.988889 | 0.118120 | +0.000082 | +0.016079 |
| `mlp64` | 0.004472 | 0.986667 | 0.102041 | 0.000000 | 0.000000 |
| `upgd_margin_tiny` | 0.004509 | 0.987778 | 0.109462 | -0.000036 | +0.007421 |
| `upgd_density_meta003_retention` | 0.005497 | 0.982222 | 0.142239 | -0.001025 | +0.040198 |

New wrapper mechanisms did not clear tracking:

| Mechanism | Representative method | Final MSE | Final acc | Test acc | Read |
|---|---|---:|---:|---:|---|
| Uniform target centering | `upgd_uniform_center` | 0.019082 | 0.935556 | 0.249227 | Better retention, tracking collapse. |
| Cumulative prior centering | `upgd_cumulative_center` | 0.017261 | 0.946667 | 0.249845 | Same tradeoff as uniform centering. |
| EMA prior centering | `upgd_ema_center_a005` | 0.017161 | 0.945556 | 0.135436 | Less retention gain, still large tracking cost. |
| Prior input context | `upgd_prior_context_a02` | 0.012305 | 0.952222 | 0.190476 | Context helps retention but not tracking enough. |
| `hidden_plus_input` | `upgd_hidden_plus_input` | 0.019474 | 0.957778 | 0.296228 | Readout shortcut improves test acc, hurts final window. |
| Softmax CE | `upgd_softmax_ce_hpi` | 0.045154 | 0.668889 | 0.773036 | High retained classifier, unusable current-block tracking. |

## 10-Seed Follow-Up

The selected follow-up artifacts are:

- `output/subagents/classblocked_mechanism_search/selected_10seed_1200/mechanism_search_results.json`
- `output/subagents/classblocked_mechanism_search/selected_10seed_1200/SUMMARY.md`

| Method | Final MSE | Final acc | Test MSE | Test acc | d final MSE vs `mlp64` | d test acc vs `mlp64` |
|---|---:|---:|---:|---:|---:|---:|
| `mlp64_64` | 0.002817 +/- 0.000055 | 0.991000 | 0.163014 | 0.099443 | +0.002063 (10/10) | -0.025974 |
| `upgd_density_repx075_meta` | 0.004239 +/- 0.000090 | 0.989333 | 0.133908 | 0.120223 | +0.000642 (10/10) | -0.005195 |
| `upgd_margin_tiny` | 0.004812 +/- 0.000128 | 0.986000 | 0.130190 | 0.127829 | +0.000068 (8/10) | +0.002412 |
| `mlp64` | 0.004881 +/- 0.000114 | 0.985333 | 0.129922 | 0.125417 | 0.000000 | 0.000000 |
| `upgd_density_repx025_meta` | 0.004899 +/- 0.000146 | 0.986333 | 0.128456 | 0.128757 | -0.000018 (3/10) | +0.003340 |
| `upgd_density_meta003_retention` | 0.005851 +/- 0.000160 | 0.982333 | 0.121291 | 0.152876 | -0.000970 (0/10) | +0.027458 |

Reads:

- `upgd_density_repx075_meta` is the cleanest MLP64-relative tracker:
  final-window MSE improves by `+0.000642` with `10/10` paired wins and
  final-window accuracy improves by `+0.004000`.  It does not retain held-out
  accuracy over `mlp64` in the 10-seed run and has worse held-out MSE.
- `upgd_margin_tiny` is the most balanced single-learner candidate in this
  run: final-window MSE improves by only `+0.000068` with `8/10` paired wins,
  final-window accuracy improves by `+0.000667`, and held-out accuracy improves
  by `+0.002412`.  The effect is small, held-out MSE is slightly worse
  (`-0.000268`), and it still loses to `mlp64_64` on final-window MSE in
  `10/10` seeds.
- `upgd_density_repx025_meta` keeps the best retained-accuracy compromise but
  does not clear MLP64 final-window MSE in the 10-seed extension.
- `upgd_density_meta003_retention` is the strongest held-out retention branch,
  but it loses final-window tracking in every paired seed.

## Mechanism Interpretation

Target centering and softmax CE mostly solve the wrong problem for this
protocol.  They reduce class-prior and calibration drift, so held-out accuracy
can increase dramatically, but they remove the aggressive current-block
overfitting that defines the final-window tracking metric.

Recent-prior input context behaves similarly.  It is causal and single-learner,
but using the final causal prior for held-out evaluation means the readout
inherits a block-state bias.  It helps retention relative to raw UPGD but does
not approach MLP tracking.

`hidden_plus_input` gives the heads a shortcut to raw pixels and often improves
held-out accuracy, but by itself it does not give fast enough current-block
tracking.  Combining it with the compromise branch worsened the final-window
metric in the 3-seed screen.

The only tracking gains came from existing head-plasticity mechanisms:
repetition pressure and tiny margin pressure.  This supports the earlier
diagnosis that the class-blocked issue is primarily output-head calibration and
plasticity under repeated one-hot blocks, not hidden-feature utility discovery.

## Recommendation

Do not promote a new wrapper mechanism from this search.

If a narrow MLP64-relative single-learner candidate is needed for follow-up,
`upgd_margin_tiny` is the least weak new-ish option because it marginally
improves final-window MSE and held-out accuracy over `mlp64` in the 10-seed
run.  It is not strong enough to claim closure: the effect is tiny, held-out
MSE is not improved, and `mlp64_64` remains far ahead on tracking.

For actual class-blocked closure, prioritize direct output-head anti-drift or
calibration work.  Wrapper-only centering, prior context, softmax CE, and
scoring temperature are useful diagnostics but do not close the retention and
tracking objectives simultaneously.

## Commands

```bash
.venv/bin/python -m py_compile \
  output/subagents/classblocked_mechanism_search/run_mechanism_search.py
```

```bash
.venv/bin/python output/subagents/classblocked_mechanism_search/run_mechanism_search.py \
  --preset smoke \
  --steps 300 \
  --n-seeds 1 \
  --seed 1020 \
  --final-window 100 \
  --output-dir output/subagents/classblocked_mechanism_search/smoke_1seed_300
```

```bash
.venv/bin/python output/subagents/classblocked_mechanism_search/run_mechanism_search.py \
  --preset screen \
  --steps 1200 \
  --n-seeds 3 \
  --seed 1020 \
  --final-window 300 \
  --output-dir output/subagents/classblocked_mechanism_search/screen_3seed_1200
```

```bash
.venv/bin/python output/subagents/classblocked_mechanism_search/run_mechanism_search.py \
  --steps 1200 \
  --n-seeds 10 \
  --seed 1020 \
  --final-window 300 \
  --methods mlp64,mlp64_64,upgd_density_repx025_meta,upgd_density_repx075_meta,upgd_margin_tiny,upgd_density_meta003_retention \
  --output-dir output/subagents/classblocked_mechanism_search/selected_10seed_1200
```
