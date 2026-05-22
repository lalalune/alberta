# Step 2 Worker K: Timescale / Guard-Margin Sweep

Date: 2026-05-05

## Question

The current universal portfolio is close on the remaining weak regimes, but the
losses look like selection errors near the MLP frontier rather than missing
experts. Worker K tested whether mathematically simple risk/timescale knobs can
reduce those errors without adding new mechanisms.

Primary ranking rule:

1. Minimize total weak3 seed losses against same-run best fair MLP on
   final-window MSE.
2. Break ties by the sum of aggregate final-window MSE deltas, where positive
   means conclusive is better than best fair MLP.

The current control is:

```bash
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 1.0 \
  --hedge-discount 0.995 \
  --selector-window 100 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse
```

On weak3 only, the control has `26/4/0` wins/losses/ties and
`+0.119143` summed aggregate final-window MSE delta.

## Completed Candidates

All candidates used the same current-best command except for the listed
timescale/guard flags and `--benchmarks
controlled_rare,synthetic_compositional,synthetic_polynomial`.

| Candidate | Output | Controlled Rare | Synthetic Compositional | Synthetic Polynomial | Weak3 total | Sum delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Control | `outputs/step2_conclusive_pruned_hedge_safe_poly_selector100_all_10seed/results.json` | `8/2/0`, `+0.014419` | `9/1/0`, `+0.043445` | `9/1/0`, `+0.061279` | `26/4/0` | `+0.119143` |
| `eta=0.5`, `window=100`, `guard_margin=0.001` | `outputs/step2_worker_k_eta0p5_disc0p995_w100_guard0p001_weak3_10seed/results.json` | `8/2/0`, `+0.014461` | `10/0/0`, `+0.045914` | `8/2/0`, `+0.061439` | `26/4/0` | `+0.121814` |
| `eta=0.5`, `window=100`, `guard_margin=0.005` | `outputs/step2_worker_k_eta0p5_disc0p995_w100_guard0p005_weak3_10seed/results.json` | `8/2/0`, `+0.011316` | `10/0/0`, `+0.046108` | `8/2/0`, `+0.060216` | `26/4/0` | `+0.117639` |
| `eta=0.5`, `window=150`, `guard_margin=0.001` | `outputs/step2_worker_k_eta0p5_disc0p995_w150_guard0p001_weak3_10seed/results.json` | `9/1/0`, `+0.021143` | `10/0/0`, `+0.047538` | `6/4/0`, `+0.061569` | `25/5/0` | `+0.130251` |
| `eta=1.0`, `window=100`, `guard_margin=0.001` | `outputs/step2_worker_k_eta1p0_disc0p995_w100_guard0p001_weak3_10seed/results.json` | `8/2/0`, `+0.014441` | `9/1/0`, `+0.043229` | `8/2/0`, `+0.061246` | `25/5/0` | `+0.118916` |

## Interpretation

The guard-margin idea was directionally useful but did not close the gap under
the strict ranking rule.

`guard_margin=0.001` with `eta=0.5` and `window=100` repaired the
synthetic-compositional seed loss, but it traded that win for an additional
synthetic-polynomial seed loss. It ties the control on total weak3 seed losses
and slightly improves summed aggregate final-window MSE delta. Because seed
losses are the primary criterion, this is not a promotion candidate.

`guard_margin=0.005` is too conservative. It also repairs compositional, but it
weakens controlled-rare aggregate delta and still creates the same second
polynomial loss. It is dominated by the `0.001` guard candidate.

Increasing `selector_window` to `150` helps controlled rare, including the
previously difficult seed 8, but it destabilizes synthetic polynomial. The
larger window appears to preserve stale route preferences in the polynomial
stream; that creates four polynomial seed losses, so this setting is not
promotable even though its aggregate delta is numerically high.

Increasing Hedge temperature from `eta=0.5` back to `eta=1.0` reintroduces the
synthetic-compositional seed loss while keeping the controlled-rare and
polynomial losses. This supports the view that the remaining failures are not
solved by faster exponential weighting alone.

## Decision

No Worker K candidate beat the control's `4` weak3 seed losses. Therefore no
all-suite confirmation was triggered.

The best Worker K candidate is:

```bash
python "examples/The Alberta Plan/Step2/step2_conclusive_learner.py" \
  --benchmarks controlled_rare,synthetic_compositional,synthetic_polynomial \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --warmup-steps 250 \
  --weighting-scheme discounted_hedge \
  --hedge-eta 0.5 \
  --hedge-discount 0.995 \
  --selector-window 100 \
  --stacker-step-size 0.006 \
  --safe-route-sources recursive_features,polynomial_features \
  --digits-deployment-objective all_h128_blend \
  --h128-blend-weight 0.5 \
  --disable-experts upgd_low_noise,dynamic_sparse \
  --guard-margin 0.001
```

It is a useful diagnostic candidate, not a promoted canonical setting.

## Next Most Plausible Direction

The sweep points away from global selector timescale tuning and toward a
contextual guard that distinguishes why a non-MLP route is being selected. The
current guard margin suppresses non-MLP routes uniformly near the MLP frontier.
That helps compositional but can hurt polynomial, where polynomial routes are
often genuinely correct. A more promising selector would condition the guard on
source identity, route stability, or recent source-specific regret, so that the
polynomial route is protected on polynomial streams while the same route is
blocked when it is merely noisy on rare/head-masked streams.

