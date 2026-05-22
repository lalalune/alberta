# Step 2 Candidate Closure Assessment

This note records the follow-up pass over the ten grounded Step 2 directions
and the simple portfolio variants added after the first candidate sweep.

## What Was Filled

All D01-D10 directions now have a runnable pilot in
`examples/The Alberta Plan/Step2/step2_new_direction_pilots.py`.

The runner evaluates each method against the same fair `MultiHeadMLPLearner`
on two common Step 2 regression streams:

- `interaction`: hidden pair-product targets where feature construction should
  help.
- `nonlinear`: tanh-latent targets used as a negative control against simple
  pair-product overfitting.

The runner also adds three causal portfolio probes:

- `portfolio_all_hedge`: loss-space Hedge over fair MLP plus all D01-D10
  candidates.
- `portfolio_signal_hedge`: loss-space Hedge over fair MLP plus the candidates
  with an initial positive signal (`D03-D08`).
- `portfolio_signal_ema_gate`: a causal EMA gate between fair MLP and the
  signal Hedge curve.

These portfolio curves are not canonical prediction mixtures. They are a
conservative loss-space diagnostic: if an actual convex prediction portfolio
uses the same expert predictions, squared loss convexity means the weighted
expert-loss curve is an upper-bound style proxy, not an over-optimistic
prediction-space score. The EMA gate is causal because it routes using only
previous EMA state.

## Scaled Result

The strongest run is the 5-seed, 900-step scale pass:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_new_direction_pilots.py" \
  --num-steps 900 \
  --num-seeds 5 \
  --final-window 180 \
  --output-dir outputs/step2_new_direction_pilots_scaled \
  --note-path docs/research/step2_new_direction_pilots_scaled.md
```

Headline result: `D03 dynamic_sparse` and the three portfolio curves beat the
fair MLP on both evaluated suites.

| Method | Interaction diff vs MLP | Interaction wins | Nonlinear diff vs MLP | Nonlinear wins |
|---|---:|---:|---:|---:|
| `d03_dynamic_sparse` | `+0.5377 +/- 0.0951` | `5/0` | `+0.0481 +/- 0.0039` | `5/0` |
| `portfolio_signal_hedge` | `+0.5377 +/- 0.0951` | `5/0` | `+0.0477 +/- 0.0039` | `5/0` |
| `portfolio_all_hedge` | `+0.5377 +/- 0.0951` | `5/0` | `+0.0476 +/- 0.0039` | `5/0` |
| `portfolio_signal_ema_gate` | `+0.5377 +/- 0.0951` | `5/0` | `+0.0475 +/- 0.0041` | `5/0` |

Positive differences mean fair MLP final-window loss minus method
final-window loss, so positive favors the candidate.

The portfolios do not discover a better expert than dynamic sparse in this
run; they mostly learn to track it. Their value is protection and routing, not
new representation. The simple read is therefore: dynamic sparse is now the
best new isolated candidate on these two synthetic Step 2 streams.

## Candidate-by-Candidate Assessment

### D01: RLS Random Features

Status: negative.

The RLS readout over fixed random nonlinear features loses on both scaled
suites. It is especially poor on nonlinear (`-0.1608`, `0/5` wins) and still
negative on interaction (`-0.1056`, `2/3` losses). The missing ingredient is
not second-order readout adaptation alone; representation quality matters.

Remaining work: do not promote. Only revisit as a last-layer module on top of a
trained/adaptive trunk, not as fixed random features.

### D02: Auxiliary Next-Projection Heads

Status: tiny/mixed.

The auxiliary-prediction pilot is basically neutral on interaction
(`+0.0017`, `3/2` wins) and negative on nonlinear (`-0.0081`, despite `3/2`
seed wins). This does not justify a core auxiliary-head path yet.

Remaining work: if revisited, it needs a stronger auxiliary target family and
explicit trunk-feature diagnostics. The current random next-observation
projection is not enough.

### D03: Dynamic Sparse Rewiring

Status: strongest new candidate on the scaled regression pass.

Dynamic sparse wins both scaled suites by paired seed count and mean
final-window loss. On interaction, the margin is large (`+0.5377`, `5/0`). On
nonlinear, the margin is smaller but clean (`+0.0481`, `5/0`).

What is promising: the mechanism is not a hand-coded oracle feature basis. It
keeps a fixed hidden-input budget, tracks utility, and reallocates hidden units.
That is closer to the Alberta Plan Step 2 goal than static quadratic or spline
features.

Remaining work:

- Run the exact scaled dynamic-sparse configuration on the existing external
  digits regimes. Earlier dynamic-sparse evidence in the repo was negative on
  class-blocked digits, so external success cannot be assumed.
- Promote the standalone pilot into a core learner only after external and
  broader synthetic results hold.
- Add config serialization, state tests, replacement-invariant tests, and
  learner-loop compatibility before treating it as a framework component.
- Compare against best fair MLP widths, not only `h64`.

### D04: Fixed Spline Basis

Status: unstable; not robust after scaling.

The two-seed run made spline look robust, but the 5-seed/900-step run reversed
that: interaction `-0.0705`, nonlinear `-0.0141`. This is a useful cautionary
case. Static local bases can look good in short runs and then lose once the MLP
has enough time.

Remaining work: do not promote the fixed basis. Adaptive knot placement would
be a different experiment, but the current fixed pilot is closed as negative.

### D05: High-Leak Homeostasis Proxy

Status: small conditional signal.

The high-leak proxy is positive on nonlinear (`+0.0060`, `4/1`) but not on
interaction (`-0.0023`, `2/3`). It may reduce dormant-unit problems, but the
effect size is small and not universal.

Remaining work: implement a real homeostatic regularizer only if paired with
diagnostics showing fewer dormant/dead units and no loss of adaptation. The
proxy alone should not be canonicalized.

### D06: Online Input Whitening Proxy

Status: negative after scaling.

The whitening proxy looked promising in the one-seed smoke run but lost on both
scaled suites. It does not solve the main Step 2 gap as implemented.

Remaining work: a true optimizer-space natural-gradient/preconditioner remains
mathematically interesting, but causal input whitening is closed as a negative
proxy.

### D07: Budgeted Kernel RLS

Status: interaction-only.

Budgeted KRLS wins interaction (`+0.1087`, `5/0`) and loses nonlinear
(`-0.1294`, `0/5`). It is useful evidence that novelty-gated nonparametric
bases can help some hidden-feature streams, but it is not broad enough.

Remaining work: keep as a specialist expert inside a portfolio, not as a
default Step 2 learner.

### D08: Independent Head Trunks

Status: interaction-only.

Independent trunks win interaction (`+0.0507`, `5/0`) and lose nonlinear
(`-0.0606`, `0/5`). The result suggests multi-head interference is real on the
interaction stream, but fully independent trunks waste sharing where shared
representation helps.

Remaining work: a true head-conditioned modulation/FiLM design remains open.
The independent-trunk proxy is not sufficient.

### D09: History Features

Status: negative on these streams.

History features lose both scaled suites. This is not a complete rejection of
history-state features; these two streams are not strongly partially observed.
It does mean D09 should move to Step 3/partial-observation settings rather than
being promoted for Step 2 regression.

Remaining work: rerun only on streams where hidden phase or temporal aliasing
is central.

### D10: Precision-Weighted Linear Readout

Status: negative.

The proxy loses badly on both suites. Simple per-head residual precision is not
enough and can amplify the wrong updates.

Remaining work: close this proxy as negative. A future uncertainty-weighted
loss would need a better uncertainty model and nonlinear representation.

## What Is Still Missing

The D01-D10 pilot gap is filled, but Step 2 is not fully closed by this pass.

Missing before canonical promotion:

1. **External validation for dynamic sparse.** The strongest new isolated
   candidate has not yet beaten fair MLP on the external digits regimes in this
   scaled configuration.
2. **Broader synthetic validation.** The scaled run covers interaction and
   nonlinear streams. It does not yet cover the canonical out-of-class
   polynomial, frequency, and compositional streams used by
   `step2_expert_mixture.py`.
3. **Prediction-space portfolio implementation.** The new portfolios are
   causal loss-space diagnostics. A canonical portfolio must actually combine
   or select predictions from live experts.
4. **Best-MLP comparator grid.** The scaled pilot uses one fair MLP setting.
   Canonical claims need best fair MLP among `h64`, `h128`, and likely
   `h64_64`, matching the stricter quadratic/centered follow-up standard.
5. **Core integration for dynamic sparse.** The dynamic-sparse implementation
   is still an examples-path pilot. It needs public API design, config
   serialization, invariant tests, docs, and compatibility with framework
   tracking utilities.
6. **Retained-generalization metrics.** The current low-noise MLP/UPGD mixture
   already showed that final-window MSE can miss held-out retention. Dynamic
   sparse and any new portfolio need final-window and held-out metrics.

## Practical Next Canonical Candidate

The next canonical attempt should be a prediction-space portfolio with three
experts:

- fair MLP fallback;
- low-noise UPGD, because it already closes the existing 8-regime fair-MLP
  promotion bar by improving or tying final-window MSE;
- dynamic sparse, because it is now the strongest new candidate on the
  interaction/nonlinear scale pass.

The portfolio should report:

- final-window MSE versus fair MLP;
- held-out test accuracy/MSE for external digits;
- best-expert regret, so routing failures are visible;
- final and mean portfolio weights;
- paired seed differences.

Until that run exists, the honest status is:

- D01-D10 pilot coverage is complete.
- D03 dynamic sparse is the leading new isolated direction.
- The existing low-noise MLP/UPGD expert mixture remains the strongest
  canonical fair-MLP candidate already backed by external regimes.
- A universal "beats MLP everywhere" Step 2 claim is still too strong.

## Follow-Up Portfolio Result

That prediction-space portfolio has now been run as
`step2_universal_portfolio.py`. The first 10-seed full-matrix result improved
the status but did not completely close the strict version. A follow-up pass
then lowered Hedge eta to `1.0` and added a causal online class-imbalance MSE
guard for class-blocked digits.

The promoted strict result now closes the current supervised matrix:

- no negative mean final-window MSE versus the best fair MLP width on any of
  the eight current regimes;
- no negative mean held-out test accuracy versus the best fair MLP width on any
  digit regime;
- class-blocked online tracking ties the best current-block MLP exactly, while
  held-out retained accuracy switches to UPGD and beats the MLP grid;
- 30-seed risk checks keep compositional and frequency positive by mean, keep
  class-blocked tracking tied, and keep non-blocked digits positive on
  final-window MSE and held-out accuracy.

Detailed assessment:
`docs/research/step2_universal_portfolio_assessment.md`.
