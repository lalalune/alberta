# D10 Precision-Calibrated Heteroscedastic Heads

## Core Hypothesis

The fair Step 2 MLP may waste plasticity by treating every online target error
as equally trustworthy. In nonstationary streams, some samples are ambiguous,
out-of-context, near drift boundaries, or simply high-variance observations. Add
a second output per prediction head that estimates target uncertainty, and use
the resulting inverse variance as a temporally-uniform loss weight. The target
codes and input features stay unchanged; only the online loss says how much each
observed error should move the shared trunk and head at that moment.

## Why Different

This is not centered targets, ECOC, RFF, hashed quadratics, residual imprinting,
fast/slow ensembling, surprise resets, prototype context, hidden
orthogonalization, or EMA prediction. It does not add a new representation,
change labels, keep multiple learners, reset weights, average parameters, or
alter the target basis. The mechanism is Bayesian target weighting: each head
learns a calibrated observation precision and uses that precision to scale the
ordinary Step 2 squared-error update.

## Mathematical Grounding

For head `k`, let the MLP produce a mean `mu_k(x_t)` and log variance
`s_k(x_t) = log sigma_k^2(x_t)`. Keep the original target `y_{t,k}`. Use a
detached-gradient surrogate for the Gaussian negative log likelihood:

```text
L_t,k = 1/2 exp(-s_t,k) stopgrad((y_t,k - mu_t,k)^2) + 1/2 s_t,k
       + 1/2 stopgrad(exp(-s_t,k)) (y_t,k - mu_t,k)^2.
```

Equivalently, the mean update is the usual squared-error gradient multiplied by
`p_t,k = clip(exp(-s_t,k), p_min, p_max)`, while the variance head is trained to
make `sigma_t,k^2` match recent squared residuals. The clipping and a weak
running-residual calibration penalty prevent the trivial solution where
uncertainty grows without bound:

```text
c_k <- (1 - beta) c_k + beta (y_t,k - mu_t,k)^2
R_cal = lambda_cal (s_t,k - stopgrad(log(c_k + eps)))^2.
```

This is the online heteroscedastic-regression analogue of Bayesian weighted
least squares: noisy targets contribute lower precision; consistent targets
contribute higher precision. Because `p_t,k` is detached for the mean update, the
method changes credit assignment without letting the mean head manipulate its
own learning weight through the same gradient.

## Why It Could Beat Previous Iteration And Fair MLP

The fair MLP has enough capacity, but its shared trunk receives the full error
from every active head and every transition. During abrupt label changes,
class-blocked digits, scale changes, or mixed-regime streams, high residuals can
mean either useful novelty or temporarily unreliable supervision. ObGD limits
the update norm after the fact; it does not decide which errors deserve high
precision. D10 can reduce destructive trunk updates from poorly calibrated
moments while amplifying stable, low-noise target relationships. It should help
especially where the previous iteration loses by chasing transient residuals,
and it should remain fair to the MLP because it uses the same architecture,
stream order, targets, hidden sizes, optimizer, and bounder plus only one
additional scalar uncertainty output per head.

## Minimal Implementation Sketch

Prototype in a new experiment script only, with no core API change:
`examples/The Alberta Plan/Step2/new_directions/d10_uncertainty_weighted_targets.py`.

- Reuse the current fair `MultiHeadMLPLearner` settings: same seeds,
  normalization, `hidden_sizes`, sparse init, optimizer, ObGD bounding, and
  stream construction.
- Replace each scalar prediction head with two script-local readouts over the
  same hidden feature: mean `mu_k` and log variance `s_k`.
- For the mean loss, multiply each active squared-error term by detached
  `p_k = clip(exp(-s_k), p_min, p_max)`, with grid values such as
  `p_min in {0.1, 0.25}`, `p_max in {4, 8}`.
- Train `s_k` every time step using the Gaussian NLL log-variance term plus the
  calibration anchor above. Keep inactive NaN targets masked exactly as in the
  existing multi-head path.
- Log both ordinary unweighted MSE and the actual Gaussian NLL. Selection uses
  unweighted task metrics only, so the method cannot win by hiding errors behind
  high variance.

## Metrics / Success Bar

Primary metrics: paired final-window unweighted MSE, final-window accuracy for
digits regimes, adaptation half-life after drift boundaries, and prequential
regret versus the fair MLP. Calibration metrics: expected calibration error over
residual bins, mean predicted variance versus realized squared residual,
precision saturation rate, NLL, and correlation between `p_t,k` and future
one-step residual.

Smoke success: across at least 5 paired seeds, improve fair MLP final-window
unweighted MSE by `>= 4%` on two nonstationary regimes, with no IID digits
accuracy loss larger than `0.005` and no precision saturation above 20% of
active head updates. Promotion success: 30 seeds, bootstrap CI excluding zero on
aggregate final-window MSE, calibrated residual bins within 25% relative error
for the central 80% of predictions, and a paired win rate of at least `20/30`
on the strongest nonstationary regime.

## Risks / Negative Controls

Risks: the variance head may learn to suppress hard but useful examples;
uncertainty clipping may become another hidden step-size knob; high predicted
variance after drift could slow necessary adaptation; calibration loss can lag
under rapid regime changes. Track both task error and calibration so a low NLL
with worse MSE is rejected.

Negative controls: fixed `p_k = 1`, random shuffled precision with the same
marginal distribution, residual-moment precision without an input-dependent
variance head, variance head trained but detached from the mean loss, oracle
regime-boundary masking if boundaries are known in synthetic streams, and IID
digits where uncertainty weighting should be neutral rather than a source of free
improvement.

## Exact First Command

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/new_directions/d10_uncertainty_weighted_targets.py" --steps 1200 --n-seeds 5 --final-window 300 --regimes iid label_drift class_blocked scale_drift --p-min 0.1 --p-max 8.0 --calibration-lambda 0.01 --output-dir outputs/step2_new_directions/d10_uncertainty_weighted_targets
```
