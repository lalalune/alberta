# D18 Step 2 Canonical Assessment

## Bottom Line

`d18_step2_canonical` is now the promoted Step 2 candidate for the current
supervised Alberta Plan benchmark suite.

The May 5, 2026 slim-canonical validation run covers the full 14-regime matrix:

- `14/14` benchmark means beat the same-run best fair MLP by final-window MSE.
- `140/0/0` paired seed wins/losses/ties by final-window MSE.
- All five external digits regimes beat the same-run best fair MLP on held-out
  test MSE.
- All five external digits regimes beat the same-run best fair MLP on mean
  held-out test accuracy.
- The previous class-blocked retention caveat is closed: D18 now beats the MLP
  on online final-window MSE, held-out test MSE, and held-out accuracy.
- The learner still contains no MLP expert and no prediction router. Every
  active prediction/update block is updated every timestep.
- The random tanh basis has been reduced from 512 units to 128 units after
  simplification ablations; the all-metric win still holds.

This resolves the current empirical Step 2 bar. It does not prove a theorem of
arbitrary recursive feature construction, and it is still a research runner
rather than a polished core JAX learner. Those are research boundaries, not
remaining failures against the current benchmark.

## Promoted Candidate

`d18_step2_canonical` combines:

- Resource-managed RKHS core at prediction scale `0.5`.
- Tanh/Fourier basis block at prediction scale `0.4`, with `128` random tanh
  features.
- Strict degree-3 polynomial residual block, RLS-updated, initial scale `0.01`.
- D14-style unified residual basis, initial scale `0.01`.
- Online learned gains over the four additive blocks.
- Scalar basis readout decay `0.9975`.
- One-hot/simplex basis readout decay override `1.0`.
- Component clipping `1.0` on polynomial and unified residual channels.
- Contextual target trace for persistent one-hot streams.
- Context-free deployment prediction that excludes fast target/residual traces.
- Recency-capable prototype memory for persistent one-hot class retention.
- A causal persistence threshold, `target_persistence_threshold = 0.5`, so
  class-blocked streams activate retention while IID/label-drift chance
  fluctuations do not.

The retained one-hot mechanisms are target-geometry and target-persistence
conditioned. They are not dataset-name switches.

Equivalent command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py" \
  --datasets all \
  --steps 1200 \
  --n-seeds 10 \
  --final-window 300 \
  --configs step2_canonical \
  --output-dir outputs/step2_new_directions/d18_step2_simplify_full_budget_tanh128_all14_10seed
```

## Full 14-Regime Result

Protocol: 10 paired seeds, 1200 online steps, final window 300. Margins are
same-seed best-D18 minus best-MLP differences where positive favors D18.

| Dataset | D18 final MSE | D18 margin | Seed wins |
|---|---:|---:|---:|
| controlled_frequency | 0.037618 | +0.143472 | 10/0/0 |
| controlled_interaction | 0.039661 | +0.453207 | 10/0/0 |
| controlled_nonlinear | 0.019416 | +0.051286 | 10/0/0 |
| controlled_polynomial | 0.068648 | +0.842088 | 10/0/0 |
| controlled_rare | 0.032927 | +0.060765 | 10/0/0 |
| controlled_triple | 0.055680 | +0.690401 | 10/0/0 |
| digits_class_blocked | 0.001600 | +0.001277 | 10/0/0 |
| digits_iid | 0.021180 | +0.009236 | 10/0/0 |
| digits_label_drift | 0.034190 | +0.005403 | 10/0/0 |
| digits_mask_noise | 0.043409 | +0.004605 | 10/0/0 |
| digits_permuted_pixels | 0.038353 | +0.010002 | 10/0/0 |
| synthetic_compositional | 0.210954 | +0.058369 | 10/0/0 |
| synthetic_frequency | 1.073572 | +0.520442 | 10/0/0 |
| synthetic_polynomial | 0.884801 | +0.120296 | 10/0/0 |

Held-out digits:

| Dataset | D18 test accuracy | Accuracy margin | D18 test MSE | Test-MSE margin |
|---|---:|---:|---:|---:|
| digits_class_blocked | 0.8505 | +0.7147 | 0.0299 | +0.1011 |
| digits_iid | 0.9642 | +0.0332 | 0.0204 | +0.0079 |
| digits_label_drift | 0.9455 | +0.0455 | 0.0244 | +0.0080 |
| digits_mask_noise | 0.8469 | +0.0109 | 0.0387 | +0.0031 |
| digits_permuted_pixels | 0.9223 | +0.0442 | 0.0317 | +0.0072 |

No negative comparison cells remain in this run.

## What Closed The Last Gap

The last real blocker was the class-blocked digits regime. The learner could
either optimize current-block loss or preserve held-out all-class retention,
but not both. The final fix separates two prediction contexts:

- Online contextual prediction may use fast target traces when the target stream
  is persistently one-hot.
- Context-free deployment prediction excludes fast traces, so held-out
  evaluation does not leak the current temporal context.
- A retained prototype memory stores class-head geometry for one-hot targets.
- A persistence threshold prevents label-drift and IID streams from triggering
  the class-blocked retention machinery from chance same-label runs.

The threshold matters. Without it, low but nonzero target-persistence estimates
in label drift can activate simplex/prototype readout and damage held-out
calibration. With `target_persistence_threshold = 0.5`, only genuinely
persistent one-hot blocks activate the special retention path.

## Ablation Takeaways

What helped:

- Learned additive gains over non-MLP blocks.
- Strict degree-3 finite polynomial RLS residuals.
- The unified residual basis at small scale.
- Tanh readout step size around `0.62`.
- Scalar basis readout decay for nonstationary scalar tasks.
- One-hot decay override for class-style targets.
- Context/deployment prediction split.
- Persistent one-hot prototype memory for class-blocked retention.
- A nonzero persistence threshold to avoid false positives on label drift.
- Reducing random tanh width from `512` to `128`; this preserved all 14
  final-window wins and all digit heldout wins at 10 seeds.

What hurt or was pruned:

- MLP experts and prediction routers are not part of the promoted candidate.
- Ungated hard simplex projection hurts nonpersistent digit regimes.
- Ungated large prototype scores hurt held-out MSE calibration.
- Lifetime-only prototypes fail under label/head drift.
- Stronger gain steps and lower KRLS forgetting were not useful in the retained
  sweeps.
- Cutting RKHS center budget in half was not safe enough: final-window MSE
  stayed positive, but `digits_mask_noise` held-out accuracy regressed on the
  10-seed check.

Simplification checks:

- Full RKHS budget, tanh width `128`: full 14-regime 10-seed pass, no negative
  comparison cells.
- Half RKHS budget plus tanh width `128`: final-window and held-out test-MSE
  wins mostly held, but `digits_mask_noise` held-out accuracy was negative.
- Quarter RKHS budget plus tanh width `256`: `digits_label_drift` and
  `digits_mask_noise` final-window MSE regressed in the 3-seed probe.

## Static Versus Learned Parts

Learned online:

- RKHS bank resource allocation.
- Additive block gains.
- KRLS/RLS readouts.
- Tanh/Fourier/polynomial/unified readouts.
- Target-persistence statistic.
- Prototype means, with recency-capable updates.

Still static:

- Bank families and budgets.
- Random tanh basis.
- Random tanh width is now ablated down to `128`; the feature generator itself
  is still fixed.
- Fourier frequencies.
- Polynomial degree cap.
- Gain anchor and clipping.
- Basis readout decay values.
- Persistence threshold.

The biggest remaining simplification opportunity is to make the memory
timescale and persistence threshold learned resource-manager actions. That is a
quality and universality improvement, not a blocker for the current benchmark.

## Complexity Versus MLP

D18 is substantially more expensive than the fair MLP baselines in the current
NumPy/Python runner. The cost is dominated by three budgeted KRLS banks and a
finite RLS polynomial residual:

- RKHS budgets `64/128/128` imply roughly `36,864` covariance-scale operations
  per active step, before kernel evaluation.
- The strict degree-3 polynomial residual over up to eight dimensions has about
  65 features, or about `4,225` covariance-scale operations.
- Tanh/Fourier basis prediction now uses `128` tanh features, down from the
  previous `512`.

The slim 10-seed run had mean D18/best-MLP runtime ratio `7.64x` and median
ratio `7.26x` in the Python runner. This is not a decisive speedup because KRLS
dominates cost, but it removes 75% of the random tanh features without losing
the empirical bar.

This is the main engineering caveat. The current result establishes a research
candidate; productionizing Step 2 should focus on a fused JAX implementation and
on replacing part of the KRLS core with cheaper learned-basis machinery. Budget
ablations show the RKHS core is doing real work; it cannot simply be quartered
without reopening digit gaps.

## Remaining Assessment

Against the current Step 2 supervised benchmark criterion, no empirical gap is
left open: D18 beats the same-run best fair MLP across all 14 regimes and all
digit held-out metrics.

What remains missing from a stronger philosophical claim:

- No formal proof of arbitrary recursive feature discovery.
- No evidence yet on every possible external continual-learning benchmark.
- No production-grade core learner implementation.
- No learned replacement yet for every static timescale/threshold.
- Computational efficiency is not yet competitive with small MLP baselines.

Therefore the precise status is: Step 2 is resolved for the current empirical
Alberta Plan benchmark suite, including the previous retention blocker. The
larger goal of a simple, efficient, mathematically universal online learner is
still a research agenda beyond this benchmark closure.
