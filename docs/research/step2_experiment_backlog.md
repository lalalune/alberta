# Step 2 Experiment Backlog

This backlog defines the experiments worth trying before claiming that Step 2
is closed. The success standard is deliberately strict: a method must beat a
fair MLP comparator under the same online protocol, with enough seeds to rule
out a lucky stream family or capacity-starved baseline.

## Agent-Pass Status

The first closure pass has now exercised all five canonical-candidate
directions below at pilot scale:

- UPGD robustness grid: implemented and run. Outcome is positive on
  polynomial/frequency synthetic streams, negative against fair MLP on shuffled
  digits final-window/test metrics, and mixed on shifted digits where UPGD
  improves retained test accuracy but loses current-window tracking.
- Guided compositional search: implemented and run. Residual-imprint/mutation
  generation improves the compositional learner and wins polynomial pilots, but
  does not beat MLP on frequency or compositional streams.
- Rare-task/context-aware utility: implemented and run. Opt-in utility
  mechanisms exist, but they did not reliably preserve rare oracle features.
- External online suite: implemented and run. MLP remains strongest on ordinary
  shuffled external data; UPGD has a retention/generalization signal on shifted
  digits.
- Plasticity baseline/hybrid: implemented and run. Low-noise UPGD is the best
  follow-up candidate; naive UPGD plus hard reset is a negative result.
- D01-D10 grounded directions: implemented and run at pilot scale. The scaled
  5-seed pass identifies dynamic sparse rewiring as the leading new isolated
  candidate on interaction and nonlinear Step 2 regression streams; most other
  directions are negative or conditional. The new loss-space portfolios are
  useful diagnostics but are not yet prediction-space canonical learners.

None of the isolated pilots satisfied the canonical promotion rule on its own.
The successful closure came from combining the two strongest signals into a
prediction-space portfolio: fair MLP width diversity, low-noise UPGD for
retention, and dynamic-sparse rewiring as an additional plasticity expert.

That strict portfolio has now been run as `step2_universal_portfolio.py`. The
promoted eta-1 result with an online class-imbalance MSE guard closes the
earlier eight-regime synthetic/digits matrix:

- no negative mean final-window MSE versus the best fair MLP width on any of
  the eight synthetic/digits regimes;
- no negative mean held-out test accuracy versus the best fair MLP width on any
  digit regime;
- 30-seed risk checks keep compositional, frequency, class-blocked, and
  non-blocked digit checks positive by the predeclared mean criterion.

The broader all-suite closure now comes from `step2_conclusive_learner.py`
with telemetry Worker-B route recovery and an MLP-floor blend. Its 10-seed
canonical result is `130/0/10` seed-level final-window MSE wins/losses/ties
against the same-run best fair MLP across controlled, synthetic, and digits
benchmarks. The 10 ties are class-blocked online-MSE ties; class-blocked
held-out accuracy wins `10/0/0`.

The stricter non-router follow-up now has a promoted simple learner candidate:
`d18_simple_universal_resource_basis.py --configs step2_persistent_trace`. It
is one additive resource-basis learner, not an output portfolio or deployment
router. The canonical all-suite result is positive on every aggregate
final-window MSE comparison (`138/2/0` seed-level wins/losses/ties over 14
regimes x 10 seeds), and the 30-seed hard digit risk check remains positive by
mean against both raw MLP MSE and a fair projected-MLP comparator.

The remaining backlog is research hardening, not a blocker for the current
supervised Step 2 benchmark promotion:

- simplify the D18 persistent-trace learner into a cleaner feature-construction
  principle, if possible;
- true published-scale external non-stationary benchmarks beyond sklearn
  digits;
- deeper integration of feature generation/testing into MLP hidden layers.
  This integration exists as an experimental native lifecycle path but remains
  negative relative to the promoted simple D18 learner and portfolio baselines.

Worker S2-ExternalScale added the first compact published-style stressor pass:
`step2_published_stressors.py`. The 5-seed canonical-ish local run writes
`outputs/step2_canonical/published_stressors_results.json` and
`outputs/step2_canonical/published_stressors_SUMMARY.md`. It is positive on the
28x28 sklearn-digits permuted-pixel fallback (`+0.0071 +/- 0.0007`
final-window MSE vs best fair MLP, `5/0/0` wins) and positive by mean but weak
by sign count on held-out fallback accuracy (`+0.0289 +/- 0.0183`, `3/2/0`).
The lightweight Slowly-Changing Regression analogue is mixed and slightly
negative by mean (`-0.0003 +/- 0.0004`, `3/2/0`). This narrows the external
gap but does not close the true OpenML MNIST / long SCR reproduction gap.

## Canonical-Candidate Experiments

These are the only experiments that can justify changing the canonical Step 2
claim.

### 1. UPGD Robustness Grid

Question: Does UPGD's synthetic out-of-class advantage survive reasonable
hyperparameter and stream variations?

Try:

- perturbation sigma grid: `0`, `1e-4`, `3e-4`, `1e-3`, `3e-3`, `1e-2`;
- hidden sizes: `(32,)`, `(64,)`, `(128,)`, `(64, 64)`;
- synthetic streams: polynomial, frequency mismatch, compositional;
- external streams: shuffled digits, class-blocked digits, permuted digits.

Success criterion:

- beats best fair MLP in at least two stream families, including one external
  or non-co-designed setting;
- paired wins at least `24/30` on synthetic canonical or an equivalent
  pre-registered threshold on a smaller external pilot;
- no large regression on any existing canonical synthetic stream.

Risk:

- UPGD may only be better when the stream rewards continual perturbation and
  worse on ordinary finite supervised streams.

### 2. Guided Compositional Search

Question: Can the feature-of-features DAG become empirically useful with a
better generator?

Try:

- utility-biased parent sampling;
- mutation of high-utility parents;
- residual/imprint-guided candidate parameter initialization;
- promotion blending that avoids output-weight churn;
- deeper candidate budget with the same active feature budget.

Success criterion:

- beats current `CompositionalFeatureLearner` and best fair MLP on at least one
  out-of-class stream;
- preserves topology/cascade-deletion invariants;
- does not rely on enumerating the oracle class.

Risk:

- a guided generator can quietly become benchmark-specific. Any positive result
  must include an out-of-generator-family stream.

### 3. Rare-Task / Context-Aware Utility

Question: Are useful features being discarded because mean utility dilutes rare
heads or recurring contexts?

Try:

- `mean` versus `max` utility aggregation over heads;
- task-balanced utility with inverse active-head frequency;
- utility retention traces for recurring contexts;
- rare-head synthetic stream where only one task occasionally uses a feature.

Success criterion:

- rare-task utility retains oracle features better than mean utility;
- improves final-window or last-cycle loss against the same active budget;
- does not freeze obsolete features in non-recurring contexts.

Risk:

- over-protection can make the learner stale and hurt adaptation.

### 4. External Online Suite

Question: Does any Step 2 method beat fair MLP outside synthetic streams?

Try:

- sklearn digits shuffled;
- sklearn digits class-blocked;
- sklearn digits with fixed per-seed pixel permutation;
- sklearn wine/breast-cancer as low-dimensional controls if multiclass target
  handling remains clean;
- optional noise/scale drift wrappers that preserve online protocol.
- compact published-style stressors:
  `step2_published_stressors.py --canonical-ish` now covers local
  28x28 permuted digits and a lightweight Slowly-Changing Regression analogue.

Success criterion:

- method beats fair MLP on prequential final-window loss or held-out test loss
  in at least one external nonstationary variant;
- gains are not explained by weaker MLP tuning;
- results include paired seed differences.

Risk:

- ordinary finite datasets can stop testing feature lifecycle and instead test
  conventional online classification.
- compact fallbacks can capture the stressor shape while still being too small
  or too easy to justify a published-scale claim.

### 5. Plasticity Baseline / Hybrid

Question: Is the useful part of UPGD simply plasticity preservation, and can CBP
or a CBP/UPGD hybrid improve it?

Try:

- MLP;
- MLP + CBP;
- UPGD;
- simple UPGD + low-utility reset protocol using existing CBP machinery if it
  can be done without a major core rewrite.

Success criterion:

- hybrid or CBP beats both MLP and UPGD on at least one stream family;
- no instability or collapse on synthetic canonical streams;
- result is interpretable as plasticity, not an MLP capacity artifact.

Risk:

- CBP may help only on very long horizons; short pilots may understate it.

## Negative-Control Experiments

These should be run to protect against overclaiming.

- Same pair-product stream with MLP(64) and MLP(64,64), to keep the old
  capacity-starvation failure visible.
- Streams where the oracle is deliberately outside the candidate generator's
  hypothesis class.
- UPGD with `perturbation_sigma=0`, which should reduce toward MLP-like
  behavior and isolate the perturbation contribution.
- Static random feature bank with the same active feature budget.
- Linear baseline on every stream to verify the stream is actually nonlinear.

## Longer-Term Research

These are real Step 2 gaps but probably not one-pass fixes.

- Future-utility estimator instead of backward-looking EMA utility.
- Meta-learned resource manager for generator choice, candidate budget,
  replacement interval, and promotion aggressiveness.
- Feature construction inside the standard `MultiHeadMLPLearner` path.
- TD/GVF-target feature finding under Horde, including bootstrapping effects.
- Partially observable sequential environments where features must combine
  present observations, traces, and learned predictions.
- Published benchmark comparison to Dohare-style Slowly-Changing Regression and
  Permuted MNIST. The compact local analogue now exists and should be treated
  as a harness plus preliminary evidence, not as closure of the published-scale
  benchmark comparison.

## Canonical Promotion Rule

A result may become canonical only if:

- the experiment has a reproducible script and committed JSON/Markdown output;
- the comparator includes fair MLP capacity;
- at least one negative control is included;
- paired seed statistics are recorded;
- the method improves on a meaningful metric without worsening the most relevant
  existing canonical stream;
- the claim text names the stream family and does not imply universality.
