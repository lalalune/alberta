# Step 2 Theory Falsification Plan

This plan turns the critique of target-structure UPGD into executable stress
tests. The goal is to find the smallest streams that either falsify the strong
universal claim or force the theory to name its assumptions.

## Runner

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_theory_falsification.py" \
  --scenarios all \
  --steps 240 \
  --n-seeds 2 \
  --methods upgd,mlp,mlp_deep \
  --upgd-width 32 \
  --mlp-width 64 \
  --output-dir /tmp/step2_theory_falsification_t2_final \
  --note-path /tmp/step2_theory_falsification_t2_final/SUMMARY.md
```

The runner compares `target_structure_upgd` against same-run fair MLP
baselines on prequential final-window MSE. Positive paired differences favor
UPGD. A negative row is not automatically a theorem falsification; first check
whether the row violates an explicit assumption such as observability, drift
rate, target calibration, excitation, or finite budget.

## Theory Boundary Under Test

The falsification suite does not refute classical universal approximation. It
tests whether the current finite target-structure UPGD learner can support a
stronger online representation-learning claim than the assumptions permit.

The defensible theorem boundary is:

- **Implemented invariants:** target-structure loss scaling, finite per-step
  displacement under bounded updates, and bounded Rademacher low-utility
  perturbation.
- **Conditional sieve theorem:** possible only for a generated dictionary with
  explicit richness, persistent evaluation, and no-regret readout or selector.
- **Empirical universality:** at most a bounded benchmark-matrix claim against
  fair online MLP comparators.

Any failing row should therefore be classified as one of:

- a violation of an assumption needed by the conditional theorem;
- a genuine failure despite the assumptions holding;
- a comparator/protocol problem;
- an expected no-free-lunch or identifiability limit.

## Stress Matrix

| Scenario | Falsification pressure | Limitation class | Assumption that saves the theory |
|---|---|---|---|
| `delayed_parity` | Target is a delayed high-order parity of past observations while the learner sees only current observation. | History/observability and generator closure. | Inputs include the needed finite history and parity-like primitives. |
| `hidden_context_aliasing` | Same observation maps to incompatible targets because context is latent. | Identifiability failure. | Inputs include enough state/history/context. |
| `adversarial_nonstationary_oos` | Target switches among high-frequency, parity, and discontinuous functions. | Generator closure plus drift-rate limits. | The generator covers the target sequence and drift is slow enough. |
| `rotating_relevant_subspace` | Relevant subspace rotates at fixed per-step rate without time/context input. | Plasticity time-scale failure. | Adaptation is faster than subspace drift, or time/context is observable. |
| `class_blocked_discontinuous_shift` | Class meanings change in discontinuous blocks without task identity. | Nonstationary tracking/retention tradeoff. | Task identity is observable or evaluation only rewards fast tracking. |
| `sparse_rare_feature_utility` | A dormant rare feature has no early utility, then dominates rare high-value events. | Rare-event/future-utility failure. | Useful rare features receive sufficient excitation or preserve option value. |

## Assumption Map

| Assumption | Encoded by scenarios | Expected failure if absent |
|---|---|---|
| Bounded calibrated targets | All rows; especially high-value rare events | Loss scale becomes task importance rather than a neutral normalization. |
| Observability/context sufficiency | `delayed_parity`, `hidden_context_aliasing`, `class_blocked_discontinuous_shift` | All causal learners have irreducible error or unstable tracking. |
| Persistent excitation | `sparse_rare_feature_utility`, `delayed_parity` | Useful heads/features receive too few informative gradients. |
| Generated-feature richness | `delayed_parity`, `adversarial_nonstationary_oos` | The target remains outside the reachable low-width feature class. |
| Resource budget sufficiency | All rows under fixed `hidden_sizes=(32,)` | Width/horizon increases help more than step-size tuning. |
| Nonstationary variation bound | `rotating_relevant_subspace`, `class_blocked_discontinuous_shift`, `adversarial_nonstationary_oos` | Static regret is the wrong comparator; dynamic regret is needed. |
| Fair baseline comparator | All rows | Apparent UPGD wins/losses become protocol artifacts. |

## Smoke Results

Run on May 6, 2026:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_theory_falsification.py" \
  --scenarios all \
  --steps 240 \
  --n-seeds 2 \
  --methods upgd,mlp,mlp_deep \
  --upgd-width 32 \
  --mlp-width 64 \
  --output-dir /tmp/step2_theory_falsification_t2_final \
  --note-path /tmp/step2_theory_falsification_t2_final/SUMMARY.md
```

Output:

```text
wrote /tmp/step2_theory_falsification_t2_final/results.json
wrote /tmp/step2_theory_falsification_t2_final/SUMMARY.md
```

| Scenario | Diff vs best MLP, positive favors UPGD | Wins/losses/ties | Decision |
|---|---:|---:|---|
| `delayed_parity` | +0.496801 | 2/0/0 | UPGD wins the baseline comparison, but all methods remain above the oracle floor; this exposes missing history/parity assumptions. |
| `hidden_context_aliasing` | +0.467691 | 2/0/0 | UPGD wins the baseline comparison, but the row is an identifiability limit, not a universal-theorem success. |
| `adversarial_nonstationary_oos` | +0.324603 | 2/0/0 | UPGD wins this smoke row; keep as an accepted stress case pending longer horizons. |
| `rotating_relevant_subspace` | -0.000283 | 1/1/0 | Mixed at 240 steps; requires focused confirmation. |
| `class_blocked_discontinuous_shift` | +0.018168 | 2/0/0 | UPGD wins this smoke row; still requires explicit tracking-vs-retention objective. |
| `sparse_rare_feature_utility` | +0.036924 | 2/0/0 | UPGD wins this smoke row; rare-event lower-bound remains an assumption. |

Focused confirmation for the rotation row:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_theory_falsification.py" \
  --scenarios rotating_relevant_subspace \
  --steps 600 \
  --n-seeds 3 \
  --methods upgd,mlp,mlp_deep \
  --upgd-width 32 \
  --mlp-width 64 \
  --output-dir /tmp/step2_theory_falsification_rotation_t2_fixedrate \
  --note-path /tmp/step2_theory_falsification_rotation_t2_fixedrate/SUMMARY.md
```

Output:

```text
wrote /tmp/step2_theory_falsification_rotation_t2_fixedrate/results.json
wrote /tmp/step2_theory_falsification_rotation_t2_fixedrate/SUMMARY.md
```

| Seed | `target_structure_upgd` | `mlp64` | `mlp64_64` | Best MLP minus UPGD |
|---:|---:|---:|---:|---:|
| 0 | 0.473641 | 0.513547 | 0.403138 | -0.070503 |
| 1 | 0.451146 | 0.504466 | 0.399738 | -0.051408 |
| 2 | 0.456077 | 0.502921 | 0.394839 | -0.061238 |

Summary: `diff_positive_favors_upgd = -0.061050`, wins/losses/ties = `0/3/0`.
This falsifies any claim that the promoted Step 2 learner is uniformly better
than same-run fair MLP baselines on observable but rapidly drifting relevant
subspaces under the fixed `hidden_sizes=(32,)` promoted budget.

## Decision Rules

- **Hard falsification:** A scenario satisfies observability, calibration,
  excitation, and budget assumptions, yet target-structure UPGD is consistently
  worse than the best fair MLP baseline.
- **Limitation exposure:** A scenario fails because an assumption is absent.
  This does not refute a conditional theorem, but the theory must state the
  assumption plainly.
- **Non-result:** All methods fail near an oracle floor, as in aliasing. This
  is an identifiability limit, not a learner ranking.

## Minimal Defensible Theory

The theory remains defensible only under these conditions:

1. Inputs are sufficient statistics for the supervised target or include the
   needed history/context.
2. Target vectors have a stable semantic contract: simplex classification,
   dense regression, masked inactivity, and task importance are not ambiguous.
3. The candidate generator can represent the target family within the stated
   hidden width and time horizon.
4. Useful features and heads receive enough informative updates before utility
   pruning or perturbation treats them as expendable.
5. The drift rate is below the learner's adaptation rate, or time/context
   variables make drift predictable.
6. The finite resource budget is part of the theorem statement.

Under those conditions, target-structure UPGD is a plausible single-learner
Step 2 mechanism. Without them, the universal representation-learning claim is
too strong.

## Falsifiable Predictions

1. On `delayed_parity`, adding history and a parity/compositional primitive
   should help more reliably than only increasing perturbation noise.
2. On `rotating_relevant_subspace`, slower rotation or observable time/context
   should reduce the MLP64_64 advantage; faster hidden-feature adaptation alone
   should show a tracking/retention tradeoff.
3. On `sparse_rare_feature_utility`, explicit rare-event memory, protection, or
   occurrence weighting should outperform a pure perturbation-scale change.
4. On `hidden_context_aliasing`, adding the missing context should reduce loss
   for both UPGD and MLP baselines; without context, a UPGD win is not evidence
   for a universal representation theorem.
5. On `adversarial_nonstationary_oos`, dictionary enrichment with the missing
   primitives should improve more than horizon increases when approximation is
   the bottleneck.

## Reporting Template

For each run, report:

- scenario, seed count, horizon, final-window size, and method widths;
- UPGD minus best fair MLP paired delta;
- which theorem assumption the scenario is meant to stress;
- whether the row is hard falsification, limitation exposure, or non-result;
- any evidence that widening, context, weighting, drift reduction, or
  dictionary enrichment closes the gap.
