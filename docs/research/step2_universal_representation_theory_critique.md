# Step 2 Universal Representation Theory Critique

This is the skeptical-review version of the target-structure UPGD claim. It is
written to strengthen the theory document by separating the claim that is
currently supported from the claim that is false under ordinary online learning
pathologies.

## Claim Under Review

The strong claim would be:

> A single target-structure UPGD learner is a universal online
> representation-learning mechanism for Step 2: with continual updates, bounded
> plasticity, utility-guided perturbation, and vector heads, it discovers or
> maintains the representation needed across target structures.

The defensible claim is narrower:

> On the current Step 2 benchmark matrix, target-structure UPGD is a compact
> single-learner empirical default that removes the mean-vs-sum target-loss
> ambiguity and beats fair MLP baselines under fixed budget and bounded updates.

The word "universal" is the problem. The present mechanism has no proof that
its feature generator contains every needed compositional primitive, no
identifiability guarantee under partial observation, no finite-time guarantee
for rare targets, no drift-rate guarantee, and no budget guarantee when the
target requires more independent features than the hidden layer can carry.

## T2 Counterexamples

| Counterexample | Attack | Assumption that removes it | Current result |
|---|---|---|---|
| Delayed parity | Target is a delayed high-order parity of past observations while the learner sees only current observation. | Inputs include the needed finite history and parity-like primitives. | UPGD beats the fair MLP baselines in the smoke run, but this is not a theorem success because the stream violates observability. |
| Hidden context aliasing | Two latent contexts emit the same observation but require different targets. | Observations are Markov/sufficient for the target, or history/context features are explicitly included. | UPGD beats the baselines in the smoke run; all causal learners still face an irreducible identifiability limit. |
| Adversarial nonstationary out-of-dictionary target | The stream switches among high-frequency, parity, and discontinuous targets. | The generator family covers the target sequence and drift is slow enough for bounded adaptation. | UPGD wins the short smoke row; keep it as an acceptance row pending longer horizons. |
| Rotating relevant subspace | The useful projection rotates at a fixed per-step rate without time/context input. | Utility/adaptation is faster than the subspace drift, or time/context is observable. | Hard empirical failure for the promoted fixed budget: UPGD loses to the best same-run MLP baseline on 3/3 seeds at 600 steps. |
| Discontinuous class-blocked shift | Class meanings change in blocks without task identity. | Task identity is observable, or evaluation only asks for fast tracking after each jump. | UPGD wins the short smoke row; the theorem still needs an explicit tracking-vs-retention objective. |
| Sparse rare-feature utility | A dormant rare feature has no early utility, then dominates rare high-value events. | Useful rare features receive sufficient excitation, or utility preserves option value for dormant features. | UPGD wins the short smoke row; finite-time rare-event guarantees remain unproved. |

## Empirical Falsification

The strongest T2 row is `rotating_relevant_subspace`, because it is observable
as a supervised stream but violates any unstated assumption that bounded
utility-guided adaptation can track arbitrary drift.

Run on May 6, 2026:

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

| Seed | `target_structure_upgd` | Best MLP baseline | Best MLP minus UPGD |
|---:|---:|---:|---:|
| 0 | 0.473641 | 0.403138 | -0.070503 |
| 1 | 0.451146 | 0.399738 | -0.051408 |
| 2 | 0.456077 | 0.394839 | -0.061238 |

Summary: `diff_positive_favors_upgd = -0.061050`, wins/losses/ties = `0/3/0`.

This invalidates the overbroad claim:

> Target-structure UPGD is a universal online representation learner that is
> uniformly better than fair MLP baselines across Step 2 target structures under
> the promoted fixed budget.

It does not invalidate the narrower claim that target-structure UPGD is a good
default on the current curated Step 2 benchmark matrix.

## Hidden Assumptions

Target-structure UPGD needs at least these assumptions for a theorem-like claim:

1. **Observability:** the target is a function of the learner's input or of an
   explicitly provided finite history.
2. **Loss calibration:** target scale, density, and task importance are encoded
   in a stable loss contract. Simplex classification is genuinely simplex mass
   1; zeros mean what the loss says they mean.
3. **Representational closure:** the feature generator can express the target
   family within the allowed width and time horizon.
4. **Sufficient excitation:** every useful feature/head receives enough
   informative updates before it is judged low-utility.
5. **Plasticity-stability compatibility:** the stream's drift rate is below the
   learner's bounded adaptation rate, or time/context makes drift predictable.
6. **Finite budget match:** hidden width, head count, and update budget exceed
   the intrinsic rank or compositional complexity of the target family.

Without these assumptions, the theorem is false. With them, the result is no
longer "universal representation learning"; it is a conditional online
adaptation claim for a calibrated, observable, budget-compatible target class
with bounded drift.

## What Would Strengthen The Theory

The theory should state a limited theorem with an explicit escape hatch:

- **Conditional theorem:** If targets are observable, calibrated, sufficiently
  excited, expressible within the finite candidate generator, and drift slower
  than the adaptation time scale, then bounded target-structure UPGD can maintain
  lower loss than fixed fair MLP baselines on the specified Step 2 stream class.
- **Non-theorem evidence:** The current benchmark results support this
  conditional claim empirically; they do not establish arbitrary recursive
  feature construction or arbitrary nonstationary tracking.
- **Falsification clause:** Any stress row that satisfies the assumptions but
  reliably loses to a fair MLP baseline falsifies the promoted mechanism for
  that target class. Any stress row that violates an assumption should be
  reported as a limitation class, not silently excluded.

The scientifically acceptable version is therefore not "universal." It is
"single-mechanism, target-structure-aware, bounded online representation
adaptation under explicit observability, calibration, excitation, drift-rate,
and resource conditions."
