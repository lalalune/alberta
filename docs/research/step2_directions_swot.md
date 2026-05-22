# Step 2 Directions and SWOT

Ranked from most obvious to most ambitious.  The ranking is about experimental
tractability, not scientific importance.

## 1. Same-budget MLP + ObGD Baseline Grid

- Status: implemented and essential; fair-width MLPs erased the old
  pair-product final-window headline.
- Strength: mandatory control; already close to state of the art for streaming
  deep supervised learning in this repo.
- Weakness: does not answer feature lifecycle questions directly.
- Opportunity: defines the minimum bar for every Step 2 proposal.
- Threat: weak baselines make generate-and-test look better than it is.

## 2. Static Random Nonlinear Feature Bank

- Status: implemented as a control in the feature-discovery scripts.
- Strength: simplest fixed-budget nonlinear representation.
- Weakness: no construction pressure after initialization.
- Opportunity: separates representation capacity from replacement policy.
- Threat: can win by luck on small synthetic tasks and mislead conclusions.

## 3. Random Generate-and-Test Replacement

- Status: scaffolded, but not supported by a committed audited win.
- Strength: literal implementation of create, score, discard.
- Weakness: disruptive; replacing active features can hurt interim performance.
- Opportunity: tests whether online utility has any predictive value.
- Threat: poor generator quality can dominate the result.

## 4. Shadow Candidate Testing

- Status: works in controlled pair-product streams; strongest as an in-class
  mechanism, not as a universal generator.
- Strength: candidates collect evidence before affecting predictions.
- Weakness: doubles bookkeeping and adds hyperparameters for promotion.
- Opportunity: directly addresses Step 2's interim-performance concern.
- Threat: candidate utility is still biased by residuals produced by the active
  bank.

## 5. Exact Interaction Feature Discovery

- Status: scientifically useful positive control. It proves bounded
  construction/testing/deletion can work when the true feature class is known.
- Strength: constructed features are unambiguous products of existing features.
- Weakness: strong inductive bias; not a general nonlinear feature generator.
- Opportunity: gives a clean falsifiable demonstration of Step 2 mechanics.
- Threat: success may not transfer to features that are not sparse low-order
  interactions.

## 6. Utility Estimator Ablation

- Status: still high priority. UPGD's success makes this more important, not
  less, because we need to know which utility signal is doing the work.
- Strength: isolates the scientific core: how utility is assigned.
- Weakness: ablation-loss estimates can be expensive or noisy online.
- Opportunity: compare weight magnitude, gradient credit, residual utility,
  rare-task utility, and ablation utility under equal streams.
- Threat: utility measures may be benchmark-specific.

## 7. Rare-Task and Context-Aware Utility

- Status: open. The context-disentanglement result shows that recurring
  context-specific slopes can dominate final-window error.
- Current implementation: `FixedBudgetInteractionLearner` has opt-in
  `utility_aggregation="max"` and `utility_retention_decay`. Smoke evidence
  shows mean utility dilutes rare-head evidence, but max utility is not a clean
  default win.
- Strength: addresses multitask nonstationarity directly.
- Weakness: requires distinguishing temporarily inactive from permanently bad
  features.
- Opportunity: preserve features that matter in recurring contexts.
- Threat: over-protection can freeze obsolete features and block adaptation.

## 8. UPGD-Style Perturbation Instead of Hard Replacement

- Status: implemented and currently the strongest out-of-class evidence:
  `UPGDLearner` beats the best fair MLP on all three canonical synthetic
  streams. External digits evidence is negative, so the claim must remain
  stream-family-specific.
- Strength: preserves learned structure while exploring low-utility features.
- Weakness: harder to reason about than slot replacement; per-weight
  perturbation does not expose explicit feature provenance or candidate tests.
- Opportunity: may reduce catastrophic interim loss during feature turnover.
- Threat: slow drift can waste resources if bad features are never fully reset.

## 9. Recursive Compositional Feature DAG

- Status: implemented as `CompositionalFeatureLearner`, with topology tests and
  cascade replacement. Current results are mostly negative: random DAG search
  gets only a tiny polynomial edge and loses to fair MLP and UPGD on the harder
  canonical streams. Candidate refresh now respects `candidate_min_age`,
  improving lifecycle validity but not solving the performance gap.
- Strength: closest to the Alberta Plan's long-term representation goal.
- Weakness: deletion becomes dependency management, not just slot replacement.
- Opportunity: features can be built from features, not only raw observations.
- Threat: credit assignment across a changing DAG can become unstable and
  opaque.

## 10. Meta-Learned Resource Manager

- Status: not implemented.
- Strength: treats generator choice, candidate budget, replacement rate, and
  promotion aggressiveness as learnable online decisions.
- Weakness: high-dimensional nonstationary meta-control problem.
- Opportunity: prior experience constructing features can shape future feature
  proposals.
- Threat: easy to overfit synthetic streams; needs strong controls before any
  claim of generality.

## 11. External Online Supervised Benchmarks

- Status: first benchmark implemented with sklearn digits. Fair MLP beats UPGD
  on final-window and held-out accuracy over 5 seeds.
- Strength: checks whether synthetic-stream conclusions survive contact with a
  dataset not designed around the methods.
- Weakness: digits is small, stationary, and not a continuing agent stream.
- Opportunity: expand to class-blocked/permuted/nonstationary dataset streams
  and tune methods under equal compute.
- Threat: external benchmarks can drift toward ordinary supervised learning and
  stop testing the Step 2 resource-management question.
