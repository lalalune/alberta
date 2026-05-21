# Step 2 Compositional No-Regret Boundary

Date: 2026-05-07.

## Claim Under Review

The claim

```text
The compositional utility/promote rule is no-regret.
```

is not proved for the current heuristic rule. The implemented
`CompositionalFeatureLearner` has causal utility traces, candidate scoring,
promotion margins, replacement, retention guards, and an optional
generator-resource manager. Those are useful finite-resource mechanisms. They
are not, by themselves, a proof of no-regret selection over all compositional
features.

The valid replacement claim is:

```text
Finite generated compositional candidates can be selected with sublinear
fixed-candidate regret when selection is analyzed through the explicit
full-information Hedge abstraction, losses are bounded, the candidate set is
finite, and the comparator is fixed. The current promote/delete heuristic
must be treated as a causal implementation heuristic unless it is reduced to
that abstraction with all residual terms named.
```

## Selector Abstraction

The repository exposes the theorem-facing selector in
`alberta_framework.core.resource_manager`:

- `optimal_hedge_learning_rate(n_actions, horizon, loss_bound)`
- `finite_candidate_hedge_regret_bound(n_actions, horizon, learning_rate, loss_bound)`
- `LearnedResourceManager.fixed_candidate_regret_bound(horizon, loss_bound)`

These functions state the static full-information finite-candidate Hedge
bound. They do not certify a compositional feature generator, promotion rule,
or replacement policy.

## Theorem 1: Finite Compositional Candidate Hedge

Let `N` candidate compositional feature banks be active for `T` steps. At every
step, candidate `i` incurs loss `ell_{t,i} in [0, L]`. The selector predicts
with the convex mixture induced by weights

```text
w_{t+1,i} proportional to w_{t,i} * exp(-eta * ell_{t,i}).
```

Then for any fixed candidate `j`,

```text
sum_t ell_t(selector) - sum_t ell_{t,j}
  <= log(N) / eta + eta * T * L^2 / 8.
```

For `L = 1` and `eta = sqrt(8 log(N) / T)`, the average regret is
`O(sqrt(log(N) / T))`.

This theorem is a selector theorem. To make it a compositional feature
discovery theorem, an analysis must add approximation error, readout regret,
generation delay, promotion delay, deletion/retention cost,
capacity/collision cost, and drift cost.

## Counterexample 1: Promotion Utility Is Not A No-Regret Proof

The current utility/promote rule can discard a dormant but future-useful
candidate before the stream supplies evidence for it, or retain a recently
useful candidate through a later regime where another candidate is better.
This does not violate the implementation contract; it shows the heuristic is
not automatically a no-regret selector.

An adversarial two-candidate stream makes the point. Candidate `a` has loss `0`
for an initial block and loss `1` later. Candidate `b` has loss `1` initially
and loss `0` later. A static regret theorem compares against the best fixed
candidate over the whole horizon, while a promote/delete heuristic may commit
based on early utility and pay a later switching or retention term. Without an
explicit Hedge reduction or a dynamic-regret theorem, "utility promotion is
no-regret" is not established.

## Assumptions Required For Any No-Regret Claim

1. Bounded losses in a known interval.
2. A finite candidate set or an expanding set with declared prior mass and
   activation delay.
3. Full-information candidate losses, or an explicit bandit estimator and
   exploration guarantee.
4. A fixed comparator for static regret, or a declared switched/dynamic
   comparator for nonstationary claims.
5. A readout guarantee for each candidate bank, if candidates are feature banks
   rather than direct predictions.
6. Capacity and retention assumptions if candidates can be deleted.
7. Temporal causality: all weights and promotion decisions use only past
   prequential losses and current learner state.

## Claim Ledger

| Claim | Status |
|---|---|
| Compositional features can represent feature-of-feature DAGs up to configured depth. | Supported by implementation invariants. |
| The current utility/promote heuristic is causal and finite-resource. | Supported by implementation and tests. |
| The current utility/promote heuristic is no-regret. | Not proved. |
| The explicit finite-candidate Hedge abstraction has sublinear fixed-candidate regret under bounded losses. | Theorem. |
| A full Step 2 compositional theorem follows automatically from the Hedge selector theorem. | False; residual approximation, readout, delay, retention, capacity, and drift terms are still required. |

## Safe Language

Use:

- "The compositional learner is a finite causal feature-of-feature DAG
  mechanism."
- "Finite compositional candidates can be selected with a Hedge no-regret
  abstraction under bounded-loss assumptions."
- "The current utility/promotion path is empirically useful and causal, but it
  is not itself proved no-regret."

Do not use:

- "The compositional promote rule is no-regret."
- "The finite DAG learner proves arbitrary recursive feature discovery."
- "Hedge over a declared candidate set proves that the generator discovers all
  useful candidates."
