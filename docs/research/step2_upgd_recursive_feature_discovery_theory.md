# Step 2 UPGD Recursive Feature Discovery Theory Boundary

Date: 2026-05-07.

## Claim Under Review

The claim

```text
UPGD discovers arbitrary recursive features.
```

is not a defensible theorem as written. Utility-based Perturbed Gradient
Descent (UPGD) is a finite causal learner with a declared architecture,
bounded perturbation schedule, and target-structure loss normalization. It can
be part of a recursive feature-discovery mechanism, but only after the
generator class, evidence process, loss bounds, capacity budget, and drift
model are stated.

The defensible replacement is:

```text
UPGD/utility selection can exploit recursively generated features that are
inside a declared finite or scheduled expanding class, receive enough evidence,
fit within capacity, use bounded losses, and face bounded drift. Remaining
error terms must include approximation, selection, optimization, capacity,
evidence-delay, and drift residuals.
```

This note gives one negative theorem/counterexample ledger and one positive
conditional theorem. It intentionally does not claim arbitrary universality.

## Assumption Registry

Every theorem or counterexample below references assumptions by these labels.

- **A1: Declared/generated class.** The allowed primitive inputs, recursive
  operations, parameter ranges, value clipping, maximum depth, and candidate
  schedule are declared before evaluation. For an expanding class, the schedule
  and prior or activation rule are declared.
- **A2: Evidence/excitation.** A useful feature is proposed, evaluated, and
  receives enough informative gradients, targets, or key hits before the
  horizon at which the claim is judged.
- **A3: Bounded losses and updates.** The prequential loss used for selection
  is clipped or normalized to `[0, 1]`; predictions, targets, feature values,
  gradients, update scales, and perturbations are bounded wherever the proof
  uses concentration, regret, or finite-displacement arguments.
- **A4: Capacity budget.** The active feature slots, candidate slots, hidden
  units, memory rows, readout degrees of freedom, and update budget are finite
  and large enough for the comparator, or a capacity residual is included.
- **A5: Drift model.** The target process is stationary, has at most `S_T`
  switches, or has bounded path variation `V_T`. Arbitrary adversarial drift
  is excluded unless the comparator and bound also pay for it.
- **A6: Causal information.** The learner and comparator use only the
  observation/history available online. Hidden state, future labels, task ids,
  replay, or oracle feature names are excluded unless explicitly declared.
- **A7: Selector/readout guarantee.** The utility, promotion, mixture, or
  readout mechanism has a no-regret, dynamic-regret, or explicitly budgeted
  optimization guarantee against the declared comparator.
- **A8: Target-structure semantics.** Multi-head targets, sparse active heads,
  simplex-like targets, dense vector targets, and masking rules are declared so
  loss normalization matches the target structure being evaluated.

## THEOREM N1: No arbitrary recursive feature discovery without declared assumptions

**Assumptions used:** A1, A2, A3, A4, A5, A6, A7, A8.

For any finite-resource causal UPGD or UPGD-plus-utility-selection learner, the
unqualified statement "discovers arbitrary recursive features" is false as a
distribution-free theorem. If any one of A1, A2, A3, A4, or A5 is omitted,
there is a target process within the informal phrase "arbitrary recursive
feature" for which the learner has a non-vanishing excess-loss term relative to
a stronger comparator that is allowed the omitted resource or assumption.

More concretely:

- Without A1, the target feature can be outside the generated/reachable class.
- Without A2, the useful feature can receive no evidence before selection.
- Without A3, a single unbounded loss can dominate any finite regret budget.
- Without A4, the target can require more independent retained features than
  the learner can store.
- Without A5, the target can drift in response to the learner faster than any
  causal fixed-budget learner can track.

**Proof sketch.** A causal finite learner has only the information, candidates,
parameters, slots, and update magnitudes made available by its protocol. Each
omitted assumption allows a standard diagonal or indistinguishability
construction: choose a target outside the class, hide evidence until after the
decision horizon, amplify one loss, require more independent features than
capacity, or switch the target adversarially after observing predictions. In
each case a comparator with the omitted privilege can have lower loss, while
the learner cannot identify, retain, or bound the needed feature. Therefore the
unqualified arbitrary-discovery claim must be replaced by a conditional theorem
with explicit residual terms.

## COUNTEREXAMPLE C1: Missing generated class

**Omitted assumption:** A1.

**Assumptions held for the construction:** A2, A3, A4, A5, A6, A8.

Let the learner declare no recursive generator, or declare a generator that
contains only a fixed finite set of feature maps through the evaluation
horizon. Choose a stationary binary target on a finite input domain whose label
function is outside the span or decision class of every feature map reachable
by that generator and readout budget. Such a function exists whenever the
domain supports more dichotomies than the finite-capacity reachable class can
realize; equivalently, choose a recursive lookup table or high-order Boolean
feature beyond the declared depth, primitives, or slots.

A comparator that is allowed the missing feature has zero loss on the same
stream. The learner has a positive approximation residual. This is not a
failure of optimization; the feature was never in the declared class.

## COUNTEREXAMPLE C2: No evidence or excitation

**Omitted assumption:** A2.

**Assumptions held for the construction:** A1, A3, A4, A5, A6, A8.

Consider two stationary environments that produce the same observations,
targets, gradients, and candidate-utility statistics for the first `T` steps,
but require different recursive features after step `T`. Any causal learner has
the same state in both worlds at time `T`, so it must make the same selection
or allocation decision in both. At least one world is wrong for that decision.

A comparator that receives evaluation evidence for the needed feature before
selection can choose correctly. The learner without excitation pays an
evidence-delay or misselection residual. No utility rule can select a feature
from evidence it has not received.

## COUNTEREXAMPLE C3: Unbounded losses

**Omitted assumption:** A3.

**Assumptions held for the construction:** A1, A2, A4, A5, A6, A8.

Take any finite candidate selector with any finite cumulative regret budget
claimed independently of loss scale. Run bounded losses until a late time, then
set one selected candidate's loss to `M` while a comparator candidate has loss
`0`. Since `M` is arbitrary, the excess loss can exceed any proposed
scale-free bound.

The usual finite-class selection proof requires losses in `[0, 1]`, or an
explicit known range. UPGD target-structure normalization and bounded updates
are useful only if the theorem states the bounds it relies on.

## COUNTEREXAMPLE C4: Finite capacity

**Omitted assumption:** A4.

**Assumptions held for the construction:** A1, A2, A3, A5, A6, A8.

Let a stream cycle through `K + 1` independent recurring contexts, each with a
different required feature/value binding, while the learner can retain only
`K` active features, hidden units, or memory rows relevant to those contexts.
By the pigeonhole principle, two required bindings must collide, be aliased, or
force destructive replacement. When both contexts recur, at least one incurs
loss unless the theorem includes a capacity residual.

A comparator with `K + 1` retained bindings can avoid this residual. A
finite-budget UPGD or utility-selection learner cannot prove arbitrary
retention without paying for capacity.

## COUNTEREXAMPLE C5: Arbitrary drift

**Omitted assumption:** A5.

**Assumptions held for the construction:** A1, A2, A3, A4, A6, A8.

For squared loss with predictions clipped to `[0, 1]`, define the next target
after observing the learner's prediction `p_t` as `y_t = 1` if `p_t < 1/2` and
`y_t = 0` otherwise. The learner's per-step squared loss is at least `1/4`.
A comparator allowed to switch after seeing the same adversarial rule can
predict `y_t` and incur zero loss.

This target is not covered by a stationary, switched-with-budget, or
bounded-variation drift model. Any positive theorem must either exclude this
case or include a dynamic-regret term large enough to pay for it.

## THEOREM P1: Conditional finite/expanding generated-class UPGD selection

**Assumptions used:** A1, A2, A3, A4, A5, A6, A7, A8.

Fix a horizon `T`. Let `G_T` be a finite generated class of recursive feature
banks, or an expanding generated class whose candidates have a declared prior
mass `pi_j > 0` and activation schedule. Let losses be prequential and bounded
in `[0, 1]`. Let `B_j` denote candidate feature bank `j` together with its
bounded target-structure UPGD/readout update rule. Suppose there exists a
comparator `j_star` in the generated class, activated by delay
`tau(j_star)`, with approximation residual `epsilon_G(T)` under the declared
target and drift model.

Assume the selector/readout satisfies the regret statement required by A7:

```text
sum_{t=1}^T loss_t(learner)
  - sum_{t=1}^T loss_t(B_{j_star})
  <= R_select(T, j_star) + R_upgd(T, j_star) + C_eval(tau(j_star)).
```

Then the average excess prequential loss against the declared target class is

```text
(1/T) sum_{t=1}^T loss_t(learner) - BayesRisk_declared
  <= epsilon_G(T)
   + R_select(T, j_star) / T
   + R_upgd(T, j_star) / T
   + C_eval(tau(j_star)) / T
   + C_capacity(T) / T
   + C_drift(T) / T.
```

For a finite class of `N` always-active candidates with exponential weights and
losses in `[0, 1]`,

```text
R_select(T, j_star) <= log(N) / eta + eta T / 8.
```

Choosing `eta = sqrt(8 log(N) / T)` gives
`R_select(T, j_star) <= sqrt(T log(N) / 2)`. For an expanding countable class,
replace `log(N)` with `log(1 / pi_{j_star})` after paying the declared
activation delay.

**Proof sketch.** The selector part is the standard exponential-weights
potential argument for bounded losses, or the assumed A7 utility-selection
regret bound. The UPGD/readout contribution is not free: it is represented by
`R_upgd`, which may include nonconvex optimization error, perturbation
hitting-time error, target-structure normalization mismatch, and local-update
residuals. Approximation, evaluation delay, capacity, and drift are separate
terms because each corresponds to a separate assumption. Dividing by `T` gives
the average-loss statement.

**Interpretation.** This theorem supports a bounded target-structure
UPGD/utility-selection claim over a finite or scheduled expanding generated
class. It proves selection or exploitation of generated recursive features
only when the useful feature is in the class, is evaluated, is retained, and is
covered by the stated regret and drift model.

## NON-THEOREM MARKER: Arbitrary universality is not claimed

This document does not prove that bare UPGD discovers arbitrary recursive
features. It does not prove that utility perturbations enumerate all recursive
programs, preserve all dormant future-useful features, or solve hidden-state
aliasing. It does not remove approximation, evidence, bounded-loss, capacity,
optimization, or drift residuals.

The strongest defensible Step 2 proof closure is conditional:

```text
declared generated class
+ sufficient evidence
+ bounded target-structure loss
+ finite compatible capacity
+ explicit drift model
+ selector/readout guarantee
=> bounded excess loss with named residual terms.
```
