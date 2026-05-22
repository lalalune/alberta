# Step 2 Distribution-Free Limits

Date: 2026-05-07.

Worker 4 conclusion: distribution-free universality under arbitrary drift or
hidden context is false. The valid Step 2 theory is conditional learnability
under explicit stream, feature, loss, evidence, drift, and regret assumptions.

This note is deliberately proof-facing. It closes the no-free-lunch and
identifiability boundary around the stronger "universal continual learner"
claim so future Step 2 writing can cite a precise negative result rather than
re-litigating it in experiment notes.

## Marker Index

- `CLAIM-REJECTED-DISTRIBUTION-FREE-UNIVERSALITY`
- `COUNTEREXAMPLE-CAUSAL-ONLINE-INDISTINGUISHABILITY`
- `COUNTEREXAMPLE-ARBITRARY-ADVERSARIAL-DRIFT`
- `COUNTEREXAMPLE-HIDDEN-CONTEXT-ALIASING`
- `REPLACEMENT-THEOREM-CONDITIONAL-LEARNABILITY`
- `ASSUMPTION-OBSERVATION-SUFFICIENCY`
- `ASSUMPTION-BOUNDED-LOSSES`
- `ASSUMPTION-ADMITTED-FEATURE-CLASS`
- `ASSUMPTION-MODELED-DRIFT`
- `ASSUMPTION-RECURRENCE-EVIDENCE`
- `ASSUMPTION-REGRET-ESTIMATION-GUARANTEE`

## Protocol

At time `t`, a causal online learner receives the next observation `x_t`, has
access only to the labeled past

```text
H_{t-1} = ((x_1, y_1), ..., (x_{t-1}, y_{t-1})),
```

chooses a prediction `p_t = A(H_{t-1}, x_t)`, then observes `y_t` and updates.
The learner may be deterministic or randomized. For an adaptive adversary,
condition on the learner's realized internal randomness before `p_t`; the
adversary then sees `p_t`. For an oblivious stochastic hidden-context stream,
the lower bounds below hold in expectation.

Unless otherwise stated, targets are binary, predictions are real-valued, and
loss is squared loss `ell(p, y) = (p - y)^2`. Clipping predictions to `[0, 1]`
can only help the learner in these examples, and the minimax one-step squared
loss remains at least `1/4`.

## Rejected Claim

`CLAIM-REJECTED-DISTRIBUTION-FREE-UNIVERSALITY`

The following claim is false:

> There exists one causal learner that is distribution-free universal for Step
> 2 under arbitrary target drift or hidden context, without modeling the drift,
> observing enough context, bounding the loss, admitting a feature/comparator
> class, or proving a regret/estimation guarantee.

This is stronger than the usual empirical Step 2 claim and stronger than the
conditional recursive-feature theorem. It asks one online mechanism to solve
all target processes compatible with the same observed past. That demand
conflicts with causal indistinguishability: if two worlds look identical before
prediction but require different next targets, the learner must make the same
prediction in both worlds and must be wrong in at least one of them.

## Counterexample 1: Causal Online Indistinguishability

`COUNTEREXAMPLE-CAUSAL-ONLINE-INDISTINGUISHABILITY`

Fix any learner `A`, any finite labeled history `h`, and any next observation
`x`. Consider two streams:

```text
S0: past h, next observation x, next target y = 0
S1: past h, next observation x, next target y = 1
```

The streams have identical past observations and, in this stronger version,
identical full labeled past. They also present the same current observation
before the prediction. Therefore the causal learner must output the same
prediction `p = A(h, x)` on both streams.

For squared loss,

```text
max{ell(p, 0), ell(p, 1)} = max{p^2, (1 - p)^2} >= 1/4.
```

The inequality is tight at `p = 1/2`. Thus no causal learner can guarantee
one-step squared loss below `1/4` on every stream continuation. For absolute
loss the corresponding lower bound is `1/2`.

This is not a weakness of UPGD, MLPs, or any specific optimizer. It is an
information-theoretic obstruction. The obstruction applies before architecture
or optimization enters the proof.

Finite-horizon version. Repeat the construction as a binary tree of stream
continuations. At each node the two children share the entire observed history
at that node and differ only in the next target. Any claimed uniform next-step
guarantee fails on at least one child. An adaptive adversary can choose that
child online after seeing `p_t`, yielding constant per-step loss.

## Counterexample 2: Arbitrary Adversarial Drift

`COUNTEREXAMPLE-ARBITRARY-ADVERSARIAL-DRIFT`

Let the observation be constant: `x_t = 0` for all `t`. After the learner
commits to `p_t`, define the target by

```text
y_t = 1 if p_t <= 1/2 else 0.
```

Then each step satisfies

```text
ell(p_t, y_t) >= 1/4.
```

The target process is allowed to flip after the learner commits. It may switch
on every step. A dynamic comparator that is allowed to move arbitrarily and see
the realized target can have zero loss, but its path variation is order `T`.
Therefore a theorem with no drift model cannot give sublinear tracking regret
against such a comparator. The missing assumption is not more features; it is a
modeled nonstationarity class, such as stationary targets, bounded switches,
bounded path variation, predictable context, or a switching-regret budget.

This counterexample also blocks a "continual adaptation beats arbitrary drift"
claim. Bounded plasticity helps only when the drift has structure or evidence
that can be used causally.

## Counterexample 3: Hidden Context Aliasing

`COUNTEREXAMPLE-HIDDEN-CONTEXT-ALIASING`

Let there be a latent context `c_t in {0, 1}`. The learner observes only
`x_t = 0`; the context is not encoded in the observation, and the target is

```text
y_t = c_t.
```

If `c_t` is an independent fair coin, then before predicting at time `t`, any
causal `p_t` is independent of the current hidden context. The expected
one-step squared loss is

```text
E[ell(p_t, y_t) | H_{t-1}, x_t]
  = 0.5 * p_t^2 + 0.5 * (1 - p_t)^2
  >= 1/4.
```

Again the bound is tight at `p_t = 1/2`. This is a stationary stream, not an
arbitrary-drift trick. The target is unidentifiable because the observation
aliases two latent states that require different predictions.

The deterministic indistinguishability version is equivalent. Two latent
worlds can share the same observed and labeled history through `t - 1`, present
the same `x_t`, and differ only in hidden `c_t`, hence in `y_t`. The learner's
causal state is identical, so it emits the same prediction and loses on at
least one world.

If the observation is changed to `x_t = (0, c_t)`, the problem disappears: the
feature class containing the context coordinate can represent `y_t = c_t`
exactly. The boundary is therefore identifiability, not optimizer quality.

## What These Counterexamples Rule Out

They rule out all theorem statements with this shape:

```text
For every stream distribution or every target path, a fixed causal learner
achieves vanishing loss or sublinear regret without restrictions on hidden
state, target drift, loss scale, comparator class, or evidence.
```

They do not rule out empirical benchmark closure, conditional universal
approximation, finite expert regret, dynamic regret under bounded variation, or
learnability when the hidden context is encoded in observations or inferred
from recurrent evidence.

## Replacement Theorem: Conditional Learnability

`REPLACEMENT-THEOREM-CONDITIONAL-LEARNABILITY`

The valid replacement theorem has the following form.

Let `F_B` be an admitted finite-resource feature or comparator class. Let the
learner run causally with bounded per-step work. Assume:

1. **`ASSUMPTION-OBSERVATION-SUFFICIENCY`.** The observation summary includes
   the current context or enough finite history/predictive state to make the
   target identifiable up to an explicit irreducible noise term `Bayes_T`.
   Hidden context not encoded in observations must be charged to `Bayes_T`; it
   cannot be learned away by a universal optimizer.
2. **`ASSUMPTION-BOUNDED-LOSSES`.** Targets, predictions, masks, and
   normalization keep the per-step loss in `[0, 1]`, or in a stated bounded
   interval after rescaling. Regret and concentration guarantees require this
   or a comparable tail condition.
3. **`ASSUMPTION-ADMITTED-FEATURE-CLASS`.** The theorem names the comparator
   class: fixed features, generated recursive features, a finite dictionary
   prefix, bounded-width MLPs, GVF-derived features, or another declared class.
   Claims are relative to this class and include its approximation error
   `Approx_T(F_B)`.
4. **`ASSUMPTION-MODELED-DRIFT`.** Nonstationarity is restricted by a model:
   stationary targets, bounded switches, bounded path variation `V_T`, a
   predictable context process, or an explicit dynamic comparator budget. With
   arbitrary adversarial drift, `V_T = Theta(T)` and sublinear tracking regret
   is unavailable.
5. **`ASSUMPTION-RECURRENCE-EVIDENCE`.** Useful contexts, features, or heads
   recur often enough, and with enough non-collinear variation, to estimate
   their value before the horizon ends. Rare or dormant utilities require a
   waiting-time, coverage, or protection assumption.
6. **`ASSUMPTION-REGRET-ESTIMATION-GUARANTEE`.** The learner or analysis
   abstraction supplies a regret, tracking-regret, or statistical-estimation
   bound over the admitted class, plus any discovery latency and retention
   penalties for generated features.

Under these assumptions, a proof can target a bound of the form

```text
(1 / T) * sum_{t=1}^T ell_t(A_t)
  <= Approx_T(F_B)
     + Bayes_T / T
     + Regret_T(F_B) / T
     + Estimation_T / T
     + Drift_T(V_T) / T
     + Discovery_T / T
     + Retention_T / T.
```

If the approximation term, irreducible aliasing term, average regret, average
estimation error, average drift charge, discovery latency, and retention
penalty all vanish under the declared resource schedule, the average loss
vanishes relative to the modeled target class. If the feature budget grows and
the admitted class is dense in the observable target family, this becomes a
conditional universal-approximation-plus-online-learning result.

Proof skeleton:

1. Observation sufficiency removes the hidden-context aliasing term, or else
   records it as `Bayes_T`.
2. The admitted feature class gives a comparator with approximation error
   `Approx_T(F_B)`.
3. Recurrence/evidence and bounded losses make the comparator's utility or
   readout estimable with error `Estimation_T`.
4. The regret or estimation guarantee bounds online excess loss relative to
   the admitted comparator.
5. The modeled drift assumption converts static regret to dynamic or switching
   regret with an explicit `Drift_T(V_T)` term.
6. Discovery and retention mechanisms are either guaranteed or charged.
7. Divide by `T`.

This is the theorem that can replace the false distribution-free universality
claim. It is useful precisely because every escape hatch is named and testable.

## Consequence For Step 2 Claims

Step 2 may claim a single empirical learner closes a declared benchmark matrix
under a fixed online protocol. It may claim conditional universality for a
specified observable target family, feature generator, drift model, loss
contract, recurrence condition, and regret/estimation proof. It must not claim
distribution-free universality under arbitrary drift or unobserved context.
