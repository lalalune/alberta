# Step 2 Formal Recursive Feature Discovery Boundary

Date: 2026-05-07.

## Purpose

This note states the strongest formal claim that is currently defensible for
Step 2 recursive feature discovery, and separates that claim from what the
implemented learners actually prove.

The phrase "arbitrary recursive feature discovery" is too strong for the
current code. The repo contains useful mechanisms:

- `UPGDLearner`: a fixed-width MLP with target-structure loss scaling,
  bounded updates, and low-utility perturbations.
- `FixedBudgetFeatureLearner`: a finite feature bank with candidate testing,
  utility, promotion, and replacement.
- `CompositionalFeatureLearner`: a finite directed acyclic graph (DAG) of raw,
  product, sum, tanh, and gated features, with bounded depth and candidate
  slots.
- `AssociativeMemoryLearner`: a fixed-size sparse key/value table over causal
  discrete context features.

These mechanisms support conditional approximation and selection claims. They
do not prove distribution-free discovery of every useful recursive feature.

## Formal Setting

Let `x_t in X` be the observation, `y_t` the active supervised target, and
`ell_t(pred, y_t)` a bounded prequential loss in `[0, 1]`. Prediction is made
before the update at each time step.

A recursive generated feature class is defined by:

```text
Phi_{K,D} = all features generated from K primitive raw/context features
            by a declared finite operation set O, with topological depth <= D,
            finite numeric parameters in a bounded set, and value clipping.
```

For the current compositional learner, `O = {raw, product, sum, tanh, gated}`,
active slots are finite, candidates are finite, and the implemented forward
pass clips feature values. For associative memory, the feature class is instead
a finite set of discrete causal key templates over the context window.

Let `A_{K,D}` be a bounded readout class over `Phi_{K,D}`. Define the
approximation term

```text
epsilon_{K,D} =
  inf_{a in A_{K,D}} E[ell(a(Phi_{K,D}(x)), y)] - BayesRisk
```

for the declared stationary target/distribution class. In nonstationary
settings, replace the static comparator with an admissible path whose switches
or path variation are explicitly bounded.

## Assumptions

The conditional theorem below requires all of these assumptions.

1. Bounded causal observations or contexts. The target is a function of the
   learner's input/history, or an irreducible unobservability term is included.
2. Bounded loss. Losses used by the selector and regret statement are clipped
   or normalized to `[0, 1]`.
3. Finite resource schedule. `K`, `D`, active slots, candidate slots, table
   rows, step sizes, update multipliers, and perturbation magnitudes are finite.
4. Declared generator class. The allowed operations, key templates, parameter
   ranges, and value clipping are fixed before evaluation.
5. Dictionary richness. For every target in the claimed class and every
   `epsilon > 0`, some finite `Phi_{K,D}` has approximation error at most
   `epsilon`, or else the theorem is only for the restricted closure of the
   fixed dictionary.
6. Evaluation/excitation. Every feature needed by the comparator is proposed,
   evaluated, and receives informative gradients or key hits often enough
   before the horizon ends.
7. Online selection/readout regret. The learner or analysis abstraction has a
   no-regret guarantee against the declared fixed comparator, switched
   comparator, or bounded-variation comparator.
8. Drift model. The target process is stationary, has at most `S_T` switches, or
   has bounded path variation `V_T`. Arbitrary adversarial drift is excluded.
9. Capacity compatibility. The useful features, readout degrees of freedom, and
   associative keys fit within the fixed budget, or a nonzero capacity error is
   included.
10. No future labels, replay, task ids, or oracle feature names are used unless
    they are explicitly part of the comparator protocol.

## Theorem 1: Conditional Recursive Feature Selection

Fix a finite generated dictionary `Phi_{K,D}` with `N` candidate feature maps or
feature banks. At each time `t`, candidate `j` incurs bounded loss
`ell_{t,j} in [0, 1]`. A selector uses exponential weights,

```text
p_{t+1,j} proportional to p_{t,j} exp(-eta ell_{t,j}),
```

and predicts with the convex mixture of candidate predictions. Then for any
fixed candidate `j`,

```text
sum_t ell_t - sum_t ell_{t,j}
  <= log(N) / eta + eta T / 8.
```

With `eta = sqrt(8 log(N) / T)`, fixed-candidate regret is at most
`sqrt(T log(N) / 2)`.

If the best readout over `Phi_{K,D}` has approximation error
`epsilon_{K,D}` and the readout learner has regret `R_readout(T, K, D)`, then
the average excess prequential loss against the target class is bounded by

```text
epsilon_{K,D}
  + R_selector(T, N) / T
  + R_readout(T, K, D) / T
  + C_eval(K, D) / T.
```

Here `C_eval` is the finite delay or cost required to propose and evaluate the
useful candidate. In a nonstationary setting, replace the fixed-comparator
regret with the appropriate dynamic-regret term, for example
`R_dyn(T, K, D, S_T)` or `R_dyn(T, K, D, V_T)`.

Proof sketch. The selector bound is the standard exponential-weights argument
for losses in `[0, 1]`: compare the log potential upper bound from Hoeffding's
lemma with the lower bound supplied by any fixed candidate. The excess-loss
decomposition then adds approximation error, selector regret, readout regret,
and evaluation delay. The dynamic version is the same reduction with a
switched or bounded-variation comparator and its corresponding dynamic-regret
bound.

What this proves. Recursive feature discovery can be reduced to finite
candidate approximation plus online selection when the useful recursive feature
appears in the evaluated dictionary and receives enough evidence.

What this does not prove. It does not prove that the current UPGD recursion,
the current utility-promotion rule, or the associative table implements the
exponential-weights selector. It also does not remove the approximation,
excitation, capacity, or drift terms.

## Proposition 1: Current UPGD Is A Bounded Local Search Ingredient

`UPGDLearner.step2_default` supplies a finite-width shared-trunk MLP,
target-structure loss normalization, ObGD-style bounded updates, and bounded
Rademacher perturbations of low-utility hidden weights. Under bounded inputs and
finite gradients on the visited parameter set, each update has finite
displacement.

This supports a resource-controlled local stochastic search view:

```text
UPGD loss <= best reachable fixed-width MLP loss
            + optimization/nonconvexity error
            + perturbation/hitting-time error
            + drift/retention error.
```

The current implementation does not prove that the perturbation path hits every
useful recursive feature, that utility estimates preserve dormant future-useful
features, or that the nonconvex update has regret against the best hidden
representation in hindsight.

Boundary: UPGD can be used as an empirical approximation mechanism for features
inside its reachable fixed-width class. It is not a theorem of arbitrary
recursive feature construction.

## Proposition 2: Current Compositional And Fixed-Budget Learners Are Finite
Generated-Dictionary Mechanisms

`FixedBudgetFeatureLearner` and `CompositionalFeatureLearner` make the
construction, testing, promotion, utility, and deletion surface explicit.
`CompositionalFeatureLearner` additionally has true feature-of-feature
recursion because composed slots can use earlier composed slots as parents.

For these learners the formal dictionary is finite at any configured budget:

```text
n_active < infinity,
n_candidates < infinity,
depth <= max_depth,
op in {raw, product, sum, tanh, gated}.
```

If a target is approximable by the configured operations and depth, if the
candidate schedule evaluates the needed feature, and if promotion/retention
acts like a no-regret selector, then Theorem 1 gives the right form of bound.

The implemented utility rules are causal and testable, but they are not
currently proved no-regret selectors. A missing primitive, insufficient depth,
candidate churn, rare-head under-excitation, or hidden context aliasing leaves a
nonzero residual term.

Boundary: these learners support formal recursive selection over a declared
finite generator class. They do not support unbounded-depth or arbitrary
operation discovery.

## Proposition 3: Associative Memory Is A Finite Causal Key/Value Mechanism

`AssociativeMemoryLearner` maps a bounded discrete context window to a finite
set of causal key templates and stores value logits in a fixed-size table. When
contexts recur, labels are consistent, generated keys are discriminative, and
the table has enough rows to avoid destructive replacement, repeated bindings
can be learned online.

Its formal approximation terms are different from the continuous feature DAG:

```text
associative loss <= best finite key-template loss
                  + collision/aliasing error
                  + table-capacity replacement error
                  + finite-repeat adaptation error.
```

Boundary: associative memory is a useful fixed-budget sequence feature
mechanism. It is not a proof of arbitrary recursive feature abstraction, and it
does not solve contexts whose needed state is not encoded by the key templates
or table budget.

## Proposition 4: Unobservability Gives An Irreducible Gap

If two latent states produce the same observation/history available to the
learner but require different targets, no causal learner using only that
observation/history can distinguish them. The best prediction collapses to the
conditional mean, leaving irreducible loss. A hidden-state-aware comparator can
have lower loss, but that comparator is outside the learner's information set.

This is not an implementation weakness. It is a formal boundary on any claim of
arbitrary discovery.

## Claim Ledger

| Claim | Status for current repo |
|---|---|
| Bounded target-structure UPGD updates are finite-resource and causal. | Supported. |
| UPGD approximates useful hidden features on declared Step 2 benchmarks. | Empirical. |
| UPGD discovers arbitrary recursive features. | Not proved. |
| Compositional features can represent feature-of-feature DAGs up to configured depth. | Supported by implementation invariants. |
| The compositional utility/promote rule is no-regret. | Not proved. |
| Finite generated dictionaries can be selected with sublinear fixed-candidate regret under bounded losses. | Theorem for the selector abstraction. |
| Associative memory can learn repeated finite key/value bindings under capacity and recurrence conditions. | Supported by mechanism and tests. |
| Associative memory is arbitrary recursive continuous feature discovery. | False. |
| Distribution-free universality under arbitrary drift or hidden context. | False. |

## Practical Test Boundary

Tests for this note should encode the following invariants rather than claiming
the full theorem for the current learners:

- A recursive dictionary missing the required feature leaves a nonzero
  approximation term.
- Adding the required finite recursive feature can remove that approximation
  term on a controlled target.
- Exponential weighting over bounded candidate losses satisfies the finite
  selection regret bound.
- Current learner configs have finite budgets and finite depth/table/hidden
  resources.
- UPGD has bounded perturbation/update ingredients but no explicit finite
  recursive candidate selector in its config.
- Unobservable latent context produces an irreducible loss gap.

## Bottom Line

The defensible formal statement is:

> Recursive feature discovery is approximable or selectable when the useful
> feature lies in a declared finite or expanding generated dictionary, the
> stream provides sufficient causal evidence, losses are bounded, drift is
> modeled, and the selector/readout has regret control. The error decomposes
> into approximation, selector regret, readout regret, evaluation delay,
> capacity/collision, and drift terms.

The current Step 2 learners provide important ingredients and empirical
evidence, but they do not prove arbitrary recursive feature discovery.
