# Step 2 Universal Recursive Feature Theorem

Date: 2026-05-07.

This note gives the strongest theorem-level statement that is defensible for
Alberta Plan Steps 1-4 in this repository. The result is conditional. It is not
a proof that the current `UPGDLearner.step2_default()` discovers arbitrary
features on arbitrary streams. It is a theorem package for a temporally uniform
recursive-feature sieve, plus an explicit map from that idealized object to the
implemented Step 2 learners.

The short version is:

> Recursive feature construction can be universal only relative to a rich
> generator, bounded observable targets, sufficient recurrence/excitation, a
> resource schedule large enough for the required feature set, and a learner or
> selector with a no-regret guarantee over generated candidates. Fixed-width
> UPGD supplies useful online plasticity and low-utility exploration, but the
> exact UPGD recursion does not yet prove the no-regret selection or retention
> assumptions.

## Known Work And Local Claim

Known foundations:

- The Alberta Plan frames Step 1 as fixed-feature continual supervised
  learning, Step 2 as supervised feature finding, Step 3 as GVF prediction, and
  Step 4 as continual control: <https://arxiv.org/abs/2208.11173>.
- Universal approximation for one-hidden-layer neural networks is classical.
  Cybenko proves uniform approximation of continuous functions on compact
  domains by finite sums of sigmoidal ridge functions:
  <https://link.springer.com/article/10.1007/BF02551274>. Hornik,
  Stinchcombe, and White establish broad approximation results for multilayer
  feedforward networks:
  <https://www.sciencedirect.com/science/article/pii/0893608089900208>.
- Prediction with expert advice and online convex optimization give no-regret
  bounds relative to a finite comparator class, not assumption-free dominance:
  <https://www.cambridge.org/core/books/prediction-learning-and-games/A05C9F6ABC752FAB8954C885D0065C8F>
  and <https://arxiv.org/abs/1804.04529>.
- No-free-lunch results block any unconditional claim that one learner is best
  over all target functions without an inductive-bias/model assumption:
  <https://dblp.org/rec/journals/neco/Wolpert96>.
- UPGD and continual backprop motivate continual random or utility-guided
  exploration to prevent loss of plasticity:
  <https://arxiv.org/abs/2302.03281> and
  <https://arxiv.org/abs/2108.06325>.
- Horde/GVF supplies the Step 3 target family into which Step 2 features are
  handed: <https://josephmodayil.com/papers/horde-final.pdf>.

Local contributions and deviations:

- `loss_normalization="target_structure"` is a local target semantics rule. It
  is not in the UPGD paper. It makes simplex targets use sum-style pressure and
  dense or multilabel targets use mean-style pressure.
- `UPGDLearner.step2_default()` is a fixed-resource empirical learner with
  target-structure loss, ObGD bounding, sparse initialization, LayerNorm, and
  bounded Rademacher low-utility perturbation.
- `UPGDLearner.step2_strict_digit_readout_default()` is a heavier one-state
  simplex/readout branch for digit-style streams.
- The theorem below is not a theorem for the exact UPGD update. It is a
  theorem for an idealized recursive sieve whose assumptions can be checked,
  approximated, or falsified by the local runners.

## Definitions

### Temporally Uniform Supervised Stream

At each time `t = 1, 2, ...`, the agent receives an observation summary
`x_t in X`, predicts `hat_y_t in R^d`, then receives a target `y_t in R^d` and
updates. `X` is compact. Targets and predictions are clipped or otherwise
calibrated so the per-step loss

```text
ell_t(g) = L(g(x_t), y_t)
```

is in `[0, 1]`. Masked heads are excluded from the loss. In Step 3, `y_t` may
be a TD or GVF target computed causally from `(x_t, r_{t+1}, x_{t+1})`; the
theorem then applies to the prediction subproblem only.

### Recursive Feature Grammar

Let `Phi_0` contain the constant feature and coordinate projections of `x`.
For depth `r >= 0`, define

```text
Phi_{r+1} = Phi_r union {
    sigma(a_0 + sum_{j=1}^k a_j phi_j(x)):
    k finite, phi_j in Phi_r, a_j in Q_grid
}
```

where `sigma` is a non-polynomial squashing or piecewise-linear activation and
`Q_grid` is a countable dense coefficient grid. Let
`Phi_* = union_r Phi_r`. A finite recursive representation is a set
`S subset Phi_*` with `|S| <= B`; its prediction class is a linear readout over
`S`.

This definition deliberately includes ordinary MLP hidden units as recursive
features. Memory features, prototype features, and GVF-derived features can be
included only if their state updates are causal and their feature maps are part
of the declared grammar.

### Budgeted Recursive Constructor

A budgeted constructor keeps at most `B` active features. On every time step it:

1. predicts using the active features and current readout;
2. updates the readout from the current prediction error;
3. updates utility or confidence statistics for active and recently proposed
   candidates;
4. proposes, perturbs, or recycles features using only past and current data;
5. never uses task identity, future targets, replay from outside the declared
   memory budget, or held-out evaluation feedback.

This is the idealized object. The current UPGD learner implements the online
update, bounded perturbation, and utility-tracking parts; it does not implement
a proved no-regret selector over all recursive candidates.

## Assumptions

The theorem needs the following assumptions. Removing any one of them produces
a known failure mode.

**A1. Observability.** The input summary `x_t` contains enough current or
history information for the target. If the same `x_t` can causally require two
different targets, the irreducible error belongs to the stream, not the learner.

**A2. Bounded calibrated loss.** Targets, predictions, and loss scaling keep
`ell_t in [0, 1]`. For vector targets, target semantics are fixed: simplex,
dense regression, masked inactivity, and multilabel targets are not confused.

**A3. Recursive richness.** For every continuous comparator target
`f: X -> R^d` and every `epsilon > 0`, there exists a finite recursive set
`S_epsilon subset Phi_*` and linear readout `w_epsilon` such that

```text
sup_{x in X} ||w_epsilon^T S_epsilon(x) - f(x)|| <= epsilon.
```

For fixed budget `B`, this holds only for functions whose required set has
`|S_epsilon| <= B`. Full universality requires a budget or dictionary prefix
that can grow as `epsilon` shrinks.

**A4. Persistent excitation.** Useful candidate features are observed often
enough, and with enough non-collinear variation, for their utility and readout
weights to be estimated before they are discarded. Rare features require either
rare-event protection, explicit memory, or a bound on the waiting time between
informative events.

**A5. No-regret readout or selector.** Conditional on a finite candidate prefix
`D_N`, the online readout or selector has regret `R_T(N)` against the best
fixed readout over that prefix:

```text
sum_{t=1}^T ell_t(h_t) <= min_{h in H(D_N)} sum_{t=1}^T ell_t(h) + R_T(N),
```

with `R_T(N) / T -> 0` for bounded losses.

**A6. Adequate retention.** Once a set `S_epsilon` is generated and empirically
validated, the resource manager either keeps it or pays an explicit retention
penalty `D_T`. Utility estimates may be noisy, but the probability of deleting
every useful representative must be controlled.

**A7. Bounded nonstationarity.** For a target path `f_1, ..., f_T`, define

```text
V_T = sum_{t=2}^T ||f_t - f_{t-1}||_infty.
```

Static-regret claims require `V_T = 0` or negligible. Tracking claims must
include `V_T` or compare against a switching/dynamic comparator.

**A8. Temporal uniformity and finite compute.** All updates are causal and
performed with bounded per-step work. A proposal or recycling process may run
periodically, but the schedule is fixed or causal and uses no special training
phase.

## Proposition 1: Recursive Density

Under A3, the recursive grammar `Phi_*` is dense in `C(X, R^d)`.

**Proof sketch.** The first recursive layer contains finite sums of
`sigma(a_0 + a^T x)` with coefficients from a dense grid. By Cybenko-style
universal approximation, these finite sums uniformly approximate any scalar
continuous function on compact `X`. Dense rational/grid coefficients preserve
approximation after a small perturbation of the real coefficients. Apply the
scalar result coordinate-wise for vector targets. Deeper recursive layers only
enlarge the class, so the union over depths remains dense.

**Implication.** This is an expressivity theorem. It says the grammar contains
good features. It does not say the online algorithm finds them, keeps them, or
learns their readout fast enough.

## Proposition 2: Finite Prefix Selection

For any finite candidate prefix `D_N` and bounded losses in `[0, 1]`, the
exponential-weights forecaster over `N` experts satisfies

```text
L_T <= min_i L_{T,i} + log(N) / eta + eta * T / 8.
```

Choosing `eta = sqrt(8 log(N) / T)` gives average regret
`O(sqrt(log(N) / T))`.

**Proof sketch.** This is the standard potential argument for prediction with
expert advice and Hoeffding's lemma. It is included here only as the clean
finite-selection component. The current UPGD learner does not literally run
Hedge over all generated recursive feature sets.

## Theorem: Conditional Universal Recursive Feature Construction

Fix `epsilon > 0`, horizon `T`, confidence `1 - delta`, and a target path
`f_1, ..., f_T` satisfying A1, A2, and A7. Suppose a temporally uniform
constructor satisfies A3-A8 and, by time `tau <= T`, has generated or retained
a set `S_epsilon` of size at most `B` whose readout class approximates the
best comparator path with average approximation error at most `epsilon^2`.
Then, with probability at least `1 - delta`,

```text
(1 / T) sum_{t=1}^T ell_t(h_t)
  <= epsilon^2
     + R_T(N, delta) / T
     + D_T(delta) / T
     + C_V * V_T / T
     + C_tau * tau / T
     + C_noise * estimation_T(delta) / T.
```

Here:

- `R_T(N, delta)` is the finite-prefix readout or selector regret;
- `D_T(delta)` is the retention/deletion penalty for useful features;
- `V_T` is the nonstationary target variation;
- `tau` is the discovery latency for the useful recursive set;
- `estimation_T` covers utility-estimation and stochastic-target error;
- constants depend on the loss Lipschitz bound and clipping scale.

If `V_T / T -> 0`, `tau / T -> 0`, `R_T / T -> 0`, `D_T / T -> 0`, and
`estimation_T / T -> 0`, the average loss approaches the approximation error.
If the budget grows so that `epsilon -> 0`, the constructor is universal over
the declared observable continuous target class.

**Proof sketch.**

1. By Proposition 1 and A3, a finite recursive feature set exists for the
   target class at accuracy `epsilon`.
2. By A4 and the definition of `tau`, the constructor receives enough evidence
   to identify or retain a useful representative after a finite latency.
3. By A5, online readout or selection over the finite prefix pays only
   sublinear regret relative to that representative.
4. By A6, any deletion of useful features is either prevented or charged to
   `D_T`.
5. By A7, replacing a stationary comparator by a moving comparator introduces
   a variation term controlled by the loss Lipschitz constant.
6. Divide the cumulative bound by `T`.

This is the theorem that can be cited. It is conditional, comparator-relative,
and budget-relative.

## Compatibility With Alberta Plan Temporal Uniformity

The theorem is compatible with temporal uniformity because every component has
a causal per-step role:

- predictions are made before observing the current target;
- readout, utility, and resource statistics update on every step;
- feature proposals and perturbations are caused by online state, not by a
  special offline training phase;
- memory or prototype features count against an explicit budget;
- Step 3 GVF targets may be constructed online from the current transition;
- Step 4 can consume the same features for control, but the theorem only
  covers prediction error, not policy optimality.

The theorem therefore supports the Alberta Plan discipline, but only under the
assumptions above.

## What The Current Repo Implements

Implemented ingredients:

- Step 1 supplies fixed-feature online supervised learners and adaptive
  step-size machinery.
- Step 2 UPGD supplies shared nonlinear hidden features, target-structure loss,
  ObGD bounded updates, utility tracking, and low-utility perturbation.
- Step 2 associative/prototype memory supplies fixed-budget retained features
  for domains where recurrence or rare views matter.
- Step 3 consumes Step 2 features as GVF/Horde prediction inputs.
- Step 4 consumes the same feature stream for SARSA or Horde actor-critic.

Missing for an exact theorem of the current UPGD recursion:

- no proof that random or utility-weighted perturbations enumerate a dense
  recursive grammar at a useful rate;
- no no-regret theorem for the nonlinear UPGD parameter recursion;
- no proved protection against deleting every rare but future-useful feature;
- no dynamic-regret theorem for abrupt class-blocked or label-drift streams;
- no proof that the same fixed budget handles all continuous target families;
- no control theorem showing policy improvement over Q/SARSA.

## Falsifiable Corollaries

**Corollary 1: Fixed budget is not globally universal.** For any fixed feature
budget `B`, there are continuous targets whose required approximation budget is
larger than `B`. Widening or enriching the generator should reduce the
approximation term if this is the bottleneck.

**Corollary 2: Missing history breaks the theorem.** If the target depends on
unobserved finite history, adding that history to `x_t` should help more than
retuning perturbation or kappa.

**Corollary 3: Rare utility needs recurrence or protection.** A feature that is
only useful after a long dormant interval can be discarded by any finite
utility-only manager unless A4/A6 are enforced. Rare-event memory or deletion
protection should outperform simple noise-scale tuning on such rows.

**Corollary 4: Drift has a measurable cost.** Faster rotating subspaces or
abrupt target remaps should increase the `V_T / T` term. Slower drift,
observable context, or dynamic-regret mechanisms should close the gap.

**Corollary 5: Target semantics matter.** Exact-zero dense targets and sparse
multilabel targets should fail under a normalizer that treats all zeros as
inactive one-hot structure. `target_structure` should preserve the correct
loss denominator in these cases.

**Corollary 6: GVF feature construction is a Step 3 extension.** The Step 2
theorem can cover GVF targets only after those targets are exposed as bounded
online prediction targets. Off-policy GVF convergence and control improvement
require separate Step 3/4 theorems.

## Acceptance Criteria For Citation

This theorem can be cited in Step 2/3 docs only with the following wording:

> A conditional recursive-feature sieve theorem shows that temporally uniform
> online feature construction can be universal over bounded observable target
> classes when the generator is dense, useful features are recurrent enough to
> be evaluated, the finite budget can contain the required feature set, and the
> readout/selector has no-regret behavior. The current UPGD default implements
> several ingredients but remains an empirical learner, not the exact theorem.

Required citation checklist:

1. Link this document.
2. State A1-A8 or a task-specific subset explicitly.
3. Say whether the claim is expressivity, online regret, tracking, or empirical
   benchmark dominance.
4. Do not say the theorem proves exact `UPGDLearner.step2_default()`.
5. Keep Step 3 off-policy convergence and Step 4 control optimality outside
   the Step 2 theorem.
6. Run the theory invariant tests:

```bash
source .venv/bin/activate
pytest tests/test_step2_theory_invariants.py -q
```

7. For empirical citation, pair the theorem with the relevant benchmark
   artifacts and report failed assumptions for negative rows.

## Machine-Checkable Invariants

The repository already encodes lightweight invariants in
`tests/test_step2_theory_invariants.py`:

- the promoted Step 2 default matches the finite-resource assumption;
- target-structure denominators separate simplex, dense, and multilabel
  semantics;
- Rademacher perturbation magnitude is bounded and monotone in low utility;
- finite-expert selection requires bounded losses;
- missing dictionary elements produce an approximation gap;
- nested generated-feature prefixes do not increase best-readout
  approximation error;
- nonstationary variation is separate from approximation error;
- average excess loss decomposes into approximation plus regret.

These tests do not prove the theorem. They prevent the most important
assumptions from being silently erased in future documentation.

## Bottom Line

The theorem-level result is a conditional universality theorem for a
temporally uniform recursive-feature sieve. It is useful because it names the
exact places where a Step 2 learner must earn the word "universal":
generator richness, recurrence, budget adequacy, retention, no-regret readout,
and bounded nonstationarity. The current UPGD learners are strong empirical
candidates inside that framework, but the remaining proof obligations are
real and should stay visible.
