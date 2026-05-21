# Step 2 Universal Representation Theory

Date: 2026-05-06.

This note formalizes the strongest defensible theory for the current
non-router Step 2 learner:

`UPGDLearner.step2_default(n_heads)`

The current default is a single shared-trunk vector-output UPGD learner with
`loss_normalization="target_structure"`, bounded ObGD updates, sparse hidden
features, low-noise Rademacher perturbations of low-utility hidden weights, and
lean bookkeeping. It is not an output router, not a portfolio over learners,
and not an MLP fallback. It is also not an unconditional universal learner.

## Thesis

Target-structure UPGD is best understood as a scale-consistent continual
representation learner with a stochastic low-utility feature exploration
channel. The supported theory is conditional:

- Target-structure normalization gives a clean causal invariant across dense
  vector regression, one-hot classification, and sparse multilabel targets.
- Bounded updates and bounded perturbations make the finite-horizon process
  resource-controlled.
- Universal approximation can be reduced to online retention only if the
  feature dictionary is allowed to expand or if the target family is restricted
  to functions approximable by the fixed hidden budget.
- A finite-budget UPGD learner can be empirically universal for the current
  Step 2 matrix, but it cannot be distribution-free universal in the
  no-free-lunch sense.

The rest of this note separates theorem, proposition, conjecture, and empirical
claim.

## Universality Taxonomy

The word "universal" is ambiguous. For Step 2, the following claims must be
kept separate.

### Expressivity

Expressivity is a statement about a class of functions that could be represented
by some parameter setting. For the current finite-width default, the relevant
class is:

```text
F(H, T, B, S) =
  vector-output MLPs with hidden budget H, parameters reachable by T bounded
  UPGD updates from sparse initialization, and Rademacher perturbation schedule S
```

This class is finite-resource and path-dependent. It may contain excellent
features for the current benchmark matrix, but it is not all continuous
functions on a compact set. A classical universal-approximation argument only
applies to a growing-width family or an explicitly dense dictionary sequence; it
does not say the online UPGD update will find the approximating parameter.

### Online Adaptation

Online adaptation is a statement about tracking the stream with prediction
before update and no replay. The current learner has causal adaptation
mechanisms: SGD, target-structure loss scaling, bounded ObGD steps, and
low-utility perturbations. These mechanisms do not by themselves imply
convergence in a nonconvex, nonstationary stream. Any adaptation theorem needs a
model of recurrence, drift, gradient excitation, and feature-retention pressure.

### Regret Or Competitiveness

Regret compares cumulative prequential loss with a comparator class. The current
default does not maintain a convex expert mixture or a proven online convex
optimization update over representations, so no sublinear regret theorem is
currently established for the implemented UPGD recursion. A regret theorem can
be proved for an idealized finite selector over candidate feature banks; that is
a design reduction, not a theorem for the exact deployed default.

### Empirical Universality

Empirical universality means one learner closes a declared benchmark matrix
without task ids, replay, output routers, or dataset-specific branches. This is
the current Step 2 claim. It is bounded by the task matrix, comparator protocol,
seeds, metrics, and resource budget.

## Setting

Let time run for a finite horizon `T`. At time `t`, the learner receives an
input `x_t in X subset R^d`, observes an active target vector
`y_t in R^m union {NaN}^m`, predicts `f_t(x_t; theta_t) in R^m`, and updates
once. NaN coordinates are inactive. Let `A_t` be the active coordinates.

The target-structure loss is

```text
L_t(theta) = 0.5 * sum_{i in A_t} a_{t,i}
             (f_i(x_t; theta) - y_{t,i})^2 / d_t
```

where `a_{t,i}` are optional positive/negative target weights and

```text
d_t = 1                      if y_t[A_t] is a non-negative simplex
d_t = |A_t|                  otherwise.
```

For this note, "simplex" means all active entries are non-negative and their
active mass is one up to the implementation tolerance.

## Assumptions

These assumptions are intentionally explicit. A result below applies only when
its listed assumptions hold.

1. Bounded inputs. There is `B_x < infinity` such that `||x_t||_2 <= B_x`.
2. Bounded active targets. There is `B_y < infinity` such that
   `|y_{t,i}| <= B_y` for all active heads.
3. Finite horizon. All claims are over `t = 1, ..., T`; asymptotic language
   means a sequence of finite-horizon problems with increasing `T` or
   increasing resource budget.
4. Smooth bounded loss on the visited parameter set. On the parameter region
   reachable under the update bounds, each active loss is `G`-Lipschitz and
   `H`-smooth in the parameters used by the update.
5. Resource constraint. The learner has finite width, finite number of heads,
   finite perturbation interval, finite step size, and finite ObGD envelope.
6. Perturbation distribution. Current Step 2 default uses bounded Rademacher
   noise with scale `sigma * (1 - u_norm)^beta`, applied to hidden weights on
   perturbation steps. A Gaussian variant needs a high-probability boundedness
   statement instead of an almost-sure one.
7. Persistent excitation. A feature or target component can only be learned if
   the stream revisits states or gradients that reveal its utility often enough
   before the horizon ends. Rare targets require explicit occurrence lower
   bounds or cannot be guaranteed.
8. Target sparsity/density is structural, not dataset identity. Dense
   regression, exact-zero dense rows, one-hot classification, and sparse
   multilabel rows are distinguished only by the current target vector.
9. Generated-feature class. For a finite-width theorem, the comparator must be
   restricted to features reachable by the configured initialization,
   perturbation distribution, step-size schedule, and update bounds, or to a
   declared finite candidate dictionary evaluated by the analysis abstraction.
10. Dictionary richness. For an asymptotic universal theorem, there must be a
   sequence of finite dictionaries `Phi_K` whose bounded readouts are dense in
   the target class under the stream distribution. Without this assumption the
   approximation term need not vanish.
11. Nonstationary environment model. Either the target process is stationary,
   piecewise stationary with a bounded number of switches, or has bounded path
   variation `V_T = sum_t d(f_t^*, f_{t-1}^*)`. A theorem must state which model
   it uses. Arbitrary adversarial target changes are not learnable.
12. Fair MLP comparator. A comparator MLP must use the same online protocol,
   no replay, no future labels, same target encoding, comparable width/compute
   sweeps, same normalization choices where applicable, and the same bounded
   update class if bounding is part of the proposed learner.

## Strongest Defensible Theorem

There is no honest theorem saying that the current finite
`UPGDLearner.step2_default` is distribution-free universal. The strongest
defensible statement has two layers:

1. For the implemented learner, target-structure normalization, bounded
   Rademacher perturbation, and ObGD-style update bounding imply finite-horizon
   scale and displacement invariants.
2. For an idealized generated-dictionary UPGD-sieve that actually evaluates a
   finite candidate dictionary and uses a no-regret selector/readout, an
   asymptotic universal approximation claim follows from dictionary richness
   plus online regret assumptions.

The second layer is the proof attempt for universal representation learning. It
is conditional, and it names the lemmas still missing for the current learner.

## Theorem 1: Target-Structure Scale Invariance

Assume bounded active targets and finite active head count. On any step where
the target vector is active:

1. If `y_t[A_t]` is a non-negative simplex, `d_t = 1`; the gradient equals the
   sum-loss multihead squared-error gradient.
2. If `y_t[A_t]` is dense regression, including exact-zero coordinates,
   `d_t = |A_t|`; the gradient equals the mean-loss vector-regression
   gradient.
3. If `y_t[A_t]` is sparse multilabel with total mass not equal to one,
   `d_t = |A_t|`; the gradient equals the mean-loss multilabel gradient.

Proof. The loss differs from ordinary active-head squared error only by the
scalar denominator `d_t` and optional per-coordinate weights `a_{t,i}`. For
simplex targets the definition sets `d_t = 1`, so differentiation gives the
same gradient as sum-loss squared error. For all non-simplex active target
vectors the definition sets `d_t = |A_t|`, so differentiation gives the mean
over active heads. Exact-zero dense coordinates remain active because activity
is determined by non-NaN supervision, not nonzero value. Sparse multilabel rows
with mass different from one are non-simplex and therefore mean-normalized.

What this proves. The normalizer has the desired update-scale invariant.

What this does not prove. It does not prove convergence, feature discovery, or
dominance over MLPs.

## Proposition 1: Bounded One-Step Displacement

Assume the active loss gradient is finite, ObGD scales the raw update by a
factor in `[0, 1]`, the base step size is finite, group/head multipliers are
clipped to finite intervals, and perturbation noise is Rademacher. Then each
parameter update has finite norm bounded by a constant determined by the ObGD
envelope, multiplier bounds, hidden width, and perturbation scale.

Proof sketch. The deterministic gradient step is a finite raw gradient times a
finite step size and finite multiplier, then ObGD can only shrink the step.
Rademacher perturbation has coordinate magnitude at most `sigma` times
`(1 - u_norm)^beta <= 1` on perturbed hidden weights. With a finite number of
parameters, the perturbation norm is finite. The sum of the deterministic and
perturbation components is therefore finite. For Gaussian perturbations, replace
the deterministic perturbation bound with a tail bound at the desired
confidence.

Interpretation. This supports finite-horizon stability and auditability. It is
not a regret bound.

## Proposition 2: Low-Utility Exploration Channel

Assume `sigma > 0`, `beta >= 0`, a perturbation step occurs, and a hidden
weight has normalized utility `u_norm < 1`. Under Rademacher perturbations that
weight receives nonzero perturbation magnitude
`sigma * (1 - u_norm)^beta` on that perturbation step. Thus lower utility gives
weakly larger perturbation scale.

Proof. The scale is a monotone function of `1 - u_norm` for `beta >= 0`.
Rademacher noise has magnitude one. If `u_norm < 1` and `sigma > 0`, the product
is positive.

Interpretation. UPGD maintains a causal exploration mechanism for hidden
weights currently estimated to be low utility. This is a representation-learning
mechanism only to the extent that perturbations can create features later
selected by gradient descent and utility. By itself it is not a complete search
algorithm over functions.

## Theorem 2: Finite Candidate Selection Bound

This theorem is not a claim about the exact current implementation. It is a
clean selection lemma for any UPGD variant or analysis abstraction that
evaluates a finite set of candidate feature maps under bounded prequential
losses and retains them by exponential weights.

Let there be `N` candidate feature maps or feature banks. At each time `t`, each
candidate incurs a loss `ell_{t,j} in [0, 1]`. The selector predicts with
weights `p_{t,j}` and suffers mixture loss
`ell_t = sum_j p_{t,j} ell_{t,j}`. Initialize `p_{1,j} = 1/N` and update

```text
p_{t+1,j} proportional to p_{t,j} * exp(-eta * ell_{t,j}).
```

Then for any candidate `j`,

```text
sum_t ell_t - sum_t ell_{t,j} <= log(N) / eta + eta * T / 8.
```

Choosing `eta = sqrt(8 * log(N) / T)` gives regret at most
`sqrt(T * log(N) / 2)` against the best fixed candidate in hindsight.

Proof. Let `W_t = sum_j w_{t,j}` and `w_{1,j} = 1`. The update gives
`W_{t+1}/W_t = sum_j p_{t,j} exp(-eta ell_{t,j})`. By Hoeffding's lemma for
`ell_{t,j} in [0, 1]`,

```text
log(W_{t+1}/W_t) <= -eta * ell_t + eta^2 / 8.
```

Summing over `t`,

```text
log(W_{T+1}/W_1) <= -eta * sum_t ell_t + eta^2 T / 8.
```

For any fixed `j`,

```text
W_{T+1} >= exp(-eta * sum_t ell_{t,j})
```

and `W_1 = N`, so

```text
-eta * sum_t ell_{t,j} - log(N)
  <= -eta * sum_t ell_t + eta^2 T / 8.
```

Rearranging proves the bound.

Why this matters for Step 2. If a finite UPGD candidate pool contains a useful
feature map and the system evaluates candidates with bounded losses, online
selection can retain that candidate with sublinear regret. This is a theorem
about the selection layer. The current default UPGD does not literally maintain
an exponential-weights pool, so this result should be used as a design reduction
or audit baseline, not as a theorem already proved for the deployed class.

## Theorem 3: Conditional Generated-Dictionary Universality

Let `X` be compact and let squared losses be clipped or otherwise normalized to
`[0, 1]`. Let `F` be a target-process class such that each environment is either
stationary, piecewise stationary with `S_T` switches, or has bounded path
variation `V_T`. Let `Phi_1 subset Phi_2 subset ...` be finite dictionaries of
bounded generated features. Assume:

1. Richness. For every stationary target function `f in F`, every stream
   distribution in the declared family, and every `epsilon > 0`, there is a
   finite `K` and a bounded linear readout over `Phi_K` whose expected loss is
   within `epsilon` of the Bayes risk.
2. Evaluation. The learner's resource schedule eventually evaluates every
   feature in the chosen `Phi_K` on enough informative examples.
3. Online readout regret. Conditional on `Phi_K`, the learner has sublinear
   regret `R_T(K) = o(T)` against the best bounded readout over `Phi_K` in the
   stationary case.
4. Nonstationary tracking. In the nonstationary case, the learner has dynamic
   regret `R_T(K, S_T)` against the best `S_T`-switch readout sequence, or
   `R_T(K, V_T)` against the best bounded-variation readout path.

Then, after the evaluation delay for `Phi_K`, the stationary average excess
prequential loss is at most

```text
epsilon + R_T(K) / T.
```

For the piecewise or bounded-variation model, replace `R_T(K)` with the
corresponding dynamic-regret term. If `R_T/T -> 0` and the environment variation
is sublinear in the sense required by that dynamic-regret bound, the average
excess loss approaches `epsilon`.

Proof sketch. Choose `K` that gives approximation error `epsilon`. Decompose
excess loss into approximation error of `Phi_K` plus online regret against the
best admissible readout or readout path in `Phi_K`. The first term is at most
`epsilon` by the dictionary assumption. The second term is `R_T/T` by the online
or dynamic-regret assumption after persistent evaluation begins.

Interpretation. This is the honest path to a universal representation theorem.
Universality comes from the expanding dense dictionary or from restricting the
target family to the closure of the fixed dictionary. A fixed finite-width UPGD
network is not universal over all continuous functions with a finite horizon.

Status for current Step 2. The implemented default supplies bounded updates,
bounded Rademacher perturbations, and the target-structure loss invariant. It
does not yet supply the evaluation, hitting-time, retention, or regret lemmas
needed to instantiate this theorem with the actual UPGD recursion.

## Conjecture 1: UPGD As Approximate Sieve Search

Under bounded inputs, bounded targets, bounded updates, persistent excitation,
low but nonzero perturbation, enough width, and a target family whose useful
features are reachable by small perturbations of the current hidden units,
target-structure UPGD behaves like a noisy resource-limited sieve: low-utility
weights explore, gradient descent amplifies useful perturbations, and utility
estimates protect features that reduce future loss.

This is plausible but unproved. The missing proof obligations are:

- hitting time for useful perturbations in high dimension;
- utility consistency under nonstationary targets;
- protection of temporarily dormant but future-useful features;
- interaction between layer norm, sparse initialization, ObGD scaling, and
  perturbation;
- finite-width interference between heads.

## Empirical Claim

The current empirical claim is narrower than the conjecture:

Target-structure UPGD is the strongest current no-router Step 2 learner in this
repo. It beats the fair MLP baselines reported in the current Step 2 synthetic
and digits matrices while using a single continually updated learner and no
dataset router. See `docs/research/step2_current_best.md` and
`docs/research/step2_upgd_critical_paper.md`.

This remains an empirical Step 2 closure claim. It is not a theorem that UPGD
dominates all MLPs or all online learners.

## Current Theory-Test Evidence

Two compact probes now make the theorem boundary sharper.

The falsification runner
`examples/The Alberta Plan/Step2/step2_theory_falsification.py` compares
target-structure UPGD against fair MLP baselines on deliberately adversarial
streams. The quick run in `outputs/step2_theory_falsification_quick/` found
positive UPGD deltas on `scale_density` and `feature_aliasing`, but negative
deltas on `delayed_utility`, `sparse_rare_targets`, `dense_zero_heads`,
`label_drift`, `outside_composition`, and `finite_resource`. Those losses are
not surprising; they are exactly the cases where a universal theorem would need
observability, calibrated targets, recurrence/excitation, future utility, and
finite-resource assumptions.

The sieve probe
`examples/The Alberta Plan/Step2/step2_universal_sieve_probe.py` tests whether
larger UPGD hidden budgets improve final-window MSE on approximation-friendly
families. The moderate run in `output/step2_universal_sieve_probe_moderate/`
supports the capacity-scaling assumption on only `1/5` families under its
monotone-or-material-improvement criterion. That result argues against using a
fixed finite-width UPGD learner as a proof object for universal approximation.
The honest theorem path remains the sieve reduction above: universality must
come from an expanding dense dictionary, a restricted target family, or an
explicit resource schedule, not from the current finite empirical default by
itself.

## Adversarial Counterarguments

Non-identifiability. Many hidden representations implement the same prediction
function. Utility based on `|w * grad|` is not an identifiable measure of
semantic feature usefulness. It is an operational credit signal, not a proof
that the learner discovered the human-intended latent variable.

Nonstationarity. A feature can be low utility now and high utility later.
Perturbing or recycling it can damage retention. Any theorem needs a recurrence
or bounded-drift assumption, or it must include dynamic regret rather than
static regret.

Off-policy and TD targets. Step 2 supervised prequential losses are simpler
than bootstrapped TD targets. In TD, targets depend on the current value
function, behavior policy, and bootstrapping operator. Bounded supervised loss
results do not automatically transfer to off-policy GVFs or control.

No-free-lunch. Without restrictions on target families or data distributions,
for any learner there are streams on which it fails. The proper claim is
conditional universality over a specified compact domain, target class,
dictionary schedule, excitation condition, and resource budget.

Finite compute. A universal approximation theorem with a growing dictionary may
be irrelevant if the required width or horizon is infeasible. Step 2 evidence
therefore needs compute-normalized comparisons, not only asymptotic existence.

Target leakage and hand-engineering risk. Target-structure normalization uses
the observed target vector to choose loss scale. This is causal and local, but
it still injects a supervised prior about target encoding. It must not be
described as discovering target structure from inputs alone.

Representation learning or loss-shaping prior. Target-structure normalization
is primarily loss shaping: it fixes gradient scale across target encodings.
UPGD perturbation plus utility tracking is the representation-learning
mechanism. The combined algorithm is a representation learner, but the
normalizer itself is not sufficient to claim representation discovery.

Fair MLP comparator. A weak MLP baseline can make UPGD look universal. Fair
comparisons require width sweeps, matched online protocol, matched target
encoding, no replay, no future labels, and comparable update bounding.

## What Cannot Currently Be Proved

The following claims are not proved by the current algorithm or evidence.

1. Discovery of every useful feature. Rademacher perturbations are local and
   utility-scaled. There is no hitting-time bound showing that they reach a
   useful hidden representation in high dimension within the finite horizon.
2. Retention of dormant features. The utility signal is based on present
   contribution to loss gradients. It cannot identify future usefulness without
   recurrence, context, a memory prior, or a dynamic-regret model.
3. Regret against the best representation in hindsight. The UPGD recursion is
   nonconvex and does not implement an exponential-weights selector over all
   hidden representations.
4. Distribution-free universality. If target functions or label mappings change
   arbitrarily, an adversary can make any causal learner fail.
5. Dominance over all MLPs. A fair comparator class can include larger widths,
   different optimizers, or different inductive biases. The current evidence is
   against declared fair baselines, not all possible MLP learners.
6. Transfer to TD/GVF/control. Supervised bounded-loss invariants do not handle
   bootstrapped targets, off-policy corrections, or policy-induced data
   distributions.

## Missing Lemmas For A Current-UPGD Theorem

A stronger theorem for the exact learner would need at least:

- Reachability lemma: nonzero probability, with finite hitting-time control, of
  generating an `epsilon`-useful feature from the perturbation schedule.
- Utility consistency lemma: low utility implies low future value, or a bounded
  penalty when this implication fails under the declared nonstationary model.
- Protection lemma: features with high future value are not erased faster than
  the learner can reconstruct them.
- Stability lemma: ObGD, layer norm, sparse initialization, and readout updates
  keep the visited parameter set in a region with bounded gradients and losses.
- Regret lemma: the coupled hidden-feature and readout updates compete with a
  declared comparator, such as best fixed generated feature bank, best
  `S_T`-switch bank, or best bounded-variation bank.
- Excitation lemma: every feature/head needed by the comparator receives enough
  informative gradients before the horizon.

## Falsifiable Predictions

The theorem boundary implies concrete predictions:

- If the target class is outside the generated dictionary, increasing horizon
  alone should not remove the approximation gap.
- If rare heads violate the occurrence lower bound, UPGD should lose to a
  baseline with explicit rare-event memory or stronger head weighting.
- If the label process has unobserved switches, all causal learners should show
  an irreducible tracking/retention tradeoff.
- If a benchmark is approximation-limited rather than optimization-limited,
  increasing hidden budget or adding the missing primitive should help more
  than tuning the perturbation scale.
- If a dormant feature later becomes useful and recurrence is weak, utility
  perturbation or recycling should damage retention unless protected by an
  explicit future-utility mechanism.

## Practical Theorem Checklist

Before making a formal claim in a paper or README, label it as one of:

- Theorem: target-structure scale invariance; finite candidate selection under
  explicit bounded-loss exponential weighting.
- Proposition: bounded one-step displacement; low-utility perturbation channel;
  sieve reduction under an expanding dictionary and regret assumption.
- Conjecture: current UPGD approximates a useful sieve search over reachable
  hidden features.
- Empirical claim: current default target-structure UPGD beats the reported
  fair MLP baselines on the Step 2 matrix.

The current learner supports the first, second, and fourth bullets. The third
is the research program for a stronger universal representation theorem.
