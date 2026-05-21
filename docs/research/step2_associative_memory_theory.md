# Step 2 Associative Memory Theory Boundary

Date: 2026-05-07.

This note closes the theory boundary for the Step 2 associative-memory branch.
It separates a defensible finite key/value result from an overbroad feature
discovery claim.

The defensible claim is conditional:

> A finite associative memory can learn repeated finite key/value bindings when
> the bindings recur, keys are separable or collisions are bounded, updates are
> stable and bounded, table capacity and retention are adequate, and target drift
> is slower than the adaptation time scale.

The overbroad claim is false:

> A finite associative memory is arbitrary recursive continuous feature
> discovery.

The current `AssociativeMemoryLearner` is best understood as a finite causal
key/value mechanism for discrete token contexts. It can be strong evidence for
repeated local associations. It is not, by itself, a universal recursive feature
constructor for arbitrary continuous functions.

## Claim Ledger

| Claim | Status | Boundary |
|---|---|---|
| Repeated finite key/value bindings can be learned. | Conditionally true. | Requires recurrence, separability or controlled collision, bounded updates, enough capacity, retention, and bounded drift. |
| Sparse associative features can help Step 2 sequence prediction. | Empirical mechanism claim. | Supported only by configured benchmarks and probes, not by an assumption-free theorem. |
| Finite associative memory is arbitrary recursive continuous feature discovery. | False. | A finite table with finite active key abstractions cannot separate every continuous target on a compact domain. |
| The Step 2 associative implementation proves universal representation learning. | Not claimed. | It is a bounded online memory branch, not a proof of arbitrary feature discovery. |

## Setup

Let a stream produce causal contexts `x_t`, target values `y_t`, and generated
associative keys `K(x_t)`. A finite associative memory has at most `M` rows.
Each row stores a key and a value vector. Prediction happens before the current
write, so the first occurrence of a binding is a warmup event rather than a
successful recall event.

A finite binding set is

```text
B = {(k_i, v_i): i = 1, ..., N}.
```

The intended interpretation is that when the stream emits a context whose
stable key is `k_i`, the desired value is `v_i`. Values may be class labels,
one-hot distributions, logits, or bounded regression targets. The theorem below
does not require the exact implementation update. It applies to any causal
finite table whose per-hit update is contractive toward the observed value and
whose inactive drift and collision error are bounded.

## Assumptions

The positive result needs all of the following assumptions. If one is removed,
the result becomes a capacity, identifiability, or tracking claim rather than a
finite associative-memory learning theorem.

**A1. Recurrence.** Each binding `(k_i, v_i)` occurs infinitely often, or at
least enough times before evaluation. For finite-horizon statements, each
binding has a maximum inter-arrival gap `R_i` after its first appearance.
Without recurrence, a prediction-before-write memory cannot learn a value that
has never been observed.

**A2. Separability and collision control.** Each binding has at least one stable
causal key that is not shared with an incompatible value. Hash collisions,
feature collisions, and many-to-one key abstractions contribute an explicit
collision error `c_i`. The collision-free case has `c_i = 0`.

**A3. Capacity and retention.** The table budget `M` is at least the number of
protected bindings plus any required collision slack. The replacement policy
does not evict every useful representative of a recurring binding; if eviction
can happen, its cost is charged as a retention penalty `r_i`.

**A4. Bounded update and bounded loss.** Row values, predictions, and losses are
bounded. On a hit for binding `i`, the row update contracts toward the current
target up to update noise:

```text
||m_i^+ - v_i(t)|| <= beta_i ||m_i - v_i(t)|| + u_i,
```

where `0 <= beta_i < 1` and `u_i` is bounded. Between hits, retention and
unrelated updates move the row by at most `d_i` over one recurrence gap.

**A5. Bounded drift.** The target value attached to a binding changes slowly:

```text
D_i(T) = sum over hits n <= T of ||v_i(n) - v_i(n - 1)||
```

is finite, and the per-gap drift is small relative to the contraction margin.
The stationary case has `D_i(T) = 0`.

**A6. Temporal causality.** The memory uses only current and past observations,
current and past targets, and its fixed resource state. It does not use task
identity, future targets, held-out feedback, or replay outside the declared
memory budget.

## Theorem 1: Conditional Finite Key/Value Binding Learning

Fix a finite binding set `B = {(k_i, v_i): i = 1, ..., N}` and a finite
associative memory with `M` rows. Suppose A1-A6 hold. Let `n_i(t)` be the number
of hits for binding `i` observed by time `t`, and let `e_i(t)` be the row error
immediately after the most recent hit:

```text
e_i(t) = ||m_i(t) - v_i(t)||.
```

Then for each binding that remains retained,

```text
e_i(t)
  <= beta_i ** n_i(t) * e_i(0)
     + (u_i + c_i + d_i + drift_i(t) + r_i) / (1 - beta_i).
```

Consequently, if the residual term

```text
(u_i + c_i + d_i + drift_i(t) + r_i) / (1 - beta_i)
```

is below an evaluation tolerance `epsilon`, then after

```text
n_i(t) >= log(epsilon / e_i(0)) / log(beta_i)
```

successful recurrent hits, binding `i` is recalled within tolerance up to the
residual error. If the largest recurrence gap is `R`, the wall-clock warmup cost
is at most `O(R * max_i n_i)` after all bindings have first appeared.

For classification with a margin `gamma`, if every retained row's residual
error is below `gamma / L`, where `L` is the prediction loss or logit Lipschitz
constant, the correct label is predicted after warmup. In the stationary,
collision-free, no-eviction case with bounded contractive updates, the row error
goes to zero as repeated hits accumulate.

### Proof Sketch

On each hit for binding `i`, A4 gives a contraction toward the current target.
Between hits, A1 bounds the number of inactive steps, and A4 bounds inactive
movement by `d_i`. A2 charges incompatible key sharing as `c_i`. A3 either
protects the row or charges eviction and relearning as `r_i`. A5 charges target
movement as `drift_i(t)`. Unrolling the one-step contraction over `n_i(t)` hits
gives a geometric series with multiplier `beta_i`; summing the bounded residual
terms gives the displayed bound. The classification statement follows from the
margin and Lipschitz assumptions.

### What The Theorem Proves

The theorem proves a finite-memory tracking statement for recurring finite
bindings. It supports language such as:

- "Associative memory can learn repeated finite key/value bindings under
  recurrence, separability, bounded update, capacity, retention, and drift
  assumptions."
- "The first occurrence and any post-eviction occurrence are warmup costs."
- "Collisions, insufficient table rows, rare events, and fast drift are theorem
  terms, not implementation details to ignore."

## Counterexample 1: Finite Associative Memory Is Not Arbitrary Recursive Continuous Feature Discovery

Consider any finite associative memory with at most `M` retained rows and a
fixed causal key abstraction used for prediction. At a given time, prediction
depends on the current continuous input only through the memory state and a
finite set of active keys. On a compact domain with more possible inputs than
active key distinctions, there exist two distinct inputs `a` and `b` that
produce the same active key abstraction, or that must collide when more than
`M` required bindings recur.

Choose the domain `X = [0, 1]`. Pick such a pair `a != b`. Because `[0, 1]` is a
normal metric space, there is a continuous target function separating the pair;
for example,

```text
f(x) = |x - a| / (|x - a| + |x - b|)
```

with the values swapped if needed, so that `f(a) = 0` and `f(b) = 1`.

Now run a recurrent stream that alternates `a, b, a, b, ...` with target
`f(x_t)`. Since `a` and `b` have the same active key abstraction or must share a
row under the finite capacity, the memory must issue the same pre-target
prediction `p` for both contexts whenever they are aliased. The average squared
loss over the two contexts is

```text
(p ** 2 + (p - 1) ** 2) / 2 >= 1 / 4.
```

Thus there is a continuous target and a recurrent stream for which the finite
associative memory has irreducible error. The failure is not optimization
noise. It is an identifiability and capacity failure caused by finite key
abstraction.

This counterexample rejects the phrase "finite associative memory is arbitrary
recursive continuous feature discovery." A recursive feature constructor with a
growing dictionary might eventually introduce a separator for `a` and `b`; a
fixed finite associative table with fixed key abstractions does not get that
power for free.

## Evidence Boundary For The Implementation

The local implementation can be described as follows:

- `AssociativeMemoryLearner` is a finite causal key/value learner over token
  contexts.
- `max_features` is a hard capacity term.
- `feature_family`, `suffix_length`, adaptive family scope, adaptive window
  scope, and adaptive budget change which finite keys are emphasized; they do
  not remove the need for recurrence, separability, capacity, bounded update,
  and drift assumptions.
- Successful repeated-binding tests are evidence for the conditional finite
  theorem's regime.
- Failed rare-event, collision, over-capacity, or fast-drift tests should be
  reported as assumption failures or mechanism failures, not hidden under a
  universal feature-discovery claim.

## Non-Claim Language

Use the following language in papers, docs, and experiment summaries:

- "This is a conditional finite-binding theorem, not an arbitrary recursive
  continuous feature-discovery theorem."
- "Associative memory can retain and recall repeated finite key/value bindings
  when recurrence, separability, capacity, bounded update, and bounded drift
  conditions hold."
- "The associative branch does not prove universal representation learning."
- "A finite associative table does not discover arbitrary continuous recursive
  features unless the needed separators are already present, generated by some
  other mechanism, or admitted by a growing resource schedule."
- "Over-capacity, aliased, non-recurrent, or fast-drifting streams are outside
  the positive theorem unless their costs are explicitly bounded."

