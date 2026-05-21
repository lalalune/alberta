# DiffEML Pure-EML Theory Audit

Worker: purity/math audit.

Goal: make the DiffEML claim hard to fake. The thing we need to show is not
"a model with some EML-shaped code trains." The thing worth showing is:

> EML can be used as a learnable computational primitive, trained by a
> legitimate relaxation, and hardened into an EML-realizable circuit whose
> behavior survives deployment without a non-EML component doing the real work.

This audit separates pure claims, acceptable relaxations, larp boundaries, and
falsification tests.

## Definitions

### Exact EML

The exact real operator is

```text
E(x, y) = exp(x) - log(y), y > 0
```

It is differentiable on its natural domain with

```text
dE/dx = exp(x)
dE/dy = -1 / y
```

`src/alberta_framework/core/diffeml.py::eml_operator()` is this object.

`stable_eml_operator()` is not exact EML. It computes exact EML after a
domain-preserving and bounded parameterization:

```text
x_safe = clip(x)
y_safe = softplus(y) + eps
out = clip(exp(x_safe) - log(y_safe))
```

That is a valid differentiable training surrogate, but any paper claim must say
whether the deployed circuit is exact EML, stable EML, or only EML-inspired.

### Pure Boolean EML Circuit

A pure Boolean EML circuit is a finite DAG with:

- leaves in `{0, 1, input bits}`;
- internal nodes of the form `bit(E(a, eps + b) >= theta)` or
  `bit(E(a, eps + b) <= theta)`, where `a` and `b` are previous Boolean
  signals and `eps > 0`;
- outputs that are either Boolean bits or count/popcount functions over Boolean
  bits.

This is the cleanest object in the current repo. It is what
`eml_template` gates harden into, and it is compatible with packed Boolean
inference because every selected EML-threshold node has a Boolean truth table.

`group_sum`, `class_vote`, and `signed_class_vote` can fit this definition if
reported as Boolean count readouts. A linear float head does not.

### Pure Continuous EML Circuit

A pure continuous EML circuit is a finite DAG with:

- leaves in `{input coordinates, learned constants, fixed constants}`;
- internal nodes that evaluate exact EML or a clearly declared stable EML
  variant;
- optional scalar affine output calibration only if reported separately.

The current `EMLTreeLearner` is close to this, except soft leaf selection and
the final affine output are non-EML training/readout components. The hardened
tree is a real EML expression plus scalar affine output.

The dense `ContinuousDiffEML` path is better described as an EML-activated
network, not pure EML. It uses dense affine projections, residual projections,
gates, normalization, and `tanh`. Those pieces may be useful, but they weaken
the purity claim.

### Train-Time Relaxation

A relaxation is legitimate when all of these are true:

1. It has a well-defined hard target object.
2. The hard object is pure EML or EML-threshold under the definitions above.
3. The training loss includes or reports the hard object, not only the soft
   surrogate.
4. The final reported accuracy is hard/deployed accuracy.
5. The soft-to-hard gap is reported and small enough not to explain the result.

Softmax source selection, softmax gate selection, sigmoid thresholding, entropy
pressure, annealing, and straight-through estimators are all acceptable under
those conditions. They are not themselves EML circuits.

## Current Claim Ledger

### Real And Defensible

- Exact EML is differentiable on `y > 0`; this is tested directly.
- Depth-2 EML-threshold templates cover all 16 two-input Boolean gates under
  the current `eps=0.05` construction.
- The executable template bank hard-evaluates to the same truth tables as the
  16-gate library.
- A differentiable selector can learn all 16 two-input Boolean gates in the
  unit tests.
- Packed hard Boolean evaluation can match JAX hard evaluation for hardened
  image circuits.
- `class_vote` and `signed_class_vote` train with a straight-through hard
  readout and deploy to class metadata plus popcounts, not float weights.

These are enough to support a finite-domain EML-threshold universality story.

### Real But Not Pure

- `truth_table` mode is a DiffLogic-style multilinear truth-table relaxation.
  It is not EML during soft training. It is acceptable only as an ablation or
  when every selected truth table is exported with an EML-threshold witness.
- `eml_template` mode is purer than `truth_table` because it executes the EML
  templates during relaxed training. It still mixes template outputs by softmax
  before hardening, so the train-time node is not a single EML expression.
- `eml_threshold` mode is the most local-EML node: one EML value, one threshold,
  one direction. It is also less expressive per node and currently less mature.
- `ContinuousDiffEML` demonstrates trainability of an EML primitive inside a
  neural layer, but dense affine projections and residual/gating/normalization
  mean it cannot carry the "pure circuit" claim.
- LUT/poly approximate kernels are inference approximations to stable EML, not
  proof about exact EML. They are performance engineering rows.

### Larp / NGMI If Promoted As The Main Claim

- Best image results with `head_mode="linear"` are not pure EML-circuit
  results. The linear head can do substantial class-evidence mixing.
- Detector-threshold features are non-EML feature engineering. They may be a
  useful input-bit pipeline, but performance gains from them cannot be credited
  to EML.
- Same-feature MLP baselines beating DiffEML mean the current image result is
  not evidence that EML is the better learner on those features.
- The pure Boolean/count readouts currently lag the linear head by a large
  margin. That gap is direct evidence that the readout is doing real work.
- Width/depth scaling by random sparse wiring regressed in the current artifacts
  and mainly increased overfitting. Blind scale is not the credible path.
- Naive `class_bank_random` and current local-tree hierarchy are negative
  results. They should stay in the ablation ledger, not the narrative.
- Soft-only success does not count. If soft accuracy is high and hard accuracy
  is low, the relaxation is doing the work.
- Continuous dense EML blocks cannot be used to claim a compressed Boolean EML
  circuit unless they are hardened and the hard circuit is the reported result.

## Universal Approximation: What We Can And Cannot Say

### Defensible Theorem: Finite Boolean Universality

The current code supports the following theorem.

**Lemma 1: EML-threshold gates can realize a functionally complete Boolean
basis.**

For `eps=0.05`, the enumerated depth-2 EML-threshold template bank contains all
16 two-input Boolean functions, including NAND. This can be turned from a test
into a constructive appendix by listing the templates.

**Lemma 2: EML-threshold circuits simulate Boolean circuits.**

Since NAND is functionally complete, any Boolean circuit can be rewritten as an
EML-threshold circuit by replacing every NAND with its EML-threshold template.
The size blowup is constant per Boolean gate, and the depth blowup is constant
per Boolean layer.

**Theorem 1: Pure Boolean EML finite-domain universality.**

For any function `f: {0,1}^n -> {0,1}^m`, there exists a pure Boolean
EML-threshold circuit that computes `f` exactly.

This is the cleanest and strongest universal approximation statement currently
available. It does not require a linear head, detector features, or continuous
MLP machinery.

### Defensible Corollary: Quantized Approximation

For a uniformly continuous bounded function on a compact domain, quantize the
domain into a finite bit grid and quantize the output to finite precision. A
pure Boolean EML-threshold circuit can compute the finite lookup table exactly.
After deterministic decoding, the approximation error can be made smaller than
epsilon by using enough input/output bits.

This is mathematically true, but the quantizer and decoder must be accounted
for. If the quantizer is hand-built threshold features and the decoder is a
linear float map, the result is not "pure EML did everything." It is "EML
implements the finite Boolean table between a quantizer and decoder."

### Defensible But Different: EML-Activated Network Universality

If affine input maps and linear readouts are allowed, an EML hidden unit with a
constant right input contains an exponential feature:

```text
E(a dot x + b, c) = exp(a dot x + b) - log(c)
```

Finite linear combinations of exponentials with arbitrary affine exponents are
dense in continuous functions on compact sets by the Stone-Weierstrass route:
they contain constants, separate points, and are closed under multiplication up
to another exponential feature.

This supports an "EML-activated network is a universal approximator" statement.
It does not support the stricter "pure EML circuit with no affine/readout help"
statement.

### Not Yet Proven

- A strict continuous EML DAG with only input leaves, constants, and exact EML
  internal nodes is a universal approximator over compact real domains.
- The current optimization reliably finds the EML circuits promised by the
  finite-domain theorem.
- The current image topology scales to DiffLogic-quality CIFAR under pure
  Boolean readout.
- The continuous sparse circuit hardening gap vanishes at larger scale.

Those are research hypotheses, not claims.

## What Counts As Cheating

The following are fine as engineering controls, but they violate a pure-EML
headline unless isolated and ablated:

- float linear heads;
- MLP teachers or distillation targets unless final hard EML metrics are primary;
- fixed detector features without same-feature controls;
- learned affine dense projections inside "EML blocks";
- residual/gating/normalization paths that can route around EML;
- reporting soft relaxed accuracy without hard deployed accuracy;
- choosing hyperparameters on the test set;
- quantized linear heads advertised as Boolean compression;
- approximate kernels reported without exact-kernel agreement.

The safe phrasing is usually:

```text
We train with non-EML relaxations, then evaluate the hardened EML-realizable
circuit. Non-EML preprocessing/readout is reported separately.
```

## Falsification Experiments

These are the tests that would make the claim credible because they can fail.

### 1. Gate And Small-Circuit Recovery

Train selectors for:

- all 16 two-input gates;
- parity on 3, 4, 5, and 8 bits;
- mux, comparator, half-adder, full-adder, and small multiplier bits;
- random teacher circuits generated from EML-threshold NAND/OR/XOR gates.

Report soft accuracy, hard accuracy, selected-gate entropy, and exact truth-table
agreement. The claim weakens if soft succeeds but hard fails, or if simple
compositional circuits cannot be learned with reasonable budgets.

### 2. Teacher-Circuit Distillation Without A Float Head

Generate a pure EML-threshold teacher circuit, sample inputs, and train a
student with the same or larger gate budget. The target is the teacher's Boolean
outputs, not image labels. This tests learnability of EML circuits directly.

Falsifier: the student cannot match the teacher even when the architecture has
enough gates and the teacher is in-distribution.

### 3. Finite Universal Approximation Bench

Use quantized scalar/vector functions:

- `sin(2*pi*x)` as output bits;
- piecewise constants;
- checkerboard/parity-like functions;
- low-degree polynomial bits;
- random lookup tables at controlled bit depth.

Only report pure Boolean EML hard metrics and a deterministic bit decoder.

Falsifier: accuracy does not improve predictably with gate budget on functions
that have compact Boolean circuits, or the hardening gap dominates.

### 4. Readout Purity Ladder

For every image or classification result, report this ladder:

1. `linear` float head;
2. int8 linear head;
3. `signed_class_vote`;
4. `class_vote`;
5. `group_sum`;
6. bit-output pure EML classifier if implemented.

The pure result is the highest row in that ladder that uses no float class
mixing. The claim is falsified if all real accuracy lives only in rows 1-2.

### 5. Non-EML Feature Ablation

Run the same circuit on:

- raw threshold pixels;
- detector thresholds;
- randomly permuted detector bits;
- label-shuffled controls;
- same-feature MLP.

Falsifier: DiffEML only works when detector engineering is strong, does not beat
weaker Boolean baselines on the same bits, or fails under permutation in ways
that indicate topology artifacts rather than learned EML composition.

### 6. Relaxation-To-Hard Accounting

For every serious run, report:

- soft train/test accuracy;
- hard train/test accuracy;
- packed hard train/test accuracy;
- hardening gap;
- selector entropy by layer;
- percent constants/dead nodes/duplicates;
- train-time parameter count;
- deployed EML node count and bytes.

Falsifier: soft/hard gaps remain large after annealing and hard loss, or packed
hard eval does not match JAX hard eval.

## Recommended Research Shape

The strongest path is not CIFAR-first. CIFAR is too easy to contaminate with
feature engineering and readout mixing. The credible path is:

1. Establish pure Boolean EML finite-domain universality in code and docs.
2. Demonstrate learnability on teacher circuits and finite functions where the
   target circuit is known to exist.
3. Add grow/prune/compact loops to improve hard circuit quality while preserving
   pure readout.
4. Only then return to image tasks with the purity ladder and same-feature
   controls.

For continuous EML, keep a separate paper track:

```text
Backprop-trained sparse continuous EML programs can harden into compact
fixed-source arithmetic circuits.
```

That is novel and clean, but it is not the same as the Boolean DiffLogic-style
claim.

## Proposed Theorem/Lemma Set

Paper-quality theory could be organized as:

1. **Differentiability lemma.** Exact EML is smooth on `R x R+`; stable EML is
   differentiable except at clipping breakpoints and has bounded outputs.
2. **Template realization lemma.** The listed depth-2 EML-threshold templates
   realize all 16 binary Boolean gates.
3. **Functional-completeness lemma.** Since NAND is in the EML-template bank,
   EML-threshold circuits simulate arbitrary Boolean circuits with constant
   local overhead.
4. **Finite-domain universality theorem.** Any finite Boolean function can be
   represented exactly by a pure EML-threshold circuit.
5. **Quantized approximation corollary.** Continuous compact-domain functions
   can be approximated by quantize -> EML-threshold table -> decode, with all
   non-EML quantizer/decoder costs explicitly reported.
6. **Relaxation consistency proposition.** As temperature goes to zero, soft
   sigmoid/softmax templates converge pointwise to the hard EML-threshold
   circuit away from threshold ties and selector ties. This justifies annealing
   as a relaxation, not as an exact optimizer.
7. **EML-activated network universality theorem.** With affine preactivations
   and linear readout, EML contains exponential networks and is universal on
   compact sets. This is a separate, weaker-purity theorem.

Do not claim a theorem that gradient descent finds the circuit. That is an
empirical learnability claim and must be tested.

## Strongest Recommendation

Make the primary DiffEML paper claim the pure Boolean one:

> EML-threshold templates are functionally complete; differentiable relaxations
> can train circuits that harden into compact EML-realizable Boolean circuits.

Then prove it on finite-domain tasks before leaning on CIFAR. Demote
linear-head image results, dense continuous EML blocks, detector features, and
approximate kernels to supporting ablations. The current setup becomes much less
larpy if the headline metric is hard packed accuracy under `group_sum`,
`class_vote`, `signed_class_vote`, or bit-output EML readouts, with the linear
head shown only as an upper-bound diagnostic.

## Changed Files

- `docs/research/diffeml_pure_eml_theory_audit.md`
