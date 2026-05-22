# Continuous EML Circuits

This note defines the clean version of DiffEML that is not tied to DiffLogic.

## Core Object

A continuous EML circuit is a sparse differentiable arithmetic circuit whose
node primitive is:

```text
z = tanh(scale * (exp(clip(a)) - log(softplus(b) + eps)))
```

The trainable model chooses source signals for `a` and `b`, then applies a small
number of per-node scale/bias parameters. This is not a Boolean gate selector.
It is an ordinary real-valued network trained by backpropagation through EML.

## Two Training Modes

### Dense Continuous EML Blocks

`ContinuousDiffEML` uses dense affine left/right projections followed by stable
EML. This is useful as a sanity check: the primitive is differentiable, finite,
and trainable with Adam.

### Sparse Continuous EML Circuits

`SparseContinuousEMLCircuit` is the more novel direction:

- each node has soft source selectors for its left and right EML inputs;
- selectors are trained by backprop;
- temperature can be annealed toward lower-entropy source choices;
- after training, selectors compile to fixed source indices;
- compiled inference stores source indices, per-node scale/bias values, and a
  readout.

This gives a path from continuous optimization to a compact fixed EML program.

## Why This Is Different From DiffLogic

Boolean-template DiffEML proves EML can realize Boolean gates. Continuous EML
circuits make a different claim: EML is a trainable arithmetic primitive for
real-valued sparse programs.

The comparison set should therefore include:

- dense MLPs;
- dense continuous EML blocks;
- sparse continuous EML circuits;
- random arithmetic circuits;
- Boolean-template DiffEML;
- DiffLogic only when the model has been hardened into a comparable discrete
  logic circuit.

## Optimization Path

The optimized inference target is a fixed-source circuit:

```text
for layer:
    left = bank[left_index] * left_scale + left_bias
    right = bank[right_index] * right_scale + right_bias
    out = tanh(scale * stable_eml(left, right))
```

Important optimization levers:

- compile soft source selectors to integer indices;
- store per-node constants in `bf16` or a small codebook;
- fuse gather, EML, normalization, and writeback in one kernel;
- approximate `exp` and `log` with lookup tables or low-degree polynomial
  kernels for inference;
- group nodes by source locality for cache-friendly evaluation;
- prune nodes with low readout magnitude or redundant source pairs;
- train with entropy pressure and temperature annealing to minimize hardening
  loss.

## Computational Advantage Hypothesis

The advantage is not the soft relaxation itself. During training, soft source
selection still multiplies source banks by selector distributions, so it can be
more expensive than a dense MLP. The advantage appears only after hardening:

- selector matrices compile to two integer source indices per node;
- each node keeps four small affine constants plus one EML primitive;
- the compiled graph can skip all inactive source edges;
- source indices expose a small arithmetic program that can be fused, quantized,
  cached, pruned, or emitted as C/accelerator code.

The current benchmark harness is:

```bash
python "examples/The Alberta Plan/Step2/step2_continuous_diffeml_performance.py" \
  --output-dir outputs/diffeml_continuous_demo/performance_probe
```

It reports trainable soft parameters, compiled scalar/index counts, estimated
per-example work, warmed-JIT forward latency, and the gap between soft,
hard-full-bank, direct-gather compiled inference, and approximate compiled
kernels.

The first approximate kernels are intentionally inference-only:

- `sparse_compiled_approx_lut`: fixed lookup tables for `exp(x)` and
  `log(softplus(y) + eps)`, with linear interpolation;
- `sparse_compiled_approx_poly`: table-free arithmetic using repeated squaring
  for the left exponential and a smooth positive/log surrogate for the right
  side.
- `_fast_tanh` variants additionally replace `tanh` with a bounded rational
  approximation.

The benchmark reports logit error against exact compiled EML. A kernel should
not be treated as a valid replacement unless it improves latency at an
acceptable hard-circuit accuracy gap.

## Current Evidence

The first trained sparse-circuit result is positive but bounded. On the
synthetic real-valued classification demo with five seeds, 300 training steps,
512 train/test examples, width 64, and straight-through hard loss:

- soft sparse EML test accuracy: `0.8617 +/- 0.0099`;
- compiled exact sparse EML test accuracy: `0.7660 +/- 0.0078`;
- hardening gap: `0.0957 +/- 0.0160`;
- LUT compiled approximation: same accuracy as exact compiled in this run;
- LUT + rational tanh: `0.7652 +/- 0.0079` with `0.9969 +/- 0.0010`
  top-1 agreement against exact compiled logits.

The matched no-hard-loss ablation compiled to only `0.5316 +/- 0.0323`, with a
hardening gap of `0.2961 +/- 0.0275`. This makes the straight-through hard
loss a necessary part of the current sparse EML hardening story.

The novelty claim should therefore be phrased narrowly:

> Backprop-trained continuous EML circuits can be hardened into compact
> fixed-source arithmetic circuits; a straight-through hard loss substantially
> reduces the soft-to-hard gap, and LUT/fast-tanh approximate kernels preserve
> the hardened circuit behavior while improving inference latency.

This is not yet a broad image-classification or MLP-beating claim.

## Paper-Quality Acceptance

A continuous EML circuit result should report:

- soft train/test metrics;
- compiled hard-source train/test metrics;
- soft selector parameter count;
- compiled scalar/index count;
- hardening gap;
- training time;
- compiled inference time;
- ablations removing EML, source sparsity, entropy pressure, normalization, and
  residual/dense paths.

The first publishable claim should be about compression or sparse-program
quality, not generic image accuracy. A strong claim would look like:

> At matched compiled scalar budget, sparse continuous EML circuits preserve
> more accuracy than dense EML blocks or MLPs after source hardening.

That is the clean novelty: backprop-trained EML programs that compile to small,
fixed arithmetic circuits.
