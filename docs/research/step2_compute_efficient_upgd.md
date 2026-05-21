# Step 2 Compute-Efficient UPGD

Date: 2026-05-06.

## Question

The earlier target-structure UPGD result beat the fair MLP baselines, but the
active feature-finding path was slower than a same-width MLP because every step
carried perturbation RNG, hidden-unit traces, previous-gradient traces, and
tuple-per-head readout work. This note records the pressure tests and
simplifications that make the promoted Step 2 learner compute-efficient as well
as stronger than MLP.

## Current Promoted Learner

The promoted no-portfolio Step 2 default is now:

```python
UPGDLearner.step2_default(n_heads)
```

Equivalent core settings:

- hidden size 32 by default
- `loss_normalization="target_structure"`
- `ObGDBounding(kappa=0.5)`
- `perturbation_sigma=1e-4`
- `perturbation_noise="rademacher"`
- `perturbation_interval=16`
- `track_unit_utilities=False`
- `track_gradient_history=False`
- no adaptive kappa and no gradient-alignment meta-plasticity

Interpretation: utility is still assigned online to hidden weights by
`|w * grad|`, and low-utility weights still receive exploratory perturbations.
The compute-efficient learner simply performs the mutation less often, uses
bounded +/-1 mutations instead of Gaussian draws, and avoids traces that are
only useful for optional unit recycling or meta-plasticity.

## Efficiency Results

Benchmark:

- `benchmarks/step2_upgd_efficiency.py`
- 4,096 online steps per timed run
- 5 repeats after one compile/warmup run
- feature dim 64, 10 heads, JAX scan timing
- output: `output/benchmarks/step2_upgd_efficiency_fused_heads_4096/SUMMARY.md`

Key throughput rows:

| Target mode | Method | Hidden | Steps/s | Relation to MLP64 |
|---|---:|---:|---:|---:|
| onehot | `mlp64` | 64 | 26,551.5 | baseline |
| onehot | `upgd32_structure_rademacher_lean_interval16` | 32 | 38,610.7 | 1.45x faster |
| onehot | `upgd16_structure_rademacher_lean_interval16` | 16 | 77,275.4 | 2.91x faster |
| dense | `mlp64` | 64 | 24,432.1 | baseline |
| dense | `upgd32_structure_rademacher_lean_interval16` | 32 | 38,322.6 | 1.57x faster |
| dense | `upgd16_structure_rademacher_lean_interval16` | 16 | 47,926.4 | 1.96x faster |

The same benchmark also shows why this change was necessary: active
width-64 UPGD with every-step perturbation remains much slower than MLP. The
compute win comes from resource scaling plus less frequent bounded mutation,
not from claiming that a fully instrumented same-width UPGD update is cheaper
than SGD.

Implementation changes:

- Reused the forward pass for loss, metrics, and readout scaling.
- Used a manual VJP path for linear-MSE UPGD instead of differentiating heads
  and trunk through a second full forward.
- Skipped perturbation RNG entirely on non-perturbation steps.
- Added optional lean tracking to avoid unit-utility and previous-gradient
  carry traffic when those mechanisms are disabled.  The deployment default now
  avoids allocating those disabled buffers at initialization as well.
- Added `perturbation_noise="rademacher"` as a bounded mutation alternative.
- Vectorized output-head forward and manual head-gradient/cotangent computation
  so the heads fuse as one matrix operation instead of a tuple of tiny heads.

## Synthetic Quality Pressure

Output:

- `output/subagents/compute_efficiency/small_rademacher_synthetic_30seed_6000/out_of_class_SUMMARY.md`

Protocol:

- 30 seeds
- 6,000 online steps
- out-of-hypothesis-class polynomial, frequency, and compositional streams
- comparison is against the same-run best fair MLP

Results:

| Stream | Width-32 diff vs best MLP | Wins |
|---|---:|---:|
| polynomial | +0.5634 +/- 0.0320 | 30/30 |
| frequency | +0.6107 +/- 0.0410 | 30/30 |
| compositional | +0.0781 +/- 0.0036 | 30/30 |

The promoted width-32 branch preserves the original target-structure UPGD
synthetic win over 90/90 paired cells. Width 16 remains the max-speed branch
from the 10-seed screen, but width 32 is the presentation default.

## Digits Quality Pressure

Output:

- `output/subagents/compute_efficiency/small_rademacher_digits_30seed_h64baseline/SUMMARY.md`

Protocol:

- sklearn digits stream
- 30 seeds x 5 regimes = 150 paired cells
- 1,200 online steps, final window 300
- regimes: iid, class-blocked, permuted pixels, mask noise, label drift
- baseline: 64-hidden-unit fair MLP

Overall results:

| Method | Final-window MSE diff | MSE wins | Test accuracy diff | Test accuracy wins |
|---|---:|---:|---:|---:|
| width-32 lean Rademacher UPGD | +0.0078 +/- 0.0004 | 147/150 | +0.0269 +/- 0.0018 | 135/150 |

By-regime weakness:

- width 32 is the conservative default: it is positive on final-window MSE in
  every regime aggregate, including class-blocked and permuted pixels.
- width 16 is the maximum-throughput branch: it is faster and has slightly
  higher overall test-accuracy diff, but it gives back class-blocked
  final-window MSE and permuted-pixel test accuracy.
- class-blocked remains the hard retention row: width 32 is positive there on
  final-window MSE (`+0.0005`) and held-out test accuracy (`+0.0287`), but
  final-window accuracy is slightly negative (`-0.0039`).

## Scientific Interpretation

This does not prove universal representation learning. It is a stronger
empirical Step 2 closure:

- no portfolio, router, replay buffer, or MLP fallback
- online vector supervised learning
- shared nonlinear hidden features
- online utility assignment to hidden weights
- resource-limited feature exploration by low-utility perturbation
- target-structure loss that handles dense vectors, exact-zero dense heads,
  sparse one-hot labels, and sparse multilabel stressors without a dataset
  switch
- compute-efficient default that beats a 64-unit MLP baseline while using a
  smaller hidden layer

The remaining caveat is class-blocked held-out retention. Width 32 is positive
there by final-window MSE and held-out test accuracy in the 30-seed digits run,
but final-window accuracy is slightly negative and absolute held-out accuracy
remains low for all learners in that regime because the stream withholds
classes for long blocks.

## Promotion

Promote width-32 lean Rademacher target-structure UPGD as the Step 2 default.
Keep width 16 as the max-speed branch. Keep readout-meta UPGD as a reference
branch, not the default, because it is slower and no longer needed for the
current quality bar.

## Tiny Transformer Demonstration

The first transformer-facing smoke demo is:

- `examples/The Alberta Plan/Step2/step2_upgd_transformer_demo.py`
- output: `output/step2_upgd_transformer_demo_smoke/SUMMARY.md`

It uses a frozen one-head attention stem to produce contextual sequence
features, then replaces the usual MLP readout with `UPGDLearner.step2_default`.
The baseline uses `MultiHeadMLPLearner` on the same attention features.

This is an integration proof, not yet a transformer-performance result. On the
5-seed, 2,000-step smoke task, UPGD is roughly at parity but slightly behind the
MLP readout:

| Metric | MLP readout | UPGD readout | Diff favoring UPGD |
|---|---:|---:|---:|
| final-window MSE | 0.2430 +/- 0.0039 | 0.2443 +/- 0.0043 | -0.0014 +/- 0.0013 |
| final-window accuracy | 0.5980 +/- 0.0154 | 0.5920 +/- 0.0156 | -0.0060 +/- 0.0024 |

The next transformer experiment should replace the frozen readout-only setup
with a trainable attention block whose feed-forward sublayer uses the same
utility/perturbation mechanism, then compare against an ordinary FFN block.

## Tiny Shakespeare Transformer Demo

The fuller language-modeling demo is:

- `examples/The Alberta Plan/Step2/step2_tiny_shakespeare_upgd_transformer.py`
- output: `output/step2_tiny_shakespeare_upgd_transformer_demo/SUMMARY.md`

It downloads Tiny Shakespeare if needed, builds a char-level online
next-token-prediction stream, and compares:

- `mlp_transformer`: a tiny one-block causal transformer with trainable
  attention, MLP/FFN, and linear softmax head.
- `upgd_transformer`: trainable causal attention plus a UPGD next-token
  learner replacing the MLP/FFN/readout learner. Attention is updated by
  cross-entropy gradient through the current UPGD predictor; UPGD updates with
  its own utility/perturbation rule. The demo now constructs this learner
  through the canonical factory:
  `UPGDLearner.step2_default(..., readout_mode="softmax_ce")`.

Smoke command:

```bash
python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_upgd_transformer.py" \
  --steps 800 \
  --seeds 2 \
  --eval-steps 256 \
  --block-size 32 \
  --d-model 32 \
  --mlp-hidden 64 \
  --upgd-hidden 32 \
  --output-dir output/step2_tiny_shakespeare_upgd_transformer_demo
```

Result:

| Metric | MLP transformer | UPGD transformer | Diff favoring UPGD |
|---|---:|---:|---:|
| final-window NLL | 3.5373 +/- 0.0080 | 3.5290 +/- 0.0064 | +0.0083 +/- 0.0143 |
| final-window accuracy | 0.1348 +/- 0.0059 | 0.1133 +/- 0.0117 | -0.0215 +/- 0.0059 |
| eval NLL | 3.5786 +/- 0.0960 | 3.5510 +/- 0.1248 | +0.0275 +/- 0.0288 |
| eval perplexity | 35.9878 +/- 3.4440 | 35.1216 +/- 4.3616 | +0.8662 +/- 0.9175 |

This demonstrates integration on a real text stream. The result is encouraging
but tiny: two seeds, 800 online updates, and a very small model. The next
scientific step is a longer run with Adam or normalized SGD for the MLP
baseline, matched parameter counts, and more seeds before making any
performance claim.
