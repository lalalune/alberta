# Alberta Plan Step 2 Presentation Readiness Audit

Date: 2026-05-06.

## Executive Answer

Yes: after the latest patch, the Tiny Shakespeare demo uses the canonical
resource-efficient Step 2 UPGD branch directly:

```python
UPGDLearner.step2_default(
    n_heads=vocab_size,
    hidden_sizes=(upgd_hidden,),
    step_size=upgd_lr,
    readout_mode="softmax_ce",
)
```

That resolves the previous drift risk where the language-model demo recreated
the branch by hand.  The softmax CE readout is the one necessary adaptation for
next-token prediction.  The resource-efficient branch knobs are still the same:
target-structure configuration, hidden size 32 by default, `ObGDBounding(kappa=0.5)`,
low-noise Rademacher perturbation, perturbation interval 16, and lean tracking.

Important nuance: with `readout_mode="softmax_ce"`, the CE path normalizes the
one-hot target distribution directly.  `loss_normalization="target_structure"`
is still part of the branch config, but the MSE target-structure denominator is
not used by the CE loss.  That is correct for language modeling and should be
said explicitly if asked.

## Presentation Thesis

Defensible thesis:

> We have a credible empirical Step 2 closure for supervised feature finding:
> a single, temporally uniform UPGD learner that assigns utility to hidden
> features, explores low-utility features under a fixed compute budget, beats
> fair MLP baselines on the current dense/sparse Step 2 test matrix, and can be
> run more compute-efficiently than the 64-unit MLP baseline by using a smaller
> resource-efficient branch.

Do not claim:

- a theorem of universal representation learning;
- superiority to modern transformers or modern language-model training;
- that UPGD is always faster than an equal-width MLP update;
- that class-blocked held-out retention is solved in an absolute sense.

## Evidence Stack

### 1. Core Step 2 Mechanism

Mechanism:

- shared nonlinear hidden features;
- vector targets with multiple supervised tasks;
- utility signal `|w * grad|` for hidden weights;
- low-utility feature exploration via bounded perturbations;
- fixed resource budget through hidden width and intervaled mutation;
- target-structure loss bridges dense vector regression and sparse one-hot
  classification without a dataset router.

Factory:

- `src/alberta_framework/core/upgd.py::UPGDLearner.step2_default`

Current default:

| Setting | Value |
|---|---|
| hidden size | 32 |
| loss config | `target_structure` |
| readout | `linear_mse` by default, `softmax_ce` for language modeling |
| bounder | `ObGDBounding(kappa=0.5)` |
| perturbation | `sigma=1e-4`, `rademacher`, interval 16 |
| unit recycling traces | off by default |
| previous-gradient traces | off by default |
| adaptive kappa/meta-plasticity | off by default |

### 2. Target-Structure Stress

Source:

- `docs/research/step2_target_structure_upgd_stress.md`

Purpose:

- close the target-density ambiguity around dense exact-zero targets and sparse
  multilabel rows.

Result:

- `target_structure` matches mean loss on dense-zero and multilabel stressors;
- it preserves one-hot classification pressure without target-count hacks;
- it removes the naive nonzero-target-count failure mode.

### 3. Synthetic Out-of-Class Pressure

Source:

- `output/subagents/compute_efficiency/small_rademacher_synthetic_30seed_6000/out_of_class_SUMMARY.md`

Protocol:

- 30 seeds;
- 6,000 online steps;
- polynomial, frequency, and compositional out-of-class streams;
- compared against the same-run best fair MLP.

Width-32 resource-efficient UPGD:

| Stream | Diff vs best MLP | Wins |
|---|---:|---:|
| polynomial | +0.5634 +/- 0.0320 | 30/30 |
| frequency | +0.6107 +/- 0.0410 | 30/30 |
| compositional | +0.0781 +/- 0.0036 | 30/30 |

Interpretation:

- this is the cleanest evidence that a smaller resource-efficient UPGD branch
  still constructs useful nonlinear features under the Step 2 synthetic
  stressors.

### 4. Sklearn Digits Pressure

Source:

- `output/subagents/compute_efficiency/small_rademacher_digits_30seed_h64baseline/SUMMARY.md`

Protocol:

- 30 seeds x 5 regimes = 150 paired cells;
- regimes: iid, class-blocked, permuted pixels, mask noise, label drift;
- baseline: 64-hidden-unit fair MLP.

Width-32 resource-efficient UPGD:

| Metric | Diff vs MLP64 | Wins |
|---|---:|---:|
| final-window MSE | +0.0078 +/- 0.0004 | 147/150 |
| test accuracy | +0.0269 +/- 0.0018 | 135/150 |
| final-window accuracy | +0.0233 +/- 0.0026 | 99/150 |
| test MSE | +0.0073 +/- 0.0004 | 139/150 |

Interpretation:

- width 32 is the conservative default;
- width 16 is faster but weaker on class-blocked and permuted-pixel rows;
- class-blocked remains the retention pressure point: it is positive on
  final-window MSE and held-out test accuracy, but final-window accuracy is
  slightly negative in the 30-seed aggregate (`-0.0039`).

### 5. Compute Efficiency

Source:

- `output/benchmarks/step2_upgd_efficiency_fused_heads_4096/SUMMARY.md`

Protocol:

- 4,096 online steps;
- 5 timed repeats after compile/warmup;
- feature dim 64, 10 heads;
- JAX scan timing.

Key rows:

| Target mode | Method | Steps/s | Relation to MLP64 |
|---|---:|---:|---:|
| one-hot | MLP64 | 26,551.5 | baseline |
| one-hot | width-32 UPGD | 38,610.7 | 1.45x faster |
| dense | MLP64 | 24,432.1 | baseline |
| dense | width-32 UPGD | 38,322.6 | 1.57x faster |

Interpretation:

- the compute win is not "same-width UPGD updates are cheaper";
- the compute win is "a smaller resource-efficient UPGD branch preserves the
  quality wins and runs faster than the larger MLP baseline."

### 6. Tiny Shakespeare Transformer Integration

Source:

- `examples/The Alberta Plan/Step2/step2_tiny_shakespeare_upgd_transformer.py`
- `output/step2_tiny_shakespeare_upgd_transformer_demo/SUMMARY.md`

Purpose:

- demonstrate that the Step 2 learner can replace a transformer-style
  MLP/readout learner in a real text stream.

Result from 2-seed smoke run:

| Metric | MLP transformer | UPGD transformer | Diff favoring UPGD |
|---|---:|---:|---:|
| final-window NLL | 3.5373 +/- 0.0080 | 3.5290 +/- 0.0064 | +0.0083 +/- 0.0143 |
| final-window accuracy | 0.1348 +/- 0.0059 | 0.1133 +/- 0.0117 | -0.0215 +/- 0.0059 |
| eval NLL | 3.5786 +/- 0.0960 | 3.5510 +/- 0.1248 | +0.0275 +/- 0.0288 |
| eval perplexity | 35.9878 +/- 3.4440 | 35.1216 +/- 4.3616 | +0.8662 +/- 0.9175 |

Interpretation:

- good integration demo;
- encouraging but not decisive;
- not core Step 2 evidence because the baseline is tiny SGD and the run is too
  short.

## Holes To Close Before A Strong Public Claim

The presentation can be strong now, but these should be handled honestly:

| Hole | Severity | Status | Presentation handling |
|---|---:|---|---|
| No theorem of universality | high | open | Say "empirical closure on current matrix," not universal proof. |
| Class-blocked held-out retention | medium | partially improved | Show final-window MSE and test accuracy, then note low absolute retained accuracy. |
| Same-width compute | medium | known limitation | Say compute efficiency comes from resource scaling and intervaled mutation. |
| Tiny Shakespeare strength | medium | integration only | Keep it in appendix or final "extension" slide. |
| MLP transformer baseline optimizer | medium | weak baseline | Do not use Shakespeare as performance proof until Adam/normalized SGD and longer seeds are added. |
| Softmax CE target-structure nuance | low | clarified | Say CE uses normalized one-hot targets; target-structure is the branch config for MSE tasks. |
| External dataset breadth | medium | sklearn digits + Tiny Shakespeare only in current public packet | Label external evidence as "digits + text smoke," not broad benchmark dominance. |

## Slide Spine

1. **Problem**: Step 2 asks for continual supervised feature finding under a
   resource budget, not just fitting fixed features.
2. **Failure of the obvious answer**: ordinary MLPs are strong but do not
   explicitly allocate utility to features or recycle/explore low-utility
   features.
3. **Mechanism**: target-structure UPGD assigns online utility to hidden
   weights and perturbs low-utility weights under bounded updates.
4. **Target structure**: the loss rule bridges dense vector targets and sparse
   one-hot tasks without a dataset router.
5. **Synthetic pressure**: width-32 UPGD beats best fair MLP on polynomial,
   frequency, and compositional streams over 90/90 paired cells.
6. **Digits pressure**: width-32 UPGD beats MLP64 over 150 regime/seed cells.
7. **Compute**: resource-efficient UPGD is faster than MLP64 in the throughput
   benchmark while preserving quality.
8. **Transformer demo**: the same branch can sit behind trainable attention on
   Tiny Shakespeare; promising integration, not a performance claim.
9. **Critical caveats**: no theorem, class-blocked retention still hard,
   same-width UPGD is not cheaper.
10. **Conclusion**: current status is a credible empirical Step 2 closure and a
    clear path to Step 3 feature discovery/GVF extensions.

## Required Repro Commands

Factory/test audit:

```bash
ruff check src/alberta_framework/core/upgd.py tests/test_upgd.py \
  "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_upgd_transformer.py"
python -m pytest tests/test_upgd.py -q
```

Tiny Shakespeare smoke:

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

Compute benchmark:

```bash
python benchmarks/step2_upgd_efficiency.py \
  --steps 4096 \
  --repeats 5 \
  --output-dir output/benchmarks/step2_upgd_efficiency_fused_heads_4096
```

Evidence gate:

```bash
python benchmarks/step2_upgd_evidence_gate.py
```

Synthetic pressure:

```bash
python "examples/The Alberta Plan/Step2/step2_out_of_class.py" \
  --seeds 30 \
  --num-steps 6000 \
  --upgd-variants structure_h32_rademacher_interval16_lean \
  --output-dir output/subagents/compute_efficiency/small_rademacher_synthetic_30seed_6000
```

Digits pressure:

```bash
python output/subagents/upgd_sweep/upgd_digits_sweep.py \
  --output-dir output/subagents/compute_efficiency/small_rademacher_digits_30seed_h64baseline \
  --n-seeds 30 \
  --seed 1600 \
  --steps 1200 \
  --final-window 300 \
  --phase-length 300 \
  --hidden-size 64 \
  --config-names upgd_h32_structure_sigma1e_4_kappa05_rademacher_interval16_lean
```

## Final Readiness Verdict

Ready for an internal Alberta Plan Step 2 research presentation if the claim is
framed as an empirical closure on the current Step 2 matrix.  Not ready for a
public theorem-style or "beats transformers" claim.  The strongest honest
headline is:

> A single resource-efficient target-structure UPGD learner now beats the fair
> MLP baseline on the current Step 2 synthetic and digits pressure tests, while
> satisfying the Alberta Plan constraints of temporally uniform online learning,
> explicit feature utility, and bounded resource use.
