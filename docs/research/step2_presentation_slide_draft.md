# Alberta Plan Step 2 Presentation Draft

Date: 2026-05-06.

Audience: Alberta Plan researchers / technically skeptical continual-learning
audience.

Tone: empirical, critical, no universality overclaim.

## Slide 1: Step 2 Status

Title:

**Supervised Feature Finding Under Resource Constraints**

Claim:

We now have a credible empirical Step 2 closure: a single resource-efficient
target-structure UPGD learner beats the fair MLP baseline on the current
synthetic and digits pressure matrix while maintaining explicit online feature
utility.

Proof object:

- one-line architecture diagram: vector targets -> shared nonlinear UPGD
  features -> multi-head outputs;
- three badges: feature utility, resource budget, MLP-beating matrix.

Speaker note:

This is not a theorem. The claim is empirical closure on the current Step 2
matrix with Alberta Plan-compatible mechanisms.

## Slide 2: What Step 2 Actually Requires

Title:

**Not Just A Better MLP**

Claim:

Step 2 asks how to construct, evaluate, and discard features continually under
a limited resource budget.

Proof object:

- three-question table:
  - How are new features constructed?
  - How is utility assigned?
  - What gets discarded or perturbed under a fixed budget?

Speaker note:

The baseline MLP has nonlinear features, but it does not expose a resource
economy over features. UPGD gives us a causal utility signal and low-utility
feature exploration.

## Slide 3: The Mechanism

Title:

**Target-Structure UPGD**

Claim:

UPGD turns a hidden layer into a simple feature economy: useful weights are
protected by utility, low-utility weights are explored by small bounded
perturbations.

Proof object:

Formula strip:

- prediction loss over vector targets;
- utility: `u_i <- decay * u_i + (1 - decay) * |w_i * grad_i|`;
- perturbation: `w_i <- w_i + sigma * (1 - norm(u_i))^beta * rademacher`.

Speaker note:

The branch uses ObGD bounding, interval-16 perturbations, and lean tracking.
Unit recycling remains optional but is not part of the default.

## Slide 4: Why Target-Structure Loss Matters

Title:

**Dense Targets And Sparse Targets Need One Rule**

Claim:

`target_structure` fixes the mean-vs-sum conflict without a dataset router.

Proof object:

Matrix:

| Target type | Rule |
|---|---|
| non-negative simplex / one-hot | sum-style pressure |
| dense regression | mean over active heads |
| exact-zero dense heads | mean, not accidental sparse boost |
| sparse multilabel | mean, not one-hot assumption |

Speaker note:

This replaced the earlier `target_density` bridge because density over-counted
zero-valued dense targets and sparse multilabel rows.

## Slide 5: Synthetic Stress Results

Title:

**Out-Of-Class Synthetic Streams**

Claim:

The resource-efficient width-32 branch keeps the original UPGD win on
polynomial, frequency, and compositional streams.

Proof object:

| Stream | Diff vs best MLP | Wins |
|---|---:|---:|
| polynomial | +0.5634 +/- 0.0320 | 30/30 |
| frequency | +0.6107 +/- 0.0410 | 30/30 |
| compositional | +0.0781 +/- 0.0036 | 30/30 |

Speaker note:

These are deliberately out-of-class for simple hand-coded interaction
features. The comparison is to the same-run best fair MLP, and the promoted
branch wins all 90 paired cells.

## Slide 6: Digits Pressure Matrix

Title:

**External Online Digits Regimes**

Claim:

Across iid, class-blocked, permuted-pixel, mask-noise, and label-drift regimes,
width-32 UPGD beats the 64-unit MLP baseline on the aggregate metrics.

Proof object:

| Metric | Diff vs MLP64 | Wins |
|---|---:|---:|
| final-window MSE | +0.0078 +/- 0.0004 | 147/150 |
| test accuracy | +0.0269 +/- 0.0018 | 135/150 |
| final-window accuracy | +0.0233 +/- 0.0026 | 99/150 |
| test MSE | +0.0073 +/- 0.0004 | 139/150 |

Speaker note:

Class-blocked retained accuracy remains the hardest row. Do not hide it.
The 30-seed class-blocked aggregate is positive on final-window MSE and
held-out test accuracy, but final-window accuracy is slightly negative
(`-0.0039`).

## Slide 7: Compute Efficiency

Title:

**The Compute Win Comes From Resource Scaling**

Claim:

The width-32 branch is faster than MLP64 while preserving the quality win.

Proof object:

| Mode | MLP64 | Width-32 UPGD | Ratio |
|---|---:|---:|---:|
| one-hot | 26,551.5 steps/s | 38,610.7 steps/s | 1.45x |
| dense | 24,432.1 steps/s | 38,322.6 steps/s | 1.57x |

Speaker note:

Do not claim same-width UPGD is cheaper. The result is stronger and cleaner:
with fewer hidden units and intervaled bounded mutation, UPGD beats MLP64 on
both speed and performance in this matrix.

## Slide 8: Transformer Integration

Title:

**Can This Sit Inside A Transformer-Shaped Loop?**

Claim:

Yes. The Tiny Shakespeare demo uses trainable causal attention plus
`UPGDLearner.step2_default(..., readout_mode="softmax_ce")` for next-token
prediction.

Proof object:

| Metric | MLP transformer | UPGD transformer | Diff favoring UPGD |
|---|---:|---:|---:|
| eval NLL | 3.5786 | 3.5510 | +0.0275 |
| eval perplexity | 35.99 | 35.12 | +0.87 |
| final-window accuracy | 0.1348 | 0.1133 | -0.0215 |

Speaker note:

This is an integration demo only. The run is too small, and the MLP transformer
baseline is simple SGD. Keep it out of the core evidence stack.

## Slide 9: What We Can And Cannot Claim

Title:

**Claims Discipline**

Claim:

The Step 2 result is strong enough for internal promotion, not for theorem
language.

Proof object:

Two-column table:

| Defensible | Not defensible |
|---|---|
| empirical closure on current matrix | universal representation theorem |
| single learner, no portfolio | beats all MLPs/transformers |
| explicit online feature utility | class-blocked retention fully solved |
| faster than MLP64 via resource scaling | same-width UPGD always cheaper |

Speaker note:

This slide is what makes the presentation scientifically credible.

## Slide 10: What Remains

Title:

**Next Holes To Close**

Claim:

The next research work is not another portfolio; it is stronger retention,
larger language-model tests, and a theory of when utility perturbation beats
plain gradient descent.

Proof object:

Prioritized list:

1. class-blocked retained accuracy without replay;
2. stronger Tiny Shakespeare baseline with Adam/normalized SGD and more seeds;
3. same-width kernel/runtime profiling;
4. formalize the target family where target-structure UPGD is expected to win;
5. Step 3: convert supervised feature finding into GVF/predictive-state
   feature finding.

Speaker note:

This is the path from "good Step 2 empirical closure" to a more universal
representation-learning claim.

## Backup Slide: Reproducibility

Title:

**Commands And Artifacts**

Content:

- `docs/research/step2_presentation_readiness_audit.md`
- `docs/research/step2_compute_efficient_upgd.md`
- `docs/research/step2_paper_and_deployment_quality_plan.md`
- `output/benchmarks/step2_upgd_evidence_gate/SUMMARY.md`
- `output/subagents/compute_efficiency/small_rademacher_synthetic_30seed_6000/out_of_class_SUMMARY.md`
- `output/subagents/compute_efficiency/small_rademacher_digits_30seed_h64baseline/SUMMARY.md`
- `output/benchmarks/step2_upgd_efficiency_fused_heads_4096/SUMMARY.md`
- `output/step2_tiny_shakespeare_upgd_transformer_demo/SUMMARY.md`

Speaker note:

Every headline number in the deck should point to one of these artifacts.
