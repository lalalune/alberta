# Step 2 Gap Follow-up Synthesis Template

Date: 2026-05-06.

## Purpose

This note coordinates the new parallel follow-up batch around one question:
what actually improves the promoted Step 2 UPGD learner against MLP baselines,
and what remains unsolved?

It is intentionally stricter than a run log. A result should be classified as
one of:

- real improvement: a paired, pre-registered metric improvement that preserves
  the relevant guardrails;
- failed improvement: a negative, underpowered, mismatched, or tradeoff-only
  result;
- presentation-quality: a real improvement with enough seeds, scale, breadth,
  diagnostics, and artifact hygiene to support a public Step 2 claim.

No new follow-up artifacts from the active batch were present when this
synthesis was written. The checklist below therefore names expected incoming
paths and how to interpret each result when it arrives.

## Source Reports Read

| Report | Current finding used here |
|---|---|
| `docs/research/step2_external_breadth_upgd.md` | Promoted `UPGDLearner.step2_default` has real sklearn classification breadth against MLP64, but loses diabetes regression and has a class-blocked tracking/retention split. |
| `docs/research/step2_image_breadth_upgd.md` | No-portfolio promoted UPGD is competitive to better on bounded CIFAR iid and compact OPMNIST, but CIFAR class-blocked still loses final-window tracking while winning held-out accuracy. |
| `docs/research/step2_class_blocked_retention_followup.md` | Simple class-blocked variants do not clear both tracking and retention; slow utility helps MSE but does not solve output-head drift or beat `mlp64_64` tracking. |
| `docs/research/step2_transformer_scale_upgd.md` | Current Tiny Shakespeare evidence is an integration/readout smoke, not a true FFN replacement; the fair parameter-matched 2,000-step run favors MLP on CE/perplexity and speed. |
| `docs/research/step2_gap_closure_evidence_gate.md` | Existing gate closes factory/state/synthetic/digits/efficiency checks only; external breadth, class-blocked retention, and transformer-scale proof remain open presentation gaps. |

## Current Bottom Line

The strongest positive evidence is not "UPGD beats MLP everywhere." It is
more specific: promoted UPGD with target-structure normalization, conservative
ObGD, Rademacher perturbations, and softmax CE readout improves many
classification streams, especially shuffled/permuted/noisy digits, wine,
breast cancer, CIFAR iid, and compact OPMNIST. It also tends to preserve
held-out accuracy better than MLPs in class-blocked regimes where MLPs overfit
the current block.

The unsolved pieces are equally clear:

- tabular regression is still a real loss in the current external sweep;
- class-blocked tracking versus retained-class calibration is not solved by
  simple repetition, margin, bias-only meta, or slow utility variants;
- class-blocked real-image tracking is mixed, not closed;
- the transformer story is currently readout/integration evidence, not an
  FFN-only replacement result;
- several positive rows are useful but underpowered for presentation-scale
  claims.

## Non-negotiable Evidence Rules

Treat any new artifact as a failed improvement, regardless of headline numbers,
if it violates one of these rules:

- The candidate is not explicitly named. It must say whether it is exact
  promoted `UPGDLearner.step2_default` or a separate ablation.
- The run uses a portfolio, replay buffer, retained MLP fallback, router
  fallback, or hidden test-set tuning while claiming single-learner UPGD proof.
- The comparator is not a same-run paired MLP baseline with the same inputs,
  target contract, update budget, and evaluation split.
- The primary metrics were chosen after seeing results.
- Smoke runs, one-seed runs, and wiring checks are used as proof rather than
  as diagnostics.
- A class-blocked claim reports only aggregate test accuracy and omits recent
  versus non-recent, macro, and min-class retention metrics.
- A transformer claim is not architecture matched. "UPGD readout" evidence
  cannot be described as "UPGD FFN replacement."
- Missing, skipped, failed, or cherry-picked rows are not listed in the JSON.

## Metric Conventions

For lower-is-better metrics, positive paired diff means `MLP - UPGD`.
For higher-is-better metrics, positive paired diff means `UPGD - MLP`.

| Area | Primary metrics | Guardrails |
|---|---|---|
| Classification breadth | held-out accuracy, final-window NLL or CE, final-window MSE | held-out accuracy diff must not be below `-0.02` on any row; report calibration/CE when softmax CE is used |
| Regression breadth | held-out MSE, final-window MSE | held-out MSE relative degradation must be `<= 5%` on every row |
| Class-blocked retention | final-window recent tracking, non-recent held-out accuracy, macro accuracy, min-class accuracy | no class collapse; report chance accuracy and recent/non-recent class sets per seed |
| Real images | final-window NLL, final-window MSE, held-out accuracy | class-blocked rows must clear tracking and retention, not only one side |
| Transformer FFN/readout | eval NLL, final-window NLL, perplexity, accuracy, bad-seed count, steps/s, state size | no NaN/Inf seeds; perplexity ratio UPGD/MLP `<= 1.05` for presentation non-inferiority |

## What Counts As Real Improvement

A new result is a real improvement only if all of these hold:

- It improves the pre-registered primary metric on the relevant paired
  comparison.
- It wins a majority of paired seeds or benchmark cells, not just the mean.
- It preserves the guardrail metrics for the same workstream.
- It is not explained by a claim mismatch, such as comparing a CE-trained UPGD
  to an MSE-only MLP and then claiming a general learner win without reporting
  the training/readout difference.
- It reports enough detail to reproduce the config and separate promoted UPGD
  evidence from exploratory ablations.

Exploratory scale can establish "mechanism clue" with 3-5 seeds. A candidate
claim needs at least 10 paired seeds for external/image breadth, at least 30
paired seeds for class-blocked retention closure, and at least 5 paired seeds
plus a meaningful token horizon for transformer evidence.

## What Counts As Failed Improvement

Classify a result as failed if any of these are true:

- It improves final-window tracking but loses retained/non-recent accuracy.
- It improves retained accuracy but still loses final-window tracking to the
  relevant MLP tracking comparator.
- It beats `mlp64` but loses to `mlp64_64` on the class-blocked tracking claim
  it was meant to close.
- It improves MSE while worsening the pre-registered accuracy/NLL primary
  metric beyond the guardrail.
- It only wins on post-hoc CE, calibration, or timing metrics while losing the
  primary task metric.
- It is positive only on smoke scale or only after dropping difficult rows.
- It adds enough state/compute that the claimed replacement story changes, and
  that cost is not reported.

Failed results should still be kept if they isolate a mechanism. Label them as
negative evidence or mechanism evidence, not as gap closure.

## Presentation-quality Bar

Presentation-quality evidence requires:

- exact artifact paths and raw JSON summaries;
- all configured rows completed or explicitly recorded as failures;
- paired seeds and per-seed records available;
- no hidden method drift from the candidate named in the report;
- positive aggregate primary diff and row/cell win rates meeting the thresholds
  below;
- no catastrophic guardrail regression;
- a short interpretation that states both wins and remaining boundaries.

Minimum presentation thresholds:

| Area | Required scale | Aggregate pass | Row/cell guard |
|---|---:|---|---|
| Sklearn/external breadth | `>= 6` rows, `>= 4` dataset families, `>= 3` task kinds, `>= 10` paired seeds per row | primary paired diff `> 0`, primary cell win rate `>= 0.60`, row positive rate `>= 0.70` | no classification held-out accuracy row below `-0.02`; no regression held-out MSE row worse by more than 5% |
| Real-image breadth/class-blocked | `>= 10` paired seeds, non-smoke horizon, fixed train/test subsets | final-window NLL/MSE and held-out accuracy positive on the pre-registered rows | class-blocked image rows cannot trade away tracking for held-out accuracy or vice versa |
| Class-blocked retention | `>= 30` paired seeds | final-window MSE diff `> 0`, final-window accuracy diff `>= 0`, non-recent held-out accuracy diff `>= +0.03` | non-recent and macro accuracy at least `chance + 0.10`; min class accuracy `>= 0.05` |
| Transformer FFN-only | `>= 5` paired seeds and `>= 100000` online token updates unless explicitly justified | eval NLL diff `>= -0.02`, final-window NLL diff `>= -0.02` | eval accuracy diff `>= -0.005`, final-window accuracy diff `>= -0.01`, no bad seeds, steps/s and state reported |

## Expected Incoming Artifacts

If workers produce reports elsewhere, keep those paths in the workstream report
and also drop or summarize the normalized result under the expected coordinator
path below.

| Workstream | Expected coordinator artifact | Required summary fields |
|---|---|---|
| Core output-head anti-drift | `output/subagents/gap_followup_synthesis/output_head_anti_drift_results.json` | candidate config, `mlp64` and `mlp64_64` baselines, final-window MSE/accuracy, test MSE/accuracy, recent/non-recent/macro/min-class accuracy, per-class bias/logit diagnostics |
| Sklearn external gap sweep | `output/subagents/gap_followup_synthesis/sklearn_external_gap_sweep_results.json` | dataset family, task kind, seeds, candidate/baseline configs, final-window and held-out metrics, CE/NLL for classification, timing, skipped rows |
| Real-image class-blocked sweep | `output/subagents/gap_followup_synthesis/real_image_class_blocked_results.json` | real data source, regime, train/test subset sizes, seeds, final-window NLL/MSE, held-out accuracy, per-block/recent/non-recent metrics if class blocked |
| Clean transformer FFN-only experiment | `output/subagents/gap_followup_synthesis/transformer_ffn_only_results.json` | architecture diff, parameter/state counts, tokens, seeds, eval/final NLL, perplexity, accuracy, steps/s, bad seeds |
| Class-blocked mechanism search | `output/subagents/gap_followup_synthesis/class_blocked_mechanism_search_results.json` | mechanism label, targeted hypothesis, tracking/retention metrics, output-head drift diagnostics, utility/perturbation diagnostics, ablation controls |

Required top-level JSON shape:

```json
{
  "gate_contract_version": 1,
  "workstream": "name",
  "evidence_level": "exploratory | candidate | presentation_candidate",
  "candidate": {},
  "baselines": [],
  "config": {},
  "primary_metrics": {},
  "guardrail_metrics": {},
  "per_seed_records": [],
  "diagnostics": {},
  "artifacts": {},
  "skipped_or_failed": []
}
```

## Workstream Interpretation Guide

### Core Output-head Anti-drift

This workstream should be read as the direct attack on the class-blocked
failure mode. The key question is whether output logits/biases for absent
classes remain calibrated while the learner still adapts to the current block.

Real improvement:

- beats `mlp64` on final-window MSE and final-window accuracy;
- beats `mlp64` on held-out test MSE and held-out accuracy;
- improves non-recent held-out accuracy by at least `+0.03`;
- does not collapse any class below min-class accuracy `0.05`;
- reports output-head drift diagnostics showing reduced absent-class decay or
  bias drift.

Presentation-quality:

- the above at 30 paired seeds;
- non-inferior to `mlp64_64` on final-window tracking, or explicitly states
  that `mlp64_64` remains the tracking ceiling while UPGD wins retention;
- no regression on existing promoted UPGD strengths outside class-blocked
  digits if a broader regression check is included.

Interpretation when it returns:

- If it clears tracking and non-recent retention, this is the first credible
  class-blocked closure candidate.
- If it improves tracking but hurts non-recent retention, it reproduces the
  known tradeoff and is not a solution.
- If it improves retention but still loses current-block tracking, it is a
  useful anti-drift mechanism but not a complete MLP replacement.

### Sklearn External Gap Sweep

This workstream should answer whether the sklearn breadth result survives a
larger, less cherry-pickable external matrix and whether diabetes/regression is
still a boundary.

Real improvement:

- promoted UPGD beats MLP64 on the aggregate primary metric;
- at least 70% of rows are positive;
- all classification rows satisfy the `-0.02` held-out accuracy guardrail;
- all regression rows satisfy the 5% held-out MSE degradation guardrail;
- CE/NLL and timing are reported, not reconstructed post hoc.

Presentation-quality:

- at least 6 rows, 4 dataset families, 3 task kinds, and 10 paired seeds per
  row;
- no missing/skipped rows hidden from the aggregate;
- diabetes or any regression loss is either fixed by a pre-registered candidate
  or stated as an open boundary.

Interpretation when it returns:

- If classification remains strong but regression remains negative, the honest
  claim is "UPGD has external classification breadth, not universal tabular
  superiority."
- If regression improves only after changing the UPGD factory, classify it as
  a new ablation candidate, not promoted-default evidence.
- If aggregate wins depend on excluding hard rows, reject the closure claim.

### Real-image Class-blocked Sweep

This workstream should determine whether the bounded image positives survive
harder class-blocked image regimes.

Real improvement:

- UPGD improves held-out accuracy and final-window NLL/MSE on class-blocked
  image rows against the best same-run MLP baseline;
- positive differences hold by paired seed majority;
- iid rows remain non-negative, so class-blocked fixes do not damage ordinary
  image learning.

Presentation-quality:

- at least 10 paired seeds;
- longer non-smoke horizons than the 600-step bounded probe;
- fixed real data sources and subset sizes reported;
- CIFAR/OPMNIST class-blocked rows include block/recent/non-recent analysis,
  not just aggregate test accuracy.

Interpretation when it returns:

- Held-out accuracy wins with final-window NLL/MSE losses mean the old
  retention/tracking split persists.
- Final-window wins with held-out losses mean the method became an MLP-like
  current-block tracker and did not preserve UPGD's retention advantage.
- Both-side wins are strong evidence that output-head anti-drift generalizes
  beyond sklearn digits.

### Clean Transformer FFN-only Experiment

This workstream should separate true FFN replacement from the existing UPGD
readout experiment.

Real improvement:

- same attention, residual stream, tokenizer, training examples, and readout
  as the MLP transformer, with only the FFN block replaced or clearly ablated;
- parameter and state counts are reported;
- eval/final NLL are non-inferior within `-0.02`;
- accuracy guards pass;
- no bad seeds.

Presentation-quality:

- at least 5 paired seeds and 100k online token updates, or a clearly justified
  smaller horizon with stable learning curves;
- compile-excluded or consistently measured timing;
- the report names the claim precisely: `upgd_ffn_replacement` if FFN-only,
  `upgd_readout_transformer` if not.

Interpretation when it returns:

- If it is not FFN-only, do not use it to support an FFN replacement claim.
- If FFN-only UPGD is non-inferior on NLL but much slower, it may be research
  evidence but not a practical replacement without a compute story.
- If it loses NLL/perplexity like the current 2,000-step readout run, the
  transformer gap remains open.

### Class-blocked Mechanism Search

This workstream should explain the failure mode, not just find another local
variant. Mechanism evidence is valuable even if it does not close the gate.

Real improvement:

- the candidate states a targeted mechanism such as bias decay, logit
  centering, absent-class calibration, per-class normalization, or perturbation
  throttling;
- it improves the targeted diagnostic and at least one primary
  tracking/retention metric;
- it includes a control showing the effect is not just extra capacity or a
  disguised MLP fallback.

Presentation-quality:

- a mechanism candidate promoted from this search must pass the same 30-seed
  class-blocked retention bar as output-head anti-drift;
- the mechanism report should include enough diagnostics to explain why it
  moves both tracking and retention, or state which side remains unsolved.

Interpretation when it returns:

- A one-sided metric gain is a mechanism clue, not a solution.
- A diagnostic gain without metric gain should feed the next design, but should
  not be promoted.
- A metric gain without diagnostics should be rerun with diagnostics before it
  is presented as a mechanism.

## Decision Labels For Returned Workstreams

Use these labels consistently in follow-up synthesis:

| Label | Meaning |
|---|---|
| `close_candidate` | Clears real-improvement criteria at candidate scale and is worth a presentation-scale rerun. |
| `presentation_close` | Clears the presentation-quality bar with raw artifacts and diagnostics. |
| `bounded_positive` | Positive and useful, but underpowered or narrower than the presentation claim. |
| `mechanism_clue` | Improves a diagnostic or one side of the tradeoff but does not close the gap. |
| `known_tradeoff_reproduced` | Repeats tracking-versus-retention or readout-versus-FFN mismatch already seen. |
| `negative_boundary` | A clean failure that should be stated as a boundary. |
| `invalid_for_claim` | Mismatched architecture, hidden fallback, missing baseline, missing rows, or post-hoc metric selection. |

## Final Synthesis Rule

When all five workstreams return, the final answer should not average away
incompatible evidence. Report improvements by claim type:

- promoted UPGD classification breadth;
- class-blocked anti-drift/retention;
- real-image class-blocked generalization;
- transformer FFN-only viability;
- remaining negative boundaries.

The Step 2 story is only stronger if a workstream closes the specific gap it
targets without breaking the guardrails that made the earlier UPGD result
interesting.
