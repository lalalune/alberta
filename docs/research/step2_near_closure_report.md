# Step 2 Near-Closure Report

Date: 2026-05-04

## Executive Read

We are close, but the honest claim is narrower than "UPGD beats MLP."

Current best evidence:

- Best single synthetic Step 2 learner: `UPGDLearner`. It beats the best fair MLP on the canonical out-of-class synthetic suite at 30 seeds.
- Best operational Step 2 system: strict universal portfolio over live MLP widths, low-noise UPGD, and dynamic sparse, with causal Hedge/retention routing. It closes the current eight-row repo matrix versus best fair MLP.
- Best single-learner progress against digits: two-signal UPGD stale/fan-in recycling beats MLP64 on permuted-pixel final-window and held-out MSE, but still loses accuracy.
- Main remaining gap: no single online learner yet beats fair MLP across digits IID, mask noise, permuted pixels, label drift, class-blocked retention, and the synthetic feature-construction suite.

So the scientific state is: Step 2 is operationally solved by a temporally-uniform portfolio, but not yet cleanly solved by one universal feature-finding learner. The strongest continuation is to make UPGD less like "hidden perturbation plus squared-error heads" and more like a complete feature economy: utility-aware hidden lifecycle, protected readout geometry, and causal retention signals.

## What Step 2 Requires

The Alberta Plan Step 2 asks for supervised feature finding in the Step 1 continual supervised-learning setting, but with vector targets and multiple tasks. The system has a fixed resource budget. It must:

- construct new nonlinear features from existing features;
- assign utility to hidden features across multiple tasks;
- decide when a feature should be discarded;
- decide how replacement features should be initialized;
- learn continually without replay, batch retraining, task IDs, or post-hoc oracle selection;
- avoid sacrificing interim performance while new features are tested.

This is stricter than "train an MLP online." It is also stricter than "find a good feature set offline." It is an online resource-allocation problem over representational elements.

## Why MLP Is Strong

The fair MLP baseline is not weak. On sklearn digits it has several real advantages:

1. Digits are friendly to dense nonlinear parametric features. A small MLP with 64 or 64x64 hidden units is already enough capacity.
2. The MLP does not disrupt its own readout. UPGD recycling can improve MSE while damaging class-discriminative head columns, which hurts argmax accuracy.
3. Squared-error one-hot learning can still work surprisingly well for small multiclass problems. It is not theoretically ideal for classification, but it gives dense negative updates to absent classes and is stable.
4. Short external streams reward current tracking. The plasticity loss literature becomes most decisive over very long streams; our 1200-step digits regimes often reward fast exploitation more than long-horizon regeneration.
5. Accuracy and MSE disagree. UPGD can win or tie MSE while having weaker true-vs-top-wrong margins, so argmax accuracy still trails.
6. Class-blocked metrics can reward forgetting. A fast MLP can look excellent on the current block while having almost no retained accuracy on absent classes.
7. A convex ensemble of MLP widths is a very strong low-complexity baseline. Existing evidence shows much of the strict portfolio's digit gain comes from prediction-space averaging over MLP widths, not exotic feature discovery.

This is why the goal should not be "beat one under-tuned MLP." The useful target is "beat a fair MLP grid under tracking, retained deployment, and feature-construction stressors."

## Existing Research Context

The Alberta Plan frames intelligence as temporally uniform continual prediction and control from experience, with explicit attention to limited computation and feature/resource management.

Generate-and-test representation search is the closest classical Step 2 ancestor. Mahmood and Sutton treat representation learning as online generation, utility testing, and replacement under fixed per-step cost. This directly matches the Step 2 question.

Continual Backprop shows that SGD's initial random diversity is not enough for lifelong learning. It maintains plasticity by continually reinitializing a small fraction of low-utility units. The later plasticity papers show that long continual streams produce dead/constant units, growing weight magnitudes, lower effective rank, and degraded learning, and that random non-gradient diversity can be necessary.

UPGD extends this line by perturbing low-utility weights/features while protecting useful ones. It is scientifically aligned with Step 2 because it is online, utility-based, fixed-resource, and explicitly about retaining plasticity.

Online convex optimization and proper-loss classification are relevant for the readout. A softmax/cross-entropy readout has competitive geometry: labels compete on a simplex rather than through independent unbounded regressors. This does not solve hidden feature construction, but it can remove a misleading bottleneck where a good representation loses by margin/ranking.

The 2026 EML paper is not direct evidence for Step 2, but it suggests one useful research idea: universal feature construction may not need a large operator vocabulary. A single stable binary primitive plus learned routing could be a cleaner generator than an expanding grab bag of feature ops. That should be tested only inside the same causal utility/replacement framework.

## Current Repo Evidence

### UPGD Synthetic Strength

`docs/research/step2_current_best.md` records the current synthetic result:

| Stream | Best MLP final MSE | UPGD final MSE | Paired diff | Wins |
|---|---:|---:|---:|---:|
| Out-of-class polynomial | 1.1458 | 0.5767 | +0.569 +/- 0.032 | 30/30 |
| Frequency mismatch | 1.1689 | 0.6335 | +0.535 +/- 0.037 | 30/30 |
| Compositional 2-layer oracle | 0.1908 | 0.1634 | +0.027 +/- 0.003 | 29/30 |

This is the best standalone feature-finding evidence in the repo.

### Strict Portfolio Closure

`docs/research/step2_universal_portfolio_assessment.md` records the strict 10-seed portfolio:

| Regime | Final-window MSE vs best MLP | Test accuracy vs best MLP |
|---|---:|---:|
| Synthetic polynomial | +0.0414 | n/a |
| Synthetic frequency | +0.0168 | n/a |
| Synthetic compositional | +0.0018 | n/a |
| Digits IID | +0.0071 | +0.0128 |
| Digits class-blocked | +0.0000 | +0.0970 |
| Digits permuted pixels | +0.0075 | +0.0156 |
| Digits mask noise | +0.0094 | +0.0158 |
| Digits label drift | +0.0055 | +0.0128 |

This is the current operational Step 2 solution, but it is complex and partly depends on MLP-width averaging.

### Simplification Evidence

`docs/research/step2_sklearn_simplification_synthesis.md` shows that digits performance is largely explained by a convex ensemble over fair MLP widths:

- uniform or slow-Hedge MLP-width ensembles preserve most digit tracking gains;
- hard selector routing loses the mixture advantage;
- dynamic sparse does not pay rent on digits;
- UPGD remains useful for synthetic structure and class-blocked retained deployment.

This makes the strict portfolio less mysterious, but it also sharpens the challenge: a single UPGD-style learner must beat a very strong simple ensemble, not just one MLP.

### Two-Signal UPGD Progress

`output/subagents/upgd_unit_recycle/TWO_SIGNAL_UPDATE.md` adds hidden-unit lifecycle signals:

- short contribution utility: row-mean `abs(weight * gradient)`;
- long retained utility: slower EMA of the same contribution;
- current demand: row-mean `abs(gradient)`;
- recycling criteria: low utility, stale gradient ratio, or low long-and-gradient;
- replacement fan-in: random or high-gradient input columns;
- optional causal fast/slow loss-spike gate.

Best permuted-pixels 10-seed result:

| Method | Final MSE diff vs MLP64 | Final Acc diff | Test MSE diff | Test Acc diff |
|---|---:|---:|---:|---:|
| `upgd_sum_stale_fanin_3e_3` | +0.0025 | -0.0743 | +0.0033 | -0.0160 |
| `upgd_sum_stale_fanin_5e_3` | +0.0023 | -0.0813 | +0.0033 | -0.0154 |
| `upgd_sum_lowboth_fanin_3e_3` | +0.0009 | -0.0850 | +0.0031 | -0.0234 |

This is the best single-learner Step 2 progress on permuted pixels. The accuracy gap is the next bottleneck.

### Label Drift

Label drift is mostly output remapping, not hidden feature discovery. Existing probes show:

- plain squared-error UPGD linear heads lose badly;
- softmax/raw readout is strong;
- UPGD trunk plus softmax readout can improve MSE, but accuracy remains mixed unless the readout geometry is handled natively.

The key issue is that the current UPGD trunk utility is still driven through squared-error linear heads. A real test must feed softmax/cross-entropy gradients into trunk utility, not just attach an observer readout.

### Class-Blocked Retention

Class-blocked digits are mostly a head-retention problem:

- fast MLP tracks the active block and forgets absent classes;
- UPGD retains slightly better but adapts too slowly;
- hidden recycling does not address missing positive evidence for absent class heads;
- retained deployment needs protected class-level readout state, not naive hidden resets.

## Problem Decomposition

The hard Step 2 problem has split into four simpler mechanisms:

1. Hidden-feature lifecycle: when inputs change, stale features should be replaced or rewired without disrupting useful class-discriminative output columns.
2. Readout geometry: classification heads should compete. Squared-error heads can hide ranking/margin failures behind acceptable MSE.
3. Retention under missing labels: when a task/class is absent, its head receives no positive evidence, so utility must not decay purely because the stream stopped showing that target.
4. Universal evaluation: a candidate that fixes one failure mode must not regress synthetic feature-construction strength, Step 1 scalar tracking, or Step 3 GVF readiness.

## Five Strongest Continuations

### 1. Native UPGD Softmax/Cross-Entropy Readout

Status: completed by Russell (`019df64d-3ad0-7052-b953-a020b9f92a48`). Notes: `/Users/shawwalters/Desktop/nca_fun/alberta-framework/output/subagents/continuation_softmax_readout/NOTES.md`.

Hypothesis: label drift and much of the UPGD accuracy gap are readout-geometry failures. A native softmax readout should improve class remapping and feed more appropriate gradients into hidden utility.

Implementation:

- Add `readout_mode="linear_mse" | "softmax_ce"` to UPGD or create `UPGDSoftmaxLearner`.
- Reuse head weights as logits: `p = softmax(W h + b)`.
- Update heads with `e = one_hot(y) - p`, normalized by `1 + ||h||^2`.
- Use the same CE-derived head gradient to update trunk weights and unit utilities.
- Keep surprise reset as an ablation, not the default.

Experiment:

- Primary: digits `label_drift`, 10 seeds, 1200 steps, final window 300.
- Baselines: MLP64, current UPGD sigma0/sigma1e-4, raw online softmax, existing probe.
- Candidates: softmax UPGD with alpha 0.75/1.5, sigma 0/1e-4, reset off/on.
- Guards: IID, class-blocked, permuted pixels, mask noise, synthetic trio.

Success:

- Beat MLP64 on label-drift paired final-window or held-out accuracy.
- Beat current UPGD on all primary label-drift metrics.
- Do not lose more than 0.01 held-out accuracy on guard digits.
- Retain at least 95% of current UPGD synthetic advantage.

Risk:

- Raw softmax may beat UPGD hidden softmax. If so, the readout is necessary but the trunk is not helping on digits.

### 2. Stale-Preserve Partial Rewiring

Status: completed by Lagrange (`019df64d-516b-7001-8a00-7947c6d79fe5`). Notes: `/Users/shawwalters/Desktop/nca_fun/alberta-framework/output/subagents/continuation_two_timescale_recycling/RESEARCH_NOTE.md`.

Hypothesis: stale/fan-in recycling wins MSE but hurts accuracy because full hidden-unit reset zeroes or damages outgoing class-discriminative columns. Replace less destructively.

Implementation:

- Add outgoing-aware utility: include `abs(head_weight * head_grad)` in unit diagnostics.
- Add outgoing preservation on replacement: `alpha_out = 0.5` and `1.0`.
- Add partial fan-in rewiring: replace only bottom-k incoming weights, e.g. k=8/16, using high-gradient input columns.
- Add a stale threshold so replacement budget accumulates when no unit is clearly stale.

Experiment:

- Primary: digits `permuted_pixels`, 10 seeds.
- Start from best current settings: sum loss, sigma 0, stale ratio, gradient fan-in, loss gate 1.08, rate 3e-3.
- Compare current full replacement, preserve outgoing, partial k8/k16, lower rate 1e-3, stricter gate 1.12, outgoing-aware selection.
- Guards: label drift and mask noise first, then synthetic compositional.

Success:

- Keep final MSE diff at least +0.001 vs MLP64.
- Improve final accuracy gap from -0.0743 to better than -0.04.
- Keep held-out accuracy no worse than current -0.0160.

Risk:

- Preserving outgoing weights may preserve stale semantics. Partial rewiring must be strong enough to adapt while weak enough not to destroy ranking.

### 3. Output-Only Margin Adapter

Status: completed by Helmholtz (`019df64d-66d5-7021-8192-12491ee1c3a5`). Notes: `/Users/shawwalters/Desktop/nca_fun/output/subagents/continuation_calibration_accuracy/NOTES.md`.

Hypothesis: UPGD accuracy failures are margin/ranking failures, not simple calibration failures. Temperature scaling cannot change argmax. A small causal margin adapter can target the true-vs-top-wrong gap directly.

Implementation:

- Keep UPGD hidden update unchanged.
- Add optional output-only normalized calibrated logits.
- Add a small perceptron/hinge update only when `q_true - max_wrong(q) < margin`.
- Track score scale, true-vs-wrong margin, and argmax flips.

Experiment:

- Baselines: MLP64, current UPGD, two-signal UPGD.
- Candidates: normalized-logit eval only, residual bias adapter, output-only margin update, normalized+bias+margin.
- Primary rows: permuted pixels, mask noise, label drift.
- Combine best margin adapter with stale/fan-in UPGD.

Success:

- Close at least half the accuracy gap on permuted pixels without losing the MSE win.
- Improve label-drift accuracy without relying on replay or memory.

Risk:

- This may improve classification but be less directly relevant to scalar GVFs. Keep it modular and do not bake it into the feature lifecycle.

### 4. Dual-Timescale Retained Heads For Class-Blocked Streams

Status: completed by Arendt (`019df64d-7ca9-7dc3-bf06-c7b65517f5e0`). Notes: `/Users/shawwalters/Desktop/nca_fun/output/subagents/continuation_class_blocked_retention/NOTE.md`.

Hypothesis: class-blocked failure is head retention, not hidden recycling. Absent classes need protected readout state when they are not currently generating positive examples.

Implementation:

- Fast trunk + fast heads update as current MLP for tracking.
- EMA trunk + retained per-class heads update only the active class row.
- Optional tiny top-1 confuser hinge update prevents retained heads from overfiring.
- Per-class retained-head utility gates blending under the causal class-coverage hazard.
- No exemplars, no replay, fixed resource.

Experiment:

- Primary: digits `class_blocked`.
- Baselines: MLP64, MLP64x64, current UPGD, low-noise UPGD, slow-trunk/headx3, EMA controls.
- Candidates: positive-only retained heads, positive+top1-negative heads, shared-trunk retained heads, EMA-trunk retained heads, utility-gated blend, fixed blend.

Metrics:

- final-window MSE/accuracy;
- held-out test accuracy;
- absent/non-recent test accuracy;
- block-first and block-tail accuracy;
- gate rate and retained/fast blend mass.

Success:

- Tie MLP64x64 final-window MSE.
- Improve retained/absent-class accuracy without memory replay.
- Do not degrade normal IID/permuted tracking.

Risk:

- This is task/class-specific. For Step 3, the analogue must be "demon/head retention under sparse cumulant evidence," not digit-label hacks.

### 5. Single-Learner Universality Matrix

Status: completed by Poincare (`019df64d-9aa6-7080-bf48-b6ebd24f8443`). Notes: `/Users/shawwalters/Desktop/nca_fun/output/subagents/continuation_universality_matrix/README.md`.

Hypothesis: the only way to avoid local overfitting is to promote a single-learner candidate only after it survives the whole matrix.

Implementation:

- Build an isolated single-learner matrix under `output/subagents/continuation_universality_matrix/`.
- Compare `mlp64`, `current_upgd`, `two_signal_upgd`, and `two_signal_upgd_better_readout`.
- Reuse existing factories from the expert-mixture, UPGD sweep, label-drift, and class-blocked runners.

Matrix:

- scalar nonstationary supervised bridge;
- digits IID;
- digits mask noise;
- digits permuted pixels;
- digits label drift;
- digits class-blocked;
- synthetic compositional.

Success:

- 10 seeds first; scale label drift, class-blocked, and synthetic compositional to 30 seeds if green.
- No MLP64 regression on primary digit metrics.
- Improve over current UPGD on weak digits rows.
- No meaningful regression versus current UPGD on synthetic compositional.
- Accuracy must be first-class for digits; MSE-only wins are insufficient.

Risk:

- Softmax readout is classification-specific. A digits win does not prove GVF readiness. The matrix must include scalar/vector regression rows.

## Cleanup Recommendation

Keep:

- UPGD core and two-signal lifecycle;
- strict portfolio as the operational baseline;
- MLP-width ensemble as the simple digits baseline;
- raw online softmax as a permanent readout control;
- resource-manager evidence for causal allocation;
- retained-head experiments as class-blocked-specific probes.

Deprioritize:

- dynamic sparse for digits deployment;
- hard selector routing;
- memory/kNN as a core Alberta Plan Step 2 solution, because it behaves like replay/exemplar storage;
- one-off moonshots that do not plug into feature utility, readout geometry, or retention.

Do not remove:

- failed experiment summaries. They are useful negative controls.

## Near-Term Implementation Order

1. Implement native `softmax_ce` UPGD readout and verify tests.
2. Implement margin diagnostics and an optional output-only margin adapter.
3. Implement stale-preserve partial rewiring behind flags.
4. Implement retained heads as a separate experimental learner, not inside core UPGD.
5. Run the single-learner universality matrix and promote only if it beats both MLP64 and current UPGD on the agreed metrics.

The best likely combined candidate is:

```text
two_signal_upgd
+ native softmax/CE readout for classification
+ stale-preserve partial rewiring
+ optional output margin adapter
+ retained-head module only under sparse-positive-evidence hazards
```

This remains Alberta Plan friendly if all components are causal, online, fixed-memory, temporally uniform, and evaluated prequentially.

## Source Links

- Alberta Plan: https://arxiv.org/abs/2208.11173
- Utility-based Perturbed Gradient Descent: https://arxiv.org/abs/2302.03281
- Continual Backprop: https://arxiv.org/abs/2108.06325
- Maintaining Plasticity in Deep Continual Learning: https://arxiv.org/abs/2306.13812
- Nature version, Loss of Plasticity in Deep Continual Learning: https://www.nature.com/articles/s41586-024-07711-7
- Representation Search through Generate and Test: https://armahmood.github.io/files/MS-RepSearch-AAAI-WS-2013.pdf
- Introduction to Online Convex Optimization: https://arxiv.org/abs/1909.05207
- All elementary functions from a single binary operator: https://arxiv.org/abs/2603.21852
