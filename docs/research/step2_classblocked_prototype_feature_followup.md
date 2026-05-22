# Step 2 Class-Blocked Prototype Feature Follow-up

Date: 2026-05-06.

## Question

Can the unresolved class-blocked tracking/retention conflict be repaired without
portfolios, replay, task ids, MLP fallback, or future labels?

Short answer: partially. Against raw MLPs, the prototype branch is strong. A
fairer sharpened-MLP readout is a harder baseline and invalidates the earlier
"solved" framing. The current best strict class-blocked learner is a
resource-bounded UPGD readout with one constructed signal and an output-head
anti-drift rule:

- a low-mix one-centroid-per-class readout for current-block tracking;
- a hysteretic utility gate that keeps constructed readouts active through
  transient harm but shuts them off under sustained negative utility.
- simplex output-head bias centering to reduce class-block drift.

The early `combobest_*` branch beats sharpened `mlp64_64` on 30-seed
final-window MSE and slightly improves retained held-out MSE, but the retention
gain is modest and the added memory costs about 3x wall-clock in this Python/JAX
probe. The more aggressive memory-only variants retain much more, but they lose
tracking. The first stateless utility gate fixed label drift but gave back the
class-blocked tracking edge. A stateful hysteretic gate recovers the edge while
preserving the label-drift repair. The newest best branch removes the
multi-prototype memory again: a 64x64 UPGD base with output-head anti-drift plus
a centroid-only hysteretic prototype feature is simpler and has the strongest
class-blocked 30-seed tracking result so far.

Current best candidates:

- `centhystbest64_center_c030_off005` as the current strict class-blocked
  tracker and default simple candidate.
- `centhystbest64_center_c033_off005` as the slightly retention-leaning
  centroid-only candidate.
- `centhystbest64_bias0_c030_off005` as the bias-free centroid-only candidate
  with stronger label-drift tracking but weaker class-blocked tracking.
- `antidrift64_center_hyst_c030_mem005_bw030` as the previous memory-assisted
  strict tracker.
- `antidrift64_bias0_reset_c033_mem008_bw020` as the 64x64 reset/lifecycle
  variant with better retained test behavior.
- `hystbest_c030_mem005_bw030_ud099_off005_on001`
- `hystbest_c030_mem005_bw020_ud099_off010_on002`
- `hystresetbest_c033_mem008_bw020_off002_reset005` as the lifecycle/recycling
  Pareto branch with better retained test behavior.
- `combobest_c030_mem005_bw030_sh010_b10` as the stronger non-gated
  class-blocked-only candidate.
- `deep_repx075_m030_t005_b10` for the simpler strict tracking win with almost
  no retained-accuracy gain.

## Mechanism

The runner is:

`output/subagents/classblocked_prototype_features/run_prototype_features.py`

The reusable constructor is:

`src/alberta_framework/core/prototype_features.py`

The original one-centroid branch works as follows. At each online step, before
updating on the current label:

1. Maintain one normalized input-space prototype per class.
2. Build a prototype probability vector from cosine similarities with
   temperature `0.05`.
3. Blend the UPGD prediction with a small prototype probability mixture
   (`0.15` or `0.20`).
4. If the resulting prediction has top-1/top-2 margin at least `0.10`, blend it
   toward its top-1 one-hot vector. The current strongest class-blocked branch
   uses hard sharpening (`blend=1.0`).
5. Update the normal UPGD learner and then update only the observed class
   prototype by EMA with `alpha=0.05`.

This is causal and temporally uniform. It stores prototypes, not examples. It is
best read as supervised feature construction: prototype-similarity features are
made from existing input features, and the readout uses them to correct the
current-block calibration/retention conflict.

The newer `combobest_*` branch keeps the same centroid signal and adds a
fixed-budget multi-prototype memory from
`src/alberta_framework/core/prototype_memory.py`. The memory keeps 8 slots per
class, allocates by novelty, and contributes only a 3-8% readout mixture in the
best class-blocked settings.

## Fair Sharpened MLP Caveat

Artifact:

- `output/subagents/classblocked_prototype_features/fair_sharpened_mlp_10seed_1200/results.json`

Applying the same top-1 confidence sharpening to `mlp64_64` makes it much
stronger on class-blocked final-window MSE:

| Method | Final MSE | Final acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mlp64_64_sharp_t0.1_b1` | `0.001533 +/- 0.000069` | `0.992333` | `0.180111` | `0.099443` |
| `mlp64_64` | `0.002817 +/- 0.000055` | `0.991000` | `0.163014` | `0.099443` |

This is the baseline to beat for current-block tracking. The earlier
one-centroid `accuracy_best` result beats raw MLPs, but not this sharpened
baseline.

## Current Best 30-Seed Class-Blocked Result

Artifact:

- `output/subagents/classblocked_prototype_features/combo_best_30seed_1200/results.json`
- `output/subagents/classblocked_prototype_features/combo_best_30seed_1200/SUMMARY.md`

Aggregate:

| Method | Final MSE | Final acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `combobest_c030_mem005_bw030_sh010_b10` | `0.001542 +/- 0.000042` | `0.991778` | `0.177475` | `0.101917` |
| `combobest_c030_mem005_bw020_sh010_b10` | `0.001551 +/- 0.000052` | `0.991889` | `0.177112` | `0.103772` |
| `combobest_c035_mem005_bw030_sh010_b10` | `0.001609 +/- 0.000054` | `0.991778` | `0.169713` | `0.107421` |
| `combobest_c033_mem008_bw020_sh010_b10` | `0.001640 +/- 0.000064` | `0.991444` | `0.168050` | `0.109709` |
| `mlp64_64_sharp_t0.1_b1` | `0.001671 +/- 0.000053` | `0.991111` | `0.180000` | `0.100000` |

Paired against `mlp64_64_sharp_t0.1_b1`:

| Method | Final MSE diff | Final acc diff | Test MSE diff | Test acc diff |
|---|---:|---:|---:|---:|
| `c030_mem005_bw030` | `+0.000129 +/- 0.000059`, `17/30` wins | `+0.000667` | `+0.002525`, `18/30` wins | `+0.001917` |
| `c030_mem005_bw020` | `+0.000120 +/- 0.000059`, `18/30` wins | `+0.000778` | `+0.002888`, `17/30` wins | `+0.003772` |

This is a real but small win. It is evidence for the mechanism, not a
production-quality universal replacement.

## Hysteretic Utility-Gated Variant

Artifact:

- `output/subagents/classblocked_prototype_features/hysteretic_combo_best_30seed_1200/`
- `output/subagents/classblocked_prototype_features/hysteretic_combo_best_guardrail_5seed_1200/`

The hysteretic gate keeps a binary on/off state for each constructed readout. It
starts on, turns off only when EMA utility is below `-off_threshold`, and turns
back on only when utility is above `on_threshold`. This matters because the
prototype readouts can be temporarily harmful early in a block even when they
are useful over the phase.

30-seed results against sharpened `mlp64_64`:

| Regime | Method | Final MSE diff | Wins | Test MSE diff | Test acc diff |
|---|---|---:|---:|---:|---:|
| `class_blocked` | `hystbest_c030_mem005_bw030_ud099_off005_on001` | `+0.000086 +/- 0.000054` | `17/30` | `+0.002509` | `+0.002474` |
| `class_blocked` | `hystbest_c030_mem005_bw020_ud099_off010_on002` | `+0.000082 +/- 0.000068` | `16/30` | `+0.002817` | `+0.004020` |
| `label_drift` | `hystbest_c030_mem005_bw020_ud099_off010_on002` | `+0.010019 +/- 0.000618` | `30/30` | `-0.001478` | `+0.001237` |
| `label_drift` | `hystbest_c030_mem005_bw030_ud099_off005_on001` | `+0.009894 +/- 0.000644` | `30/30` | `-0.001610` | `+0.001484` |

Five-seed guardrail for one single balanced config,
`hystbest_c030_mem005_bw030_ud099_off005_on001`:

| Regime | Final MSE diff vs sharp `mlp64_64` | Wins | Test MSE diff | Test acc diff |
|---|---:|---:|---:|---:|
| `iid` | `+0.006948` | `5/5` | `+0.005927` | `+0.030798` |
| `permuted` | `+0.012692` | `5/5` | `+0.045886` | `+0.217811` |
| `mask_noise` | `+0.018041` | `5/5` | `+0.023278` | `+0.131354` |

This is the best current Step 2 class-blocked/label-drift compromise. It is
still not a universal claim: label-drift held-out MSE is slightly worse than
the sharpened MLP because the final test labels use original semantics while
the stream's label mapping has drifted. The online final-window result is the
right primary metric for this stressor.

## Reset/Recycling Variant

Artifact:

- `output/subagents/classblocked_prototype_features/hysteretic_reset_best_30seed_1200/`

The reset variant recycles the constructed prototype state when hysteretic
utility becomes strongly negative. This is closer to the Alberta Step 2 feature
lifecycle requirement: stale features are not merely hidden; their resource is
made available for new target semantics.

30-seed results against sharpened `mlp64_64`:

| Regime | Method | Final MSE diff | Wins | Test MSE diff | Test acc diff |
|---|---|---:|---:|---:|---:|
| `class_blocked` | `hystresetbest_c033_mem008_bw020_off010_reset010` | `+0.000066 +/- 0.000068` | `14/30` | `+0.011595` | `+0.009586` |
| `class_blocked` | `hystresetbest_c033_mem008_bw020_off002_reset005` | `+0.000038 +/- 0.000058` | `15/30` | `+0.012967` | `+0.015646` |
| `label_drift` | `hystresetbest_c033_mem008_bw020_off002_reset005` | `+0.008136 +/- 0.000761` | `30/30` | `+0.000494` | `+0.002907` |

Interpretation: reset/recycling is not the best strict class-blocked tracker,
but it is the best lifecycle-flavored branch so far because it preserves a
small tracking win while improving retained test behavior and avoiding the
label-drift held-out MSE regression seen in the non-reset hysteretic branch.

## 64x64 Output-Head Anti-Drift Variant

Artifacts:

- `output/subagents/classblocked_prototype_features/anti_drift64_best_30seed_1200/`
- `output/subagents/classblocked_prototype_features/anti_drift64_best_guardrail_5seed_1200/`
- `output/subagents/classblocked_prototype_features/centroid_hysteretic64_best_30seed_1200/`
- `output/subagents/classblocked_prototype_features/centroid_hysteretic64_best_guardrail_5seed_1200/`

This branch keeps the same 64x64 UPGD base used by the best hysteretic
prototype/memory runs, but changes the output head dynamics:

- `bias0`: freezes the output-head bias updates;
- `center`: recenters simplex-classification output biases after each update.

The motivation is that class-blocked retention failure is partly output-head
drift, not just hidden-feature utility. This is still a single learner; there
is no portfolio, replay, task id, MLP fallback, or future-label access.

### Centroid-Only Hysteretic Branch

The strongest and simplest current candidate removes the fixed-budget
multi-prototype memory and keeps only one causal centroid per class plus a
hysteretic utility gate. This better matches the Step 2 feature-finding core:
construct a bounded feature, assign utility to it, suppress it when stale, and
keep the resource footprint small.

30-seed results against sharpened `mlp64_64`:

| Regime | Method | Final MSE diff | Wins | Test MSE diff | Test acc diff | Extra elapsed |
|---|---|---:|---:|---:|---:|---:|
| `class_blocked` | `centhystbest64_center_c030_off005` | `+0.000154 +/- 0.000055` | `17/30` | `+0.001137` | `+0.000433` | `+0.609s` |
| `class_blocked` | `centhystbest64_center_c033_off005` | `+0.000097 +/- 0.000061` | `18/30` | `+0.002334` | `+0.002721` | `+0.639s` |
| `label_drift` | `centhystbest64_bias0_c030_off005` | `+0.009954 +/- 0.000719` | `30/30` | `-0.001909` | `+0.001979` | `+0.720s` |
| `label_drift` | `centhystbest64_center_c033_off005` | `+0.009277 +/- 0.000713` | `30/30` | `-0.001595` | `+0.001175` | `+0.701s` |
| `label_drift` | `centhystbest64_center_c030_off005` | `+0.008900 +/- 0.000775` | `29/30` | `-0.001504` | `+0.002103` | `+0.718s` |

Five-regime guardrail against sharpened `mlp64_64`:

| Regime | `center_c030` final MSE diff | Wins | Test MSE diff | Test acc diff |
|---|---:|---:|---:|---:|
| `iid` | `+0.005720` | `5/5` | `+0.004739` | `+0.024861` |
| `permuted` | `+0.011030` | `5/5` | `+0.061674` | `+0.347681` |
| `class_blocked` | `+0.000154` | `17/30` | `+0.001137` | `+0.000433` |
| `label_drift` | `+0.008900` | `29/30` | `-0.001504` | `+0.002103` |
| `mask_noise` | `+0.017722` | `5/5` | `+0.020949` | `+0.116512` |

This supersedes the memory-assisted non-reset anti-drift branch as the default
strict learner because it is simpler and has the better class-blocked margin.
The memory/reset branch remains useful for retained-test behavior and feature
lifecycle demonstrations.

### Subagent Review and Fresh Ablation

Artifacts:

- `output/subagents/centroid_ablation_worker_a/`
- `output/subagents/centroid_learnable_worker_b/`
- `output/subagents/centroid_compute_worker_c/`
- `output/subagents/centroid_external_worker_d/`
- `output/subagents/centroid_mechanism_worker_e/`
- `output/subagents/centroid_nextdirs_worker_f/`
- `output/subagents/centroid_nextdirs_worker_f/perclass_gate_3seed_1200/`

The parallel review did not find a replacement for
`centhystbest64_center_c030_off005`, but it clarified the mechanism and the
next useful search directions:

| Track | Result | Decision |
|---|---|---|
| Ablation worker | Interrupted before evidence-grade output. | No claim. Rerun only with checkpointing or smaller chunks. |
| Learnable parameters | One-seed, 80-step smoke found learned temperature and no-gate learned mix beating fixed centroid and sharp MLP on class-blocked. | Promising lead, not evidence. Needs 5-seed blocker/regime run before promotion. |
| Compute simplification | Minimal telemetry preserves metrics and reduces overhead in the probe; soft/no sharpening degrades MSE. | Keep centroid-only simplification; do not remove hard sharpening. |
| External breadth | Classification transfer is mixed; centroid helps some shuffled/classification rows but does not address regression and is weaker than promoted UPGD on some rows. | Do not call the centroid branch universal. Add a regression analogue rather than forcing class prototypes. |
| Mechanism diagnostics | Existing 30-seed telemetry shows centroid gate is on in class-blocked and off in label-drift. | Confirms the intended hysteretic utility behavior. |
| Next simple variants | Per-class gate and confidence mixes looked locally promising on class-blocked. | Run focused validation before considering changes. |

Focused per-class gate result:

| Regime | Candidate vs current reference | Wins | Gate read | Decision |
|---|---:|---:|---:|---|
| `class_blocked` | `+0.000319 +/- 0.000160` final-window MSE | `2/3` | `0.596` | Useful diagnostic: class-specific suppression can improve current-block tracking. |
| `label_drift` | `-0.006824 +/- 0.002415` final-window MSE | `0/3` | `0.406` | Reject as default: partial per-class reactivation lets stale centroid signal leak back in. |

This strengthens the scalar hysteretic gate choice. The good behavior is not
"more gates everywhere"; it is specifically a conservative constructed-feature
utility gate that fully suppresses the centroid branch when target semantics
drift. Per-class gates are still worth revisiting only if their utility signal
can be tied to a detected stable class semantics, not merely per-output recent
loss.

### Follow-up on Best Leads

Detailed report:

- `docs/research/step2_leads_followup_learned_compute_regression.md`

Three leads were tested after the subagent review:

- Learned centroid temperature/mix: not promoted. Capped learned mix improves
  class-blocked MSE on the 5-seed screen (`+0.000147` vs fixed for
  `learnmix_eta001_cap035_center_c030`), but it hurts label drift. The combined
  mix+temperature scalar slightly helps label drift but loses class-blocked.
- Minimal telemetry: promoted as a safe production/evidence option. With shared
  initialization, full and minimal telemetry give identical online/test metrics
  on both class-blocked and label-drift 5-seed checks. Timing is mixed, so the
  claim is reduced metric bandwidth and smaller artifacts, not a robust speedup.
- Regression analogue: positive but partial. A bounded local target-prototype
  regressor beats same-key UPGD on diabetes final-window MSE by
  `+0.028963 +/- 0.008221` with `5/5` wins and improves diabetes test MSE by
  `+0.067903 +/- 0.020881`. On Friedman1 it gates off and ties UPGD, so this
  closes the classification-only hole partially but is not a universal
  regression solution.

### Memory-Assisted Branch

30-seed results against sharpened `mlp64_64`:

| Regime | Method | Final MSE diff | Wins | Test MSE diff | Test acc diff | Extra elapsed |
|---|---|---:|---:|---:|---:|---:|
| `class_blocked` | `antidrift64_center_hyst_c030_mem005_bw030` | `+0.000105 +/- 0.000050` | `18/30` | `+0.002212` | `+0.001113` | `+0.730s` |
| `class_blocked` | `antidrift64_bias0_hyst_c030_mem005_bw030` | `+0.000095 +/- 0.000065` | `17/30` | `+0.002712` | `+0.004638` | `+0.651s` |
| `class_blocked` | `antidrift64_bias0_reset_c033_mem008_bw020` | `+0.000044 +/- 0.000069` | `15/30` | `+0.013101` | `+0.020037` | `+0.651s` |
| `label_drift` | `antidrift64_bias0_hyst_c030_mem005_bw030` | `+0.009868 +/- 0.000579` | `30/30` | `-0.000931` | `+0.004576` | `+0.667s` |
| `label_drift` | `antidrift64_center_hyst_c030_mem005_bw030` | `+0.009604 +/- 0.000775` | `30/30` | `-0.001360` | `+0.001793` | `+0.675s` |
| `label_drift` | `antidrift64_bias0_reset_c033_mem008_bw020` | `+0.008722 +/- 0.000652` | `30/30` | `-0.000004` | `+0.001917` | `+0.655s` |

Five-seed guardrail against sharpened `mlp64_64`:

| Regime | Best anti-drift64 final MSE diff | Wins | Best retained/test read |
|---|---:|---:|---|
| `iid` | `+0.006887` | `5/5` | Reset variant: `+0.005257` test MSE, `+0.030427` test acc. |
| `permuted` | `+0.015415` | `5/5` | Non-reset center: `+0.057468` test MSE, `+0.313173` test acc. |
| `class_blocked` | `+0.000211` | `3/5` | Reset center: `+0.016812` test MSE, `+0.012245` test acc. |
| `label_drift` | `+0.010596` | `5/5` | Reset center: `+0.001017` test MSE, `+0.003711` test acc. |
| `mask_noise` | `+0.018704` | `5/5` | Non-reset center: `+0.020878` test MSE, `+0.118738` test acc. |

Interpretation: this is the strongest single-learner Step 2 candidate so far.
It preserves the small fair-MLP class-blocked tracking win, preserves the large
label-drift online win, and adds a plausible output-head anti-drift mechanism
instead of relying only on constructed-feature utility. It is not yet a
production replacement for MLPs: the Python/JAX probe is still slower than the
MLP baseline, and the class-blocked margin remains small enough that it needs
external breadth and longer streams before stronger claims are justified.

A follow-up sweep over larger centroid/memory mixtures looked promising on a
5-seed screen but did not scale:

- Artifact: `output/subagents/classblocked_prototype_features/anti_drift64_sweep_best_30seed_1200/`
- `ad64best_bias0_c033_mem008_bw020_off010`: class-blocked final MSE diff
  `-0.000007 +/- 0.000073`, label-drift final MSE diff
  `+0.009967 +/- 0.000616`.
- `ad64best_center_c033_mem008_bw020_off010`: class-blocked final MSE diff
  `+0.000032 +/- 0.000080`, label-drift final MSE diff
  `+0.009782 +/- 0.000702`.

This rejects the wider-memory setting as the default despite its better
5-seed retained/test read. The simpler `c030_mem005_bw030_off005` setting is
the stronger 30-seed tracker.

Two additional readout probes were rejected as defaults:

- Native `softmax_ce` UPGD readouts improved held-out accuracy but failed the
  online MSE objective badly. On the 5-seed class-blocked screen, the best
  softmax variant had final MSE diff `-0.007091`; on label-drift it was still
  negative. This is not acceptable for the Step 2 squared-error task.
- Reducing negative-class MSE pressure to `0.5` improved label-drift but did
  not improve the 30-seed class-blocked margin. The best scaled variant,
  `negbest64_center_neg05_hyst_c030_mem005_bw030`, had class-blocked final MSE
  diff `+0.000060 +/- 0.000061` and label-drift diff
  `+0.010719 +/- 0.000589`; useful evidence, not the default.

A compute probe was also rejected: replacing separate `learner.predict` and
`learner.update` calls with `learner.update(...).predictions` preserved metrics
but made the JAX runner slower, likely because it forced the full learner update
onto the critical path before prototype scoring. The reverted implementation is
kept.

## Earlier Utility-Gated Variant

Artifact examples:

- `output/subagents/classblocked_prototype_features/utility_combo_threshold3020_5seed_1200/`
- `output/subagents/classblocked_prototype_features/utility_combo_threshold_5seed_1200/`

The stateless causal utility gate tracks EMA loss advantage for the centroid and memory
readouts and suppresses a constructed readout when it becomes harmful. It fixes
the label-drift failure: across two 5-seed blocks, all common utility-gated
variants beat sharpened `mlp64_64` on label-drift final-window MSE, typically by
about `0.006-0.008`.

The same utility gate currently loses the class-blocked edge by about
`0.00007-0.00017` final-window MSE over the same 10 seeds. Interpretation:
feature utility is the right mechanism, but the gate needed hysteresis before
it could replace the non-gated class-blocked candidate.

## Previous Raw-MLP Result

Artifact:

- `output/subagents/classblocked_prototype_features/accuracy_best_30seed_1200/results.json`
- `output/subagents/classblocked_prototype_features/accuracy_best_30seed_1200/SUMMARY.md`

Command:

```bash
.venv/bin/python output/subagents/classblocked_prototype_features/run_prototype_features.py \
  --config-preset accuracy_best \
  --steps 1200 \
  --n-seeds 30 \
  --seed 1020 \
  --final-window 300 \
  --output-dir output/subagents/classblocked_prototype_features/accuracy_best_30seed_1200
```

Aggregate:

| Method | Final MSE | Final acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `protomix_m030_t005_sh010_b10` | `0.002313 +/- 0.000116` | `0.987778` | `0.132240` | `0.235065` |
| `protomix_m025_t005_sh010_b10` | `0.002405 +/- 0.000126` | `0.987000` | `0.151890` | `0.169821` |
| `protomix_m030_t010_sh010_b05` | `0.002828 +/- 0.000098` | `0.987333` | `0.102620` | `0.201299` |
| `mlp64_64` | `0.002893 +/- 0.000030` | `0.991000` | `0.164008` | `0.100000` |
| `margin_m001` | `0.004914 +/- 0.000067` | `0.985333` | `0.132483` | `0.122635` |
| `mlp64` | `0.004991 +/- 0.000077` | `0.985556` | `0.134777` | `0.118677` |

Paired result for `protomix_m030_t005_sh010_b10`:

| Baseline | Final MSE | Final acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `mlp64` | `+0.002678 +/- 0.000095`, `30/30` wins | `+0.002222 +/- 0.000721`, `20/30` wins | `+0.002537 +/- 0.003497`, `16/30` wins | `+0.116388 +/- 0.016828`, `30/30` wins |
| `mlp64_64` | `+0.000581 +/- 0.000107`, `25/30` wins | `-0.003222 +/- 0.000723`, `1/30` wins | `+0.031768 +/- 0.003667`, `29/30` wins | `+0.135065 +/- 0.016852`, `30/30` wins |

This superseded the earlier softer branch when the baseline was raw `mlp64_64`.
It is now superseded by the fair sharpened-MLP comparison above. It remains
useful as evidence that stronger prototype mixtures can buy retained accuracy,
but it is not the current strict tracking result.

## Earlier 30-Seed Class-Blocked Result

Artifact:

- `output/subagents/classblocked_prototype_features/sharpen_best_30seed_1200/results.json`
- `output/subagents/classblocked_prototype_features/sharpen_best_30seed_1200/SUMMARY.md`

Command:

```bash
.venv/bin/python output/subagents/classblocked_prototype_features/run_prototype_features.py \
  --config-preset sharpen_best \
  --steps 1200 \
  --n-seeds 30 \
  --seed 1020 \
  --final-window 300 \
  --output-dir output/subagents/classblocked_prototype_features/sharpen_best_30seed_1200
```

Aggregate:

| Method | Final MSE | Final acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `protomix_m015_t005_sh010_b05` | `0.002710 +/- 0.000077` | `0.987000` | `0.127601` | `0.136673` |
| `protomix_m020_t005_sh010_b05` | `0.002755 +/- 0.000105` | `0.986222` | `0.118107` | `0.153618` |
| `mlp64_64` | `0.002893 +/- 0.000030` | `0.991000` | `0.164008` | `0.100000` |
| `margin_m001` | `0.004914 +/- 0.000067` | `0.985333` | `0.132483` | `0.122635` |
| `mlp64` | `0.004991 +/- 0.000077` | `0.985556` | `0.134777` | `0.118677` |

Paired against `mlp64_64`:

| Method | Final MSE | Final acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `m015` | `+0.000183 +/- 0.000068`, `19/30` wins | `-0.004000`, `0/30` wins | `+0.036406 +/- 0.001084`, `30/30` wins | `+0.036673 +/- 0.007293`, `22/30` wins |
| `m020` | `+0.000138 +/- 0.000093`, `20/30` wins | `-0.004778`, `1/30` wins | `+0.045901 +/- 0.001285`, `30/30` wins | `+0.053618 +/- 0.006760`, `27/30` wins |

Paired against `mlp64`:

| Method | Final MSE | Final acc | Test MSE | Test acc |
|---|---:|---:|---:|---:|
| `m015` | `+0.002281 +/- 0.000068`, `30/30` wins | `+0.001444`, `12/30` wins | `+0.007176 +/- 0.001248`, `24/30` wins | `+0.017996 +/- 0.005049`, `21/30` wins |
| `m020` | `+0.002235 +/- 0.000078`, `30/30` wins | `+0.000667`, `12/30` wins | `+0.016670 +/- 0.000997`, `30/30` wins | `+0.034941 +/- 0.006102`, `25/30` wins |

## Five-Regime Digit Guardrail

Artifact root:

`output/subagents/classblocked_prototype_features/guardrail_5seed/`

The newer hard-sharpened family was checked at:

`output/subagents/classblocked_prototype_features/accuracy_best_guardrail_5seed/`

The two sharpened prototype variants beat both MLP baselines on final-window
MSE in all five checked digit regimes: `iid`, `permuted`, `class_blocked`,
`label_drift`, and `mask_noise`.

Held-out accuracy improved strongly on `iid`, `permuted`, `class_blocked`, and
`mask_noise`. On `label_drift`, held-out accuracy was essentially tied/slightly
lower versus `mlp64` (`-0.0022` for `m015`, `-0.0011` for `m020`) while held-out
MSE still improved. This is expected because the label-drift stream permutes
class-head meanings by phase, while the held-out test keeps the original label
semantics.

For the strongest raw-MLP prototype branch, `protomix_m030_t005_sh010_b10`,
the 5-seed guardrail read was:

| Regime | Final MSE | Test acc | Read |
|---|---:|---:|---|
| `iid` | `0.015129` vs `mlp64` `0.031515` | `0.925789` vs `0.922449` | Wins MSE and slight retained accuracy. |
| `permuted` | `0.044288` vs `0.061791` | `0.490538` vs `0.215955` | Large retained accuracy win. |
| `class_blocked` | `0.002179` vs `mlp64_64` `0.003018` | `0.211874` vs `0.099072` | Reconfirms the repair on a different seed block. |
| `label_drift` | `0.051275` vs `0.053899` | `0.054545` vs `0.046753` | Smaller but positive MSE and retained accuracy. |
| `mask_noise` | `0.031758` vs `0.052861` | `0.897588` vs `0.822635` | Large MSE and retained accuracy win. |

## Scientific Read

The current evidence supports a narrower claim: constructed prototype features
plus hysteretic utility gates can beat a fair sharpened deep MLP on
class-blocked final-window MSE, but the margin is small. The best current
strict learner is centroid-only, not memory-assisted: one causal prototype per
class, a hysteretic utility gate, and 64x64 output-head anti-drift. Reset and
multi-prototype memory give the cleanest feature-lifecycle story and better
retained behavior, but they give back some online tracking margin.

The mechanism is not magic and should not be oversold:

- It is classification-specific.
- It uses target labels to update prototypes, so regression needs a different
  analogue.
- The confidence sharpening improves MSE by making confident predictions more
  one-hot; it must be applied to MLP baselines too.
- The utility-gated branch is scientifically important because it addresses
  stale features under label drift. Hysteresis is necessary; the stateless gate
  over-suppresses helpful features on class-blocked streams.
- Output-head anti-drift matters. The best current results are not explained by
  hidden-feature utility alone.
- Compute efficiency is improved by the centroid-only simplification but not
  solved. The best probe is still slower than `mlp64_64`; a production
  candidate needs fused prototype reads and less redundant scoring before it can
  be called resource-efficient.
- The next production-quality version should integrate prototypes as explicit
  constructed features with resource accounting, feature utility, and a
  deletion/replacement rule.

## Next Step

The prototype feature constructor now has focused tests in
`tests/test_prototype_features.py`. Promote the full sharpened UPGD+prototype
learner from an output-folder probe into a tested learner module only after
adding:

- simplex-target gating so non-classification tasks skip the mechanism;
- prototype utility and replacement for bounded class/resource budgets;
- output-head anti-drift as an explicit, tested option;
- external image and sklearn breadth checks;
- ablation of prototype features, sharpening, mixture weight, UPGD margin,
  anti-drift, hysteresis, and reset/recycling.
