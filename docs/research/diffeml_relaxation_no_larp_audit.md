# DiffEML Relaxation No-Larp Audit

This audit covers the image DiffEML training path in `src/alberta_framework/core/diffeml_image.py`.
The goal is narrow: demonstrate that EML-derived Boolean circuits can be learned by gradient-based
relaxation and then hardened to an actual discrete circuit. Results should not rely on a continuous
classifier doing the real work after hardening.

## What Is Non-Pure During Training

The current differentiable path intentionally uses continuous optimization machinery:

- Gate selection is relaxed with softmax mixtures for `gate_mode="eml_template"` and
  `gate_mode="truth_table"`.
- Raw EML-threshold nodes use sigmoid relaxations over threshold margin and direction for
  `gate_mode="eml_threshold"`.
- Learned count readouts (`class_vote`, `signed_class_vote`) use softmax during training and
  straight-through hard class metadata when `hard=True`.
- Adam, cross-entropy, entropy penalties, dropout, and gradient clipping are training procedures,
  not deployable circuit components.

These are acceptable only as training relaxations. The credible claim is the hardened result, not the
soft model.

## What Is Pure After Hardening

The clean deployment path is:

`gate_mode in {"eml_template", "truth_table"}` plus
`head_mode in {"group_sum", "class_vote", "signed_class_vote"}`.

In that path, selected gates compile to fixed Boolean masks derived from EML templates, and the head is
a count/readout rule:

- `group_sum`: fixed class-bank popcounts.
- `class_vote`: one learned class id per final Boolean feature.
- `signed_class_vote`: one learned class id plus one polarity bit per final Boolean feature.

For classification, the positive `group_sum_tau` scaling is argmax-invariant and is not needed by a
hard deployed classifier. The packed evaluator is the no-larp path because it executes Boolean gates
and popcount-style readouts.

## Current Non-Pure Deployment Paths

These paths should be treated as ablations or diagnostics, not evidence for a pure DiffEML circuit:

- `head_mode="linear"`: the circuit features are Boolean after hardening, but the classifier head is
  still continuous fp32 or int8 arithmetic.
- `gate_mode="eml_threshold"`: this is closer to raw EML nodes, but deployment retains learned real
  thresholds; there is no packed Boolean evaluator for it yet.
- Soft accuracy alone: this can be inflated by train-time mixtures and does not prove a hardened
  circuit works.

## Added Guardrails

`deployment_purity_summary(config)` now records whether a run uses:

- soft gate mixtures during training,
- soft threshold relaxations during training,
- soft learned readout mixtures during training,
- a continuous head at deployment,
- learned real thresholds at deployment,
- a Boolean count readout,
- a packed Boolean evaluation path,
- a hard deployment that is pure Boolean for classification.

`run_one_dataset()` now reports soft-vs-hard prediction disagreement:

- `train_soft_hard_prediction_disagreement`
- `test_soft_hard_prediction_disagreement`
- `test_hard_packed_prediction_disagreement` when packed eval is enabled

The primary no-larp metric is `packed_hard_test_accuracy` when the deployment path is pure Boolean and
packed evaluation is enabled. Otherwise the run should be described with the relevant caveat from the
purity metadata.

## Remaining Research Risk

The math story is coherent if we frame DiffEML as a relaxation-to-discrete-circuit method:

1. Train a continuous relaxation of a finite EML-derived Boolean circuit family.
2. Harden every gate selector and learned readout selector.
3. Evaluate only the hardened circuit for the main claim.
4. Use soft-hard disagreement as a failure signal.

The remaining hard problem is optimization, not expressivity. EML-derived Boolean gates can represent
Boolean circuits; Boolean circuits can approximate binarized finite-input tasks. What still has to be
shown experimentally is that the relaxation consistently finds useful circuits without a continuous
head carrying the result.
