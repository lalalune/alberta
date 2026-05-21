# DiffEML Acceptance Criteria

This note defines the bar for claiming that DiffEML works as a learnable EML
circuit rather than as a neural model with an EML-themed component.

## Required Claim Boundary

A primary DiffEML result must report the deployable hardened circuit, not only
the relaxed training graph. The deployed object must be expressible as:

- binary input features;
- fixed wiring;
- one selected EML-threshold template or EML-derived Boolean mask per gate;
- Boolean activations;
- a pure discrete readout such as fixed class buckets, class ids, or class ids
  plus polarity bits.

Linear heads, int8 heads, learned real-valued thresholds that remain real at
deployment, and fixed detector features can be useful controls, but they are
not the pure claim.

## Minimum Metrics

Every promoted run should include:

- train/test soft accuracy;
- train/test hard accuracy;
- packed hard accuracy;
- soft-hard accuracy gap;
- pure-readout accuracy, or an explicit label that the run uses a float head;
- hardened gate entropy or selected-mask histogram;
- deployed byte count split into wiring, gate masks, and readout metadata;
- same-feature non-EML baseline when feature engineering changes.

## Pass Conditions

A result is credible when:

- packed hard accuracy exactly matches JAX hard accuracy within numerical
  tie-breaking expectations;
- hard accuracy is close enough to soft accuracy that the relaxation is not the
  only model that works;
- pure readout accuracy is the promoted number, or the result is labeled as a
  linear-head ablation;
- the feature pipeline is fixed and disclosed, or the feature learner is itself
  compiled into the deployed circuit;
- topology storage is counted honestly.

## Red Flags

Treat a result as larp until fixed if:

- the best result uses `head_mode="linear"` and is presented as pure DiffEML;
- the relaxed soft model works but the hard or packed model fails;
- detector features change without a same-feature MLP or logic baseline;
- readout metadata is omitted from compression accounting;
- width/depth grows without pruning, topology accounting, or validation
  controls;
- the hard circuit cannot be rendered as EML-threshold templates.

## Near-Term Target

The next paper-quality target is not "beat MLP." It is:

1. train a relaxed DiffEML circuit;
2. harden it to EML-template gates;
3. use only a pure discrete readout;
4. evaluate it with packed Boolean inference;
5. beat a matched DiffLogic-style topology or a simple Boolean-gate baseline on
   at least one nontrivial task while reporting bytes and soft-hard gap.

