# DiffEML Paper-Quality Checklist

This checklist separates three claims that should not be blurred:

1. Boolean-template DiffEML can harden to EML-realizable logic circuits.
2. Continuous DiffEML can train real-valued EML networks with backprop.
3. DiffEML beats DiffLogic on a logic-network benchmark.

Only the first claim has strong local evidence today. The second is the next
research direction. The third requires a new benchmark campaign.

## Claim Boundary

### Boolean-Template DiffEML

This is the current selector/template path:

- inputs are binary features;
- nodes choose among EML-threshold Boolean templates;
- training uses differentiable selector relaxations;
- deployment can be a hardened Boolean circuit or packed Boolean evaluator.

This path is allowed to compare to DiffLogic because the hard model class is a
logic circuit. The comparison must report hard accuracy, packed accuracy when
available, gate count, and topology.

### Continuous DiffEML

This path should not be framed as DiffLogic-like. It should use real-valued EML
layers trained by backprop:

- real-valued affine left/right projections;
- stable right-input parameterization for the logarithm;
- residual or normalized EML blocks for depth;
- optional hardening only as a later compression step.

Continuous DiffEML should be evaluated as its own neural architecture. It can
compare to MLPs, kernel baselines, and Boolean-template DiffEML, but a win over
continuous baselines is not the same claim as beating DiffLogic.

### Beat-DiffLogic Claim

A paper can only claim DiffEML beats DiffLogic when all of these are true:

- the DiffLogic baseline is either locally reproduced or clearly labeled as
  paper-reported;
- DiffEML and DiffLogic use the same dataset split and comparable preprocessing;
- the headline metric is hard/discrete test accuracy;
- both models report gate counts, not just trainable parameters;
- the claim is made at matched or better gate budget;
- at least five seeds are reported with confidence intervals;
- all tuning decisions are based on validation data, not test data;
- raw JSON artifacts include commit hash, seed, environment, and command.

## Required Result Fields

Every paper table row should include:

- `schema_version`;
- `run_id`;
- dataset name, source, split, seed, train/test counts;
- feature mode and preprocessing;
- model kind: `diffeml`, `continuous_diffeml`, `difflogic`, or `logictreenet`;
- topology and head mode;
- gate count for logic-network rows;
- parameter count for trainable models;
- optimizer, epochs, batch size, and elapsed time;
- train/test soft accuracy when applicable;
- train/test hard accuracy for logic-network rows;
- packed hard accuracy when packed inference is used;
- baseline provenance: `paper_reported`, `local_reproduced`, or `pending`.

The helper module `src/alberta_framework/core/diffeml_results.py` enforces the
minimal version of these rules.

## External Baseline Matrix

The first target is the original DiffLogic frontier:

| Target | Dataset | Gate budget | Required DiffEML result |
| --- | --- | ---: | --- |
| Original DiffLogic small | MNIST | 48k to 1.28M | higher hard accuracy at same or fewer gates |
| Original DiffLogic CIFAR | CIFAR-10 | 5.12M | greater than 62.14% hard test accuracy |

The second target is the newer LogicTreeNet frontier:

| Target | Dataset | Gate budget | Required DiffEML result |
| --- | --- | ---: | --- |
| LogicTreeNet small | CIFAR-10 | 0.40M | greater than 60.38% hard test accuracy |
| LogicTreeNet medium | CIFAR-10 | 3.08M | greater than 71.01% hard test accuracy |
| LogicTreeNet large | CIFAR-10 | 16M to 61M | improve the Pareto curve or inference cost |

These values are external targets until reproduced locally.

## Ablations Required Before Submission

Boolean-template DiffEML:

- EML templates versus truth-table selector on identical topology;
- group-sum head, learned-discrete class-vote readout, signed class-vote
  readout, and learned linear head as separate rows;
- class-bank wiring versus generic random wiring under the same pure
  Boolean/count readout;
- random wiring versus structured spatial topology;
- hard-loss and straight-through ablation;
- temperature and entropy schedule ablation;
- packed hard evaluation equality;
- same gate budget across all logic-network rows.

Continuous DiffEML:

- exact EML versus stable EML;
- residual EML block versus plain stack;
- normalization on/off;
- affine right path versus softplus-positive right path;
- continuous output versus optional hardened/compressed output;
- backprop optimizer and learning-rate schedule sweep.

## Current Status

Current local artifacts show that Boolean-template DiffEML works, but do not
support a beat-DiffLogic claim. The strongest CIFAR artifact is approximately
47.00% packed hard test accuracy with detector-threshold features, random
sparse wiring, stronger hard loss, input dropout 0.02, feature dropout 0.30,
and a learned linear head. Original DiffLogic and LogicTreeNet targets are
substantially higher. The exact same-feature MLP-512 control for this feature
set reaches 49.32%. A larger random sparse `4096 x 8` circuit reached only
43.82% packed hard accuracy, so capacity scaling by itself is not a credible
closure path.

Under the stricter Boolean-compression goal, the linear-head result is not the
right headline. The current pure readout rows are 31.78% for fixed `group_sum`
and 35.50% for learned-discrete `class_vote` on the same detector4 protocol.
Int8 linear-head quantization preserves the current best linear-head result
within 0.02 percentage points, but it is a deployment ablation for that model,
not evidence that the whole classifier has been reduced to Boolean metadata.
The first `class_bank_random` smoke is negative at tiny CIFAR scale, so it is
an ablation candidate rather than a promoted topology.
The first `signed_class_vote` smokes are neutral or negative, so polarity bits
are an implemented readout ablation rather than a promoted result.

The most important next step is not more narrative polish. It is a matched
gate-budget benchmark runner plus a continuous DiffEML implementation that is
not constrained to Boolean gate templates. For the Boolean-template path, the
next credible step is class-bank topology/readout design that improves
`group_sum` or `class_vote` without adding a float head.
