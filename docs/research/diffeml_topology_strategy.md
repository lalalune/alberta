# DiffEML Topology Strategy

## Current State

The current baseline detector-threshold CIFAR run uses random sparse wiring,
width 2048, 6 layers, four thresholds, stronger hard loss, and packed hard
evaluation:

`outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_hard08_packed.json`

That is a 12,288 binary-gate budget before the linear head. The packed hard
CIFAR accuracy is 44.22%, which is credible for the current prototype but still
below the smallest published DiffLogic CIFAR row at 51.27%.

Regularization improves the same topology:

`outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout_packed.json`

Training-only input bit dropout 0.02 and final feature dropout 0.10 raise the
packed hard result to 46.40% and reduce the gap to the smallest published
DiffLogic CIFAR row to 4.87 points. The int8-quantized linear head keeps
46.36%, which is a useful deployment ablation for the current best linear-head
model, but it is not the corrected pure Boolean compression target.

The stronger regularized artifact is:

`outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_packed.json`

The same input bit dropout with final feature dropout 0.30 reaches 47.00%
packed hard accuracy and 46.98% after int8 linear-head quantization, reducing the
DiffLogic-small gap to 4.27 points and the same-feature MLP-512 gap to 2.32
points. The sweep peaked at 0.30 and fell back at 0.40, so regularization has
real signal but is not an unbounded knob.

The corrected Boolean-compression readout rows are much lower:

`outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_groupsum_tau10_packed.json`

The fixed `group_sum` readout reaches 31.78% packed hard accuracy with 61,440
compiled bytes and no readout weights. This is the cleanest deployment story,
but it shows that the generic final features are not already organized into
class banks.

`outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_classvote_tau10_packed.json`

The learned-discrete `class_vote` readout reaches 35.50% packed hard accuracy
with 62,464 compiled bytes. It stores one class id per final feature and uses a
packed popcount vote at inference. This is a real Boolean/count readout, but
the 11.50-point gap to the linear head means readout/topology design is now the
main bottleneck. This full artifact predates the straight-through hard-readout
gradient fix; post-fix smokes improved the direct hard-vote path, but the
detector4 full run still needs to be rerun before updating the headline number.

A direct random-sparse capacity increase did not help:

`outputs/diffeml_image_demo/cifar_detector4_random_w4096_l8_train20000_hard08_packed.json`

This 32,768-gate run reached 43.82% packed hard accuracy while increasing train
hard accuracy to 70.70%, so it mainly amplified overfitting. The next topology
work should change inductive bias or regularization, not just width and depth.

The first explicit topology change, `residual_random`, forces deeper nodes to
combine one previous EML feature with one raw detector input. With the same
dropout settings it reached 45.96%, slightly below regularized random wiring.
It is more EML-feature-construction-like, but not yet a promoted topology.

The local-tree hierarchy path is useful evidence, but it is currently a negative
result. Existing smoke artifacts put it well below matched random sparse wiring.
That makes topology and gate efficiency the next bottleneck rather than another
large CIFAR training run.

## Why `local_tree_hierarchy` Underperforms

The current local tree is closer to image structure, but it spends gates too
early and too locally. The first stage repeatedly samples small neighborhoods
from thresholded input bits. Those nodes can detect local conjunctions or OR-like
patterns, but they do not receive enough channel-mixed or long-range context to
separate CIFAR classes. Once fixed OR pooling is inserted, much of the sign and
threshold information is collapsed before the classifier head can decide whether
it was useful.

The hierarchy also has a width mismatch. It keeps a fixed layer width while the
spatial grid coarsens, so some cells receive many repeated nodes and other
useful cross-cell interactions still require later random coincidences. This is
not the same as a convolutional stage with controlled channel count per spatial
cell.

Finally, the local tree has weak rescue paths. Random sparse wiring is not image
aware, but it mixes detector features broadly and gives the head diverse global
features. The local tree constrains mixing before it has proven that the local
features are class-relevant, so bad early choices propagate.

## Accounting Layer

`src/alberta_framework/core/diffeml_topologies.py` separates four quantities:

- binary gate nodes, which are the fair DiffLogic gate-budget comparison;
- trainable node parameters, which differ between template selectors and
  threshold/direction EML nodes;
- fixed gate nodes, especially OR-pool layers, which cost packed inference gates
  but do not add selector parameters;
- head parameters and packed uint64 inference work.

This matters because an apparent small model can hide extra trainable selector
parameters, and an apparent fixed operation can still consume packed gate work.
For acceptance, the paper should report both node budget and trainable parameter
budget.

## Topology Families To Test Next

1. **Matched random sparse baseline.** Keep the current 2048 x 6 random sparse
   plan as the control. Every new topology must first match this exact 12,288
   binary-gate budget or report a gate-normalized comparison.

2. **Permuted butterfly / Benes schedules.** These are cheap global mixers. They
   preserve the same node count as random sparse wiring while giving deterministic
   coverage and easier packed compilation.

3. **Continuous EML blocks.** Train blocks with continuous EML mixing and harden
   them to the same two-input gate count. This is not just DiffLogic-style random
   wiring; it tests whether training with block structure can discover useful
   intermediate features before committing to hard gates.

4. **Conv/Tree spatial stages.** Use variable-width stages of
   `grid_height * grid_width * channels`, with local tree layers inside a stage
   and explicit pooling between stages. This is the structured replacement for
   the current fixed-width local tree. It should spend gates per spatial cell,
   not per global layer.

5. **Hybrid spatial-to-global mixers.** Use one or two spatial stages, then spend
   the remaining gates on Benes or random global mixing. This keeps local
   inductive bias while avoiding the current failure mode where global evidence
   reaches the head too late.

6. **Class-bank final layers.** Allocate the final one or two layers into
   per-class banks and score with `group_sum` or `class_vote` popcounts. This is
   the most direct way to make the deployed circuit purely Boolean while still
   giving training a reason to build class-specific evidence before the readout.
   The control is the current generic final layer plus `class_vote`.

Two structured versions are now integrated in the image runner:

- `affine_expander` is a deterministic sparse global mixer. It keeps the same
  gate count as random wiring but replaces explicit source-index storage with a
  small affine descriptor per layer. In scaled full continuous runs, it matched
  random on XOR and slightly beat random on diagonal halfspace with about 4.7x
  fewer deployed bytes.
- `butterfly_class_bank` combines deterministic butterfly mixer layers with a
  single class-local bank tail. The one-bank version is now the cleanest pure
  `group_sum` topology: in full continuous runs it beat random on XOR with no
  soft-hard flag, stayed close on diagonal halfspace, and recovered
  checkerboard above majority. It still trails random on real-image smokes, so
  promote it as a compression/topology candidate rather than the current
  accuracy winner.

The first implemented `class_bank_random` smoke is not promoted. On the bounded
CIFAR 1200/300, width-512, layer-3 smoke it reached 15.67% with `group_sum` and
16.67% with `class_vote`, below the generic random `group_sum` and `class_vote`
smokes at 18.33% and 19.33%. That suggests naive bank isolation removes useful
global mixing; a better variant should use global/Benes mixers before class
banks or class-conditioned skip inputs, not fully isolate the last two layers.

`signed_class_vote` adds one polarity bit per final feature and remains a pure
Boolean/count readout. Its first smokes are not promoted: it tied unsigned
`class_vote` at 19.33% on the bounded CIFAR smoke and was worse on digits. It
should stay as an expressivity ablation until a detector-scale rerun proves a
benefit.

## Acceptance Criteria Versus DiffLogic

The primary comparison is binary gate nodes, excluding the classifier head unless
the DiffLogic baseline includes its head in the same way. The initial target
budget is 12,288 gates because it matches the current width-2048, layer-6
artifact.

A topology is paper-worthy only if it satisfies all of the following:

- exact accounting artifact includes total nodes, fixed nodes, trainable nodes,
  selector parameters, head parameters, and packed uint64 work;
- hard packed inference matches the JAX hard circuit on smoke tests after runner
  integration;
- pure Boolean/count readout rows report deployed readout metadata bytes
  separately from train-time readout parameters;
- CIFAR hard accuracy beats the matched random sparse baseline at the same gate
  budget across multiple seeds;
- if it exceeds 12,288 gates, it also wins after normalizing to DiffLogic-style
  budget tiers such as 8k, 12k, 16k, and 24k gates;
- local-tree descendants must beat the current random sparse baseline before
  being promoted as image-structured evidence.

The next production step is not another blind CIFAR sweep at the same topology.
It is to use the accounting specs to choose two or three budget-matched topology
candidates, integrate only those into the image runner, and then run small
deterministic smoke checks before launching full experiments. The detector4 run
shows that hyperparameter pressure can buy about two points, and input/feature
dropout buys almost three more over the unregularized packed baseline. The
larger random-sparse run shows that capacity alone can lose those points again.
The remaining DiffLogic-small gap is 4.27 points for the linear-head artifact,
but 15.77 points for the cleaner `class_vote` artifact. If the goal is a
Boolean-compressed EML circuit, topology and readout structure must carry the
next gain.
