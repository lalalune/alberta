# DiffEML Scaling, Compression, And Performance Assessment

This assessment separates four different goals that are easy to blur:

- make the model more accurate;
- keep the deployed object a Boolean EML-derived circuit;
- reduce deployed memory and inference cost;
- scale training without just overfitting larger random circuits.

The current conclusion is blunt: DiffEML already compresses well after
hardening, but the pure Boolean/count readout is not accurate enough yet. The
next credible improvements are topology, readout supervision, feature
construction, and pruning/growth workflows, not another raw width increase.

## Current Evidence

| Artifact | Head | Wiring | Gates | Packed hard | Deployed bytes | Notes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `cifar_detector4_random_w2048_l6_train20000_dropout030_packed.json` | `linear` | random | 12,288 | 47.00% | 82,000 | strongest accuracy, not pure Boolean readout |
| `cifar_detector4_random_w2048_l6_train20000_dropout030_groupsum_tau10_packed.json` | `group_sum` | random | 12,288 | 31.78% | 61,440 | pure fixed count readout |
| `cifar_detector4_random_w2048_l6_train20000_dropout030_classvote_tau10_packed.json` | `class_vote` | random | 12,288 | 35.50% | 62,464 | learned discrete metadata, pre-STE fix |
| `cifar_detector4_random_w4096_l8_train20000_hard08_packed.json` | `linear` | random | 32,768 | 43.82% | not recorded | larger model overfit and regressed |
| `cifar_class_vote_ste_smoke_w512_l3.json` | `class_vote` | random | 1,536 | 19.33% | 7,936 | post-STE bounded CIFAR smoke |
| `cifar_group_sum_smoke_w512_l3.json` | `group_sum` | random | 1,536 | 18.33% | 7,680 | matched pure count smoke |
| `cifar_class_bank_group_sum_smoke_w512_l3.json` | `group_sum` | class-bank | 1,536 | 15.67% | 7,680 | naive class-bank isolation underperformed |
| `cifar_signed_class_vote_smoke_w512_l3.json` | `signed_class_vote` | random | 1,536 | 19.33% | 8,000 | polarity bit tied unsigned on CIFAR smoke |

The important negative result is the 32,768-gate run: scaling random sparse
capacity alone made train hard accuracy high but test hard accuracy worse. That
rules out "just make it wider/deeper" as a paper-quality path.

## Scaled Structured Topology Update

On 2026-05-07, the image runner gained two executable compressed topologies:

- `affine_expander`: deterministic degree-2 modular wiring. The hard circuit can
  be reconstructed from small per-layer affine descriptors instead of explicit
  source-index arrays.
- `butterfly_class_bank`: deterministic butterfly mixing followed by one final
  class-local bank layer. This is the cleanest route to `group_sum`, because
  class evidence is built directly inside fixed readout banks.

These are real runner modes, not planning-only accounting rows. Packed hard
evaluation matched JAX hard evaluation in all scaled smoke/full runs.

Continuous full scale, 1,024 gates, pure Boolean/count readout:

| Task | Wiring | Head | Packed hard | Majority | Bytes | Compacted gates | Flags |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| XOR quadrants | random | `class_vote` | 80.86% | 52.34% | 5,152 | 436 / 1,024 | large soft-hard gap |
| XOR quadrants | `affine_expander` | `class_vote` | 80.86% | 52.34% | 1,088 | 425 / 1,024 | large soft-hard gap |
| XOR quadrants | `butterfly_class_bank` | `group_sum` | 82.03% | 52.34% | 1,028 | 446 / 1,024 | none |
| Diagonal halfspace | random | `class_vote` | 89.65% | 51.56% | 5,152 | 363 / 1,024 | large soft-hard gap |
| Diagonal halfspace | `affine_expander` | `class_vote` | 90.23% | 51.56% | 1,088 | 355 / 1,024 | large soft-hard gap |
| Diagonal halfspace | `butterfly_class_bank` | `group_sum` | 89.84% | 51.56% | 1,028 | 172 / 1,024 | none |
| Checkerboard-4 | random | `class_vote` | 60.74% | 52.54% | 5,152 | 404 / 1,024 | large soft-hard gap |
| Checkerboard-4 | `affine_expander` | `class_vote` | 54.30% | 52.54% | 1,088 | 398 / 1,024 | large soft-hard gap |
| Checkerboard-4 | `butterfly_class_bank` | `group_sum` | 58.01% | 52.54% | 1,028 | 465 / 1,024 | large soft-hard gap |

Digits smoke, two layers, threshold-pixel bits, pure Boolean/count readout:

| Wiring | Head | Gates | Packed hard | Same-feature MLP | Bytes | Compacted gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| random | `class_vote` | 192 | 20.31% | 24.22% | 624 | 106 / 192 |
| `affine_expander` | `class_vote` | 192 | 26.56% | 24.22% | 248 | 121 / 192 |
| `butterfly_class_bank` | `group_sum` | 200 | 14.84% | 24.22% | 202 | 129 / 200 |

Digits scaled smoke, four layers, width 512:

| Wiring | Head | Gates | Packed hard | Same-feature MLP | Bytes | 4-bit bytes | Compacted gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | `class_vote` | 2,048 | 79.00% | 92.00% | 10,496 | 9,472 | 644 / 2,048 |
| `affine_expander` | `class_vote` | 2,048 | 75.33% | 92.00% | 2,336 | 1,312 | 823 / 2,048 |
| `butterfly_class_bank` | `group_sum` | 2,048 | 72.33% | 92.00% | 2,056 | 1,032 | 173 / 2,048 |

CIFAR detector smoke, width 512, three layers:

| Wiring | Head | Gates | Packed hard | Same-feature MLP | Bytes | 4-bit bytes | Compacted gates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| random | `class_vote` | 1,536 | 17.67% | 21.33% | 7,936 | 7,168 | 728 / 1,536 |
| `affine_expander` | `class_vote` | 1,536 | 16.33% | 21.33% | 1,816 | 1,048 | 1,041 / 1,536 |
| `butterfly_class_bank` | `group_sum` | 1,536 | 16.00% | 21.33% | 1,542 | 774 | 889 / 1,536 |

Interpretation:

- `affine_expander` is the first credible scale/compression win. It matched
  random on XOR, slightly beat random on diagonal halfspace, and beat the
  low-budget digits random smoke while using far fewer deployed bytes.
- `butterfly_class_bank` became credible after changing it to one final bank
  layer. It beat the random/affine continuous XOR rows with no soft-hard flag,
  stayed competitive on diagonal halfspace, and recovered above majority on
  checkerboard. On real-image smokes it is still less accurate than random, but
  it is the cleanest and smallest pure `group_sum` circuit.
- Compaction is not cosmetic. The full continuous runs prune 55-86% of gates
  after hardening depending on task/topology, so grow/prune/compact/fine-tune is
  now the credible scale path.
- Gate masks now have explicit 4-bit packing support. The `Bytes` column above
  keeps the legacy one-byte-per-mask estimate for compatibility; `4-bit bytes`
  is the deployable mask-packed estimate.
- The remaining larp risk is the soft-hard gap, not packed deployment. The
  deployed rows are packed Boolean EML-template circuits with count readouts, but
  many winning `class_vote` runs still have large soft-hard gaps.

## Improve Accuracy

### 1. Readout Expressivity Without Float Weights

Keep these as separate rows, never mixed with the linear-head frontier:

| Readout | Deploy form | Status | Next action |
| --- | --- | --- | --- |
| `group_sum` | fixed contiguous class buckets, zero metadata | cleanest and smallest | improve topology so final features land in class banks |
| `class_vote` | class id per final feature | works, post-STE fix helps | rerun detector4 full artifact after STE update |
| `signed_class_vote` | class id plus polarity bit per final feature | implemented, first smokes neutral/negative | keep as ablation; try distillation/longer training before promotion |

High-value next readout ideas:

- **Teacher-distilled Boolean readout:** train the circuit/readout to match the
  linear-head logits or same-feature MLP logits, then deploy only class ids,
  polarity bits, and popcounts. This tests whether the readout gap is
  supervision, not representation.
- **Two-bank signed readout:** explicitly split final features into positive and
  negative class banks instead of letting `signed_class_vote` choose freely.
- **Top-k class metadata:** allow each feature to vote for two classes using two
  small class ids. This doubles readout metadata but remains discrete and may
  approximate the linear head better.
- **Density-normalized counts:** divide each class logit by a learned or fixed
  per-class activity prior measured on train data. If fixed after training, this
  is small scalar metadata; report separately from pure Boolean rows.

### 2. Topology

Promising:

- **Global mixer then class bank:** the naive `class_bank_random` isolated the
  last two layers too aggressively. A better version should use Benes/permuted
  butterfly or random global mixers through most layers, then one final
  class-bank layer.
- **Residual random with stronger regularization:** current residual-random
  trailed random by about one point but is closer to feature construction. It
  deserves a matched dropout/hard-loss sweep before rejection.
- **Hybrid spatial-to-global:** local features need early rescue paths. Use one
  local detector/OR stage, then global mixers. Pure local tree is negative.
- **Structured Benes/permuted butterfly:** deterministic coverage reduces
  source-index storage and may improve packed GPU locality.

Rejected for now:

- **Blind width/depth scaling:** `4096 x 8` regressed while overfitting.
- **Naive class-bank isolation:** first smoke underperformed generic random.
- **Current local tree hierarchy:** consistently below random sparse smokes.

### 3. Feature Construction

Current CIFAR gains depend heavily on fixed detector thresholds. Ways to make
this more EML-like without pretending the detector is learned EML:

- add a learned binary stem as a clearly separate preprocessing module;
- learn threshold positions per detector map with STE and then freeze them;
- select bits by train split mutual information or density balance, not only
  variance;
- add multi-scale detector features, then let pruning remove weak bits;
- test lower-density input bits, because dense detector bits make count readouts
  less discriminative.

Feature changes need same-feature MLP controls. If the feature pipeline alone
raises the MLP equally, it is not DiffEML progress.

### 4. Training

Useful:

- hard-readout STE, already added for `class_vote` and `signed_class_vote`;
- feature dropout around 0.30 for the linear-head detector4 row;
- input bit dropout around 0.02;
- entropy pressure for gate hardening;
- readout entropy/balance as small class-vote controls.

Still missing:

- validation-set early stopping instead of fixed epoch/test-set pilots;
- distillation loss from linear-head or MLP teacher into pure Boolean readout;
- label smoothing for count heads;
- longer post-STE `class_vote` detector4 rerun;
- annealed readout temperature separate from gate temperature;
- per-layer hard-loss schedule, because early hard pressure may freeze weak
  gates before the readout has learned.

## Compress Deployed Circuits

The current compiled format stores:

- two source indices per gate;
- one 8-bit gate mask per gate;
- optional readout metadata;
- no train-time selector logits.

Compression opportunities:

1. **Pack gate masks to 4 bits.** Implemented in the image runner storage
   helpers. There are only 16 Boolean gates, so this halves gate-mask storage.
   For 12,288 gates, masks drop from 12,288 bytes to 6,144.

2. **Use structured topology to remove wiring indices.** Random wiring needs
   source indices. Butterfly/Benes schedules can store a tiny schedule plus
   permutations. This is the biggest memory win. A 12,288-gate random circuit
   spends 49,152 bytes on wiring; a structured schedule can reduce that by an
   order of magnitude.

3. **Delta-code or width-code source indices.** If random wiring remains, sort
   or group sources by layer and store 16-bit deltas where possible. Current
   detector4 source indices need 2 bytes; this is already reasonable, but
   locality-aware wiring could make deltas smaller.

4. **Canonicalize duplicate gates.** If two nodes in the same layer have the
   same `(left, right, mask)`, store once and reference it, or prune the
   duplicate if downstream does not need both copies.

5. **Prune constant gates.** Gates that are constant on the train set can be
   replaced by constant `0`/`1` sources. This also allows recursive pruning of
   now-unused parents.

6. **Prune unused readout features.** For `class_vote` and `signed_class_vote`,
   remove final features whose class metadata is rarely active or contributes
   only to consistently wrong votes.

7. **Gate-set compression.** If a trained artifact uses a small subset of the
   16 masks, store a compact remap table and smaller per-node codes.

8. **Readout metadata packing.** `class_vote` needs `ceil(log2(C))` bits per
   feature; `signed_class_vote` needs one extra polarity bit. For CIFAR-10 and
   width 2048 this is about 1,024 to 1,280 bytes, which is not the bottleneck.

The biggest deployment-size lever is structured wiring, not head quantization.

## Improve Inference Performance

### CPU / SIMD

Use a structure-of-arrays layout:

```text
layer_offsets: int32[L + 1]
left_idx: uint16[nodes]
right_idx: uint16[nodes]
gate_mask: uint4[nodes]
class_id: uint4[final_width]
polarity: uint1[final_width]       # signed readout only
activations: uint64[(input + width) * word_batches]
```

Then evaluate one 64-example word at a time:

1. pack input bits by column;
2. for each layer, load left/right words, apply truth-table mask by bit ops;
3. write final packed features;
4. popcount grouped class features without unpacking examples.

The current Python/NumPy packed evaluator proves semantics, not speed. A real
kernel should be C++/CUDA/Triton or JAX custom call.

### GPU

GPU inference should use one block per layer tile and word batch:

- coalesce reads by storing source indices in layer-major order;
- put class-group readout indices in constant or shared memory;
- fuse final layer and popcount readout when possible;
- avoid dense unpacking entirely;
- benchmark `uint32` versus `uint64`; some GPUs handle 32-bit bitwise ops more
  efficiently even though they pack fewer examples per word.

Potential GPU bottlenecks:

- random source reads are memory-latency dominated;
- structured Benes/butterfly improves coalescing;
- per-example Python loops in the current evaluator must disappear.

### Train-Time Performance

The expensive path is evaluating all candidate gates/templates at every node.
For width `W`, layers `L`, library size `G`, and batch `B`, relaxed training is
roughly `O(B * W * L * G)` before the head. Ways to reduce this:

- use a smaller gate library after early training by pruning low-probability
  gates;
- use top-k selector relaxation instead of full softmax over all gates;
- train with truth-table multilinear relaxation and only render selected gates
  as EML templates for verification/export;
- cache template bank outputs where left/right source pairs repeat;
- split training into grow/prune phases with smaller active gate sets.

## Grow The Network

Do not grow by raw width/depth alone. Use staged growth:

1. train a compact random/Benes mixer;
2. measure per-feature utility and residual class errors;
3. add gates only for classes or regions with high residual error;
4. initialize new gates by mutating high-utility parent sources;
5. freeze or slow old gates for a short warmup;
6. retrain with hard loss and feature dropout;
7. prune back to the target budget.

Growth candidates:

- **Class-targeted growth:** add gates to classes with low margin under
  `group_sum` or `class_vote`.
- **Parent mutation:** clone a useful final feature, replace one parent with a
  nearby detector bit or another high-utility feature.
- **Layer insertion:** add one global mixer layer before the readout and prune
  the least useful older gates.
- **Budget ladder:** evaluate 8k, 12k, 16k, 24k, 48k gates with the same
  protocol. Promote only if accuracy improves gate-normalized.

## Prune The Network

Pruning should be part of training, not just postprocessing. Candidate signals:

- activation is nearly always 0 or nearly always 1;
- selected gate entropy stayed high and hard choice is unstable;
- duplicate `(left, right, mask)` within a layer;
- feature has low mutual information with labels;
- feature's readout vote is consistently anti-correlated with correctness;
- removing the feature has low validation loss impact;
- source node has no path to any retained final feature.

Pruning workflow:

1. harden the circuit;
2. compute activation density and validation ablation scores;
3. remove constants, duplicates, and unreachable nodes first;
4. remove low-utility nodes in small batches;
5. compact source indices and layer widths;
6. fine-tune selectors/readout at the smaller budget;
7. verify packed hard equality after every compaction.

The first pruning target should be the 12,288-gate detector4 linear-head model.
If pruning preserves 47% while reducing gates, it gives a strong efficiency
claim even before pure Boolean readout catches up.

## Scale Plan

### Short Term

- Rerun detector4 `class_vote` and `signed_class_vote` after STE/readout updates.
- Add a pruning report for existing artifacts: constants, duplicates, density,
  readout utility, and estimated compacted bytes.
- Add a file-format/export path that writes the implemented 4-bit gate-mask
  packing directly instead of reporting it only in storage summaries.
- Add a CPU microbenchmark for packed inference that reports gates/s and
  examples/s.
- Add validation split support so future sweeps stop tuning on test metrics.

### Medium Term

- Implement global-mixer-to-class-bank topology.
- Add teacher-distilled Boolean readout loss.
- Add top-k selector relaxation and gate-library pruning.
- Implement source-index compaction and structured topology export.
- Run budget ladder at 8k, 12k, 16k, and 24k gates.

### Long Term

- Triton/CUDA packed inference kernel.
- Dynamic grow/prune training loop.
- Learned binary feature stem with same-feature MLP controls.
- Full five-seed CIFAR 20k/5k and 50k/10k matrices.

## Promotion Criteria

A proposed improvement is real only if it satisfies one of these:

- higher packed hard accuracy at the same gate budget and same readout class;
- same accuracy with fewer gates or fewer deployed bytes;
- same accuracy with faster packed inference;
- better pure Boolean/count readout accuracy without adding float readout
  weights;
- a negative result that removes a tempting but weak path from the roadmap.

Everything else remains an ablation, not a claim.
