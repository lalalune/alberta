# DiffEML LARP Audit

This note separates what is actually EML from what is Boolean-equivalent
compilation, image-demo scaffolding, and ordinary neural baselines. The goal is
to keep the DiffEML package direction honest enough to be useful.

## Bottom Line

DiffEML is currently a promising EML-constrained logic-circuit research
prototype, not a demonstrated replacement for DiffLogic or a competitive image
classifier.

The strongest current image evidence is a single-seed CIFAR-10 split using
fixed detector-threshold features, random sparse wiring, executable
depth-2 EML-template gates, and a learned linear head:

| Artifact | Train/test | Features | Circuit | Test hard acc. | Baseline |
| --- | ---: | --- | --- | ---: | --- |
| `outputs/diffeml_image_demo/cifar_detector3_random_w2048_l6_train20000.json` | 20k/5k | detector thresholds, 16,384 selected bits | random, 6 x 2048 | 42.24% | none in artifact |
| `outputs/diffeml_image_demo/cifar_detector3_random_w2048_l6_train20000_vs_mlp512.json` | 20k/5k | same detector bits | same circuit | 42.24% | MLP-512: 46.70% |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_hard08_packed.json` | 20k/5k | detector thresholds, 24,576 selected bits | random, 6 x 2048 | 44.22% packed hard | MLP-512: 49.32% in paired baseline artifact |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout_packed.json` | 20k/5k | same detector4 bits, training-only bit/feature dropout | random, 6 x 2048 | 46.40% packed hard | MLP-512: 49.32% |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_packed.json` | 20k/5k | same detector4 bits, input dropout 0.02, feature dropout 0.30 | random, 6 x 2048 | 47.00% packed hard | MLP-512: 49.32% |

That is a real improvement over the original toy CIFAR runs near 21%, but it
does not beat a same-feature MLP or the smallest published DiffLogic CIFAR row
at 51.27%. On the exact detector4 feature set, MLP-512 reaches 49.32%. The
credible claim is "a regularized hard EML-template Boolean circuit can reach
nontrivial CIFAR-10 accuracy from fixed binary detector features, compile to
packed Boolean evaluation, and expose how much the readout matters." The
non-credible claim is "DiffEML is already competitive with neural baselines or
DiffLogic."

The user's corrected compression target is also right: the deployable object
should be a Boolean circuit plus compact Boolean/count readout metadata, not a
1-bit quantized neural head. The current code now has two readouts that satisfy
that boundary:

| Artifact | Head mode | Packed hard acc. | Compiled bytes | Interpretation |
| --- | --- | ---: | ---: | --- |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_groupsum_tau10_packed.json` | `group_sum` | 31.78% | 61,440 | fixed class buckets, no learned readout |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_classvote_tau10_packed.json` | `class_vote` | 35.50% | 62,464 | learned class id per final feature, packed popcount readout |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_packed.json` | `linear` | 47.00% | 82,000 with int8 head | strongest accuracy, but not a pure Boolean readout |

This is a useful negative result. The pure Boolean readouts work and compress
cleanly, but the linear head is still doing substantial class-evidence mixing.
The next honest research path is readout/topology design, not sign-quantizing
the linear head.

The `class_vote` full CIFAR artifact above predates the straight-through
hard-readout gradient fix. The post-fix smoke artifacts
`outputs/diffeml_image_demo/class_vote_ste_smoke_digits_tau1.json`,
`outputs/diffeml_image_demo/cifar_class_vote_ste_smoke_w512_l3.json`, and
`outputs/diffeml_image_demo/cifar_group_sum_smoke_w512_l3.json` show that the
path now trains the deployed hard vote directly: digits moved to 37.89% packed
hard, and a bounded CIFAR smoke reached 19.33% for `class_vote` versus 18.33%
for `group_sum`. That is not paper evidence, but it fixes the earlier soft-only
readout failure mode.

Two follow-up smokes are negative or neutral:
`outputs/diffeml_image_demo/cifar_class_bank_group_sum_smoke_w512_l3.json`
and
`outputs/diffeml_image_demo/cifar_class_bank_class_vote_smoke_w512_l3.json`
show that the first `class_bank_random` wiring underperformed generic random
wiring at this tiny budget. `readout_entropy_weight=0.01` and
`readout_balance_weight=5.0` nudged the digits class-vote smoke from 37.89% to
38.28% packed hard, but did not move the bounded CIFAR smoke. These are useful
controls, not promoted results.

`signed_class_vote` is also implemented as a pure Boolean readout ablation. It
stores one class id plus one polarity bit per final feature and evaluates
`popcount(positive) - popcount(negative)`. The first smokes are neutral or
negative: `outputs/diffeml_image_demo/signed_class_vote_smoke_digits_tau1.json`
fell to 31.25% packed hard on digits, while
`outputs/diffeml_image_demo/cifar_signed_class_vote_smoke_w512_l3.json` tied
unsigned `class_vote` at 19.33% packed hard on the bounded CIFAR smoke.

## Genuinely EML

These parts evaluate or learn expressions whose primitive is the EML operator
`exp(x) - log(y)` or a stable relaxation of it:

- `src/alberta_framework/core/diffeml.py`
  - `eml_operator()` implements the exact real operator.
  - `stable_eml_operator()` keeps the right input positive with `softplus` and
    clips for trainable real-valued circuits.
  - `EMLTemplateExpr`, `EMLTemplateBank`, `build_eml_template_bank()`,
    `evaluate_eml_template()`, and `evaluate_eml_template_bank()` represent and
    execute nested EML-threshold programs.
  - `eml_threshold_gate_library(depth=2, eps=0.05)` enumerates depth-2
    EML-threshold expressions that span all 16 two-input Boolean gates.
- `src/alberta_framework/core/diffeml_image.py`
  - `gate_mode="eml_template"` evaluates the executable nested EML template
    bank at each selector node during relaxed and hard circuit evaluation.
  - `gate_mode="eml_threshold"` is a compressed ablation: one EML operation,
    one learned threshold, and one learned direction per node. It is more
    directly EML at each node, but weaker and less mature than the template
    selector path.

The most defensible EML identity is local: each selected hard gate can be
rendered as an EML-threshold template. A hard trained circuit is therefore
compilable to a Boolean circuit whose gates each have an EML-threshold
realization.

## Boolean-Equivalent Compilation

The package also has a Boolean-equivalent path. This is valuable, but it should
not be oversold as "running EML" at inference time:

- The 16 depth-2 EML templates cover the 16 two-input Boolean functions.
- Once a selector hardens to a gate mask, its input-output behavior is exactly a
  Boolean truth table.
- `packed_hard_logits()`, `packed_hard_features()`, `pack_feature_columns()`,
  `unpack_feature_columns()`, and `eval_packed_binary_gates()` in
  `src/alberta_framework/core/diffeml_image.py` evaluate the hardened circuit
  with packed `uint64` Boolean operations.
- `packed_group_sum_logits()`, `packed_class_vote_logits()`, and
  `packed_signed_class_vote_logits()` keep readout inference in the same packed
  Boolean/popcount representation. `group_sum` stores no readout weights;
  `class_vote` stores a compact class id per final feature; `signed_class_vote`
  adds one polarity bit.
- `tests/test_diffeml_image_demo.py::test_packed_binary_gates_match_truth_table_masks`
  checks truth-table mask ordering.
- `tests/test_diffeml_image_demo.py::test_packed_group_sum_logits_match_jax_without_float_head`
  and
  `tests/test_diffeml_image_demo.py::test_packed_class_vote_logits_match_jax_with_discrete_readout`
  verify the two pure Boolean readout paths against JAX hard logits.
- `outputs/diffeml_image_demo/cifar_detector3_random_w1024_l5_train1200_packed.json`
  reports `test_hard_accuracy = 37.6667%` and
  `packed_hard_test_accuracy = 37.6667%`, so packed hard eval matched JAX hard
  eval on that smoke run.
- `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_hard08_packed.json`
  reports `test_hard_accuracy = 44.22%` and
  `packed_hard_test_accuracy = 44.22%`, so the detector4 CIFAR path compiles
  exactly under packed hard evaluation.
- `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_packed.json`
  reports `test_hard_accuracy = 47.00%`,
  `packed_hard_test_accuracy = 47.00%`, and
  `packed_int8_head_test_accuracy = 46.98%`, so the regularized best run also
  survives head quantization.

This is the DiffLogic-style deployment story: train with relaxations, harden to
discrete gates, and evaluate cheaply. It is Boolean-equivalent compilation of
EML-derived gates, not proof that the deployed packed evaluator performs
floating-point EML operations.

## Non-EML Scaffolding

The image results depend heavily on non-EML components:

- Fixed feature engineering in `diffeml_image.py`:
  - `variance_pixels` selects high-variance pixel bits.
  - `threshold_pixels` creates fixed per-pixel threshold bits.
  - `detector_thresholds` builds raw, edge, Laplace, and color-difference maps,
    then thresholds them.
- Fixed sparse wiring:
  - random, butterfly, Benes, permuted variants, local hierarchy, and local tree
    hierarchy are all topology choices, not learned EML mechanisms.
- Classification readout:
  - the best current CIFAR artifact uses `head_mode="linear"`, so the class
    decision is not itself an EML circuit.
  - `head_mode="group_sum"`, `head_mode="class_vote"`, and
    `head_mode="signed_class_vote"` remove the float linear readout at
    deployment, but their current CIFAR accuracy is much lower than the
    linear-head result.
- Adam training and temperature schedules:
  - current image experiments use ordinary minibatch optimization, not continual
    online Alberta-style temporal-uniform learning.
- Same-feature MLP baselines:
  - the MLP in
    `outputs/diffeml_image_demo/cifar_detector3_random_w2048_l6_train20000_vs_mlp512.json`
    is a non-EML baseline trained on the exact same binary features.

These pieces are acceptable package scaffolding, but claims must attribute
performance to "DiffEML circuit plus fixed binary feature pipeline" rather than
to EML alone.

## Current Evidence

### Gate-Level Evidence

- `tests/test_diffeml.py::test_eml_threshold_gate_library_depth_two_is_universal`
  verifies that depth-2 EML-threshold templates span all 16 binary gates.
- `tests/test_diffeml.py::test_executable_eml_template_bank_matches_truth_table_library`
  checks hard executable templates against the truth-table library.
- `tests/test_diffeml.py::test_diffeml_gate_selector_learns_all_binary_gates`
  trains one selector seed per gate and recovers every Boolean mask.
- `outputs/diffeml_gate_selection/` and `outputs/diffeml_ste/` contain earlier
  gate-selection and straight-through explorations. They are useful for
  mechanism history, not image-level package claims.

### Image Evidence

| Artifact | Dataset | Notes | Test hard acc. |
| --- | --- | --- | ---: |
| `outputs/diffeml_image_demo/diffeml_current_default_digits_mnist_cifar.json` | digits / MNIST / CIFAR | original default-scale image demo; CIFAR remains toy | CIFAR 21.00% |
| `outputs/diffeml_image_demo/cifar_threshold3_random_w2048_l6_train20000.json` | CIFAR | threshold pixels, random sparse wiring | 38.76% |
| `outputs/diffeml_image_demo/cifar_detector3_random_w2048_l6_train20000.json` | CIFAR | detector thresholds, random sparse wiring | 42.24% |
| `outputs/diffeml_image_demo/cifar_detector3_random_w2048_l6_train20000_vs_mlp512.json` | CIFAR | same as above plus MLP-512 baseline | DiffEML 42.24%, MLP 46.70% |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_hard08_packed.json` | CIFAR | detector thresholds, 4 thresholds, stronger hard loss, packed eval | JAX hard 44.22%, packed 44.22% |
| `outputs/diffeml_image_demo/cifar_detector4_same_features_mlp512_seed0.json` | CIFAR | MLP-512 on exact detector4 binary features | MLP 49.32% |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout_packed.json` | CIFAR | input bit dropout 0.02, final feature dropout 0.10, packed eval | JAX hard 46.40%, packed 46.40%, int8 head 46.36% |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout020_packed.json` | CIFAR | input bit dropout 0.02, final feature dropout 0.20, packed eval | JAX hard 46.66%, packed 46.66%, int8 head 46.76% |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_packed.json` | CIFAR | input bit dropout 0.02, final feature dropout 0.30, packed eval | JAX hard 47.00%, packed 47.00%, int8 head 46.98% |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_groupsum_tau10_packed.json` | CIFAR | same circuit scale with fixed grouped Boolean count readout | JAX hard 31.78%, packed 31.78%, compiled 61,440 bytes |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout030_classvote_tau10_packed.json` | CIFAR | same circuit scale with learned discrete class-vote readout | JAX hard 35.50%, packed 35.50%, compiled 62,464 bytes |
| `outputs/diffeml_image_demo/cifar_detector4_random_w2048_l6_train20000_dropout040_packed.json` | CIFAR | input bit dropout 0.02, final feature dropout 0.40, packed eval | JAX hard 46.78%, packed 46.78%, int8 head 46.56% |
| `outputs/diffeml_image_demo/cifar_detector4_residual_random_w2048_l6_train20000_dropout_packed.json` | CIFAR | residual-random topology plus same dropout | JAX hard 45.96%, packed 45.96%, int8 head 45.94% |
| `outputs/diffeml_image_demo/cifar_detector4_random_w4096_l8_train20000_hard08_packed.json` | CIFAR | larger random sparse circuit, 32,768 gates | JAX hard 43.82%, packed 43.82% |
| `outputs/diffeml_image_demo/cifar_detector3_random_w1024_l5_train1200_packed.json` | CIFAR smoke | packed hard eval check | JAX hard 37.6667%, packed 37.6667% |
| `outputs/diffeml_image_demo/cifar_threshold3_local_tree_2222_w1024_train1200.json` | CIFAR smoke | exact local tree hierarchy | 17.67% |
| `outputs/diffeml_image_demo/cifar_threshold3_local_tree_2222_left05_w1024_train1200.json` | CIFAR smoke | local tree with left residual bias | 21.67% |
| `outputs/diffeml_image_demo/cifar_threshold3_local_tree_2222_left05_maxpool_w1024_train1200.json` | CIFAR smoke | local tree with OR pooling / residual bias | 23.00% |
| `outputs/diffeml_image_demo/cifar_threshold3_random_w1024_l5_train1200.json` | CIFAR smoke | random sparse wiring at similar smoke scale | 34.00% |

The local tree hierarchy is currently a negative result. It is closer to an
image-structured story, but it underperforms random sparse wiring in the
available smoke artifacts.

## Claims We Can Safely Make

- EML-threshold templates of depth 2 can realize all 16 two-input Boolean gates
  under the current `eps=0.05` construction.
- The executable template bank and truth-table library agree in tests.
- A differentiable selector can learn and harden to each two-input Boolean gate
  in the unit tests.
- The image demo can train a selector-based EML-template circuit on real
  CIFAR-10 data when the local CIFAR archive is available.
- The best current single-seed CIFAR result is 47.00% packed hard test accuracy
  on a 20k/5k split with detector-threshold features, random wiring, 6 layers,
  width 2048, four thresholds, stronger hard loss, lower final temperature,
  input bit dropout 0.02, and final feature dropout 0.30. The feature-dropout
  sweep improved through 0.30 and fell back at 0.40.
- The same run's int8-quantized linear head reaches 46.98%, and its compiled
  int8 artifact is estimated at 82,000 bytes versus 868,392 bytes for the soft
  train-time selector/head representation.
- The pure Boolean readout variants compile exactly: `group_sum` reaches
  31.78% with 61,440 compiled bytes and `class_vote` reaches 35.50% with
  62,464 compiled bytes on the same detector4 CIFAR protocol.
- On the same 20k/5k split and exact detector4 binary features, an MLP with one
  hidden layer of width 512 reaches 49.32% test accuracy.
- Packed hard Boolean evaluation can match JAX hard evaluation on both the smoke
  artifact and the current strongest CIFAR artifact.

## Claims We Cannot Make Yet

- We cannot claim DiffEML beats same-feature neural baselines.
- We cannot claim DiffEML beats DiffLogic; the best current CIFAR packed hard
  accuracy is 47.00%, below the smallest published DiffLogic CIFAR row at
  51.27%.
- We cannot claim the current image result is statistically robust; the key
  CIFAR result is single-seed.
- We cannot claim local EML hierarchy is the right inductive bias; current
  local-tree artifacts are worse than random sparse wiring.
- We cannot claim the linear classification head has been eliminated from the
  strongest result. The pure Boolean readouts exist, but currently trail by
  11.50 to 15.22 percentage points on the same CIFAR protocol.
- We cannot claim packed inference has been benchmarked for speed, memory, or
  accuracy across datasets and seeds.
- We cannot claim CIFAR-10 competitiveness against standard CNNs, ViTs, or even
  tuned MLPs on pixels.
- We cannot claim continual-learning relevance yet; the image demo is
  minibatch-supervised and does not use the framework's scan-based continual
  learner interfaces.
- We cannot claim a standalone package-quality API, CLI, benchmark suite, or
  reproducible release artifact exists yet.

## Open Concerns

1. **Attribution:** detector thresholds and random wiring may explain more of
   the CIFAR gain than the EML templates. The package needs ablations against
   truth-table DiffLogic gates, random Boolean gates, and linear-only heads.
2. **Baseline strength:** the same-feature MLP is already better by 2.32 points
   on the strongest detector4 feature set. A package launch should position
   DiffEML as interpretable/compilable logic research, not accuracy leadership.
3. **Seed variance:** all headline image numbers need multiple seeds with fixed
   train/test protocol and confidence intervals.
4. **Hard/soft gap:** hard accuracy can exceed or trail soft accuracy depending
   on setup. Release results should always report both.
5. **Compilation semantics:** packed eval validates the hardened Boolean
   behavior, but it erases the EML arithmetic at deployment. Documentation must
   call this Boolean-equivalent EML compilation.
6. **Topology:** random sparse wiring is currently strongest. Structured local
   hierarchy is scientifically interesting but not yet a positive result. A
   larger random sparse detector4 circuit (`4096 x 8`, 32,768 gates) also failed
   to improve the best result, and `residual_random` trailed regularized random
   wiring. Blind capacity scaling is not the answer; topology changes need
   matched-budget evidence.
7. **Head expressivity:** a learned linear head over many circuit features is a
   real model component. The pure readout rows show that fixed buckets and
   learned per-feature class votes are not yet expressive enough. The next
   topology work should make class evidence emerge inside the circuit instead
   of relying on a float readout to mix it at the end.
8. **Package boundary:** current implementation lives inside
   `alberta_framework.core`. A standalone DiffEML package should split reusable
   circuit logic, image demos, benchmarks, and Alberta-specific experiments.

## Recommendation

Proceed with DiffEML as a small research package only if the public narrative is
strict:

> DiffEML trains differentiable selectors over EML-threshold realizable Boolean
> gates, hardens them into Boolean circuits, and can compile those circuits to
> packed Boolean inference. Current image results are nontrivial but trail a
> same-feature MLP.

That direction is honest, useful, and close to the DiffLogic package pattern.
The package should not imply that EML arithmetic itself is currently the source
of superior image performance.
