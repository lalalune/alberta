# DiffEML Package Roadmap

This roadmap treats DiffEML like a DiffLogic-style research package: train a
relaxed logic circuit, harden it, compile it, benchmark it, and publish complete
artifacts. The package should lead with reproducibility and compilation
semantics, not inflated accuracy claims.

## Package Positioning

DiffEML should be packaged as:

> A JAX research package for training differentiable selectors over
> EML-threshold-realizable Boolean gates, then hardening and compiling the
> resulting circuit to Boolean-equivalent inference.

Primary value:

- EML-derived gate library and executable template bank.
- Trainable soft selectors with hard circuit extraction.
- Packed Boolean inference for hardened circuits.
- Reproducible benchmarks against DiffLogic-style and neural baselines.
- Clear result artifacts that report soft, hard, and packed metrics.

Non-goals for the first release:

- Beating CNNs or ViTs on image benchmarks.
- Claiming superiority over same-feature MLPs before evidence exists.
- Hiding fixed feature engineering behind EML language.
- Providing a full continual-learning solution.

## Proposed Module Layout

```text
diffeml/
|-- __init__.py
|-- operators.py          # exact/stable EML operator, threshold helpers
|-- templates.py          # EMLTemplateExpr, EMLTemplateBank, render/evaluate
|-- gates.py              # Boolean masks, truth tables, EML gate libraries
|-- circuits.py           # circuit params/state, soft/hard forward passes
|-- wiring.py             # random, butterfly, Benes, local, local-tree wiring
|-- packed.py             # uint64 packing and hard Boolean circuit eval
|-- features.py           # binary image feature builders
|-- training.py           # losses, optimizers, temperature schedules
|-- baselines.py          # same-feature linear, MLP, optional DiffLogic-style
|-- datasets.py           # digits, MNIST/OpenML, CIFAR local/download loaders
|-- benchmarks.py         # benchmark runner internals
|-- results.py            # result schema, validation, aggregation
|-- export.py             # circuit export: JSON, masks, expressions, packed cfg
`-- cli.py                # command-line entrypoints
```

Optional later modules:

```text
diffeml/alberta.py        # adapters to alberta-framework continual streams
diffeml/torch.py          # optional PyTorch baseline interop, not core
diffeml/cuda.py           # optional packed-kernel experiments
```

Current source mapping:

| Current file | Package destination |
| --- | --- |
| `src/alberta_framework/core/diffeml.py` | `operators.py`, `templates.py`, `gates.py`, early `circuits.py` |
| `src/alberta_framework/core/diffeml_image.py` | `features.py`, `wiring.py`, `packed.py`, `training.py`, `datasets.py`, `benchmarks.py` |
| `examples/The Alberta Plan/Step2/step2_diffeml_image_demo.py` | `diffeml.cli` commands |
| `tests/test_diffeml.py` | operator, template, gate, selector tests |
| `tests/test_diffeml_image_demo.py` | feature, wiring, packed eval tests |

## CLI Commands

Use subcommands with JSON output by default and optional Markdown summaries.

```bash
diffeml gates enumerate \
  --depth 2 \
  --eps 0.05 \
  --output artifacts/gates_depth2.json

diffeml gates verify \
  --depth 2 \
  --eps 0.05

diffeml train image \
  --dataset cifar \
  --data-dir outputs/diffeml_image_data \
  --feature-mode detector_thresholds \
  --input-bits 24576 \
  --pixel-thresholds 4 \
  --wiring random \
  --layers 6 \
  --width 2048 \
  --gate-mode eml_template \
  --hard-loss-weight 0.8 \
  --input-drop-rate 0.02 \
  --feature-drop-rate 0.30 \
  --min-temperature 0.05 \
  --entropy-weight 0.01 \
  --epochs 10 \
  --max-train 20000 \
  --max-test 5000 \
  --seed 0 \
  --output runs/cifar_detector_random_seed0.json

diffeml eval hard \
  --checkpoint runs/cifar_detector_random_seed0.json \
  --dataset cifar \
  --split test \
  --output runs/cifar_detector_random_seed0_hard.json

diffeml compile packed \
  --checkpoint runs/cifar_detector_random_seed0.json \
  --output runs/cifar_detector_random_seed0_packed.json

diffeml eval packed \
  --compiled runs/cifar_detector_random_seed0_packed.json \
  --dataset cifar \
  --split test \
  --output runs/cifar_detector_random_seed0_packed_eval.json

diffeml baseline image \
  --dataset cifar \
  --feature-mode detector_thresholds \
  --input-bits 24576 \
  --pixel-thresholds 4 \
  --baseline mlp \
  --hidden-sizes 512 \
  --epochs 10 \
  --max-train 20000 \
  --max-test 5000 \
  --seed 0 \
  --output runs/cifar_detector_mlp512_seed0.json

diffeml bench matrix \
  --matrix configs/benchmark_matrix.yaml \
  --output-dir runs/matrix

diffeml results summarize \
  --input-dir runs/matrix \
  --group-by dataset feature_mode wiring gate_mode baseline \
  --output docs/results/diffeml_matrix.md
```

## Benchmark Matrix

The first public matrix should be small enough to rerun but broad enough to
prevent cherry-picking.

### Datasets

| Dataset | Purpose | Required for release |
| --- | --- | --- |
| sklearn digits | fast smoke, gate/circuit sanity | yes |
| MNIST/OpenML | simple image benchmark | yes |
| CIFAR-10 5k/1k | fast external image smoke | yes |
| CIFAR-10 20k/5k | headline package-scale result | yes |
| full CIFAR-10 50k/10k | stretch, not v0.1 blocker | no |

### Feature Modes

| Feature mode | Status | Release use |
| --- | --- | --- |
| `threshold_pixels` | simple DiffLogic-style binary input | required baseline |
| `detector_thresholds` | current strongest fixed image feature pipeline | required headline |
| `variance_pixels` | older toy mode | legacy comparison only |
| learned binary stem | future work | not v0.1 |

### Circuit Modes

| Axis | Values |
| --- | --- |
| gate mode | `eml_template`, `eml_threshold`, `truth_table` |
| wiring | `random`, `butterfly`, `permuted_butterfly`, `benes`, `class_bank_random`, `local_hierarchy`, `local_tree_hierarchy` |
| head | `linear`, `group_sum`, `class_vote`, `signed_class_vote` |
| hardening | soft eval, hard JAX eval, packed hard eval |
| size | width 512/1024/2048, layers 3/5/6 |

### Baselines

Every benchmark row that makes an image claim should include:

- same-feature linear softmax;
- same-feature MLP-512;
- truth-table selector with the same topology;
- random frozen Boolean gates plus trained head;
- optional DiffLogic implementation if dependency/setup is acceptable.

The MLP baseline from
`outputs/diffeml_image_demo/cifar_detector4_same_features_mlp512_seed0.json`
is the current minimum bar: 49.32% on the exact strongest detector4 binary
features.

## CI/Test Matrix

### Unit Tests

Required tests before package release:

- exact EML gradients for positive right input;
- stable EML finite outputs under unconstrained inputs;
- Boolean mask conversion and truth-table ordering;
- depth-2 EML template coverage of all 16 two-input gates;
- executable template bank matches truth-table library;
- selector learns every two-input Boolean gate;
- soft and hard circuit shape/dtype checks;
- packed gate evaluation matches truth-table masks;
- packed hard features/logits match JAX hard features/logits on random small
  circuits;
- feature builders are deterministic and report density/selection metadata;
- wiring builders are deterministic by seed and validate invalid configs;
- result schema validation rejects missing soft/hard/packed fields.

### Smoke Tests

Run on every PR:

```bash
pytest tests/test_diffeml.py tests/test_diffeml_image_demo.py -q
diffeml gates verify --depth 2 --eps 0.05
diffeml train image --dataset digits --max-train 256 --max-test 128 --epochs 1 --width 128 --layers 2 --output /tmp/diffeml_smoke.json
diffeml compile packed --checkpoint /tmp/diffeml_smoke.json --output /tmp/diffeml_smoke_packed.json
diffeml eval packed --compiled /tmp/diffeml_smoke_packed.json --dataset digits --split test
```

### Nightly or Manual Tests

Run outside normal PR CI:

- CIFAR-10 5k/1k across seeds `0..4`;
- CIFAR-10 20k/5k across seeds `0..4`;
- same-feature MLP baselines for every feature mode;
- packed vs JAX hard equality checks on all hardened benchmark circuits;
- runtime and memory benchmarks for soft, hard JAX, and packed eval;
- local hierarchy and local tree hierarchy sweeps, clearly labeled as negative
  unless they beat random wiring.

## Result Table Schema

Every benchmark JSON should validate against a stable schema. Required top-level
fields:

```json
{
  "schema_version": "diffeml.result.v1",
  "run_id": "cifar_detector_random_w2048_l6_seed0",
  "created_at": "ISO-8601 timestamp",
  "git": {
    "repo": "alberta-framework or diffeml",
    "commit": "required when available",
    "dirty": true
  },
  "environment": {
    "python": "3.x",
    "jax": "...",
    "platform": "...",
    "device": "cpu/gpu/tpu"
  },
  "dataset": {
    "name": "cifar",
    "source": "cifar-10-python",
    "train_examples": 20000,
    "test_examples": 5000,
    "seed": 0,
    "split": "stratified train_fraction=0.8"
  },
  "features": {
    "mode": "detector_thresholds",
    "raw_features": 3072,
    "expanded_bits": 61440,
    "selected_bits": 24576,
    "threshold_values": [0.2, 0.4, 0.6, 0.8],
    "train_density": 0.2631,
    "test_density": 0.2658
  },
  "model": {
    "kind": "diffeml",
    "gate_mode": "eml_template",
    "template_depth": 2,
    "eml_eps": 0.05,
    "wiring": "random",
    "layers": 6,
    "width": 2048,
    "head": "linear",
    "nodes": 12288,
    "active_node_parameters": 196608,
    "head_parameters": 20490,
    "compiled_storage": {
      "deployed_readout_bytes": 20560,
      "compiled_packed_bytes": 82000,
      "compiled_int8_bytes": 82000,
      "soft_train_bytes": 868392,
      "soft_to_compiled_packed_compression": 10.59
    }
  },
  "training": {
    "optimizer": "adam",
    "epochs": 10,
    "batch_size": 128,
    "updates": 1570,
    "elapsed_s": 220.8,
    "final_temperature": 0.05
  },
  "metrics": {
    "train_soft_accuracy": 0.5607,
    "train_hard_accuracy": 0.5784,
    "test_soft_accuracy": 0.4658,
    "test_hard_accuracy": 0.4700,
    "packed_hard_test_accuracy": 0.4700,
    "packed_int8_head_test_accuracy": 0.4698
  },
  "baselines": {
    "linear_same_features": null,
    "mlp_same_features": {
      "hidden_sizes": [512],
      "parameters": 12588554,
      "test_accuracy": 0.4932
    }
  },
  "artifacts": {
    "checkpoint": "optional path",
    "compiled_packed": "optional path",
    "summary_markdown": "optional path"
  }
}
```

CSV/Markdown summaries should flatten these fields into columns:

```text
run_id, dataset, seed, train_examples, test_examples,
feature_mode, selected_bits, gate_mode, wiring, layers, width, head,
nodes, active_node_parameters, head_parameters,
epochs, updates, elapsed_s,
train_soft_accuracy, train_hard_accuracy,
test_soft_accuracy, test_hard_accuracy, packed_hard_test_accuracy,
packed_int8_head_test_accuracy, linear_test_accuracy, mlp512_test_accuracy,
hard_minus_mlp512, packed_matches_jax_hard
```

`packed_int8_head_test_accuracy` is only valid for `head = "linear"`. For
`group_sum`, `class_vote`, and `signed_class_vote`, the deployed readout is
already Boolean/count metadata, so result rows should report
`packed_hard_test_accuracy` and `compiled_packed_bytes` instead of a fake
int8-head metric.

## Release Criteria

### v0.1 Research Release

Required:

- standalone package import works without `alberta_framework`;
- public docs explain EML arithmetic, EML-template gates, hardening, and
  Boolean-equivalent packed compilation;
- all unit and smoke tests pass;
- `diffeml gates verify --depth 2 --eps 0.05` passes;
- packed hard eval equals JAX hard eval on randomized unit tests and at least
  one image smoke artifact;
- reproduced artifacts for digits, MNIST, CIFAR 5k/1k, and CIFAR 20k/5k;
- same-feature linear and MLP baselines included in every image result table;
- README states that current CIFAR 20k/5k DiffEML trails same-feature MLP unless
  newer replicated evidence changes that.

Minimum acceptable public claim:

> On the current single-seed 20k/5k CIFAR-10 detector-threshold split, DiffEML
> reaches 47.00% packed hard test accuracy with a 46.98% int8-head packed
> variant, while the same-feature MLP-512 control reaches 49.32%. The result
> demonstrates nontrivial hard
> EML-template circuit learning and exact packed compilation, not accuracy
> superiority.

Pure Boolean/count readout rows are a separate, stricter claim. On the same
split, `group_sum` reaches 31.78% with 61,440 compiled bytes and `class_vote`
reaches 35.50% with 62,464 compiled bytes. These rows are cleaner under the
Boolean-compression goal, but they are not yet accuracy-competitive.

### v0.2 Evidence Release

Required:

- 5-seed CIFAR 20k/5k matrix with confidence intervals;
- packed inference timing and memory report;
- quantized linear-head report for the legacy/current-best linear readout;
- Boolean/count readout report for `group_sum`, `class_vote`, and class-bank
  topology variants;
- same-feature MLP and truth-table selector baselines for all headline rows;
- ablation report separating feature mode, wiring, gate mode, head mode, and
  packed compilation;
- local-tree hierarchy either improved or documented as rejected for now.

### v1.0 Claim Release

Required before stronger claims:

- replicated win over same-feature MLP on at least one pre-registered benchmark,
  or a clearly different value claim such as substantially better packed
  inference cost at acceptable accuracy;
- full CIFAR-10 or another standard benchmark with no test-set tuning;
- package docs and examples stable enough for external users;
- archival result bundle with configs, seeds, commit hashes, and raw JSON.

## Immediate Work Items

1. Split `diffeml.py` and `diffeml_image.py` into package-shaped modules without
   changing behavior.
2. Extend the current result schema adapter into a CLI converter for existing
   artifacts in `outputs/diffeml_image_demo/`.
3. Add same-feature linear and MLP baselines to the default benchmark matrix.
4. Extend packed-vs-JAX hard equality tests from small random circuits to saved
   benchmark artifacts.
5. Re-run CIFAR detector-threshold random wiring for seeds `0..4`.
6. Re-run the detector4 `class_vote` and `group_sum` rows after the
   straight-through hard-readout update, then decide whether class-bank topology
   replaces generic final features for the pure Boolean readout path.
7. Keep the first `class_bank_random` result as a negative smoke unless a
   global-mixer-to-bank variant beats generic random wiring under packed
   Boolean readout.
8. Keep `signed_class_vote` as a pure Boolean readout ablation. Promote it only
   if a detector-scale rerun beats unsigned `class_vote`.
9. Create a short negative-result note for `local_tree_hierarchy` unless a new
   variant beats random sparse wiring at matched train/test scale.
10. Keep Alberta integration as an adapter, not the package core.
