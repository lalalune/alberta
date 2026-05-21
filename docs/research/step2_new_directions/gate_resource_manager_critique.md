# Gate Resource Manager Critique

Worker scope: inspect and run the existing
`examples/The Alberta Plan/Step2/step2_tiny_shakespeare_gated_proto_transformer.py`.
The script existed before this task, so it was not edited. Outputs were written
only under `outputs/step2_new_directions/gate_resource_manager_critique/`.

## Implementation Finding

The current file is not a learned residual gate/resource manager yet. Despite
the filename, the implemented slow path is a prototype residual block whose
residual is always added:

`hidden = basis_input + block.transform(params["proto"], activations)`

or, in the hybrid model:

`hidden = basis_input + block.transform(params["proto"], activations)`.

There is no learned scalar/vector gate, no entropy or margin feature, and no
recent-loss EMA feature. Novelty only controls prototype center allocation via
`PrototypeBasisBlock.update_centers`; it does not decide whether the residual
path should be used.

Required gate inputs versus current implementation:

| Required signal | Present? | Current behavior |
| --- | --- | --- |
| Novelty | Partially | Used only for prototype allocation/replacement. |
| Entropy uncertainty | No | Logit entropy is not computed. |
| Margin uncertainty | No | Top-1/top-2 margin is not computed. |
| Recent loss EMA | No | Online loss is logged but no EMA state drives a gate. |
| Learned slow-path gate | No | Prototype residual is always added when activations are nonzero. |
| Gate collapse diagnostics | No | Only proxy diagnostics exist: active features, active prototypes, allocation rate, nearest distance. |

## Commands

```bash
source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_gated_proto_transformer.py" --steps 64 --seeds 1 --eval-steps 64 --final-window 32 --block-size 16 --d-model 16 --mlp-hidden 32 --proto-count 16 --output-dir outputs/step2_new_directions/gate_resource_manager_critique/smoke

source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_gated_proto_transformer.py" --steps 800 --seeds 2 --eval-steps 256 --final-window 256 --output-dir outputs/step2_new_directions/gate_resource_manager_critique/default_2seed_800

source .venv/bin/activate && python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_gated_proto_transformer.py" --steps 800 --seeds 2 --eval-steps 256 --final-window 256 --proto-bandwidth 0.1 --proto-novelty-threshold 0.2 --proto-update-rate 0.1 --proto-adaptive-bandwidth --output-dir outputs/step2_new_directions/gate_resource_manager_critique/adaptive_bw0p1_thr0p2_2seed_800
```

## Smoke Result

Small model, 64 online steps, 1 seed.

| Method | Final-window NLL | Eval NLL | Eval perplexity | Eval accuracy | Train s | Slow-path diagnostics |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FFN baseline | 4.039907 | 4.066769 | 58.368050 | 0.031250 | 0.255264 | n/a |
| Prototype basis | 4.029035 | 4.040460 | 56.852493 | 0.031250 | 0.262235 | active features 1.000000; active prototypes 1.000000; allocation rate 0.015625; final allocation 0.000000 |
| FFN + prototype | 4.007315 | 4.036068 | 56.603340 | 0.031250 | 0.287148 | active features 1.000000; active prototypes 1.000000; allocation rate 0.015625; final allocation 0.000000 |

Smoke passed mechanically. The slow path did not collapse to zero because it
is always connected, but it collapsed functionally to one active prototype in
the final window.

## Default 2-Seed 800-Step Result

Configuration: block size 32, `d_model=32`, FFN hidden 64, 64 prototypes,
`proto_bandwidth=0.01`, novelty threshold 0.08, update rate 0.3.

| Method | Final-window NLL | Final-window acc | Eval NLL | Eval acc | Eval perplexity | Train s | Steps/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FFN baseline | 3.503801 | 0.142578 | 3.548977 | 0.132812 | 34.894415 | 0.212259 | 3769.144190 |
| Prototype basis | 3.530154 | 0.142578 | 3.566871 | 0.132812 | 35.532389 | 0.265365 | 3017.218995 |
| FFN + prototype | 3.508608 | 0.142578 | 3.553778 | 0.132812 | 35.072241 | 0.287251 | 2785.300541 |

Diffs versus FFN baseline, positive favors the prototype method:

| Method | Final NLL diff | Eval NLL diff | Eval PPL diff | Eval acc diff | Train-time diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prototype basis | -0.026353 | -0.017893 | -0.637974 | +0.000000 | -0.053106 |
| FFN + prototype | -0.004807 | -0.004801 | -0.177826 | +0.000000 | -0.074992 |

Slow-path proxy diagnostics:

| Method | Active features | Active prototypes | Allocation rate | Final allocation rate | Final nearest distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prototype basis | 1.000000 | 1.000000 | 0.001250 | 0.000000 | 0.001388 |
| FFN + prototype | 1.000000 | 1.000000 | 0.001250 | 0.000000 | 0.002333 |

Interpretation: the slow path is not being adaptively selected. It is always
present, but only a single prototype is active in the final window. No new
prototypes are allocated late in training. This is closer to a static residual
bias/nearest-center path than a learned memory manager.

## Tuned 2-Seed 800-Step Result

Configuration change: `proto_bandwidth=0.1`, novelty threshold 0.2, update
rate 0.1, adaptive bandwidth enabled.

| Method | Final-window NLL | Final-window acc | Eval NLL | Eval acc | Eval perplexity | Train s | Steps/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| FFN baseline | 3.503801 | 0.142578 | 3.548977 | 0.132812 | 34.894415 | 0.205108 | 3902.288940 |
| Prototype basis | 3.530154 | 0.142578 | 3.566871 | 0.132812 | 35.532394 | 0.291113 | 2749.743481 |
| FFN + prototype | 3.508608 | 0.142578 | 3.553778 | 0.132812 | 35.072245 | 0.319081 | 2512.937928 |

Diffs versus FFN baseline, positive favors the prototype method:

| Method | Final NLL diff | Eval NLL diff | Eval PPL diff | Eval acc diff | Train-time diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prototype basis | -0.026353 | -0.017893 | -0.637980 | +0.000000 | -0.086005 |
| FFN + prototype | -0.004807 | -0.004801 | -0.177830 | +0.000000 | -0.113973 |

Slow-path proxy diagnostics:

| Method | Active features | Active prototypes | Allocation rate | Final allocation rate | Final nearest distance |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prototype basis | 1.000000 | 1.000000 | 0.001250 | 0.000000 | 0.001243 |
| FFN + prototype | 1.000000 | 1.000000 | 0.001250 | 0.000000 | 0.002209 |

The tuned bandwidth/novelty setting did not change the final-window behavior
or improve metrics. It increased runtime and left the slow path effectively
single-prototype.

## Gate Collapse Assessment

There is no learned gate, so the strict zero/one collapse question cannot be
answered for this implementation. The closest equivalent is:

| Collapse mode | Evidence |
| --- | --- |
| Gate-to-zero | Not applicable. The residual path is structurally always added. |
| Gate-to-one | Structurally yes: there is no learned shutoff. |
| Slow memory diversity collapse | Yes. Final-window active features and active prototypes are exactly 1.000000 in every prototype run. |
| Allocation collapse | Yes after initialization. Final-window allocation rate is 0.000000; total allocation rate is only 0.001250 in 800-step runs. |
| Performance collapse | Mild but consistent. Both prototype variants underperform FFN baseline on final-window NLL, eval NLL, eval perplexity, and runtime in the 800-step runs. |

## What A Real Learned Gate Should Add

A proper gate/resource manager for this experiment should maintain online gate
state and use at least these features at every token:

1. Novelty: nearest-center distance, allocation event, active prototype count.
2. Uncertainty: logit entropy and top-1/top-2 probability margin.
3. Recent loss: EMA of cross-entropy and optionally EMA of improvement from
   using the slow residual.
4. Cost: prototype activation count and state/update cost.

The residual should be:

`hidden = fast_hidden + gate * slow_residual`

where `gate` is learned online and logged each step. Minimum diagnostics:
mean gate, final-window mean gate, gate standard deviation, fraction below
0.05, fraction above 0.95, gate/loss correlation, gate/novelty correlation,
gate/entropy correlation, and slow-residual norm.

## Recommendation

Do not promote the current `step2_tiny_shakespeare_gated_proto_transformer.py`
as evidence for a learned residual gate. It is useful as a prototype-residual
baseline, but the requested resource manager is missing.

The next implementation should be a separate gated wrapper or a coordinated
edit to Pascal's file that adds an online gate with novelty, entropy/margin,
and loss-EMA features. The acceptance bar should be: no gate collapse, positive
NLL or perplexity margin over the FFN baseline on the 2-seed 800-step comparison,
and explicit evidence that the slow path is used selectively rather than always
or never.
