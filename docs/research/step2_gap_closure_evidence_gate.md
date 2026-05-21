# Step 2 Gap Closure Evidence Gate

Date: 2026-05-06.

## Purpose

This note defines the evidence that would make the three remaining Step 2 gaps
presentation-quality, and how that evidence should be wired into
`benchmarks/step2_upgd_evidence_gate.py` after the experiment agents produce
stable artifacts.

The current gate is artifact-driven. It validates `UPGDLearner.step2_default`,
then reads fixed JSON files for synthetic out-of-class streams, sklearn digits,
and efficiency. The extension should keep that pattern: experiment scripts
produce JSON summaries; the gate consumes those summaries and fails on missing,
underpowered, or negative evidence.

## Current Gate Baseline

`benchmarks/step2_upgd_evidence_gate.py` currently checks:

| Area | Current gate source | Current status | Gap status |
|---|---|---:|---|
| Factory freeze | `UPGDLearner.step2_default` | pass | closed for current default |
| Lean state | `UPGDLearner.step2_default(...).init(...)` | pass | closed for current default |
| Synthetic breadth | `output/subagents/compute_efficiency/small_rademacher_synthetic_30seed_6000/out_of_class_results.json` | pass | internal only |
| Digits matrix | `output/subagents/compute_efficiency/small_rademacher_digits_30seed_h64baseline/upgd_digits_sweep_results.json` | pass | external but narrow |
| Class-blocked caveat | same digits artifact | pass with caveat | not closed |
| Efficiency | `output/benchmarks/step2_upgd_efficiency_fused_heads_4096/efficiency_results.json` | pass | closed for width-32 vs MLP64 scan benchmark |

Important current class-blocked facts:

| Metric | Current promoted UPGD vs MLP64 | Read |
|---|---:|---|
| final-window MSE | `+0.00047`, `27/30` wins | clears current tracking-loss gate |
| held-out test MSE | `+0.00334`, `20/30` wins | positive but modest |
| final-window accuracy | `-0.00389`, `1/30` wins | still a caveat |
| held-out test accuracy | `+0.02870`, `22/30` wins | positive relative to a weak retained baseline, but absolute mean is only `0.1466` |

The current external-breadth artifacts are not enough to close the gap:

| Artifact | Why it is useful | Why it is insufficient |
|---|---|---|
| `output/subagents/external_breadth_sklearn/smoke_promoted/external_suite_results.json` | proves the promoted factory can run through `step2_external_suite.py` | only 1 seed, 200 steps, 2 digit benchmarks, and no external-domain breadth |
| `output/worker_s2c_external/external_suite_results.json` | broader sklearn suite with 5 seeds | older/non-promoted config, UPGD loses several rows, and hidden/resource contract is not the promoted width-32 vs MLP64 comparison |
| `output/subagents/retention_tracking_split/results.json` | contains useful recent/non-recent retention metrics | tests MLP/EMA split probes, not promoted UPGD; current guarded split does not improve class-blocked held-out accuracy |
| `output/step2_tiny_shakespeare_upgd_transformer_demo/results.json` | shows transformer-shaped integration with `softmax_ce` | only 2 seeds and 800 steps; baseline is a tiny SGD demo; accuracy is mixed |

## Claim Levels

Keep two bars separate.

| Claim level | What it can support | What it cannot support |
|---|---|---|
| Alberta Plan Step 2 presentation | A scientific internal claim that the promoted Step 2 learner survived prespecified external, retention, and transformer-facing checks. Negative rows must be shown, not hidden. | Production replacement, theorem language, modern transformer superiority, or deployment beyond online supervised workloads. |
| Production MLP replacement | A domain-specific decision to replace an MLP/FFN under a known traffic, latency, reliability, and monitoring envelope. | Broad generalization from the research matrix alone. Each deployment domain needs a shadow evaluation and rollback plan. |

## Acceptance Matrix

### External Breadth

Presentation-quality external breadth requires a prespecified, non-synthetic
online matrix that is large enough to detect cherry-picking and config drift.

| Requirement | Presentation gate | Production deployment bar |
|---|---|---|
| Protocol | prequential online: predict, score, then update; held-out test split never used for updates | same, plus domain replay/shadow logs with production feature preprocessing |
| Candidate | exact promoted `UPGDLearner.step2_default` branch unless the artifact explicitly declares a separate ablation | exact production candidate config, frozen and serialized |
| Baseline | fair MLP64 comparator with same inputs, target contract, step budget, and update budget | current production MLP/FFN plus any incumbent safety wrappers |
| Breadth | at least 6 non-synthetic benchmark rows across at least 4 dataset families and at least 3 task kinds | the actual deployment distribution and stress slices, not a generic suite |
| Seeds | at least 10 paired seeds/splits per benchmark row | at least 30 paired runs or a domain-specific power analysis |
| Scale | no smoke runs; each row must run long enough for the final window to represent a stable phase | production-length or traffic-representative horizon |
| Primary metric | classification: held-out accuracy and final-window MSE; regression: held-out MSE and final-window MSE; multilabel: held-out macro/micro accuracy plus MSE | deployment KPI plus guardrail metrics |
| Aggregate pass | primary paired diff positive across all benchmark/seed cells and wins on at least 60% of cells | non-inferior or better on the deployment primary KPI with confidence interval |
| Row-level pass | at least 70% of benchmark rows have positive primary paired diff; no row has catastrophic degradation | no critical slice may regress beyond the approved non-inferiority margin |
| Catastrophic degradation | classification held-out accuracy diff must be `>= -0.02`; regression/multilabel held-out MSE relative degradation must be `<= 5%` | margins set by product/service SLOs |
| Reporting | all configured rows, seeds, failures, and skipped rows recorded in JSON | same, plus operational monitoring and rollback evidence |

Recommended presentation thresholds for the gate:

| Check name | Threshold |
|---|---|
| `external.config.n_benchmark_rows` | `>= 6` |
| `external.config.n_dataset_families` | `>= 4` |
| `external.config.n_task_kinds` | `>= 3` |
| `external.config.min_seeds_per_row` | `>= 10` |
| `external.aggregate.primary_diff` | `> 0` |
| `external.aggregate.primary_win_rate` | `>= 0.60` |
| `external.aggregate.row_positive_rate` | `>= 0.70` |
| `external.rows.no_catastrophic_degradation` | all rows pass |
| `external.rows.no_missing_or_skipped_rows` | all configured rows completed |

### Class-Blocked Retention

The current gate treats class-blocked retention as a caveat. To close the gap,
the next artifact must prove retained-class behavior directly, not infer it
from aggregate test accuracy.

| Requirement | Presentation gate | Production deployment bar |
|---|---|---|
| Protocol | class-blocked stream with explicit final-window recent classes and held-out recent/non-recent split | domain-specific rare-class or stale-task retention slices |
| Candidate | promoted UPGD single learner, not a router/portfolio unless the claim is explicitly changed | exact deployment candidate and fallback policy |
| Baseline | MLP64 under the same stream and evaluation split | incumbent production model |
| Seeds | at least 30 paired seeds | at least 30 paired runs or domain power analysis |
| Tracking loss | final-window MSE diff `mlp_minus_upgd_mean > 0` and wins `>= 20/30` | non-inferior or better on tracking KPI |
| Tracking accuracy | final-window accuracy diff `method_minus_mlp_mean >= 0` and wins `>= 15/30` | non-inferior with approved margin |
| Retained accuracy | non-recent held-out accuracy diff `>= +0.03` and wins `>= 20/30` | domain-specific retained-slice KPI, with alert thresholds |
| Absolute retention | non-recent held-out accuracy `>= chance + 0.10` and macro accuracy `>= chance + 0.10` | slice-specific absolute floor |
| Class collapse guard | minimum per-class held-out accuracy `>= 0.05` | no protected/critical class collapse |
| Diagnostics | utility mean/min/max, perturbation energy share for low-utility units, hidden effective rank, and class coverage recorded | same, plus live monitoring |

Recommended presentation thresholds for the gate:

| Check name | Threshold |
|---|---|
| `retention.config.regime` | `class_blocked` |
| `retention.config.n_seeds` | `>= 30` |
| `retention.final_window_mse.diff` | `> 0` |
| `retention.final_window_mse.wins` | `>= 20 / 30` |
| `retention.final_window_accuracy.diff` | `>= 0` |
| `retention.final_window_accuracy.wins` | `>= 15 / 30` |
| `retention.test_accuracy.diff` | `>= +0.02` |
| `retention.test_non_recent_accuracy.diff` | `>= +0.03` |
| `retention.test_non_recent_accuracy.absolute` | `>= chance + 0.10` |
| `retention.test_macro_accuracy.absolute` | `>= chance + 0.10` |
| `retention.test_min_class_accuracy.absolute` | `>= 0.05` |
| `retention.diagnostics.required_fields` | all present |

This is intentionally stricter than the current gate. The present artifact
would fail the new retention gate because final-window accuracy is still
negative and retained/non-recent class accuracy is not measured for the
promoted UPGD learner.

### Transformer-Scale Proof

The current Tiny Shakespeare run should stay an integration demo until a
stronger artifact exists. A presentation-quality transformer result does not
need to beat modern language models, but it must prove that the promoted Step 2
learner remains competitive inside a transformer-shaped online sequence model
at a meaningful scale.

| Requirement | Presentation gate | Production deployment bar |
|---|---|---|
| Candidate | `UPGDLearner.step2_default(..., readout_mode="softmax_ce")` behind the same attention front-end | exact FFN/readout replacement candidate in the production model family |
| Baseline | same transformer front-end with MLP64 FFN/readout; include a stronger optimizer baseline such as AdamW or normalized SGD if used by the script | production transformer baseline and training recipe |
| Seeds | at least 5 paired seeds | enough runs for deployment power analysis |
| Scale | no smoke runs; at least `100000` online token updates or an explicitly justified smaller horizon | production-scale token budget or offline replay/shadow test |
| Evaluation | held-out stream not used for updates; report eval NLL, eval perplexity, eval accuracy, final-window NLL, final-window accuracy | same, plus latency, memory, and failure-rate metrics |
| Primary result | eval NLL diff favoring UPGD `>= -0.02`, final-window NLL diff `>= -0.02` | non-inferior on deployment loss/KPI |
| Accuracy guard | eval accuracy diff `>= -0.005`; final-window accuracy diff `>= -0.01` | non-inferior on task KPI |
| Stability | no seed has NaN/Inf metrics or divergence; max eval perplexity ratio vs MLP `<= 1.05` | operational stability under realistic load |
| Efficiency | report trainable parameters, float state, and steps/s for the transformer loop | must satisfy production latency/cost budget |

Recommended presentation thresholds for the gate:

| Check name | Threshold |
|---|---|
| `transformer.config.seeds` | `>= 5` |
| `transformer.config.online_token_updates` | `>= 100000` |
| `transformer.config.upgd_readout_mode` | `softmax_ce` |
| `transformer.config.baseline_optimizer` | declared, not missing |
| `transformer.eval_nll.diff_favoring_upgd` | `>= -0.02` |
| `transformer.final_window_nll.diff_favoring_upgd` | `>= -0.02` |
| `transformer.eval_accuracy.diff_favoring_upgd` | `>= -0.005` |
| `transformer.final_window_accuracy.diff_favoring_upgd` | `>= -0.01` |
| `transformer.eval_perplexity.max_ratio_upgd_over_mlp` | `<= 1.05` |
| `transformer.stability.no_bad_seed` | true |
| `transformer.efficiency.required_fields` | all present |

The current 2-seed, 800-step Tiny Shakespeare artifact would fail this gate.
It is still useful as evidence that the CE branch can be wired into a
transformer-shaped loop.

## Results Contract

Each new artifact should expose a top-level `gate_contract_version` and use the
same paired-summary vocabulary as the current gate where possible. A concrete
example is also written to
`output/subagents/evidence_gate_extension/proposed_results_contract.json`.

Required top-level fields:

```json
{
  "gate_contract_version": 1,
  "evidence_area": "external_breadth | class_blocked_retention | transformer_scale",
  "evidence_level": "presentation_candidate",
  "created_by": "script path or agent name",
  "config": {},
  "candidate": {},
  "baseline": {},
  "benchmarks": {},
  "aggregate": {},
  "records": [],
  "diagnostics": {},
  "artifacts": {}
}
```

Common method fields:

| Field | Required value |
|---|---|
| `candidate.name` | promoted method name or explicit ablation name |
| `candidate.factory` | `UPGDLearner.step2_default` for promoted evidence |
| `candidate.hidden_size` | `32` for promoted width-32 evidence |
| `candidate.loss_normalization` | `target_structure` |
| `candidate.perturbation_sigma` | `0.0001` for promoted MSE evidence unless a CE transformer adaptation declares otherwise |
| `candidate.perturbation_noise` | `rademacher` |
| `candidate.perturbation_interval` | `16` |
| `baseline.name` | `mlp64` or explicit transformer MLP baseline |
| `baseline.hidden_size` | `64` for Step 2 MLP replacement comparisons |

Common paired metric fields:

| Metric family | Required fields |
|---|---|
| lower-is-better | `mlp_minus_method_mean`, `diff_stderr`, `wins_for_method`, `wins_for_mlp`, `ties`, `n` |
| higher-is-better | `method_minus_mlp_mean`, `diff_stderr`, `wins_for_method`, `wins_for_mlp`, `ties`, `n` |
| absolute candidate | `method_mean`, `method_stderr` |
| absolute baseline | `mlp_mean`, `mlp_stderr` |

Area-specific required metrics:

| Area | Required metrics |
|---|---|
| external breadth | `primary_metric`, `final_window_mse`, `test_mse`, `final_window_accuracy`, `test_accuracy`, row-level `task_kind`, `dataset_family`, `n_seeds`, `completed` |
| class-blocked retention | `final_window_mse`, `final_window_accuracy`, `test_mse`, `test_accuracy`, `test_recent_accuracy`, `test_non_recent_accuracy`, `test_macro_accuracy`, `test_min_class_accuracy`, `chance_accuracy`, per-seed recent class sets |
| transformer scale | `final_window_nll`, `final_window_accuracy`, `eval_nll`, `eval_accuracy`, `eval_perplexity`, `tokens_seen`, `online_token_updates`, `trainable_params`, `float_state_size`, `steps_per_second`, bad-seed list |

## Gate Integration Plan

Do not add experiment execution to `benchmarks/step2_upgd_evidence_gate.py`.
Add optional artifact arguments and pure JSON checks.

Proposed CLI additions:

```bash
python benchmarks/step2_upgd_evidence_gate.py \
  --external-breadth-results output/subagents/evidence_gate_extension/external_breadth_results.json \
  --class-blocked-retention-results output/subagents/evidence_gate_extension/class_blocked_retention_results.json \
  --transformer-scale-results output/subagents/evidence_gate_extension/transformer_scale_results.json
```

Recommended implementation shape:

1. Add constants for the three new default artifact paths under
   `output/subagents/evidence_gate_extension/`.
2. Add `_validate_contract_header(payload, evidence_area)` to fail fast on
   missing contract version, area, candidate, baseline, or config.
3. Add `_check_external_breadth(checks, path)` using the thresholds above.
4. Add `_check_class_blocked_retention(checks, path)` using recent/non-recent
   and per-class metrics.
5. Add `_check_transformer_scale(checks, path)` using NLL, accuracy, stability,
   and efficiency metrics.
6. Keep missing artifacts as failures only after the workstream declares them
   required. During transition, the gate can expose `--require-gap-closure`
   so current CI keeps checking the existing evidence while long runs complete.
7. Include each new check in `write_summary` with the existing `GateCheck`
   format.

Suggested default files once produced:

| Evidence area | Proposed path |
|---|---|
| external breadth | `output/subagents/evidence_gate_extension/external_breadth_results.json` |
| class-blocked retention | `output/subagents/evidence_gate_extension/class_blocked_retention_results.json` |
| transformer scale | `output/subagents/evidence_gate_extension/transformer_scale_results.json` |

## Current Gaps To Hand Off

External-breadth agents should produce a non-smoke artifact with promoted
width-32 UPGD versus MLP64, not an older same-width or width-64 UPGD run.
`step2_external_suite.py` is close in metric shape, but the output must expose
separate candidate/baseline configs and enough datasets/seeds.

Retention agents should add promoted-UPGD class-blocked runs that report
recent/non-recent held-out accuracy, macro accuracy, minimum per-class
accuracy, and feature/utility diagnostics. The current digits artifact cannot
answer retained-class behavior directly.

Transformer agents should keep the existing Tiny Shakespeare script as the
integration seed, but produce an aggregate paired JSON with at least 5 seeds,
a meaningful token horizon, declared baseline optimizer, stability flags, and
efficiency/resource metrics.

Until those artifacts exist, the honest Step 2 statement is unchanged: the
current gate supports internal synthetic/digits/efficiency promotion, while
external breadth, class-blocked retention, and transformer-scale proof remain
open presentation gaps.
