# Confirmatory Validation and Lockbox Run Log

Role: Worker B, validation/lockbox orchestrator.

Local run date: 2026-05-06. Summary generated UTC:
`2026-05-07T04:58:13.510638+00:00`.

## Commands

Initial frozen validation attempt, before the eval-batching patch was visible to
this worker:

```bash
source .venv/bin/activate
set -o pipefail
PYTHONUNBUFFERED=1 python benchmarks/step2_transformer_confirmatory_paperworthy_runner.py \
  --preset validation \
  --seeds 30 \
  --steps 3000 5000 10000 \
  --eval-steps 4096 \
  --final-window 512 \
  --output-root outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed \
  2>&1 | tee outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/validation_console.log
```

Outcome: stopped during the first 3000-step horizon after the legacy full eval
path hit repeated XLA constant-folding slow-operation alarms, including a
16.933461s compile for a `f32[4096,32,32]` batched dot. The partial process
did not write a horizon `results.json`.

Batched validation rerun after the wrapper exposed `--eval-batch-size`:

```bash
source .venv/bin/activate
set -o pipefail
PYTHONUNBUFFERED=1 python benchmarks/step2_transformer_confirmatory_paperworthy_runner.py \
  --preset validation \
  --seeds 30 \
  --steps 3000 5000 10000 \
  --eval-steps 4096 \
  --eval-batch-size 512 \
  --final-window 512 \
  --output-root outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed \
  2>&1 | tee outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/validation_batched_console.log
```

Lockbox command was not run because validation did not clear the preregistered
bar. If a later validation run clears the bar, the corresponding one-shot
command is:

```bash
source .venv/bin/activate
python benchmarks/step2_transformer_confirmatory_paperworthy_runner.py \
  --preset lockbox \
  --seeds 30 \
  --steps 3000 5000 10000 \
  --eval-steps 4096 \
  --eval-batch-size 512 \
  --final-window 512 \
  --output-root outputs/step2_new_directions/advantage_memory_transformer_confirmatory_lockbox_30seed
```

## Validation Artifacts

- `outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/confirmatory_wrapper_manifest.json`
- `outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/confirmatory_decision_summary.json`
- `outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/confirmatory_decision_summary.md`
- `outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/validation_paperworthy_report/paired_stats.csv`
- `outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/validation_paperworthy_report/paperworthy_report.json`
- `outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/validation_paperworthy_report/paperworthy_report.md`
- `outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/validation_3000_30seed_eval4096_fw512_eb512_replay128_scalar_glr05_l2_01_gmax015/results.json`
- `outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/validation_5000_30seed_eval4096_fw512_eb512_replay128_scalar_glr05_l2_01_gmax015/results.json`
- `outputs/step2_new_directions/advantage_memory_transformer_confirmatory_validation_30seed/validation_10000_30seed_eval4096_fw512_eb512_replay128_scalar_glr05_l2_01_gmax015/results.json`

No `advantage_memory_transformer_confirmatory_lockbox_30seed*` output directory
was created by this run.

## Decision

Validation completed with `clears_confirmatory_bar = false`, so lockbox was
skipped.

Failure reasons:

- Five of six primary rows failed Holm correction within the primary family.
- The 10000-step held-out `eval_nll` CI lower bound was `0.00042690797937876626`,
  below the preregistered SEI of `0.005`.
- Mechanism flags were present: open gate with negative advantage at 3000 and
  5000 steps, gate saturation at 3000 steps, and active-prototype saturation at
  all three horizons.
- Two paired seed outliers were reported, not dropped.

Non-finite metric count was zero, all primary paired means were positive, and
the run was not underpowered (`30` paired seeds).

## Primary Metrics

Positive paired differences favor `advantage_post_ffn_memory` over
`baseline_ffn_transformer`.

| Horizon | Metric | N | Baseline mean | Candidate mean | Diff | SE | 95% CI low | 95% CI high | paired t p | Wilcoxon p | sign p | Holm p | W/L/T |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3000 | `final_window_nll` | 30 | 2.8837228218714395 | 2.883696699142456 | 2.6122728983561197e-05 | 4.5822585531609766e-05 | -6.759498122484826e-05 | 0.00011984043919197065 | 0.5730111801495225 | 0.4521643426269293 | 0.8555355519056321 | 0.9802914840126125 | 16/14/0 |
| 3000 | `eval_nll` | 30 | 3.1164541274309157 | 3.1159138798713686 | 0.0005402475595474243 | 0.00028739288008676993 | -4.753687774392717e-05 | 0.0011280319968387758 | 0.07021312557674746 | 0.15188691951334476 | 0.36159460805356514 | 0.2808525023069898 | 18/12/0 |
| 5000 | `final_window_nll` | 30 | 2.824163071314494 | 2.823769728342692 | 0.0003933429718017578 | 0.0001264789065354878 | 0.0001346645630508464 | 0.0006520213805526693 | 0.0041719490104515 | 0.00466480478644371 | 0.016124801710247997 | 0.025031694062709 | 22/8/0 |
| 5000 | `eval_nll` | 30 | 3.1526694893836975 | 3.1510925283034643 | 0.001576961080233256 | 0.0016249321566772046 | -0.0017463983330575858 | 0.004900320493524098 | 0.33983344387337533 | 0.3284698575735092 | 0.20048842206597334 | 0.9802914840126125 | 19/11/0 |
| 10000 | `final_window_nll` | 30 | 2.7218604485193887 | 2.7212431033452353 | 0.0006173451741536459 | 0.0006188807778484951 | -0.0006484081376482411 | 0.001883098485955533 | 0.32676382800420417 | 0.4399667661637068 | 1.0 | 0.9802914840126125 | 15/15/0 |
| 10000 | `eval_nll` | 30 | 3.069663936893145 | 3.0553389062484104 | 0.0143250306447347 | 0.006795384918665363 | 0.00042690797937876626 | 0.028223153310090635 | 0.04378569333749567 | 0.0027663204818964005 | 0.016124801710247997 | 0.21892846668747834 | 22/8/0 |

## Split

Validation corpus: train bytes `[0, 800000)` plus validation bytes
`[800000, 950000)`.

Source sha256:
`86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed`.

Derived validation sha256:
`555877fa1338ff1fc441129607e4a763182aa6ccc651aa111709f4a388b678f2`.
