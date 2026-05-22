# Advantage-Memory Transformer Paperworthy Benchmark Protocol

## Verdict

The replay-capped advantage-memory transformer is a promising Step 2 candidate,
but the current fast/slow Tiny Shakespeare evidence is not yet paperworthy as a
standalone empirical claim. The existing runs are useful exploratory evidence:
they use paired seeds, a held-out text suffix, config JSON, resource counts, and
per-method timing. They do not yet meet the standard required for a paper claim
about a replay-capped slow memory mechanism beating a tuned fast transformer.

The main issue is not a missing table. It is the combination of tiny margins,
post hoc hyperparameter selection, one dataset, only 10 paired seeds, and
insufficient manifest capture.

## Gap Register

| Area | Current state | Paperworthy gap | Required fix |
| --- | --- | --- | --- |
| Data protocol | Tiny Shakespeare is split by `train_fraction`; evaluation uses the held-out suffix. | No frozen train/validation/test byte ranges, no lockbox split, and no record that the final split was untouched during tuning. | Freeze byte offsets before running: train for online updates, validation for model selection, lockbox test for one final report. |
| Data provenance | `data_path` is captured in config. | The runner artifact does not store corpus sha256, byte count, source URL, or preprocessing hash. | Record data sha256 and byte count in every result artifact; publish or cache the exact corpus. |
| Held-out size | Current canonical runs use `eval_steps=512`. | This is small relative to the observed NLL/perplexity margins. | Use thousands of held-out contexts per seed or evaluate the full fixed validation/test shard. |
| Seeds | Canonical runs use 10 paired seeds. | The claimed effects are often below `0.002` NLL; 10 seeds is underpowered. | Use at least 30 paired seeds or include a power analysis and minimum detectable effect. |
| Pairing | Baseline and memory variants share seed, initialization key, and stream construction. | Per-seed train offsets are not stored, so exact streams cannot be reconstructed from the artifact alone. | Store train offset, eval offset, and PRNG derivation per seed and method. |
| Statistics | The runner summary reports means, standard errors, and raw diffs. | No paired confidence intervals, paired tests, effect sizes, win/loss/tie counts, or corrected p-values in the canonical summary. | Use paired seed-level diffs, 95% CIs, paired effect sizes, sign/Wilcoxon checks, and Holm correction. |
| Multiple comparisons | The winning config follows many exploratory sweeps over gate, cap, placement, replay, and horizon. | The current result does not account for selection over many knobs and outcomes. | Label prior sweeps exploratory; preregister one confirmatory family before the final rerun. |
| Primary metrics | Follow-up text discusses final-window NLL, held-out NLL, and perplexity. | Perplexity is a monotone transform of NLL and should not be treated as an independent primary test. | Primary family: final-window prequential NLL and held-out NLL. Perplexity is report-only. |
| Practical effect size | Margins are directionally consistent but very small at 3000 and 5000 steps. | Statistical significance alone would not establish a meaningful method advantage. | Predeclare a smallest effect of interest, such as held-out NLL improvement >= `0.005` or a compute-normalized threshold. |
| Baselines | The current benchmark compares against a tuned FFN transformer. | A single baseline cannot isolate whether memory, extra parameters, replay, cap, or extra compute caused the effect. | Add parameter-matched FFN, compute-matched FFN, replay-only memory, no-cap memory, fixed-cap ablations, no-memory slow-parameter controls, and a tuned wider FFN. |
| Compute reporting | Artifacts record train seconds, steps/sec, params, trainable bytes, and state bytes. | No hardware identity, backend, JAX/JAXLIB versions, warmup vs hot-loop split, compile time, peak memory, or energy. | Record device, platform, package versions, compile time, hot-loop time, peak memory, and total wall clock. |
| Config capture | `ExperimentConfig` captures many runner flags. | It omits some CLI/prototype details and does not include argv, git revision, dirty status, environment, source hash, or exact command. | Add a training-run manifest emitted by the runner, not only a post hoc report. |
| Reproducibility | Existing JSON is enough to approximate commands. | Approximate commands are not equivalent to an immutable run manifest. | Store full argv, config, git status, data hash, package versions, and source file hash. |
| Failure modes | Gate, advantage, allocation, and active-feature diagnostics are recorded. | There are no preregistered failure thresholds; the leading config can show negative final-window advantage while the gate stays open. | Fail or flag runs for NaNs, open gate with negative advantage, held-out regression, over-allocation, and seed outliers. |
| External validity | Evidence is Tiny Shakespeare only. | No evidence that the mechanism transfers beyond one small character-level corpus and tiny model shape. | Add at least one different text stream and one non-text Step 2 stream with the same confirmatory protocol. |
| Reporting artifacts | Runner writes `results.json` and `SUMMARY.md`. | No raw paired CSV, no machine-readable statistical table, no benchmark checklist, and no manifest. | Use the paperworthy report generator below and then move manifest generation into the runner. |

## Implemented Reporting Improvement

New helper:

```bash
source .venv/bin/activate
python benchmarks/step2_transformer_paperworthy_benchmark_suite.py \
  --output-dir output/benchmarks/step2_transformer_paperworthy_benchmark_suite
```

The helper reads existing `results.json` artifacts and writes:

- `config_manifest.json`: source hashes, configs, prototype block configs,
  method counts, data hash if available, git status, Python/platform metadata;
- `paired_stats.csv`: seed-paired baseline-vs-candidate tables with CIs,
  paired effect sizes, paired t-test p-values, Wilcoxon p-values, sign-test
  p-values, win/loss/tie counts, and Holm correction;
- `paperworthy_report.json`: machine-readable report payload;
- `paperworthy_report.md`: human-readable audit, primary/secondary tables,
  compute table, failure-mode flags, and reconstructed commands.

Default inputs are the current canonical replay-capped 10-seed runs:

- `3000` online steps, `gate_max=0.15`;
- `5000` online steps, `gate_max=0.15`;
- `10000` online steps, `gate_max=0.15`.

The default confirmatory family is deliberately narrow:

- method: post-FFN replay-capped memory;
- metrics: `final_window_nll`, `eval_nll`;
- horizons: all supplied result artifacts;
- correction: Holm over that family.

Pre-KV placement, fast-only deployment metrics, perplexity, accuracy, and
timing are reported as secondary or exploratory.

## Confirmatory Protocol

Use this protocol before making a paper claim.

1. Freeze the data before running:
   - training byte range for online updates;
   - validation byte range for all tuning;
   - lockbox test byte range used once after the method and metrics are frozen.
2. Freeze the primary hypothesis:
   - candidate: replay-capped post-FFN advantage memory;
   - baseline: tuned FFN transformer;
   - metrics: final-window prequential NLL and held-out NLL;
   - horizons: `3000`, `5000`, and `10000` online steps;
   - correction: Holm within the six primary tests.
3. Run at least 30 paired seeds:
   - same initialization seed for baseline and candidate;
   - same online stream offset per seed;
   - same evaluation offset per seed;
   - store every offset in the artifact.
4. Add required baselines:
   - tuned FFN;
   - parameter-matched FFN;
   - compute-matched FFN or wider FFN trained under equal wall-clock;
   - no-replay memory;
   - no-cap replay memory;
   - cap-only or fixed-gate memory;
   - pre-KV placement as secondary.
5. Report the statistical table:
   - paired mean diff where positive favors memory;
   - 95% CI for paired diff;
   - paired Cohen `dz`;
   - paired t-test and Wilcoxon p-values;
   - sign-test p-value;
   - win/loss/tie counts;
   - Holm-corrected p-values;
   - smallest effect of interest and whether the CI clears it.
6. Report compute:
   - trainable params and bytes;
   - runtime state bytes, including replay buffer;
   - compile time and hot-loop time;
   - steps/sec after warmup;
   - device, backend, JAX/JAXLIB versions, CPU/GPU model;
   - peak memory if available.
7. Enforce failure thresholds:
   - any NaN or non-finite metric fails;
   - held-out NLL regression fails the primary claim for that horizon;
   - open gate with negative final-window advantage is a flagged mechanism failure;
   - memory allocation saturation is flagged;
   - outlier seeds are reported, not dropped.

## Minimum Acceptance Bar

The replay-capped memory result should not be promoted beyond exploratory until
all of these are true:

- 30 paired seeds pass the frozen protocol;
- post-FFN memory has positive paired diff on both primary metrics at all
  horizons;
- Holm-corrected primary p-values support the directional claim, or the paper
  states that the result is descriptive rather than statistically confirmed;
- the 95% CI clears the smallest effect of interest on held-out NLL for at
  least the main horizon;
- parameter-matched and compute-matched baselines do not explain the win;
- lockbox held-out NLL is positive for the final frozen config;
- complete manifests make the runs exactly reproducible.

## Current Assessment

The existing follow-up can support this wording:

> In exploratory Tiny Shakespeare runs, replay-gated slow memory under a tight
> resource cap produced small but consistent paired improvements over the tuned
> FFN baseline across three horizons.

It should not yet support this wording:

> Replay-capped advantage memory transformers beat tuned fast transformers.

That stronger claim requires the confirmatory protocol above.
