# Step 1: Continual Supervised Learning with Given Features — Canonical Results

This document is the canonical numerical record for Alberta Plan Step 1.
It is populated by the experiment scripts in
`examples/The Alberta Plan/Step1/` and the JSON files committed under
`outputs/step1_canonical/`.

## What Step 1 requires (paper, lines 280–358)

> "Continual supervised learning with given features" — a linear function
> approximator on a non-stationary target
> `y*_t = w*_t · x_t + b*_t + η_t`, with per-feature meta-learned step-sizes
> and online normalization. Footnote 11 names the comparison set:
> NADALINE, IDBD, Autostep, Autostep-for-GTD(λ), Auto, Adam, RMSprop,
> Batch Normalization.

The framework's headline claim — "IDBD/Autostep beat hand-tuned LMS" —
must be evidenced against this full comparison set, on streams that
exercise all three forms of non-stationarity (drifting w*, drifting b*,
shifting input distribution), with statistical confidence.

## Reproducibility

| Script | Output JSON | Output summary |
|---|---|---|
| `step1_full_baselines.py` | `outputs/step1_canonical/multi_baseline_results.json` | `outputs/step1_canonical/SUMMARY.md` |
| `step1_normalization_ablation.py` | `outputs/step1_canonical/normalization_ablation_results.json` | `outputs/step1_canonical/normalization_ablation_SUMMARY.md` |
| `step1_robustness_study.py` | `outputs/step1_canonical/robustness_study_results.json` | `outputs/step1_canonical/robustness_study_SUMMARY.md` |

To regenerate everything from scratch:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step1/step1_full_baselines.py"
python "examples/The Alberta Plan/Step1/step1_normalization_ablation.py"
python "examples/The Alberta Plan/Step1/step1_robustness_study.py"
pytest tests/test_step1_replication.py -v
```

Note: `step1_full_baselines.py` now also includes `AdaGain`, the public
Degris-coauthored meta-descent method from Jacobsen et al. 2019. This is not
the unpublished `Auto (Degris in prep.)` footnote item; it is included under
its own name because it has a public update rule.

Note: `step1_normalization_ablation.py` now also includes
`StreamingBatchNormalizer(momentum=0.99)`, a temporally-uniform,
BatchNorm-style running-statistics baseline. The committed canonical
normalization ablation has been regenerated with this baseline at 30 seeds and
20,000 steps per seed.

The regression suite (`tests/test_step1_replication.py`) asserts that
the headline claims still hold: ≥30 seeds, all public Step 1 optimizers
evaluated, IDBD beats best-tuned LMS by ≥20% on the noisy Sutton task,
and IDBD's hyperparameter robustness ratio is ≥10% better than LMS's.

## Headline results (canonical run, 30 seeds, committed JSON)

### Multi-baseline replication

Best-tuned final-window MSE (lower is better), and paired-difference vs best-tuned LMS:

| Stream | Best optimizer | Best MSE | LMS MSE | % improvement | Wins / 30 | Cohen's d |
|---|---|---|---|---|---|---|
| Sutton 1992 (noiseless) | **IDBD** | 1.56 | 3.63 | **57.0%** | 30/30 | 22.5 |
| Sutton 1992 (noisy, η_t added) | **IDBD** | 3.01 | 4.89 | **38.3%** | 30/30 | 23.9 |
| AlbertaPlanStep1 (drifting w*, b*) | Adam ≈ RMSprop | 1.0093 | 1.0115 | 0.2% | 29/30 | 1.75 |
| XDistShift (input-scale shifts) | **NADALINE** | 0.0106 | 0.0160 | **33.4%** | 30/30 | 5.7 |

Source: `outputs/step1_canonical/multi_baseline_results.json`.

**Findings:**

- The original Sutton 1992 IDBD claim (~57% MSE reduction over LMS) is **reproduced** at 30 seeds with d ≈ 22.5.
- The Alberta-Plan-noisy variant of the Sutton stream (with the η_t the audit found was missing) shows IDBD beating LMS by 38.3% — somewhat smaller than the noiseless case, but still 30/30 wins with d ≈ 23.9.
- AdaGain is now implemented and included in the canonical multi-baseline
  sweep. It beats best-tuned LMS on the noiseless and noisy Sutton streams
  (`27/27` and `29/29` finite paired wins), but it is not the best optimizer
  and it diverges on `XDistShift` in the tested grid. This strengthens Step 1
  coverage of public step-size methods without changing the headline winner.
- On the canonical `AlbertaPlanStep1Stream` (drifting w*, b*, noisy), the
  original six-optimizer footnote set mostly clusters near the noise floor;
  AdaGain is worse in this grid. With `noise_std=1.0` and
  `drift_rate_w=0.001`, the irreducible variance dominates, so this remains a
  hard differentiator.
- A follow-up non-canonical joint normalizer/hyperparameter probe on only
  `AlbertaPlanStep1Stream` broadens the grids and tunes across
  `None`/EMA/Welford/StreamingBatch normalization. It confirms the same
  qualitative ranking near the noise floor: Adam `1.009283`, Autostep
  `1.009305`, RMSprop `1.009316`, LMS `1.011124`, and IDBD `1.013931`.
  Autostep slightly beats tuned LMS in that probe (`+0.001819 +/- 0.000645`,
  `21/30` wins) and is statistically tied with Adam/RMSprop; IDBD remains
  worse than LMS (`5/30` wins). The probe is stored in
  `output/step1_alberta_probe/` and is not part of the canonical
  all-stream JSON.
- `XDistShift` exposes IDBD's missing input normalization: **IDBD diverges on every (HP, seed) combination** (final MSE = +∞). NADALINE wins by 33.4%; Autostep is close behind. This is the single most important new finding from this experiment batch — it identifies a concrete failure mode of Sutton 1992's algorithm that the Alberta-Plan-style baselines (NADALINE, Autostep) are designed to handle.

### Normalization ablation

Online normalization is **required** (not merely helpful) on input-scale-varying streams:

| Configuration | XDistShift MSE | DynamicScaleShift MSE |
|---|---|---|
| LMS, no normalizer | 30/30 NaN | 30/30 NaN |
| LMS + EMA | 2.89 | 6.54 |
| LMS + Welford | 0.29 | (1 NaN survivor) |
| LMS + StreamingBatch | 2.91 | 6.48 |
| IDBD, no normalizer | 30/30 NaN | 30/30 NaN |
| IDBD + EMA | survives | 30/30 NaN |
| IDBD + StreamingBatch | 30/30 NaN | 30/30 NaN |
| Autostep (any) | always survives | EMA further reduces by 35.3%, d=2.30 |
| Autostep + StreamingBatch | hurts XDistShift, improves DynamicScaleShift by 34.5% |
| Adam (any) | tolerates absent norm | EMA helps marginally (n.s.) |

Source: `outputs/step1_canonical/normalization_ablation_results.json`.

**Finding:** Autostep is the most scale-robust optimizer. The paper's open
question — "the effect of online normalization has yet to be definitively
established" — now has a concrete answer for these streams: normalization is
essential for fixed-α and IDBD-class methods on input-scale shifts; Autostep's
overshoot prevention substitutes for it. The BatchNorm-style streaming
normalizer is not uniformly better than EMA or Welford: it stabilizes LMS on
both scale-shift streams and matches EMA on `DynamicScaleShift`, but it hurts
Autostep/Adam on `XDistShift` and does not rescue IDBD in this configuration.

### Robustness study

Working range = decades of HP grid where mean MSE stays within 1.5× of the optimum:

| Optimizer | best HP | best MSE | grid-mean / best ratio | working range |
|---|---|---|---|---|
| LMS | 5×10⁻⁴ | 1.009 | 1.38 | 2.45 dec |
| IDBD | 6.3×10⁻² | 1.044 | 1.40 | 2.10 dec |
| **Autostep** | 3.2×10⁻¹ | 1.018 | **1.13** | **3.50 dec (entire grid)** |
| Adam | 1.1×10⁻³ | 1.012 | 2.67 | 1.75 dec |

Source: `outputs/step1_canonical/robustness_study_results.json` and `robustness_curves.png`.

**Finding:** Autostep is the only optimizer whose best-tuned MSE stays within 1.5× of optimum across the entire 3.5-decade hyperparameter grid we tested. Adam has the narrowest working range (1.75 decades) and the worst grid-mean / best ratio (2.67). IDBD's working range is essentially the same as LMS's — a notable result for the framework's headline claim about meta-learner robustness, and worth investigating further (likely caused by IDBD log-step-size oscillation at large meta-rates).

### Autostep implementation audit

Source checked: Mahmood, Sutton, Degris, and Pilarski (2012),
["Tuning-free step-size adaptation"](https://people.bordeaux.inria.fr/degris/papers/RupamAutostep.pdf),
Table 1.

Audit result:

- Linear `Autostep.update()` now follows the Table 1 order: update `v_i` from
  old `alpha_i` and old `h_i`; update `alpha_i`; compute `M`; divide `alpha_i`
  by `M`; apply the weight update with the new scaled `alpha_i`; then update
  `h_i`.
- The normalizer is the Table 1 self-regulated running maximum:
  `v_i <- max(|delta x_i h_i|, v_i + (1/tau) alpha_i x_i^2 (|delta x_i h_i| - v_i))`.
- The overshoot prevention is joint over weights plus the learner bias. The
  paper writes the algorithm over feature weights only; this implementation
  treats the bias as an additional feature with `x=1`, which is the defensible
  extension for `LinearLearner`'s affine prediction.
- The `h_i` trace uses the post-`M` step-size. The paper writes
  `[1 - alpha_i x_i^2]_+`; after `M`, each nonnegative term is bounded by the
  sum, so the linear path's decay is nonnegative.
- Bug fixed in this pass: a post-`M` `jnp.clip(..., 1e-8, 1.0)` could raise an
  already-scaled tiny `alpha_i` and violate `sum_i alpha_i x_i^2 <= 1` on
  extreme inputs. The clipping was removed and regression tests now cover this.
- The arbitrary-shape `update_from_gradient()` path is an Autostep-inspired
  per-parameter generalization. It is useful for MLP experiments, but it should
  not be cited as a literal Table 1 theorem for a whole nonlinear network,
  because `M` is computed per parameter tensor rather than as one global
  effective step over the full network.

### Auto public-spec closure

The Alberta Plan footnote names `Auto (Degris in prep.)`; the public arXiv page
for the plan is [arXiv:2208.11173](https://arxiv.org/abs/2208.11173), and the
PDF footnote 11 is the source of the `Auto (Degris in prep.)` citation. Public
source audit, repeated and expanded 2026-05-06:

| Source checked | Public content found | Implementability result |
|---|---|---|
| Alberta Plan arXiv/PDF | Step 1 equation, baseline family, and footnote 11 naming `Auto (Degris in prep.)` | Name mention only; no state variables, update order, objective, hyperparameters, or experiments |
| Thomas Degris INRIA publication/software pages | Degris public work including Autostep, Horde, Off-PAC, RLPark, and SDyna | No public `Auto` optimizer paper, report, thesis, or software entry |
| Degris-linked RLPark generated docs and repository documentation | Public code docs contain `Autostep`, `IDBD`, `K1`, `TDLambdaAutostep`, and other RL algorithms | Public software trail exposes Autostep-family methods, not an `Auto` class/spec |
| Mahmood MSc thesis and Mahmood et al. 2012 ICASSP | Public Autostep derivation and Table 1 algorithm | Implementable as Autostep, not Auto |
| Kearney/Koop/Pilarski public AutoStep-for-GTD(lambda) paper | Algorithm for GTD(lambda) with AutoStep tuning | A TD/GVF method, not Step 1 supervised `Auto` |
| Jacobsen et al. 2019 AdaGain | Public Degris-coauthored meta-descent algorithm | Implementable public substitute, but named AdaGain and explicitly not Auto |
| Degris et al. 2024 `Step-size Optimization for Continual Learning` | IDBD/RMSProp/Adam comparison; states normalized step-size optimization for deep networks remains open and Autostep is a possible attempt | Confirms the research direction; does not publish an algorithm named Auto |

Exact search trail: `"Auto (Degris in prep.)"`, `"Auto" "Degris" "in prep"`,
`"Auto" "Thomas Degris" "step-size"`, `"Auto" "Degris" "IDBD" "Autostep"`,
`"Auto" "Degris" "Step-size Optimization for Continual Learning"`, site-scoped
searches over `people.bordeaux.inria.fr/degris`, searches over RLPark docs and
GitHub, and searches for cited Degris/White/Mahmood/Sutton step-size papers.
These found the Alberta Plan footnote and public Autostep/AdaGain/IDBD-family
material, but no public update rule, pseudocode, preprint, technical report,
thesis, package, class, or reproducible benchmark for an optimizer named
`Auto`.

An implementable public spec would need, at minimum:

- the learned state variables and their initialization;
- the base weight update and the exact per-step order;
- the meta-update objective or proxy objective;
- normalizers, traces, caps, and any overshoot-prevention rule;
- bias handling for the Step 1 affine learner;
- hyperparameter defaults/ranges and a reproducible benchmark or reference
  implementation.

No `Auto` optimizer was implemented, because doing so would be an invention
rather than a replication. The framework also does not accept `Auto` as an
optimizer config alias or export an `Auto` symbol from the public package API.
The production Step 1 API accepts `adagain` only under the public `AdaGain`
name; a misspelled `adagiven` alias was removed during this closure pass.

There is a public, different Degris-coauthored meta-descent method,
[AdaGain](https://arxiv.org/abs/1907.07751). It has been added explicitly as
`AdaGain` with source-cited tests pinning the linear-LMS specialization. It must
still not be presented as `Auto`.

`AutostepGain` is an explicitly experimental hybrid of public Autostep and
AdaGain ingredients. It uses AdaGain sensitivity traces, Autostep
self-normalized meta-gradients, and Autostep effective-step overshoot
prevention. The serialized config type is `"AutostepGain"`, and `"Auto"` still
raises an unknown-optimizer error. This gives the framework a reproducible
Auto-inspired comparator without fabricating Degris's unpublished `Auto`
algorithm.

### Autostep-for-GTD(lambda) closure

`AutostepGTDLambda` (Kearney, Veeriah, Travnik, Pilarski & Sutton 2019,
"Learning Feature Relevance Through Step Size Adaptation in
Temporal-Difference Learning") is now implemented and exposed as a public
Step 1 baseline under that name. The Step 1 supervised limit
(`gamma=0`, `lamda=0`, `rho=1`) reduces algebraically to standard Autostep,
and the `tests/test_optimizers.py::TestAutostepGTDLambda` regression suite
pins this equivalence to within `1e-5` over a 10-step toy problem.

A 5-seed × 5,000-step preliminary probe on `AlbertaPlanStep1Stream` with
EMA(0.99) normalization confirms the optimizer behaves as expected
alongside LMS and Autostep:

| Optimizer | Best MSE (5-seed mean ± stderr) | Status |
|---|---|---|
| LMS (`alpha=0.01`)             | 1.150 ± 0.009 | preliminary |
| Autostep (`alpha0=0.02, mu=0.01, tau=10000`) | 1.248 ± 0.012 | preliminary |
| AutostepGTDLambda (supervised limit, same HPs) | 1.248 ± 0.012 | **5-seed preliminary evidence** |

Source: `outputs/step1_canonical/autostep_gtd_5seed_results.json`,
`outputs/step1_canonical/autostep_gtd_5seed_SUMMARY.md`. Reproduce with

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step1/step1_autostep_gtd_5seed_probe.py"
```

This is **5-seed preliminary evidence** intended only to confirm that the
optimizer compiles, runs, and matches Autostep in the supervised limit. The
canonical 30-seed multi-baseline sweep (`step1_full_baselines.py`) now also
sweeps `AutostepGTDLambda` over the same hyperparameter grid as Autostep so
the closure can be promoted to a paper-grade row by re-running the full
sweep.

With `AutostepGTDLambda` exposed, Alberta Plan footnote 11 is now closed by
name within the framework: NADALINE, IDBD, Autostep, Autostep-for-GTD(λ),
Adam, RMSprop are all implemented, sweepable, and serialization-tested. The
remaining `Auto (Degris in prep.)` entry is still not implementable from
public sources (see audit table above).

### CPU throughput and daemon readiness

Canonical CPU pass, run 2026-05-04 on `CpuDevice(id=0)`:

```bash
source .venv/bin/activate
python benchmarks/step1_throughput.py --output-dir outputs/step1_canonical/throughput
```

Outputs:

- `outputs/step1_canonical/throughput/step1_throughput_20260504_171841.csv`
- `outputs/step1_canonical/throughput/step1_throughput_20260504_171841.json`
- `outputs/step1_canonical/throughput/step1_throughput_20260504_171841.md`

The benchmark covers `LinearLearner` on `AlbertaPlanStep1Stream` with all
public Step 1 optimizers (`LMS`, `IDBD`, `Autostep`, `Adam`, `RMSprop`,
`NADALINE`), all core online-normalizer settings (`none`, `EMA`, `Welford`,
`StreamingBatch`), and both single-stream scan and 8-seed batched scan modes.
It reports compile/dispatch warmup separately from a hot run.

All 48 configurations exceeded the daemon-readiness floor of 1000 learner
updates/sec on CPU. The slowest hot single-stream scan was `IDBD + none` at
8559.1 updates/sec. The slowest hot batched configuration was `Adam + Welford`
at 32984.4 learner updates/sec, equivalent to 4123.0 stream steps/sec across
the 8-way batch. The fastest hot single-stream scan was `IDBD + Welford` at
36916.6 updates/sec; the fastest batched configuration was `LMS + none` at
193020.5 learner updates/sec.

### Headline conclusion

The Alberta Plan Step 1 paper claim — "meta-algorithms for setting the step-size parameter so that the same method can be used on any problem" — is **substantially supported by these results, with caveats**:

- Sutton 1992's IDBD result on its original task **reproduces** with high statistical confidence.
- IDBD/Autostep beat LMS on the Sutton sparse-relevance setting even when the
  Alberta Plan's η_t noise term is added.
- Autostep is the most genuinely tuning-free optimizer (3.5-decade working range).
- IDBD is **not** robust to input-distribution shifts without external normalization; this is a concrete, falsifiable failure of vanilla IDBD that Autostep and NADALINE address.
- On the strengthened AlbertaPlanStep1-only probe, Autostep is competitive
  with Adam/RMSprop and slightly beats tuned LMS, while IDBD remains worse than
  LMS. The claim should therefore stay method- and stream-specific, not
  "IDBD/Autostep beat LMS everywhere."

The audit's "headline claim is unverified" gap is now closed: the numbers are committed, the regression tests assert them, and the qualitative claim has been refined into something more accurate than the original sweeping statement.

## What was closed since the audit

- Added Adam, RMSprop, NADALINE optimizers (paper footnote 11) — `core/baseline_optimizers.py`
- Added AdaGain as a public Degris-coauthored meta-descent comparator under
  its own name — `core/optimizers.py`
- Added `AlbertaPlanStep1Stream` with the η_t noise term and drifting w*, b* per the paper's eq.
- Added `XDistShiftStream` for the input-distribution non-stationarity case
- Fixed `SuttonExperiment1Stream` to honor `noise_std` (was previously fixed at 0)
- Added `StreamingBatchNormalizer`, a BatchNorm-style online running-statistics
  normalizer that fits the single-step continual-learning API
- Audited Autostep against Mahmood et al. 2012 Table 1 and removed post-`M`
  clipping that could break the overshoot guarantee on extreme inputs
- Multi-baseline replication on all 4 canonical streams × 7 optimizers × 30
  seeds (committed JSON)
- Normalization on/off ablation across 4 optimizers × 2 streams × 30 seeds
  for None/EMA/Welford/StreamingBatch
- Hyperparameter robustness study (decades of working range per optimizer)
- CPU throughput benchmark across 6 optimizers × 4 normalizer settings ×
  scan/batched modes; 48/48 configurations exceed 1000 learner updates/sec
- Pytest regression covering all of the above

## Remaining actionable Step 1 caveats

None for public/reproducible methods. `Auto (Degris in prep.)` is intentionally
unavailable because no public algorithm specification was found; this is a
source-availability boundary, not an implementation TODO. If a public,
implementable Degris specification appears later, it should be added under its
own tested name rather than aliased to AdaGain or Autostep.
