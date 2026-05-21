# Step 2 Class-Blocked Retention Follow-up

Date: 2026-05-06.

## Question

Can a simple single UPGD learner clear both class-blocked final-window tracking
and held-out retention against `mlp64` and `mlp64_64`, without portfolios,
replay, or MLP fallback?

Short answer: no. The best simple variants still split the objective. Several
clear parts of the `mlp64` comparison, but none clear `mlp64_64` final-window
tracking, and none simultaneously clear `mlp64` held-out MSE and held-out
accuracy.

Positive differences below favor the UPGD variant. For MSE this is
`MLP - UPGD`; for accuracy this is `UPGD - MLP`.

## Runs

Artifacts:

- `output/subagents/class_blocked_retention/repro_3seed`
- `output/subagents/class_blocked_retention/followup_3seed`
- `output/subagents/class_blocked_retention/slowutil_10seed`

All runs used `examples/The Alberta Plan/Step2/step2_upgd_ablation.py`,
`digits_regimes=class_blocked`, seeds from `1020`, `steps=1200`, and
`final_window=300`.

## Ten-Seed Extension

This extended only the most balanced three-seed variant, slow utility memory,
against the original compromise.

| Method | Final MSE | Final acc | Test MSE | Test acc | d final MSE vs `mlp64` | d final MSE vs `mlp64_64` | d test MSE vs `mlp64` | d test acc vs `mlp64` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mlp64_64` | 0.002817 +/- 0.000055 | 0.991000 +/- 0.000711 | 0.163014 +/- 0.003592 | 0.099443 +/- 0.000410 | +0.002063 (10/10) | 0.000000 | -0.033092 (0/10) | -0.025974 (0/10) |
| `upgd_density...repx025_meta001_slowutil999_notrunk_tight` | 0.004760 +/- 0.000132 | 0.986667 +/- 0.000861 | 0.127101 +/- 0.002586 | 0.122078 +/- 0.008723 | +0.000121 (8/10) | -0.001943 (0/10) | +0.002821 (7/10) | -0.003340 (4/10) |
| `mlp64` | 0.004881 +/- 0.000114 | 0.985333 +/- 0.000737 | 0.129922 +/- 0.002289 | 0.125417 +/- 0.009469 | 0.000000 | -0.002063 (0/10) | 0.000000 | 0.000000 |
| `upgd_density...repx025_meta001_notrunk_tight` | 0.004895 +/- 0.000103 | 0.987000 +/- 0.000923 | 0.130059 +/- 0.002220 | 0.130241 +/- 0.010835 | -0.000014 (6/10) | -0.002077 (0/10) | -0.000136 (6/10) | +0.004824 (4/10) |

Read:

- `mlp64_64` is the tracking bar in this harness. Neither UPGD variant beats it
  on final-window MSE or final-window accuracy.
- Slow utility memory improves final-window MSE and test MSE versus `mlp64`,
  but loses held-out test accuracy versus `mlp64`.
- The original compromise has the better held-out test accuracy versus `mlp64`,
  but does not clear final-window MSE or held-out test MSE versus `mlp64`.

## Three-Seed Variant Screen

| Variant | d final MSE vs `mlp64` | d final MSE vs `mlp64_64` | d test MSE vs `mlp64` | d test acc vs `mlp64` | Read |
|---|---:|---:|---:|---:|---|
| `repx05_meta001` | +0.000228 (2/3) | -0.001514 (0/3) | -0.007007 (0/3) | +0.000000 (2/3) | More tracking, worse retained MSE. |
| `repx025_biasmeta001` | +0.000195 (2/3) | -0.001547 (0/3) | -0.002742 (2/3) | +0.001237 (1/3) | Bias-only meta helps online acc, not retention enough. |
| `repx025_meta001_margin_tiny` | +0.000139 (2/3) | -0.001603 (0/3) | -0.002428 (1/3) | +0.004329 (1/3) | Margin helps test acc, costs retained MSE. |
| `repx025_meta001_slowutil999` | +0.000070 (2/3) | -0.001672 (0/3) | +0.002901 (3/3) | +0.000000 (1/3) | Best balanced screen; extended above. |
| `repx025_meta001` | -0.000259 (1/3) | -0.002001 (0/3) | -0.003769 (1/3) | -0.001237 (1/3) | Original compromise did not reproduce as a clear win here. |

The known extremes reproduced the expected shape in the separate three-seed
run: `repx075_meta001` improved final-window MSE versus `mlp64`
(`+0.000504`, 3/3) but not versus `mlp64_64` and lost test MSE versus `mlp64`;
`meta003_notrunk_tight` improved held-out test MSE and accuracy versus both
MLPs but lost final-window tracking.

## Mechanism Hypothesis

The class-blocked weakness is not primarily a hidden-feature utility problem.
Making utility memory slower (`utility_decay=0.999`) reduced destructive
perturbation churn enough to improve held-out MSE, but it did not recover
held-out accuracy and still failed the deeper MLP tracking bar.

The dominant tradeoff is output-head plasticity under repeated one-hot blocks:
fixed repetition and margin/bias variants help the current block, while the
retained classifier needs old class logits to remain calibrated. `mlp64_64`
solves current-block tracking by capacity and adaptation but collapses held-out
retention to near chance, so clearing it on tracking and `mlp64` on retention
requires an explicit anti-drift mechanism, not just another repetition gain.

## Recommendation

Do not promote any new simple variant from this workstream.

If a single learner must be chosen today, keep the current compromise
`upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight` as
the least bad retained-accuracy branch, with the caveat that it does not clear
the official `mlp64_64` tracking comparator and did not clear `mlp64` final MSE
in the 10-seed extension.

The next patch should target output-head anti-drift directly: bias centering or
bias decay under repeated class blocks, ideally coupled to the existing
single-learner repetition pressure. Slow utility decay is useful as a secondary
stabilizer, but the evidence says utility memory alone is not enough.

## Commands

```bash
.venv/bin/python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite digits \
  --upgd-configs upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight,upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight,upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight \
  --digits-regimes class_blocked \
  --steps 1200 \
  --n-seeds 3 \
  --seed 1020 \
  --final-window 300 \
  --output-dir output/subagents/class_blocked_retention/repro_3seed
```

```bash
.venv/bin/python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite digits \
  --preset class_blocked_retention_followup \
  --digits-regimes class_blocked \
  --steps 1200 \
  --n-seeds 3 \
  --seed 1020 \
  --final-window 300 \
  --output-dir output/subagents/class_blocked_retention/followup_3seed
```

```bash
.venv/bin/python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite digits \
  --upgd-configs upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight,upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_slowutil999_notrunk_tight \
  --digits-regimes class_blocked \
  --steps 1200 \
  --n-seeds 10 \
  --seed 1020 \
  --final-window 300 \
  --output-dir output/subagents/class_blocked_retention/slowutil_10seed
```

```bash
.venv/bin/python -m pytest tests/test_step2_upgd_ablation_configs.py -q
```
