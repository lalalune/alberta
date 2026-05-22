# Step 2 Class-Blocked UPGD Plasticity Ablation

Date: 2026-05-05.

## Scope

This note tracks the class-blocked digits stability/plasticity ablation for
target-density UPGD.  The ablation axes are repeated-target head plasticity,
head-only versus trunk+head meta-plasticity, and retained held-out accuracy.

The latest 30-seed source is:

`output/subagents/combined_upgd/digits_density_repetition_compromise_30seed/SUMMARY.md`

## Named Configs

The runnable harness configs live in:

`examples/The Alberta Plan/Step2/step2_upgd_ablation.py`

Preset: `class_blocked_retention`.

It includes:

- repetition off: `upgd_density_sigma1e_4_adaptk035_065_lr06_repx0_notrunk_tight`
- fixed repetition `0.25`: `upgd_density_sigma1e_4_adaptk035_065_lr06_repx025`
- fixed repetition `0.25` plus head-only meta: `upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight`
- fixed repetition `0.75`: `upgd_density_sigma1e_4_adaptk035_065_lr06_repx075`
- fixed repetition `0.75` plus head-only meta: `upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight`
- learned readout multiplier, head-only: `upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight`
- learned readout multiplier, trunk+head: `upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_trunk_head_tight`
- learned repetition/readout multiplier: `upgd_density_sigma1e_4_adaptk035_065_lr06_rep_learned_notrunk_tight`

## Current Recommendation

For the class-blocked compromise, prefer:

`upgd_density_sigma1e_4_adaptk035_065_lr06_repx025_meta001_notrunk_tight`

Against `mlp64` on 30 class-blocked seeds, it is the only checked branch here
with positive online final-window MSE, positive final-window accuracy, positive
test MSE, and positive retained test accuracy:

| Metric | Paired diff | Wins/Losses |
|---|---:|---:|
| Final-window MSE | `+0.000103 +/- 0.000060` | `18/12` |
| Final-window accuracy | `+0.000556 +/- 0.000661` | `11/7` |
| Test MSE | `+0.002281 +/- 0.001085` | `21/9` |
| Test accuracy | `+0.005318 +/- 0.002991` | `16/3` |

The stronger online tracker,
`upgd_density_sigma1e_4_adaptk035_065_lr06_repx075_meta001_notrunk_tight`,
wins class-blocked final-window MSE in all 30 seeds but has negative retained
test accuracy (`-0.005937 +/- 0.002493`) and negative test MSE
(`-0.002929 +/- 0.001029`).

The strongest retained branch,
`upgd_density_sigma1e_4_adaptk035_065_lr06_meta003_notrunk_tight`, improves
class-blocked test accuracy (`+0.019790 +/- 0.004052`) but loses online
final-window MSE in all 30 seeds (`-0.001029 +/- 0.000071`).

## Commands

Small smoke:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite digits \
  --preset class_blocked_retention \
  --class-blocked-mode blocked \
  --steps 300 \
  --n-seeds 1 \
  --seed 1020 \
  --final-window 100 \
  --output-dir output/subagents/combined_upgd/digits_class_blocked_retention_smoke
```

Full class-blocked rerun:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite digits \
  --preset class_blocked_retention \
  --class-blocked-mode blocked \
  --steps 1200 \
  --n-seeds 30 \
  --seed 1020 \
  --final-window 300 \
  --output-dir output/subagents/combined_upgd/digits_class_blocked_retention_30seed
```

Full five-regime digits rerun:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_upgd_ablation.py" \
  --suite digits \
  --preset class_blocked_retention \
  --class-blocked-mode both \
  --steps 1200 \
  --n-seeds 30 \
  --seed 1020 \
  --final-window 300 \
  --output-dir output/subagents/combined_upgd/digits_class_blocked_retention_both_30seed
```

## Unresolved Risk

The retained-test risk is still open: class-blocked online tracking and held-out
retention are not optimized by the same repetition setting.  Repetition `0.75`
is the clean online tracker but gives back retained accuracy; learned head-only
meta is the clean retained branch but loses class-blocked online MSE.  The
`0.25` plus head-only meta branch is therefore a compromise, not closure.

## 2026-05-06 Softmax Closeout

The stricter `MLP(64,64)` class-blocked online-MSE gap is now closed by the
64-64 target-structure softmax branch:

`upgd64_64_structure_sigma1e_4_adaptk035_065_lr07_repx075_meta001_softmax_notrunk_tight`

Source:
`output/subagents/class_blocked_retention/softmax_closeout_10seed_6000/digits_ablation_SUMMARY.md`.

| Method | Final-window MSE | Wins vs `MLP(64,64)` | Retained test accuracy |
|---|---:|---:|---:|
| softmax UPGD 64-64 lr07 | `0.0019926` | `6/10` | `0.8692 +/- 0.0080` |
| `MLP(64,64)` | `0.0019993` |  | `0.1007 +/- 0.0006` |

This turns class-blocked from a blocker into a readout-selection problem. The
same softmax branch wins IID, class-blocked, mask-noise, and permuted digit
final-window MSE in the five-regime check, but loses label drift:

`output/subagents/class_blocked_retention/softmax_lr07_allregime_10seed_6000/digits_ablation_SUMMARY.md`.

Label drift still prefers the linear-MSE 64-64 structure branch recorded in the
adaptive structure rerun. The next mechanism should therefore be a causal
readout rule that can favor softmax under repeated/stable class blocks and
linear-MSE-style updates under label remapping drift.
