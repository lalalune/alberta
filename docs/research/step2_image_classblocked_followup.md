# Step 2 Image Class-Blocked Follow-Up

Date: 2026-05-06

## Scope

This sweep targeted the real CIFAR-10 `class_blocked` gap from
`docs/research/step2_image_breadth_upgd.md`: promoted no-portfolio UPGD kept a
held-out accuracy advantage over MLP, but tracked the current class block worse
on final-window MSE/NLL.

The runner is standalone:
`output/subagents/image_gap_classblocked/run_image_gap_sweep.py`.
It reuses the no-portfolio image breadth data factories and metric accounting
from `output/subagents/external_breadth_image/run_no_portfolio_image_breadth.py`.
No core UPGD files, example scripts, portfolios, replay buffers, MLP fallback
routes, or task-id inputs were used.

## Commands

Smoke:

```bash
.venv/bin/python output/subagents/image_gap_classblocked/run_image_gap_sweep.py \
  --scenarios cifar_class_blocked \
  --variants promoted_softmax_h32,softmax_hidden_plus_input \
  --steps 20 \
  --n-seeds 1 \
  --final-window 5 \
  --cifar-max-train 100 \
  --cifar-max-test 50 \
  --output-dir output/subagents/image_gap_classblocked/smoke \
  --result-prefix smoke
```

Main screen:

```bash
.venv/bin/python output/subagents/image_gap_classblocked/run_image_gap_sweep.py \
  --scenarios cifar_class_blocked,cifar_iid \
  --steps 600 \
  --n-seeds 3 \
  --final-window 200 \
  --cifar-max-train 2000 \
  --cifar-max-test 500 \
  --output-dir output/subagents/image_gap_classblocked/screen_3seed_600 \
  --result-prefix image_gap_screen_3seed_600
```

Focused margin follow-up:

```bash
.venv/bin/python output/subagents/image_gap_classblocked/run_image_gap_sweep.py \
  --scenarios cifar_class_blocked \
  --variants promoted_softmax_h32,softmax_margin0p5_m0p01,softmax_margin0p2_m0p0025,softmax_margin0p2_m0p005,softmax_margin0p2_m0p01,softmax_margin0p5_m0p005,softmax_margin1p0_m0p005,softmax_no_perturb,softmax_perturb_int1 \
  --steps 600 \
  --n-seeds 3 \
  --final-window 200 \
  --cifar-max-train 2000 \
  --cifar-max-test 500 \
  --output-dir output/subagents/image_gap_classblocked/focused_margin_3seed_600 \
  --result-prefix image_gap_focused_margin_3seed_600
```

## Data

- CIFAR source: local `data/cifar-10-batches-py`, loaded through the existing
  direct CIFAR archive path as `cifar10_python_archive`.
- Real CIFAR evidence: `true`.
- Train/test subset: 2,000 train examples and 500 held-out test examples.
- Online stream: 600 steps, 3 seeds, final window 200.

## Class-Blocked Results

Positive paired deltas below favor the UPGD variant against
`promoted_softmax_h32`. Test accuracy is also compared to the best same-run MLP
held-out accuracy, which was `0.1180`.

| Method | FW MSE | FW NLL | FW Acc | Test MSE | Test NLL | Test Acc | Time s | FW NLL delta | Test acc vs MLP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp_h32` | 0.002045 | 1.473203 | 0.9967 | 0.177142 | 2.342022 | 0.1180 | 1.81 | n/a | n/a |
| `mlp_h64` | 0.002569 | 1.466149 | 1.0000 | 0.176194 | 2.340992 | 0.1180 | 1.94 | n/a | n/a |
| `promoted_softmax_h32` | 0.068632 | 1.579984 | 0.6167 | 0.104671 | 2.639659 | 0.1673 | 1.26 | +0.000000 | +0.0493 |
| `softmax_margin0p2_m0p0025` | 0.062962 | 1.444099 | 0.6683 | 0.108782 | 2.725471 | 0.1660 | 1.41 | +0.135885 | +0.0480 |
| `softmax_margin0p2_m0p005` | 0.058767 | 1.344650 | 0.6933 | 0.111750 | 2.792825 | 0.1580 | 1.54 | +0.235334 | +0.0400 |
| `softmax_margin0p5_m0p005` | 0.057836 | 1.325787 | 0.6933 | 0.112933 | 2.820503 | 0.1553 | 1.26 | +0.254197 | +0.0373 |
| `softmax_margin0p5_m0p01` | 0.049672 | 1.140619 | 0.7283 | 0.119243 | 2.977313 | 0.1493 | 1.40 | +0.439365 | +0.0313 |
| `softmax_no_perturb` | 0.068466 | 1.576745 | 0.6183 | 0.104622 | 2.638880 | 0.1667 | 0.93 | +0.003239 | +0.0487 |
| `softmax_perturb_int1` | 0.068037 | 1.566398 | 0.6133 | 0.105040 | 2.651115 | 0.1693 | 1.89 | +0.013586 | +0.0513 |

All focused margin variants improved class-blocked final-window MSE and NLL
while keeping held-out test accuracy above MLP. The cleanest balance is
`softmax_margin0p2_m0p0025`: it reduces NLL from `1.5800` to `1.4441`, reduces
MSE from `0.0686` to `0.0630`, raises final-window accuracy from `0.6167` to
`0.6683`, and essentially preserves the promoted held-out accuracy
(`0.1660` vs `0.1673`).

The aggressive margin variants track the current block much better, including
lower final-window NLL than both MLP rows, but they give up more held-out
accuracy. They still keep a positive held-out advantage over MLP.

## Positive Control: CIFAR IID

The main screen kept a CIFAR iid positive-control row. UPGD variants continued
to beat the best MLP held-out accuracy (`0.1633`).

| Method | FW MSE | FW NLL | FW Acc | Test MSE | Test NLL | Test Acc | Time s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mlp_h32` | 0.107289 | 2.225389 | 0.1983 | 0.118653 | 2.246706 | 0.1633 | 2.41 |
| `promoted_softmax_h32` | 0.085749 | 2.117201 | 0.2417 | 0.086159 | 2.134287 | 0.2440 | 1.74 |
| `softmax_h64` | 0.085313 | 2.099471 | 0.2483 | 0.085586 | 2.119017 | 0.2613 | 2.00 |
| `softmax_hidden_plus_input` | 0.085867 | 2.112966 | 0.2867 | 0.085671 | 2.153837 | 0.2793 | 1.93 |
| `softmax_margin0p5_m0p01` | 0.085766 | 2.112189 | 0.2833 | 0.086194 | 2.130067 | 0.2627 | 2.33 |

This positive control says the class-blocked margin fix does not depend on a
global failure of the promoted UPGD image path. The iid row remains favorable to
UPGD, and direct readout input improves iid held-out accuracy, although it is
not a class-blocked NLL fix.

## Failed Or Weak Variants

- `promoted_linear_h32`: improves class-blocked MSE and final-window accuracy
  (`0.0323` MSE, `0.8267` FW accuracy), but worsens NLL (`1.6736`) and reduces
  held-out accuracy (`0.1333`). Not a good classification default.
- Higher kappa (`0.75`, `1.0`): improves held-out accuracy slightly in
  class-blocked but worsens final-window NLL/MSE and hurts iid.
- `softmax_hidden_plus_input`: improves class-blocked held-out accuracy
  (`0.1940`) and MSE, but NLL degrades badly (`2.1161`), suggesting
  miscalibrated direct pixel readout under class blocks.
- `softmax_headnorm`: improves retained accuracy (`0.1847`) but worsens
  tracking NLL/MSE.
- Width 64: helpful on CIFAR iid but neutral to slightly negative on
  class-blocked tracking.
- Perturbation changes (`sigma=0`, interval 1, normal noise): only tiny effects.
  `interval=1` is the safest small tracking/retention nudge, not a real gap
  closer.

## Mechanism Hypotheses

1. The class-blocked gap is mostly an output-head/readout tracking issue, not a
   hidden-trunk capacity or utility-perturbation issue. Width, perturbation
   cadence, and kappa did not close the NLL gap.
2. MLP gets near-perfect final-window current-class accuracy by specializing to
   the current block, but its held-out test accuracy collapses. UPGD retains
   broader class information, so it needs a bounded head-local correction rather
   than stronger shared-trunk drift.
3. The readout-margin adapter supplies exactly that head-local correction:
   it pushes the true head above the strongest wrong head when the margin is
   too small, improving current-block NLL/MSE without replay or task identity.
4. The margin strength is the tradeoff knob. Large margin steps solve tracking
   most aggressively but spend retained accuracy; a small margin and small
   margin step preserve the original retained accuracy advantage.

## Recommendation

Do not change core anti-drift from this worker. The next best patch is an
opt-in classification readout-margin preset, starting with:

`readout_mode="softmax_ce", readout_margin=0.2, readout_margin_step_size=0.0025`

This candidate should be promoted only after a stronger run: 10 seeds, 1,000
steps or more, the same real CIFAR subsets, and compact OPMNIST as a second
positive-control row. If that run holds, expose the preset through a
non-default config path first; avoid changing `UPGDLearner.step2_default` until
the core anti-drift worker has finished.
