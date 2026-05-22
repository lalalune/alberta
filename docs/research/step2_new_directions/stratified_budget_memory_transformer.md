# Stratified-Budget Transformer Memory

Date: 2026-05-07

## Question

The prior learned-budget runner replaced static `gate_max=0.15`, but it
under-opened memory and lost slightly to the static cap. This pass keeps the
same paired static comparator and attacks three additional static choices:
FIFO replay, fixed gate L2, and fixed slow-memory learning-rate scale.

## New Runner

- `examples/The Alberta Plan/Step2/step2_tiny_shakespeare_stratified_budget_memory_transformer.py`

The learned path keeps a scalar budget cap, but changes the utility estimator:

- replay is sampled from a deterministic four-phase cycle:
  hard-negative, positive, uncertainty, then recent;
- hard negatives close runaway memory when replay says memory hurts;
- positive replay lets the budget and gate open when measured memory utility is
  useful;
- uncertainty replay revisits stale or volatile advantage estimates;
- `gate_l2` is controlled online inside a bounded interval;
- the slow-memory LR is controlled online through a bounded multiplier.

## Static-Knob Audit

| Static choice | Current value | Role | Evidence | Learnability path | Risk | Priority |
| --- | --- | --- | --- | --- | --- | --- |
| Memory cap | Static comparator: `gate_max=0.15`; learned path starts `budget_init=0.15` | Bounds residual memory contribution | Static cap is the current strongest memory-enabled candidate; previous learned cap shrank to about `0.115` and lost slightly | Keep learned `budget_logit`; improve utility with stratified replay and pressure floor | Too loose over-opens memory; too tight collapses to fast-only | P0 |
| Budget bounds | `budget_min=0.02`, `budget_max=0.35` | Keeps learned cap numerically and behaviorally bounded | Loose caps previously hurt long-horizon online NLL; lower caps preserved small gains | Learn bounds from long-run risk, or set per-placement/per-regime bounds | Static bounds can hide true optimum or block useful memory | P1 |
| Budget update LR | `budget_lr=0.0005` | Timescale of cap adaptation | Conservative learned-budget run was safe but under-opened | Meta-adapt from replay-advantage variance or use Autostep-style normalizer | Too high collapses or over-opens; too low behaves static | P1 |
| Budget utility EMA | `budget_ema_decay=0.995` | Smooths replay utility and variance | Replay advantage is noisy and near zero | Learn decay from replay variance/drift | Slow decay misses regime shifts; fast decay chases noise | P2 |
| Budget utilization target | `budget_target_utilization=0.85` | Opens budget mostly when gate approaches cap | Prior controller rarely opened because gate stayed below cap | Learn target from budget regret | Static target can enforce under-use | P1 |
| Budget resource cost | `budget_cost=0.005` | Penalizes open budget without utility | Aggressive learned cap collapsed harder; static cap stayed useful | Learn cost from hard-negative replay frequency | Cost can dominate weak positive utility | P1 |
| Budget pressure floor | `0.25` in this runner | Lets positive utility affect budget before gate reaches target utilization | Added because prior learned budget under-opened | Learn floor from positive-vs-negative replay mix | Can slowly open budget on noisy positives | P1 |
| Advantage floor | `budget_advantage_floor=0.01` | Stabilizes normalized utility | Replay variance can be near zero early | Learn floor from observed variance | Too large mutes signal; too small amplifies noise | P2 |
| Replay sampling | Prior FIFO; this runner cycles hard-negative/positive/uncertainty/recent | Chooses delayed examples for gate and budget utility | FIFO learned-budget run saw near-zero utility and under-opened | Learn stratum probabilities from downstream regret | Biased replay can optimize stale edge cases | P0 |
| Replay size | `128` | Capacity for delayed utility | Prior winning static-cap config used `128` | Learn size/frequency from utility diversity or memory pressure | Larger buffers cost state and can stale; smaller buffers miss rare utility | P1 |
| Gate regularization | Prior fixed `gate_l2=0.1`; this runner learns bounded `0.0..0.2` | Closes memory unless replay advantage pays for it | Previous scalar gate stayed open under negative final-window advantage | Learn `gate_l2_logit` from utility and budget pressure | Too high collapses useful memory; too low runaway | P0 |
| Gate L2 controller LR | `gate_l2_lr=0.001` | Timescale for learned regularization | Needed movement without fast oscillation | Meta-adapt from sign stability of replay advantage | Fast controller can fight gate updates | P2 |
| Gate L2 pressure weight | `0.25` | Increases L2 when gate/budget pressure is high | Prevents cap opening from becoming residual runaway | Learn from over-budget/hard-negative regret | Can over-penalize useful memory | P2 |
| Gate init | `gate_init_logit=-3.0` | Initial open probability | Winning static-cap config uses a low initial gate | Learn from prototype maturity or uncertainty | Too closed delays utility discovery; too open harms early fast features | P1 |
| Gate LR | `gate_lr=0.5` | Converts replay utility into gate logit movement | Static-cap candidate uses this safely | Autostep-style meta-step for gate updates | Too high oscillates; too low ignores replay | P1 |
| Gate decay | `gate_decay=0.995` | Passive gate decay | Decay `1.0` closed memory and removed useful signal; `0.998/0.999` were too weak | Learn decay from replay sign persistence | Drift can keep harmful gates open | P1 |
| Advantage margin | `0.0` | Utility threshold before opening memory | Current advantages are tiny; extra margin would close path | Learn margin from held-out proxy or replay calibration | Positive margin may suppress small true gains | P2 |
| Slow-memory LR | Base `slow_lr=0.1`; this runner learns multiplier `0.5..1.5` | Updates prototype value rows separately from transformer | Static-cap candidate uses `0.1`; prior reports list slow LR as still static | Learn multiplier from replay utility and cost | Higher slow LR can memorize noise; lower slow LR can underfit memory | P0 |
| Slow-LR controller LR | `0.0005` | Timescale for slow-LR multiplier | Kept bounded movement in smoke/3000/5000 | Meta-adapt from utility variance | Can couple badly with budget/gate controllers | P2 |
| Slow-LR control cost | `0.1` | Pulls multiplier toward `1.0` absent utility | Avoids permanent collapse or runaway | Learn cost from hard-negative replay rate | Too high makes multiplier static | P2 |
| Placement | `post_ffn` | Where memory residual enters transformer | Prior placement selector had tiny 2-seed gains but did not replace static post-FFN | Learn placement weight jointly with budget, or per-placement budgets | Pre-KV can help held-out later but hurt online stability; two branches cost compute | P0 |
| Train loss mode | `memory` | Optimizes memory-enabled logits directly | Blended loss weakened prior fast-only gain | Learn blend weight from fast-vs-memory deployment regret | Static memory-only can hurt fast-only eval | P1 |
| Reset mode | `meta_ema` | Initializes replaced prototype values | Used by prior winning config | Learn reset value from replay utility by prototype age | Bad reset can erase useful rows or preserve stale rows | P2 |
| Prototype count | `64` | Memory capacity | Current candidate budget and throughput measured at 64 | Learn allocate/split count by utility, or dynamic birth/death | More prototypes cost compute; fewer miss rare contexts | P1 |
| Prototype update rate | `0.3` | Center tracking speed | Inherited from prior prototype-memory runners | Learn per-prototype update from activation utility | Too fast tracks noise; too slow stale | P2 |
| Novelty threshold | `0.0002` | Controls prototype replacement | Inherited static choice | Learn from replay hard-negative density | Too low churns; too high freezes | P2 |
| Prototype bandwidth | `0.01`, adaptive bandwidth off by default | Activation locality | Prior memory work used narrow local activations | Learn bandwidth per prototype | Too narrow under-activates; too broad blurs memory | P1 |
| Fast LR | `0.15` | Transformer update step | Tuned FFN baseline and static comparator use this | Learn or decouple per module | Changing it confounds memory evidence | P2 |
| Gradient clip | `1.0` | Stabilizes online SGD | Inherited from transformer runners | Learn clip scale from gradient norm history | Too small undertrains; too large unstable | P2 |
| Model shape | `block_size=32`, `d_model=32`, `mlp_hidden=64` | Baseline capacity | Current Tiny Shakespeare protocol uses this small model | Scale after mechanism is stable | Larger models change margins and throughput | P2 |

## Commands

Smoke:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_stratified_budget_memory_transformer.py" --steps 64 --seeds 1 --eval-steps 64 --final-window 32 --block-size 16 --d-model 16 --mlp-hidden 32 --proto-count 16 --replay-size 16 --output-dir outputs/step2_new_directions/stratified_budget_memory_transformer_smoke
```

3000-step comparison:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_stratified_budget_memory_transformer.py" --steps 3000 --seeds 3 --eval-steps 512 --final-window 512 --output-dir outputs/step2_new_directions/stratified_budget_memory_transformer_3000_3seed_default
```

5000-step follow-up:

```bash
source .venv/bin/activate && PYTHONUNBUFFERED=1 python "examples/The Alberta Plan/Step2/step2_tiny_shakespeare_stratified_budget_memory_transformer.py" --steps 5000 --seeds 3 --eval-steps 512 --final-window 512 --output-dir outputs/step2_new_directions/stratified_budget_memory_transformer_5000_3seed_default
```

## Results

Smoke passed and wrote artifacts. The controller moved slightly from the prior:
final budget `0.1496`, gate L2 `0.101`, slow-LR multiplier `0.996`.

3000 steps, 3 seeds:

| Metric | Static `gate_max=0.15` | Stratified budget | Stratified minus static |
| --- | ---: | ---: | ---: |
| final-window NLL | 2.9218 +/- 0.0345 | 2.9217 +/- 0.0344 | +0.0001 +/- 0.0001 |
| held-out NLL | 3.1016 +/- 0.0641 | 3.1014 +/- 0.0644 | +0.0002 +/- 0.0002 |
| held-out perplexity | 22.3260 +/- 1.4379 | 22.3212 +/- 1.4429 | +0.0049 +/- 0.0050 |
| train steps/s | 2219.8567 +/- 280.7440 | 2740.6466 +/- 193.3933 | +520.7899 +/- 121.7971 |

5000 steps, 3 seeds:

| Metric | Static `gate_max=0.15` | Stratified budget | Stratified minus static |
| --- | ---: | ---: | ---: |
| final-window NLL | 2.7879 +/- 0.0044 | 2.7871 +/- 0.0046 | +0.0008 +/- 0.0003 |
| held-out NLL | 3.1038 +/- 0.1399 | 3.1004 +/- 0.1375 | +0.0035 +/- 0.0027 |
| held-out perplexity | 22.7281 +/- 3.2263 | 22.6335 +/- 3.1513 | +0.0945 +/- 0.0788 |
| held-out accuracy | 0.2051 +/- 0.0085 | 0.2070 +/- 0.0081 | +0.0020 +/- 0.0011 |
| train steps/s | 1904.4314 +/- 275.4744 | 2354.6693 +/- 262.1849 | +450.2379 +/- 88.9071 |

The 5000-step stratified result also beats the tuned FFN on the target
memory-enabled metrics in this 3-seed run:

| Metric | FFN baseline | Stratified budget |
| --- | ---: | ---: |
| final-window NLL | 2.7876 +/- 0.0045 | 2.7871 +/- 0.0046 |
| held-out perplexity | 22.7002 +/- 3.2025 | 22.6335 +/- 3.1513 |

## Controller Diagnostics

| Horizon | Budget | Gate | Gate L2 | Slow-LR multiplier | Replay advantage EMA |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 3000 | 0.1274 +/- 0.0008 | 0.0792 +/- 0.0042 | 0.1556 +/- 0.0012 | 0.8436 +/- 0.0048 | -0.0107 +/- 0.0017 |
| 5000 | 0.1225 +/- 0.0011 | 0.0725 +/- 0.0040 | 0.1628 +/- 0.0012 | 0.8083 +/- 0.0045 | -0.0126 +/- 0.0059 |

The learned controllers did not open memory more than the static cap. They
opened it less, raised regularization, and reduced slow-memory LR. The positive
memory-enabled result therefore comes from better replay selection and safer
resource use, not from larger memory exposure.

## Assessment

Memory-enabled metrics improved over static `gate_max=0.15` at 3000 and 5000
steps, but margins are still tiny and only 3 seeds. This is a stronger learned
resource result than the prior learned-budget run because it no longer loses to
the static cap on the target memory-enabled comparison.

The main failure mode remains under-opening. Replay hard negatives dominate the
final controller state, so the learned path becomes a safer, lower-gate memory
regularizer. That improves memory-enabled held-out NLL here, but it slightly
hurts fast-only eval versus the static comparator at 5000 steps.

Static knobs still remaining after this pass:

- placement is still static `post_ffn`;
- budget bounds and controller constants are static;
- gate init, gate LR, gate decay, and advantage margin are static;
- replay size and stratum frequencies are static;
- base slow LR and slow-LR controller constants are static;
- prototype count, bandwidth, novelty threshold, and update rate are static;
- fast LR, gradient clip, train loss mode, reset mode, and model shape are
  static.

Next follow-up: combine this safer stratified budget with the prior placement
selector, but give each placement its own learned budget and hard-negative
replay queue. A single shared placement weight without per-branch resource
control is likely too weak because the useful post/pre signal is near the noise
floor.
