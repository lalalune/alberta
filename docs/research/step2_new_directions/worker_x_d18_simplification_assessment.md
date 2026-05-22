# Worker X D18 Simplification Assessment

## Scope

Ownership: D18 simplification and ablation evidence only. No source files were
edited. All new artifacts are under `outputs/step2_new_directions/worker_x_*`
and `docs/research/step2_new_directions/worker_x_*`.

Question: can `step2_canonical` be simplified into one fixed universal learner
that still beats the best fair MLP on the requested blocker set?

Blocker set: `digits_mask_noise`, `digits_class_blocked`, `controlled_rare`,
`synthetic_compositional`, `synthetic_polynomial`, `synthetic_frequency`.
Protocol: 3 paired seeds, 1200 online steps, final window 300.

## D18 Mechanism Inspection

Mathematically generic blocks:

- Resource-managed RKHS core: generic nonparametric approximation with budgeted
  centers; this is a legitimate universal-function-approximation mechanism.
- Random tanh features: fixed random nonlinear basis; generic ridgelet-style
  approximation, not tailored to one dataset.
- Fourier features: generic spectral basis for smooth/periodic structure.
- Learned block gains: generic online residual-gradient adaptation over additive
  components, if kept as a small continuous mechanism rather than a selector.
- Global residual update: generic additive residual correction; all included
  blocks update every step.

Blocks that look like rescue paths rather than clean universal core:

- Strict degree-3 polynomial RLS residual: mathematically valid, but highly
  matched to the synthetic polynomial blocker and previously introduced to close
  that exact failure.
- D14 unified residual basis: a broad fixed basis, but in D18 it functions as a
  second residual rescue channel after core/basis/polynomial conflicts.
- Component clipping on polynomial/unified predictions: pragmatic spillover
  control for digit streams, not a representation principle.
- Strong gain anchoring after adaptive gains: pragmatic conflict management
  between polynomial suppression and digit retention.

Overall: D18 is one additive learner and not a router, but the canonical form is
not yet a clean simple learner. It is a generic core plus two residual rescue
channels and guardrails.

## Commands Run

```bash
source .venv/bin/activate && python 'examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py' --datasets digits_mask_noise,digits_class_blocked,controlled_rare,synthetic_compositional,synthetic_polynomial,synthetic_frequency --steps 1200 --n-seeds 3 --final-window 300 --configs step2_canonical,step2_no_unified,step2_no_poly,step2_gain_l2_0p1 --output-dir outputs/step2_new_directions/worker_x_core_ablations_3seed --note-path docs/research/step2_new_directions/worker_x_core_ablations_3seed.md

source .venv/bin/activate && python 'examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py' --datasets digits_mask_noise,digits_class_blocked,controlled_rare,synthetic_compositional,synthetic_polynomial,synthetic_frequency --steps 1200 --n-seeds 3 --final-window 300 --configs gain_safecore_poly_unified_0p01 --gain-step-size 0.0 --gain-l2 0.05 --component-clip 1.0 --output-dir outputs/step2_new_directions/worker_x_fixed_gains_3seed --note-path docs/research/step2_new_directions/worker_x_fixed_gains_3seed.md

source .venv/bin/activate && python 'examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py' --datasets digits_mask_noise,digits_class_blocked,controlled_rare,synthetic_compositional,synthetic_polynomial,synthetic_frequency --steps 1200 --n-seeds 3 --final-window 300 --configs step2_canonical --tanh-width 256 --output-dir outputs/step2_new_directions/worker_x_tanh256_3seed --note-path docs/research/step2_new_directions/worker_x_tanh256_3seed.md

source .venv/bin/activate && python 'examples/The Alberta Plan/Step2/new_directions/d18_simple_universal_resource_basis.py' --datasets digits_mask_noise,digits_class_blocked,controlled_rare,synthetic_compositional,synthetic_polynomial,synthetic_frequency --steps 1200 --n-seeds 3 --final-window 300 --configs step2_canonical --raw-poly-budget 32 --algebraic-budget 64 --arccosine-budget 64 --total-center-budget 160 --output-dir outputs/step2_new_directions/worker_x_half_centers_3seed --note-path docs/research/step2_new_directions/worker_x_half_centers_3seed.md
```

No 10-seed run was triggered because no ablation survived all blockers at 3
seeds.

## 3-Seed Results

Margin is paired `best MLP final-window MSE - D18 final-window MSE`, so positive
favors D18. W/L/T is seed-level D18 versus the best MLP on that seed.

| Ablation | Dataset | D18 MSE | Best MLP MSE | Margin | W/L/T |
|---|---|---:|---:|---:|---:|
| canonical | digits_mask_noise | 0.044755 | 0.047837 | +0.003082 | 3/0/0 |
| canonical | digits_class_blocked | 0.004527 | 0.002992 | -0.001535 | 0/3/0 |
| canonical | controlled_rare | 0.018128 | 0.073166 | +0.055038 | 3/0/0 |
| canonical | synthetic_compositional | 0.240080 | 0.271702 | +0.031623 | 3/0/0 |
| canonical | synthetic_polynomial | 0.798105 | 0.947464 | +0.149360 | 3/0/0 |
| canonical | synthetic_frequency | 0.882529 | 1.148462 | +0.265933 | 3/0/0 |
| no unified | digits_mask_noise | 0.044608 | 0.047837 | +0.003230 | 3/0/0 |
| no unified | digits_class_blocked | 0.004478 | 0.002992 | -0.001486 | 0/3/0 |
| no unified | controlled_rare | 0.017915 | 0.073166 | +0.055252 | 3/0/0 |
| no unified | synthetic_compositional | 0.250559 | 0.271702 | +0.021144 | 2/1/0 |
| no unified | synthetic_polynomial | 0.796623 | 0.947464 | +0.150841 | 3/0/0 |
| no unified | synthetic_frequency | 0.902734 | 1.148462 | +0.245728 | 3/0/0 |
| no polynomial | digits_mask_noise | 0.044725 | 0.047837 | +0.003112 | 3/0/0 |
| no polynomial | digits_class_blocked | 0.004606 | 0.002992 | -0.001614 | 0/3/0 |
| no polynomial | controlled_rare | 0.035047 | 0.073166 | +0.038119 | 3/0/0 |
| no polynomial | synthetic_compositional | 0.251120 | 0.271702 | +0.020582 | 3/0/0 |
| no polynomial | synthetic_polynomial | 0.918262 | 0.947464 | +0.029202 | 2/1/0 |
| no polynomial | synthetic_frequency | 0.886133 | 1.148462 | +0.262328 | 3/0/0 |
| gain l2 0.1 | digits_mask_noise | 0.045459 | 0.047837 | +0.002379 | 3/0/0 |
| gain l2 0.1 | digits_class_blocked | 0.004636 | 0.002992 | -0.001644 | 0/3/0 |
| gain l2 0.1 | controlled_rare | 0.021430 | 0.073166 | +0.051736 | 3/0/0 |
| gain l2 0.1 | synthetic_compositional | 0.235651 | 0.271702 | +0.036051 | 3/0/0 |
| gain l2 0.1 | synthetic_polynomial | 0.839099 | 0.947464 | +0.108366 | 2/1/0 |
| gain l2 0.1 | synthetic_frequency | 0.887679 | 1.148462 | +0.260783 | 3/0/0 |
| fixed gains | digits_mask_noise | 0.045132 | 0.047837 | +0.002705 | 3/0/0 |
| fixed gains | digits_class_blocked | 0.005550 | 0.002992 | -0.002558 | 0/3/0 |
| fixed gains | controlled_rare | 0.035728 | 0.073166 | +0.037439 | 3/0/0 |
| fixed gains | synthetic_compositional | 0.239766 | 0.271702 | +0.031936 | 2/1/0 |
| fixed gains | synthetic_polynomial | 1.034771 | 0.947464 | -0.087306 | 1/2/0 |
| fixed gains | synthetic_frequency | 0.910325 | 1.148462 | +0.238137 | 3/0/0 |
| tanh width 256 | digits_mask_noise | 0.044899 | 0.047837 | +0.002938 | 3/0/0 |
| tanh width 256 | digits_class_blocked | 0.004580 | 0.002992 | -0.001588 | 0/3/0 |
| tanh width 256 | controlled_rare | 0.017904 | 0.073166 | +0.055263 | 3/0/0 |
| tanh width 256 | synthetic_compositional | 0.242361 | 0.271702 | +0.029342 | 3/0/0 |
| tanh width 256 | synthetic_polynomial | 0.796318 | 0.947464 | +0.151146 | 3/0/0 |
| tanh width 256 | synthetic_frequency | 0.885242 | 1.148462 | +0.263220 | 3/0/0 |
| half centers | digits_mask_noise | 0.046381 | 0.047837 | +0.001456 | 2/1/0 |
| half centers | digits_class_blocked | 0.004756 | 0.002992 | -0.001764 | 0/3/0 |
| half centers | controlled_rare | 0.016496 | 0.073166 | +0.056670 | 3/0/0 |
| half centers | synthetic_compositional | 0.244586 | 0.271702 | +0.027116 | 3/0/0 |
| half centers | synthetic_polynomial | 0.814820 | 0.947464 | +0.132644 | 3/0/0 |
| half centers | synthetic_frequency | 0.902511 | 1.148462 | +0.245951 | 3/0/0 |

## Minimal Failure Mode

The minimal blocker is `digits_class_blocked`. Every tested D18 variant loses
that dataset by final-window MSE on all three seeds, even though D18 improves
held-out test accuracy. This is not a polynomial-residual issue: removing the
unified block, removing the polynomial block, freezing gains, shrinking tanh
width, and halving centers all preserve the class-blocked MSE failure.

The simplest near-survivor is `tanh_width=256`: it keeps the canonical positive
seed profile on the five non-class-blocked blockers, but still loses
`digits_class_blocked` 0/3/0. This suggests the D18 core can be made cheaper,
but not promoted on the requested blocker matrix.

## Recommendation

Keep researching. Do not promote D18 as-is on this blocker set, and do not
promote a simpler ablation. The current D18 evidence remains strong for five of
six requested blockers, but `digits_class_blocked` is a hard counterexample to
the fixed final-window MSE criterion.

Next work should target the class-blocked online MSE/retention conflict without
adding a router, MLP expert, or hand-engineered feature search. A clean direction
would be one generic calibration or loss-normalization mechanism for multi-head
classification under blocked labels, not another output portfolio.
