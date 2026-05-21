# Step-Size Adaptation and Target-Structure UPGD for Early Alberta Plan Continual Learning

## Abstract

We present the Step 1 and supervised Step 2 kernels of the Alberta Framework, a
JAX implementation of early Alberta Plan continual-learning problems. Step 1
reproduces and extends the IDBD/Autostep step-size adaptation setting with the
full public baseline set named by the Alberta Plan footnote where public
algorithms exist. Step 2 promotes target-structure UPGD, a single non-router
online learner that uses hidden-weight utility, low-utility perturbation, and
bounded updates to beat same-run best fair MLP baselines on the repository's
current supervised empirical acceptance matrix. We distinguish implementation
closure from scientific closure: Step 1 is complete for public reproducible
methods; Step 2's supervised empirical matrix is closed, but full 800-task
OPMNIST and a theorem of universal representation learning remain open.

## 1. Problem Setting

The Alberta Plan begins with continual supervised learning where every
component updates at every timestep. Step 1 fixes the feature representation
and asks whether meta-learned step sizes can remove manual step-size tuning.
Step 2 introduces nonlinear function approximation and feature utility under
non-stationary supervised targets.

The framework implements these settings with JAX scan-compatible state,
single-step update APIs, explicit online normalizers, and reproducible
multi-seed experiment runners.

## 2. Step 1: Given Features

Step 1 targets the equation

`y*_t = w*_t . x_t + b*_t + eta_t`

with non-stationarity coming from drifting target weights, drifting target
bias, or shifting input distributions. The canonical implementation includes:

- `AlbertaPlanStep1Stream` for drifting `w*`, drifting `b*`, and additive
  noise;
- `XDistShiftStream` for non-stationary input distributions;
- LMS, IDBD, Autostep, AdaGain, Adam, RMSprop, and NADALINE;
- EMA, Welford, and StreamingBatch online normalization.

The unpublished `Auto (Degris in prep.)` item is not implemented. No public
algorithm specification was found, and the package deliberately refuses to
fabricate a symbol or config alias for it. The 2026-05-06 public-source audit
checked the Alberta Plan footnote, Degris publication/software pages, the
RLPark public code/documentation trail, Mahmood's Autostep thesis and ICASSP
paper, Kearney/Pilarski AutoStep-for-GTD(lambda), AdaGain, and Degris et al.
2024. These sources publish Autostep, AdaGain, IDBD-family variants, and open
directions for normalized step-size optimization, but not an implementable
`Auto` update rule.

### Step 1 Result

The canonical 30-seed results support the Step 1 claim with caveats:

- IDBD reproduces Sutton's original step-size adaptation result on the Sutton
  stream and the noisy Alberta-compatible variant.
- Autostep is the most robust tuning-light optimizer across the tested
  hyperparameter grid.
- Online normalization is essential under input-scale shifts.
- Adam/RMSprop can slightly edge LMS near the noise floor on the drifting
  Alberta stream, so the precise claim is not that IDBD/Autostep dominate every
  modern optimizer everywhere.

The production package exposes this as `Step1KernelConfig` and
`make_step1_learner`.

## 3. Step 2: Target-Structure UPGD

The promoted supervised Step 2 learner is target-structure UPGD:

- vector-output shared hidden representation;
- utility `|w * grad|` tracked online on hidden weights;
- low-utility hidden weights receive small perturbations;
- updates are ObGD-bounded;
- loss normalization uses sum-style pressure for non-negative simplex targets
  and mean-style pressure otherwise.

The target-structure rule is the smallest robust bridge found between dense
synthetic vector regression and sparse one-hot classification. It replaces the
earlier target-density compromise, which over-boosted exact-zero dense rows and
sparse multilabel targets.

### Step 2 Result

The supervised Step 2 empirical acceptance matrix is closed by a single
non-router learner against same-run best fair MLP baselines. The evidence
includes controlled nonlinear/interaction/rare/polynomial/frequency probes,
out-of-class synthetic probes, hard sklearn-digits regimes, and focused
target-structure boundary rows.

This is an empirical result, not a universality theorem. The theory note proves
only conditional statements: scale behavior, bounded displacement, finite
candidate selection, and sieve-style reductions under explicit assumptions.
Falsification probes show that unrestricted universal representation learning
is not established by the current mechanism.

## 4. External Scale Boundary

OpenML MNIST OPMNIST evidence now includes a completed one-seed 800-task,
48M-example run for the latest UPGD-memory learner against raw and sharpened
fair MLP baselines. The result supports the online-tracking claim, with
UPGD-memory winning online MSE, online accuracy, and final-window MSE. It does
not support an unqualified retained-view claim: the best fair MLP still wins
final-window accuracy and the all-permutation held-out test metrics.

This boundary is deliberate. A production package can be useful before every
scientific replication is complete, but the paper claim must not hide the
difference.

## 5. Reproducibility

Step 1:

```bash
python "examples/The Alberta Plan/Step1/step1_full_baselines.py"
python "examples/The Alberta Plan/Step1/step1_normalization_ablation.py"
python "examples/The Alberta Plan/Step1/step1_robustness_study.py"
pytest tests/test_step1_replication.py -q
```

Step 2:

```bash
alberta-step2-smoke --steps 128
pytest tests/test_step2_canonical.py tests/test_upgd.py -q
```

Evidence gate:

```bash
alberta-evidence-gate --step all
```

## 6. Package Surface

The stable package surface is:

- `alberta_framework.steps.Step1KernelConfig`;
- `alberta_framework.steps.Step2KernelConfig`;
- `make_step1_learner`, `make_step1_stream`;
- `make_step2_learner`, `make_step2_stream`;
- `run_step1_smoke`, `run_step2_smoke`;
- `alberta-step1-smoke`, `alberta-step2-smoke`, `alberta-evidence-gate`.

The broader `alberta_framework.core` namespace remains available for research,
ablation work, and later Alberta Plan steps.

## 7. Conclusion

Step 1 is complete for public reproducible methods and packaged as a stable
kernel. Step 2 is packaged as a stable supervised empirical kernel with an
honest claim boundary. The next scientific targets are multi-seed external
scale confirmation, retained-view OPMNIST generalization, and a stronger
mathematical theory that either proves a restricted universality result or
cleanly characterizes the target families for which utility-guided perturbation
is sufficient.
