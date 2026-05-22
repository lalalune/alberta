# Core Concepts

This guide explains the foundational concepts of the Alberta Framework.

## The Alberta Plan

The Alberta Plan (Sutton et al., 2022) is a 12-step roadmap for building continual
learning AI systems. This framework implements **all 12 steps**, from fixed-feature
supervised learning through multi-agent augmentation with the Prototype-IA architecture.

| Step | Focus |
|------|-------|
| 1 | Fixed-feature continual supervised learning (IDBD/Autostep) |
| 2 | Nonlinear function approximation (MLP, ObGD, IDBD-MLP) |
| 3 | GVF/Horde prediction with per-head trace decay |
| 4 | Continual control (SARSA, actor-critic) |
| 5 | Off-policy nonlinear Horde prediction |
| 6 | Average-reward optimization (DifferentialTD, DifferentialSARSA) |
| 7 | Dyna planning with one-step world model |
| 8 | World model facade with ensemble prediction |
| 9 | Guarded dreaming with error-gated imagined transitions |
| 10 | STOMP temporal abstraction (subtasks, intra-option policies) |
| 11 | OaK architecture (utility tracking, curation, option keyboard) |
| 12 | Prototype-IA (exo-cerebellum and exo-cortex multi-agent augmentation) |

## Temporal Uniformity

The framework's core principle is **temporal uniformity**: every component updates at
every time step. This means:

- No batch processing, no epochs, no passes over data
- Learning happens incrementally, one sample at a time
- All components — optimizers, normalizers, world models — update on every step

This reflects the reality of continual learning where data arrives as an unbounded stream.

## Key Components

### Experience Streams

Streams generate `TimeStep` objects containing:

- `observation`: Feature vector `x_t ∈ R^d`
- `target`: Value to predict `y_t ∈ R`
- `reward`: Optional reward signal (for RL streams)

Available streams: `RandomWalkStream`, `AbruptChangeStream`, `CyclicStream`,
`PeriodicChangeStream`, `ScaledStreamWrapper`, `DynamicScaleShiftStream`,
`ScaleDriftStream`, and Gymnasium-wrapped environments.

### Optimizers

Optimizers compute weight updates given a prediction error. All implement the
`Optimizer` protocol: `init(feature_dim)` and `update(state, error, observation)`.

| Optimizer | Description |
|-----------|-------------|
| **LMS** | Fixed step-size baseline |
| **IDBD** | Per-weight adaptive step-sizes via gradient correlation (Sutton 1992) |
| **Autostep** | Tuning-free adaptation with self-regulated normalizers (Mahmood et al. 2012) |
| **ObGD** | Online gradient descent with dynamic update bounding (Elsayed et al. 2024) |
| **TDIDBD** | IDBD adapted for temporal difference learning |
| **AutoTDIDBD** | Autostep adapted for temporal difference learning |

Bounding is decoupled via the `Bounder` ABC: `ObGDBounding`, `AdaptiveObGDBounding`,
and `AGCBounding` (Brock et al. 2021).

### Learners

Learners combine a prediction model with an optimizer. All accept composable
`Optimizer`, `Bounder`, and `Normalizer` ABCs.

```python
from alberta_framework import LinearLearner, MLPLearner, IDBD, ObGD, ObGDBounding
import jax.random as jr

# Linear learner with IDBD
learner = LinearLearner(optimizer=IDBD())
state = learner.init(feature_dim=10, key=jr.key(0))
prediction = learner.predict(state, observation)
result = learner.update(state, error, observation)

# MLP learner with ObGD and bounding
mlp = MLPLearner(
    hidden_sizes=(64, 64),
    optimizer=ObGD(step_size=0.01),
    bounder=ObGDBounding(kappa=1.0),
)
```

The `MLPLearner` uses: `Input -> [Dense -> LayerNorm -> LeakyReLU] x N -> Dense(1)`,
with sparse initialization (90% sparsity by default).

### GVF / Horde (Steps 3–5)

General Value Functions (GVFs) represent knowledge as value functions with four
question functions: policy π, pseudo-termination γ, pseudo-reward (cumulant), and
pseudo-terminal-reward. `HordeLearner` wraps `MultiHeadMLPLearner` with per-demon
gamma/lambda, TD target computation, and GVF metadata.

```python
from alberta_framework import HordeLearner, GVFSpec, create_horde_spec

spec = create_horde_spec(
    feature_dim=32,
    num_demons=4,
    gamma=0.99,
)
horde = HordeLearner(spec=spec)
```

### Control (Step 4)

`SARSAAgent` wraps `HordeLearner` with epsilon-greedy action selection. Each action
maps to a control demon. SARSA target `r + γ Q(s', a')` is computed externally.

`HordeActorCriticAgent` and `NonlinearHordeActorCriticAgent` provide actor-critic
control with MLP actor and critic.

### Average-Reward Optimization (Steps 5–6)

`DifferentialTDLearner` and `DifferentialSARSAAgent` implement average-reward
prediction and control for continuing (non-episodic) tasks, using a running estimate
of the average reward to compute differential TD errors.

### Planning and Dreaming (Steps 7–9)

`OneStepWorldModel` and `ActionConditionedWorldModel` learn forward models of the
environment. `GuardedDreamer` uses error-gated, buffer-anchored imagined transitions
to augment real experience — only accepting dream transitions that pass an
error-based plausibility gate.

### Temporal Abstraction (Steps 10–11)

`STOMPAgent` implements subtasks, intra-option policies, and option outcome models
over an extended action space. `OaKAgent` adds utility tracking, curation, and an
option keyboard over STOMP.

### Prototype-IA (Step 12)

`PrototypeAgent` integrates all 12 Alberta Plan steps into a single agent with an
exo-cerebellum (future-feature prediction) and exo-cortex (OaK-based action
recommendation) augmenting a partner agent.

## Immutable State

All state uses `@chex.dataclass(frozen=True)` for JAX PyTree compatibility:

- `LearnerState`: Weights and optimizer state
- `LMSState`, `IDBDState`, `AutostepState`: Optimizer-specific state
- `MLPLearnerState`, `MultiHeadMLPState`: MLP-specific state
- `HordeLearnerState`, `SARSAState`: RL-specific state
- `NormalizerState`: Running statistics for normalization

This enables JAX transformations (`jit`, `vmap`, `grad`) and easy serialization.

## The Learning Loop

The `run_learning_loop` function encapsulates the training process via
`jax.lax.scan` for JIT-compiled execution:

```python
from alberta_framework import LinearLearner, IDBD, RandomWalkStream, run_learning_loop
import jax.random as jr

stream = RandomWalkStream(feature_dim=10, drift_rate=0.001)
learner = LinearLearner(optimizer=IDBD())
state, metrics = run_learning_loop(learner, stream, num_steps=10000, key=jr.key(42))
```

Analogous loops exist for each learner type: `run_mlp_learning_loop`,
`run_horde_learning_loop`, `run_sarsa_from_arrays`, `run_differential_td_from_arrays`,
`run_world_model_learning_loop`, etc.

## Metrics and Analysis

Use `compute_tracking_error` to aggregate prediction error over a window:

```python
from alberta_framework.utils import compute_tracking_error

final_error = compute_tracking_error(metrics, window=1000)
```

The `run_multi_seed_experiment` utility runs batched multi-seed experiments with
statistical comparisons and publication-quality visualization.
