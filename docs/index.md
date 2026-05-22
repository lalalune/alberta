# Alberta Framework

A research-first framework for the Alberta Plan: Building the foundations of Continual AI.

## Overview

The Alberta Framework implements all 12 steps of the Alberta Plan (v0.26.0). The
framework provides production-facing kernels for the complete continual learning
roadmap: from adaptive step-size supervised learning through nonlinear GVF/Horde
prediction, SARSA and actor-critic control, average-reward optimization, Dyna
planning with guarded dreaming, STOMP temporal abstraction, and the OaK option
architecture with Prototype-IA multi-agent augmentation.

**Core Philosophy**: Temporal uniformity — every component updates at every time step.

## Key Features

- **All 12 Alberta Plan Steps**: Complete implementation confirmed by end-to-end benchmarks
- **Adaptive Optimizers**: IDBD and Autostep with per-weight meta-learned step-sizes
- **Nonlinear Learners**: MLPLearner with ObGD, HordeLearner, MultiHeadMLPLearner
- **Control Agents**: SARSAAgent, HordeActorCriticAgent, DifferentialSARSAAgent
- **Planning**: GuardedDreamer with error-gated, buffer-anchored imagined transitions
- **Temporal Abstraction**: STOMPAgent with intra-option policies and option outcome models
- **Non-stationary Streams**: Random walk, abrupt change, and cyclic target generators
- **Gymnasium Integration**: Wrap RL environments as prediction or control streams
- **Publication-Quality Analysis**: Multi-seed experiments, statistical tests, and visualization

## Quick Example

```python
import jax.random as jr
from alberta_framework import LinearLearner, IDBD, RandomWalkStream, run_learning_loop

# Non-stationary stream where target weights drift over time
stream = RandomWalkStream(feature_dim=10, drift_rate=0.001)

# Train with IDBD meta-learned step-sizes
learner = LinearLearner(optimizer=IDBD())

# JIT-compiled training via jax.lax.scan
state, metrics = run_learning_loop(learner, stream, num_steps=10000, key=jr.key(42))
print(f"Final tracking error: {metrics['squared_error'][-1]:.4f}")
```

## Installation

```bash
pip install alberta-framework
```

For development with all optional dependencies:

```bash
pip install alberta-framework[dev,gymnasium,analysis,docs]
```

## Alberta Plan Status

| Step | Focus | Status |
|------|-------|--------|
| 1 | Fixed-feature continual supervised learning (IDBD/Autostep) | **Complete** |
| 2 | Nonlinear function approximation (MLP, ObGD, IDBD-MLP) | **Complete** |
| 3 | GVF/Horde prediction (HordeLearner, per-head trace decay) | **Complete** |
| 4 | Continual control (SARSA, Horde actor-critic) | **Complete** |
| 5 | Off-policy nonlinear Horde prediction | **Complete** |
| 6 | Average-reward optimization (DifferentialTD, DifferentialSARSA) | **Complete** |
| 7 | Dyna planning with one-step world model | **Complete** |
| 8 | World model facade with ensemble prediction | **Complete** |
| 9 | Guarded dreaming (error-gated, buffer-anchored imagined transitions) | **Complete** |
| 10 | STOMP temporal abstraction (subtasks, intra-option policies) | **Complete** |
| 11 | OaK architecture (utility tracking, curation, option keyboard) | **Complete** |
| 12 | Prototype-IA (exo-cerebellum + exo-cortex multi-agent augmentation) | **Complete** |

## Design Principles

- **Immutable State**: `@chex.dataclass(frozen=True)` for JAX PyTree compatibility
- **Functional Style**: Pure functions enable `jit`, `vmap`, `grad`
- **Composition**: Learners accept independent Optimizer, Bounder, and Normalizer ABCs
- **Temporal Uniformity**: Every component updates at every time step

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{alberta_framework,
  title = {Alberta Framework},
  author = {Walters, Shaw},
  year = {2026},
  url = {https://github.com/lalalune/alberta}
}
```

## Questions & Contact

Open an issue on [GitHub](https://github.com/lalalune/alberta/issues).

## License

Apache-2.0
