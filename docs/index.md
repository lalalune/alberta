# Alberta Framework

A research-first framework for the Alberta Plan: Building the foundations of Continual AI.

## Overview

The Alberta Framework implements production-facing kernels through Step 7 of
the Alberta Plan. Step 1 is complete for fixed-feature continual supervised
learning. Step 2 is accepted for the current supervised empirical matrix.
Step 3 is accepted for given-feature GVF/Horde-style prediction. Step 4 is
accepted for SARSA control, while actor-critic remains implemented but
provisional as the canonical control learner. Steps 5-7 have primitive
production surfaces for average-reward prediction, differential SARSA, and
bounded one-step Dyna planning.

**Core Philosophy**: Temporal uniformity — every component updates at every time step.

## Key Features

- **Adaptive Optimizers**: IDBD and Autostep with per-weight meta-learned step-sizes
- **Non-stationary Streams**: Random walk, abrupt change, and cyclic target generators
- **Gymnasium Integration**: Wrap RL environments as prediction streams
- **Publication-Quality Analysis**: Multi-seed experiments, statistical tests, and visualization

## Quick Example

```python
import jax.random as jr
from alberta_framework import LinearLearner, IDBD, run_learning_loop
from alberta_framework.streams import RandomWalkTarget

# Create a non-stationary prediction problem
stream = RandomWalkTarget(
    feature_dim=10,
    key=jr.PRNGKey(0),
    walk_std=0.01,
)

# Train with adaptive step-sizes
learner = LinearLearner(optimizer=IDBD(initial_step_size=0.01))
state, metrics = run_learning_loop(
    learner=learner,
    stream=stream,
    num_steps=10000,
    key=jr.PRNGKey(42),
)

print(f"Final error: {metrics[-1]['squared_error']:.4f}")
```

## Installation

```bash
pip install alberta-framework
```

For development with all optional dependencies:

```bash
pip install alberta-framework[dev,gymnasium,analysis,docs]
```

## Design Principles

- **Immutable State**: All state uses NamedTuples for JAX compatibility
- **Functional Style**: Pure functions enable `jit`, `vmap`
- **Composition**: Learners accept optimizers as parameters
- **Temporal Uniformity**: Every component updates at every time step

## Project Status

This is an active research framework. The package API may change as the
remaining research boundaries are closed, especially TD/GVF feature discovery,
off-policy nonlinear Horde learning, average-reward actor-critic, model-bias
control in planning, and actor-critic promotion.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{alberta_framework,
  title = {Alberta Framework},
  author = {Lawson, Keith},
  year = {2026},
  url = {https://github.com/j-klawson/alberta-framework}
}
```

## Questions & Contact

Open an issue on [GitHub](https://github.com/j-klawson/alberta-framework/issues).

## License

Apache-2.0
