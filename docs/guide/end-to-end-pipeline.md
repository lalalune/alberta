# End-to-End Pipeline

This guide defines the production-facing contract for composing the packaged
Alberta Plan pieces through Step 4.

## Contract

The online transition contract is:

```text
raw observation
  -> Step 2 causal, UPGD, associative, or identity features
  -> Step 3 Horde predictions and GVF update
  -> Step 4 SARSA or Horde actor-critic control update
```

Step 1 enters this pipeline through the optimizers used by Step 3 and Step 4.
The facade keeps optimizer choice explicit and reproducible; the default control
path uses LMS with ObGD bounding, while callers can configure the Step 4 facade
for IDBD or Autostep.

The pipeline state is immutable and checkpoint-friendly. The config dataclasses
provide `to_dict()` and `from_dict()` methods using JSON-compatible values.

## Minimal Use

```python
import jax.numpy as jnp
import jax.random as jr

from alberta_framework.pipeline import (
    AlbertaPipelineConfig,
    Step2FeatureConfig,
    make_alberta_pipeline,
)
from alberta_framework.steps import Step3HordeConfig, Step4SARSAConfig

config = AlbertaPipelineConfig(
    features=Step2FeatureConfig.identity(observation_dim=4),
    horde=Step3HordeConfig(
        gammas=(0.0, 0.9),
        lamdas=(0.0, 0.0),
        hidden_sizes=(),
    ),
    control=Step4SARSAConfig(n_actions=3, hidden_sizes=()),
)

pipeline = make_alberta_pipeline(config)
state = pipeline.init(jr.key(0), jnp.zeros(4))

result = pipeline.update(
    state,
    observation=jnp.ones(4),
    reward=jnp.asarray(0.1, dtype=jnp.float32),
    terminated=jnp.asarray(0.0, dtype=jnp.float32),
    horde_cumulants=jnp.asarray([1.0, 0.5], dtype=jnp.float32),
)
state = result.state
action = result.action
```

For a smoke probe:

```python
from alberta_framework.pipeline import run_pipeline_smoke

result = run_pipeline_smoke(config, steps=24, seed=0)
assert result.finite
```

## Shapes

`AlbertaPipelineConfig.feature_dim()` is the feature size consumed by both Step
3 and Step 4. It is derived from `Step2FeatureConfig`, `Step2UPGDConfig`,
`Step2AssociativePipelineConfig`, or identity features.

For each transition:

- `observation`: raw vector with shape `(observation_dim,)`;
- `horde_cumulants`: vector with shape `(n_demons,)`, where `n_demons` is
  `len(Step3HordeConfig.gammas)`;
- `reward`: scalar SARSA reward;
- `terminated`: scalar flag; nonzero means the SARSA bootstrap discount is zero;
- `result.features`: Step 2 feature vector;
- `result.horde_predictions`: Step 3 predictions before the Horde update;
- `result.q_values`: Step 4 action values for the next feature vector;
- `result.action`: selected next action.

For array scans, initialize the pipeline with the observation preceding the
first row, then pass arrays of next observations, rewards, termination flags,
and Horde cumulants to `AlbertaPipeline.run_arrays()`.

## Step 2 Modes

The end-to-end facade exposes these Step 2 modes:

- `identity`: raw observations pass through unchanged.
- `temporal_context`: causal phase-product features for continuous streams with
  observable temporal drift.
- `upgd`: target-structure UPGD hidden features for supervised vector-output
  streams.
- `associative`: fixed-budget associative prediction features for discrete
  token-context streams, with optional adaptive feature-family, suffix-window,
  and effective-budget scope controls.

For `associative`, observations are integer token context windows with shape
`(block_size,)`. Single-step updates require `associative_label`; array scans
require `associative_labels`. The pipeline uses the pre-write associative
prediction as the Step 2 feature vector, then updates the table on the same
transition.

```python
import jax.numpy as jnp
import jax.random as jr

from alberta_framework.pipeline import (
    AlbertaPipelineConfig,
    Step2AssociativePipelineConfig,
    make_alberta_pipeline,
)
from alberta_framework.steps import Step3HordeConfig, Step4SARSAConfig

config = AlbertaPipelineConfig(
    step2="associative",
    associative=Step2AssociativePipelineConfig(
        vocab_size=16,
        block_size=4,
        max_features=64,
    ),
    horde=Step3HordeConfig(gammas=(0.0,), lamdas=(0.0,), hidden_sizes=()),
    control=Step4SARSAConfig(n_actions=2, hidden_sizes=()),
)

pipeline = make_alberta_pipeline(config)
state = pipeline.init(jr.key(0), jnp.asarray([1, 2, 3, 4], dtype=jnp.int32))

result = pipeline.update(
    state,
    observation=jnp.asarray([2, 3, 4, 5], dtype=jnp.int32),
    reward=jnp.asarray(0.0, dtype=jnp.float32),
    terminated=jnp.asarray(0.0, dtype=jnp.float32),
    horde_cumulants=jnp.asarray([1.0], dtype=jnp.float32),
    associative_label=jnp.asarray(6, dtype=jnp.int32),
)
```

Step 4 can be used through the SARSA facade or the Horde actor-critic facade.
In SARSA mode, the Step 3 GVFs are also mirrored as prediction demons inside
the SARSA Horde, so the control learner can update Q heads and auxiliary GVF
heads from the same transition. In Horde actor-critic mode, the pipeline keeps a
single synchronized critic state instead of running a parallel discarded Step 3
update.

## Boundary

This is pipeline glue, not a new research result. Step 2 feature augmentation is
exposed through identity, temporal-context, UPGD, and associative modes. The
UPGD and associative modes are useful integration surfaces, but they do not by
themselves prove arbitrary recursive feature discovery. The associative mode
can learn soft scope gates inside a declared finite feature family/window/table
budget, but those outer resource bounds remain explicit configuration. The
UPGD mode remains a supervised hidden-feature learner rather than a theorem of
representation optimality.

The accepted control path is still SARSA. Horde actor-critic is implemented and
tested, but it is not promoted as the canonical Step 4 learner until its
evidence improves against the Q/SARSA baselines. Average-reward control, richer
continuing benchmarks, and full Step 5/6 integration remain separate research
milestones.
