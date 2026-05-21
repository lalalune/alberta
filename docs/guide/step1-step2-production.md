# Step 1/2 Production Kernels

This page defines the package-quality surface for the completed Step 1 kernel
and the promoted supervised Step 2 kernel.

## Scope

Step 1 is complete for public, reproducible methods. The package exposes LMS,
IDBD, Autostep, AdaGain, Adam, RMSprop, NADALINE, and the online normalizers
used in the canonical ablations. It does not expose `Auto (Degris in prep.)`
because no public algorithm specification was found.

Step 2 is complete for the current supervised empirical acceptance matrix. The
base production learner is target-structure UPGD. For retained class-view
streams, the package also exposes a single UPGD-memory learner that combines
UPGD with fixed-budget D20-style prototype memory. For discrete sequence
streams, the package exposes a fixed-budget associative feature learner that
turns token contexts into prediction features with learned row utility and
replacement. These components update every step; none of them is a post hoc
route-selecting portfolio.

## Imports

```python
from alberta_framework.steps import (
    Step1KernelConfig,
    Step2AssociativeConfig,
    Step2HybridConfig,
    Step2KernelConfig,
    Step2StrictDigitReadoutConfig,
    Step2TemporalContextConfig,
    make_step1_learner,
    make_step1_stream,
    make_step2_associative_learner,
    make_step2_hybrid_learner,
    make_step2_learner,
    make_step2_strict_digit_readout_learner,
    make_step2_temporal_context,
    make_step2_temporal_learner,
    run_step2_associative_smoke,
)
```

## Step 1 Kernel

```python
from alberta_framework.steps import Step1KernelConfig, make_step1_learner

config = Step1KernelConfig(
    optimizer="autostep",
    normalizer="ema",
    feature_dim=20,
    num_relevant=5,
)
learner = make_step1_learner(config)
state = learner.init(config.feature_dim)
```

Use `run_step1_smoke()` in integration tests. Use the canonical scripts under
`examples/The Alberta Plan/Step1/` for paper claims.

## Step 2 Kernel

```python
import jax.random as jr
from alberta_framework.steps import Step2KernelConfig, make_step2_learner

config = Step2KernelConfig(n_heads=10, feature_dim=64)
learner = make_step2_learner(config)
state = learner.init(config.feature_dim, jr.key(0))
```

The default learner config is equivalent to:

```python
from alberta_framework import UPGDLearner

learner = UPGDLearner.step2_default(n_heads=10)
```

Key defaults:

- hidden size 32;
- `loss_normalization="target_structure"`;
- `ObGDBounding(kappa=0.5)`;
- `perturbation_sigma=1e-4`;
- Rademacher perturbation noise;
- perturbation every 16 updates;
- no default router, portfolio, MLP fallback, or dataset switch.

For strict one-hot online digit/readout consistency, use the separate heavier
two-timescale simplex branch:

```python
import jax.random as jr
from alberta_framework.steps import (
    Step2StrictDigitReadoutConfig,
    make_step2_strict_digit_readout_learner,
)

config = Step2StrictDigitReadoutConfig(n_heads=10)
learner = make_step2_strict_digit_readout_learner(config)
state = learner.init(feature_dim=64, key=jr.key(0))
```

This branch corresponds to
`UPGDLearner.step2_strict_digit_readout_default(n_heads)`. It is promoted for
the digit readout conflict, not as the broad synthetic/vector default.

For compact OPMNIST-style retained class views, use the packaged hybrid:

```python
import jax.random as jr
from alberta_framework.steps import Step2HybridConfig, make_step2_hybrid_learner

config = Step2HybridConfig(n_heads=10, feature_dim=784)
learner = make_step2_hybrid_learner(config)
state = learner.init(jr.key(0))
```

The hybrid default uses 20 prototype slots per class, an adaptive novelty
allocation target of 0.18, a neutral memory blend prior, and an update-time
target-trace prior for persistent supervised streams.  Ordinary `predict()`
calls remain observation-based UPGD plus memory; the trace prior only affects
prequential `update()` predictions.

For streams with observable temporal drift, use the phase-context helper:

```python
import jax.random as jr
from alberta_framework.steps import (
    Step2TemporalContextConfig,
    make_step2_temporal_context,
    make_step2_temporal_learner,
)

config = Step2TemporalContextConfig(feature_dim=12, n_heads=1)
context = make_step2_temporal_context(config)
learner = make_step2_temporal_learner(config)
context_state, features = context.step(context.init(), observation)
learner_state = learner.init(features.shape[0], jr.key(0))
```

This helper uses a dense bank of causal phase-product features.  It is promoted
for rotating-subspace stressors, not as a default replacement for raw UPGD.

For sparse discrete contexts, use the associative Step 2 kernel:

```python
import jax.random as jr
from alberta_framework.steps import (
    Step2AssociativeConfig,
    make_step2_associative_learner,
)

config = Step2AssociativeConfig(
    vocab_size=32,
    block_size=4,
    max_features=256,
    feature_family="token_suffix_pair",
)
learner = make_step2_associative_learner(config)
state = learner.init(jr.key(0))
```

The associative learner stores context-derived feature rows in fixed JAX arrays,
predicts before writing the current target, credits rows by one-step loss
advantage, and replaces low-utility rows when the budget is full. It is the
production version of the sparse-KV sequence-memory probe: it closes the
"experiment-only mechanism" gap. Optional adaptive controls can learn soft
feature-family, suffix-window, and effective-budget gates from causal loss
advantage/replacement pressure. They are disabled by default, so canonical
claims must state whether the adaptive scope controls were used.

Use `run_step2_associative_smoke()` as the package-level sanity check for this
kernel.

## CLI

```bash
alberta-step1-smoke --steps 256
alberta-step2-smoke --steps 128
alberta-evidence-gate --step all
```

The smoke commands are deployment checks. They do not replace the canonical
multi-seed evidence tests:

```bash
pytest tests/test_step1_replication.py tests/test_step2_canonical.py -q
```

## Claim Boundary

Do not claim that Step 2 proves universal representation learning. The current
package supports this claim:

> The current supervised Step 2 empirical acceptance matrix is closed by a
> single non-router target-structure UPGD learner against same-run best fair MLP
> baselines.

For retained one-hot image-view streams, the packaged UPGD-memory learner has
positive compact OPMNIST evidence, but it is a separate retained-view kernel
rather than a theorem of universal representation learning.

For sparse discrete sequence streams, the packaged associative learner provides
continual feature construction, learned utility, and budgeted replacement inside
the core package and end-to-end pipeline. It now has optional learned scope
controllers for feature family, suffix window, and effective budget, but the
outer operation set and maximum table budget remain declared finite resources.
This is evidence for a useful Step 2 mechanism rather than a formal solution to
arbitrary recursive feature discovery.

Full 800-task OPMNIST remains an external-scale replication boundary until the
48M-example protocol completes.
