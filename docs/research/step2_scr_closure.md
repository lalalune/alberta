# Step 2 Slowly-Changing Regression Closure

Date: 2026-05-05

## Scope

This note records the follow-up on the Slowly-Changing Regression (SCR)
external-evidence gap. The earlier `--long-scr` run used a Dohare-style local
configuration (`m=20`, `f=15`, target hidden units=100) for 20,000 online
steps and showed that the default convex portfolio slightly lost to the best
fair MLP. The question here is whether a causal router can close that feasible
local comparator without claiming the million-step public reproduction scale.

## Protocol Boundary

The local SCR stream matches the important structural ingredients: binary
inputs, slow-changing bits, iid random bits, a constant bias bit, and a fixed
LTU target network. The `--long-scr` preset uses:

- 20,000 online steps;
- 3 seeds;
- final window 5,000;
- `m=20`, `f=15`;
- flip interval `T=1000`;
- target hidden units=100;
- `beta=0.7`.

This is not the full Dohare public scale. The published-scale flag remains
false unless the run uses the paper SCR preset and at least 1,000,000 online
examples.

## Commands

Negative convex reference:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_published_stressors.py" \
  --benchmarks slowly_changing_regression \
  --long-scr \
  --output-dir outputs/step2_canonical \
  --result-prefix published_stressors_long_scr
```

Focused causal router search:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_scr_router_search.py" \
  --long-scr \
  --output-dir outputs/step2_canonical \
  --result-prefix scr_router_search_long
```

The recovered worker outputs for the three long router variants are under
`outputs/step2_scr_router_probe/`.

## Results

Positive paired differences mean best fair MLP final-window MSE minus portfolio
final-window MSE.

| Run | Router | Final-window MSE vs best fair MLP | Wins/losses/ties | Status |
|---|---|---:|---:|---|
| `published_stressors_long_scr` | `convex` | `-0.0011 +/- 0.0003` | `0/3/0` | negative reference |
| `mlp_selector_long` | `mlp_selector` | `+0.000069 +/- 0.000060` | `3/0/0` | feasible local comparator closed |
| `guarded_best_mlp_long` | `guarded_best_mlp` | `+0.000099 +/- 0.000074` | `3/0/0` | best recovered router |
| `meta_slow_long` | `meta` | `+0.000067 +/- 0.000056` | `3/0/0` | feasible local comparator closed |

The causal guarded route is the cleanest interpretation: it uses the full
portfolio only when its online router EMA is within a small tolerance of the
better fair-MLP route, otherwise it falls back to the MLP-safe path. This fixes
the local SCR regression without changing the stream or the best-fair-MLP
comparison.

## Conclusion

Status: partial.

The feasible 20,000-step local SCR comparator is closed by causal guarded/MLP
routing. That matters because it turns the earlier negative result into a
bounded routing failure rather than a learner-capacity failure.

This still does not support an unqualified Step 2 claim. The margin is tiny,
the run is only 3 seeds, `T=1000` is shorter than the public `T=10000` setting,
and no million-step public-scale reproduction has been run. Keep SCR in the
external-evidence gap list until a true paper-scale run is nonnegative against
the fair MLP grid.
