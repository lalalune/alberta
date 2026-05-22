# Step 2 Plasticity Hybrid Assessment

Worker: S2E  
Scope: CBP, UPGD, and a simple UPGD/CBP-style reset hybrid for Step 2
plasticity.

## Question

The Step 2 gap is not whether the framework can train a nonlinear MLP. It can.
The open question is whether an online feature-construction or plasticity
mechanism improves over a fair MLP on streams that stress continual
nonstationarity and representation turnover.

This worker tested the narrow plasticity hypothesis:

> Continual Backprop (CBP), UPGD, or a simple UPGD plus low-utility unit reset
> hybrid might improve Step 2 over a fair MLP baseline without changing core
> learner APIs.

## Implemented Experiment

The script is
`examples/The Alberta Plan/Step2/step2_plasticity_hybrid.py`.

It compares these methods with matched hidden width, step size, sparsity, layer
norm, and ObGD bounding:

| Method | Mechanism |
|---|---|
| `mlp` | Fair `MultiHeadMLPLearner` baseline |
| `upgd` | UPGD, perturbation sigma `1e-3` |
| `upgd_low_noise` | UPGD, perturbation sigma `1e-4` |
| `cbp` | CBP replacement rate `1e-4` |
| `cbp_fast` | CBP replacement rate `5e-4` |
| `upgd_reset_hybrid` | UPGD plus script-local low-utility unit reset, rate `1e-4` |
| `upgd_reset_hybrid_fast` | Same hybrid, reset rate `5e-4` |

The hybrid did not modify core code. It uses UPGD's existing per-weight
utility tensors, averages them into per-unit utilities, maintains script-local
unit ages and replacement accumulators, then periodically reinitializes the
lowest-utility mature hidden unit while zeroing outgoing weights. This is a
simple feasibility test, not a proposed framework API.

## Experiment Matrix

Main command:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_plasticity_hybrid.py" \
  --suite both \
  --synthetic-steps 1500 \
  --digits-steps 1200 \
  --n-seeds 3 \
  --final-window 300 \
  --include-tuned \
  --output-dir output/worker_s2e_plasticity_main
```

Smoke commands:

```bash
source .venv/bin/activate
python "examples/The Alberta Plan/Step2/step2_plasticity_hybrid.py" \
  --suite synthetic \
  --synthetic-stream out_of_class_polynomial \
  --synthetic-steps 80 \
  --n-seeds 1 \
  --final-window 20 \
  --output-dir output/worker_s2e_plasticity_smoke_synth

python "examples/The Alberta Plan/Step2/step2_plasticity_hybrid.py" \
  --suite digits \
  --digits-variant iid \
  --digits-steps 80 \
  --n-seeds 1 \
  --final-window 20 \
  --output-dir output/worker_s2e_plasticity_smoke_digits
```

Scenarios:

| Family | Scenario | Purpose |
|---|---|---|
| Synthetic | `out_of_class_polynomial` | Triple-product oracle, outside pair-product features |
| Synthetic | `frequency_mismatch` | Sinusoidal oracle, outside tanh/pair-product feature banks |
| Synthetic | `compositional` | Two-layer tanh oracle |
| External digits | `digits_iid` | Ordinary online prequential digit classification |
| External digits | `digits_class_blocked` | Class-blocked epochs; strong distribution shift |
| External digits | `digits_permuted_blocks` | Alternating pixel permutations; feature-coordinate shift |

## Main Results

Positive paired differences below mean the method beat MLP on final-window MSE.

| Scenario | Best plasticity signal | Paired result vs MLP |
|---|---|---:|
| `out_of_class_polynomial` | UPGD family | `upgd_low_noise`: `+0.48598`, 3/3 wins |
| `frequency_mismatch` | UPGD family | `upgd_low_noise`: `+0.66790`, 3/3 wins |
| `compositional` | CBP, marginal | `cbp`: `+0.00243`, 2/3 wins |
| `digits_iid` | Tie/no clear winner | `upgd_low_noise`: `+0.00032`, 2/3 wins |
| `digits_class_blocked` | MLP/CBP for online tracking | `upgd_low_noise`: `-0.01747`, 0/3 wins |
| `digits_permuted_blocks` | CBP or low-noise UPGD, tiny | `cbp`: `+0.00123`, 2/3 wins |

The automatic broad scorer chose `upgd_low_noise` because it won 4/6 scenarios
on final-window MSE and had the largest mean MLP-minus-method final-window MSE:
`+0.18505`. That score is dominated by the two synthetic streams where UPGD
cuts MLP error by a large amount.

External digits test accuracy tells a different but important story:

| Scenario | UPGD low-noise test accuracy effect vs MLP |
|---|---:|
| `digits_iid` | `-0.001` average, no reliable gain |
| `digits_class_blocked` | `+0.098`, 3/3 wins |
| `digits_permuted_blocks` | `+0.183`, 3/3 wins |

This means UPGD low-noise tends to preserve broader test generalization under
block/permutation shifts, but it often pays for that with worse tracking of the
current online block.

## Critical Findings

1. UPGD is promising, but not solved.

UPGD and low-noise UPGD are clearly valuable on two out-of-class synthetic
streams. This is not a tiny statistical artifact: on polynomial and frequency
mismatch streams, UPGD variants beat MLP in every seed and by large paired
margins. This is the strongest S2E result.

The same method fails the `compositional` stream on final-window MSE and loses
badly to MLP/CBP on class-blocked digits final-window MSE. It should therefore
not be described as a robust canonical Step 2 improvement over MLP.

2. CBP alone is conservative and mostly neutral.

CBP does not meaningfully improve the synthetic construction tasks. It is
near-identical to MLP on polynomial and frequency mismatch, slightly positive
on compositional, slightly positive on digits permutation final-window MSE, and
roughly neutral on iid digits. Faster replacement helps frequency mismatch and
digits permuted-block final-window MSE slightly, but hurts compositional.

The interpretation is that CBP maintains/reset units but does not by itself
create the right features in these short Step 2 streams. It is more of a
plasticity maintenance baseline than a feature-construction advance.

3. The simple UPGD plus reset hybrid should not be canonicalized.

The hybrid mostly tracks ordinary UPGD on synthetic tasks and is worse on
digits final-window MSE. Faster reset does not fix the class-blocked or iid
digits regressions. The reset mechanism did not add a reliable benefit beyond
UPGD perturbation, and in several settings it made online tracking worse.

The likely reason is that hard unit reset and UPGD perturbation both inject
plasticity. Combining them naively can destroy features that UPGD was already
adjusting smoothly. A better hybrid would need gating, reset maturity tied to
utility confidence, or context/loss-change triggers.

4. The metric matters.

On class-blocked digits, MLP has excellent final-window accuracy because the
final window contains the current block distribution, but its held-out test
accuracy is near chance. UPGD has worse final-window MSE but better test
accuracy. This is not a contradiction; it exposes a Step 2 evaluation issue.

For Step 2, a canonical benchmark should report both:

- online tracking of the current nonstationary stream; and
- broad retained generalization across contexts/classes/permutations.

A method that wins only one of these should be called metric-specific, not
solved.

## Best Candidate

`upgd_low_noise` is the best candidate from this worker, but only as a
candidate.

It should be carried into the final Step 2 review because:

- it strongly improves two hard synthetic out-of-class streams;
- it roughly ties iid digits final-window MSE;
- it improves held-out test accuracy under class-blocked and permuted-block
  external shifts; and
- it requires no core API changes.

It should not be called the canonical Step 2 solution yet because:

- it loses on the compositional synthetic stream;
- it loses class-blocked digits final-window MSE in every seed;
- the result uses only 3 seeds and short horizons; and
- the gains depend on the evaluation metric.

## Recommended Follow-Up Experiments

Before Step 2 canonicalization, run the following focused experiments:

1. UPGD sigma sweep: `0`, `1e-5`, `3e-5`, `1e-4`, `3e-4`, `1e-3`, `3e-3`.
   Low noise was better on digits than default UPGD, so the optimum is likely
   below `1e-3`.

2. Delayed perturbation schedule: no perturbation until an initial maturity
   threshold, then gradual ramp. This may preserve MLP's early/current-block
   tracking while recovering UPGD's retained generalization.

3. Context-shift-gated perturbation: increase perturbation only after online
   loss rises or feature statistics shift. The class-blocked digits result
   suggests always-on perturbation trades away current-block fit.

4. CBP utility variants: test activation-gradient utility, outgoing-weight
   utility, and UPGD per-weight utility aggregated to units. The current CBP
   wrapper and hybrid use different utility definitions, and neither was
   uniformly best.

5. Reset-gated hybrid: apply hard resets only when unit utility remains low
   after perturbation has had time to act. The naive simultaneous hybrid is too
   destructive.

6. Longer horizons: repeat the main grid at 6000+ synthetic steps and multiple
   full digits epochs. CBP should be most useful after enough time for
   low-utility dead units to accumulate.

7. Dual-metric selection: choose candidates by a Pareto criterion over
   final-window MSE and retained test accuracy, not by one scalar. The
   class-blocked digits result makes a single metric misleading.

## Current Conclusion

No S2E plasticity method robustly improves over a fair MLP across all tested
Step 2 scenarios.

The strongest direction is low-noise UPGD. It is a real candidate for the final
review, especially if the goal includes retained external generalization under
feature/context shifts. CBP is useful as a maintenance baseline. The naive
UPGD-reset hybrid should be kept as a negative result and not adopted without a
better gating rule.
