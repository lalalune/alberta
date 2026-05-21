# Step 2 Universal Sieve Probe

Worker: THEORY-TESTS

## Question

Does the current target-structure UPGD candidate behave like a useful online
sieve on theory-friendly target classes when only the feature/capacity budget is
grown?

This is an empirical check of assumptions behind a universal representation
learning story. It is not a proof. The protocol intentionally avoids routers,
hand-picked per-regime feature search, and post-hoc expert selection.

## Protocol

- Target families: low-degree polynomial, Fourier/sinusoidal, radial bumps,
  sparse interactions, and piecewise threshold functions.
- Learner under test: `UPGDLearner.step2_default(..., loss_normalization="target_structure")`.
- Comparator: capacity-matched `MultiHeadMLPLearner`.
- Capacities: one hidden layer with 8, 16, 32, or 64 units.
- Metric: prequential final-window MSE, averaged across seeds.
- Smoke: 2 seeds, 600 steps, final window 150, capacities 8/16/32.
- Moderate: 5 seeds, 2500 steps, final window 600, capacities 8/16/32/64.

## Moderate Result

| family | learner | h=8 | h=16 | h=32 | h=64 | monotone/tol | material gain |
|---|---|---|---|---|---|---|---|
| polynomial | upgd | 0.0671 | 0.0372 | 0.0380 | 0.0506 | False | 44.5% |
| polynomial | mlp | 0.0811 | 0.0423 | 0.0377 | 0.0505 | False | 53.5% |
| fourier | upgd | 0.4014 | 0.4275 | 0.4701 | 0.5597 | False | 0.0% |
| fourier | mlp | 0.4031 | 0.4266 | 0.4738 | 0.5212 | False | 0.0% |
| radial_bumps | upgd | 0.5685 | 0.5641 | 0.5712 | 0.6162 | False | 0.8% |
| radial_bumps | mlp | 0.5934 | 0.6317 | 0.6841 | 0.7248 | False | 0.0% |
| sparse_interactions | upgd | 0.1961 | 0.1810 | 0.1798 | 0.2005 | False | 8.3% |
| sparse_interactions | mlp | 0.1986 | 0.1858 | 0.1793 | 0.1859 | False | 9.7% |
| piecewise_threshold | upgd | 0.1367 | 0.1416 | 0.1557 | 0.2329 | False | 0.0% |
| piecewise_threshold | mlp | 0.1445 | 0.1427 | 0.1626 | 0.2429 | False | 1.2% |

## Smoke Result

| family | learner | h=8 | h=16 | h=32 | monotone/tol | material gain |
|---|---|---|---|---|---|
| polynomial | upgd | 0.1643 | 0.1002 | 0.1025 | True | 39.0% |
| polynomial | mlp | 0.2118 | 0.0946 | 0.0877 | True | 58.6% |
| fourier | upgd | 0.3778 | 0.4167 | 0.5304 | False | 0.0% |
| fourier | mlp | 0.3956 | 0.4427 | 0.5251 | False | 0.0% |
| radial_bumps | upgd | 1.2750 | 1.2514 | 1.2582 | True | 1.9% |
| radial_bumps | mlp | 1.3945 | 1.4081 | 1.5256 | False | 0.0% |
| sparse_interactions | upgd | 0.3656 | 0.3176 | 0.3154 | True | 13.7% |
| sparse_interactions | mlp | 0.3163 | 0.2957 | 0.3385 | False | 6.5% |
| piecewise_threshold | upgd | 0.1758 | 0.1794 | 0.1977 | False | 0.0% |
| piecewise_threshold | mlp | 0.2059 | 0.1884 | 0.2033 | False | 8.5% |

## Interpretation

The smoke run was partly encouraging: UPGD showed monotone-with-tolerance or
material improvement on polynomial, radial bumps, and sparse interactions. The
moderate run did not preserve that pattern. It found material UPGD improvement
only on the polynomial family, and no family was monotone with the configured
3% adjacent-capacity tolerance.

This is evidence against the strong empirical form of the sieve assumption for
the current target-structure UPGD setup. The learner can exploit extra capacity
on the smooth polynomial target, but simply adding hidden units does not reliably
improve final-window online MSE across Fourier, radial, sparse-interaction, or
piecewise families. Wider models often degrade, which suggests optimization,
plasticity, initialization, or online stability constraints dominate the
existence of a universal approximating representation in this protocol.

The fair MLP comparator shows a similar pattern, so the negative result is not
unique to UPGD perturbation. The evidence says the proof assumptions may remain
mathematically plausible but are not operationally supported by this compact
online benchmark without additional mechanisms or better capacity scaling.

## Artifacts

- Script: `examples/The Alberta Plan/Step2/step2_universal_sieve_probe.py`
- Tests: `tests/test_step2_universal_sieve_probe.py`
- Smoke output: `output/step2_universal_sieve_probe_smoke/`
- Moderate output: `output/step2_universal_sieve_probe_moderate/`
