# Step 2 Native Deep Lifecycle Worker Diagnosis

Date: 2026-05-05.

Worker: `DEEP-LIFECYCLE`.

## Question

Can a compact native deep feature lifecycle variant fix the three hard-blocker
failures against the fair MLP baseline while preserving one learner and temporal
uniformity?

The prior 5-seed x 800-step hard audit left native deep lifecycle at `3/6`.
The recurrent failed probes were:

- `nonlinear`
- `compositional`
- `digits_iid`

## Changes Tested

Core changes:

- Corrected soft-gated candidate utility to score current loss minus proposed
  loss under the gate change.
- Added opt-in normalized candidate updates:
  - candidate readout update divided by candidate activation energy;
  - candidate incoming-row update divided by layer-input energy.

Experiment-runner changes:

- Added `--streams` for focused probe subsets.
- Added shallow native lifecycle variants using one hidden layer:
  - `deep_shallow_nlms`
  - `deep_shallow_soft_gate`
  - `deep_shallow_net2net`

All candidate and active components still update every time step.

## Focused Failed-Probe Evidence

Exploratory run, recorded in
`outputs/step2_deep_worker_failed_probe_5seed_800/deep_feature_lifecycle_results.json`.
This run also included `deep_shallow_safe`, which was trimmed from the runner
after it failed to become the best variant on any probe.

```bash
python "examples/The Alberta Plan/Step2/step2_deep_feature_lifecycle.py" \
  --seeds 5 --num-steps 800 --final-window 200 \
  --streams nonlinear compositional digits_iid \
  --methods mlp_64 mlp_64_64 deep_shallow_nlms deep_shallow_safe \
    deep_shallow_soft_gate deep_shallow_net2net deep_active_perturb_low \
  --output-dir outputs/step2_deep_worker_failed_probe_5seed_800 \
  --note-path docs/research/step2_deep_worker_failed_probe_5seed_800.md
```

Positive `best_mlp - method` means the lifecycle variant beat the best fair MLP.

| Probe | Best variant in focused run | `best_mlp - method` | Paired wins |
|---|---:|---:|---:|
| `nonlinear` | `deep_shallow_nlms` | `+0.00225 +/- 0.00034` | `5/5` |
| `compositional` | `deep_shallow_net2net` | `+0.00403 +/- 0.00257` | `3/5` |
| `digits_iid` | `deep_shallow_net2net` | `-0.00072 +/- 0.00068` | `2/5` |

Diagnosis: the failed probes were partly a two-layer-trunk penalty. Shallow
lifecycle variants reduce the gap and can win individual probes, but the wins
are mechanism-specific rather than one robust variant.

## Six-Probe Matrix

Run:

```bash
python "examples/The Alberta Plan/Step2/step2_deep_feature_lifecycle.py" \
  --seeds 5 --num-steps 800 --final-window 200 \
  --methods mlp_64 mlp_64_64 deep_shallow_nlms deep_shallow_soft_gate \
    deep_shallow_net2net \
  --output-dir outputs/step2_deep_worker_six_probe_5seed_800 \
  --note-path docs/research/step2_deep_worker_six_probe_5seed_800.md
```

| Probe | `deep_shallow_nlms` | `deep_shallow_soft_gate` | `deep_shallow_net2net` |
|---|---:|---:|---:|
| `nonlinear` | `+0.00225` | `-0.00016` | `-0.00056` |
| `interaction` | `-0.12251` | `-0.00083` | `+0.02175` |
| `out_of_class_polynomial` | `-0.72135` | `-0.04420` | `-0.04055` |
| `frequency_mismatch` | `-0.59659` | `-0.01635` | `+0.02043` |
| `compositional` | `-0.17147` | `+0.00193` | `-0.00012` |
| `digits_iid` | `-0.00198` | `-0.00105` | `-0.00095` |

Best single variant: `deep_shallow_net2net`, `2/6` positive probes.

## Verdict

Native deep lifecycle remains rejected as the promoted Step 2 path.

The compact changes produce useful diagnostics and one real single-probe
improvement (`deep_shallow_nlms` on nonlinear), but no single native lifecycle
variant beats the best fair MLP across the six-probe matrix. The blocker is now
sharper: candidate learning can help a specific probe, but utility and promotion
signals do not transfer across nonlinear, polynomial/frequency, compositional,
and digits regimes.
