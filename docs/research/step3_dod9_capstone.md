# Step 3 DoD-9 Capstone Evidence

This note summarizes the canonical local DoD-9 critic-for-control artifacts in
`output/step3_dod9`.

## Command

The 10-seed result was completed by appending seeds 8 and 9 to the existing
8-seed 30k-step run, using the same `run_one`/`build_agent` code path from:

```bash
python "examples/The Alberta Plan/Step3/dod9_capstone_sweep.py"
```

Canonical output files:

- `output/step3_dod9/results.csv`
- `output/step3_dod9/summary.json`

## Configuration

- Seeds: 10
- Steps per seed: 30000
- Final evaluation window: 5000 steps
- Hidden sizes: `[32]`
- Conditions: `baseline_sarsa`, `sarsa_prediction_horde`,
  `sarsa_horde_cbp_history`

## Results

| Condition | Total reward mean | Total reward std | Last-window reward mean | Last-window std | Seeds |
|---|---:|---:|---:|---:|---:|
| `baseline_sarsa` | 10681.1 | 2164.3 | 2088.6 | 275.0 | 10 |
| `sarsa_prediction_horde` | 11556.3 | 1059.5 | 2173.3 | 73.0 | 10 |
| `sarsa_horde_cbp_history` | 11883.0 | 302.3 | 2197.7 | 33.1 | 10 |

## Verdict

PASS. Auxiliary prediction demons improve the SARSA control baseline, and the
history+CBP condition is best on both total reward and final-window reward.
