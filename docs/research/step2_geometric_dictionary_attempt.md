# Step 2 Geometric Dictionary Attempt

Seeds: 3; steps: 1500; final-window: 300.

Command:

```bash
python "examples/The Alberta Plan/Step2/step2_geometric_feature_probe.py" --suite --seeds 3 --num-steps 1500 --final-window 300 --methods geometric,single_mechanism,mlp_32x32_no_ln,mlp_64x64_no_ln
```

## Numeric results

### frequency

| Method | Final MSE | Heldout MSE | Active centers | Insertions |
|---|---:|---:|---:|---:|
| `geometric` | 0.8692 +/- 0.0420 | 0.7422 +/- 0.1040 | 48.0 | 736.3 |
| `mlp_32x32_no_ln` | 0.1966 +/- 0.0563 | 0.2179 +/- 0.0660 | 0.0 | 0.0 |
| `mlp_64x64_no_ln` | 0.2594 +/- 0.0167 | 0.2828 +/- 0.0619 | 0.0 | 0.0 |
| `single_mechanism` | 0.0994 +/- 0.0314 | 0.1126 +/- 0.0241 | 0.0 | 0.0 |
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `geometric`: -0.6726; `geometric` wins 0/3 seeds.
Paired delta `single_mechanism` minus `geometric`: -0.7698; `geometric` wins 0/3 seeds.

### interaction

| Method | Final MSE | Heldout MSE | Active centers | Insertions |
|---|---:|---:|---:|---:|
| `geometric` | 0.8232 +/- 0.0271 | 0.6809 +/- 0.1184 | 48.0 | 866.7 |
| `mlp_32x32_no_ln` | 0.7814 +/- 0.1315 | 0.5489 +/- 0.1094 | 0.0 | 0.0 |
| `mlp_64x64_no_ln` | 0.7826 +/- 0.1217 | 0.6928 +/- 0.1775 | 0.0 | 0.0 |
| `single_mechanism` | 0.0970 +/- 0.0237 | 0.1138 +/- 0.0459 | 0.0 | 0.0 |
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `geometric`: -0.0418; `geometric` wins 1/3 seeds.
Paired delta `single_mechanism` minus `geometric`: -0.7262; `geometric` wins 0/3 seeds.

### nonlinear

| Method | Final MSE | Heldout MSE | Active centers | Insertions |
|---|---:|---:|---:|---:|
| `geometric` | 0.2978 +/- 0.0090 | 0.2110 +/- 0.0268 | 48.0 | 765.3 |
| `mlp_32x32_no_ln` | 0.0847 +/- 0.0087 | 0.0771 +/- 0.0223 | 0.0 | 0.0 |
| `mlp_64x64_no_ln` | 0.1092 +/- 0.0065 | 0.0938 +/- 0.0179 | 0.0 | 0.0 |
| `single_mechanism` | 0.1568 +/- 0.0049 | 0.2437 +/- 0.0780 | 0.0 | 0.0 |
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `geometric`: -0.2131; `geometric` wins 0/3 seeds.
Paired delta `single_mechanism` minus `geometric`: -0.1410; `geometric` wins 0/3 seeds.

### polynomial

| Method | Final MSE | Heldout MSE | Active centers | Insertions |
|---|---:|---:|---:|---:|
| `geometric` | 1.6868 +/- 0.1585 | 1.4990 +/- 0.4404 | 48.0 | 858.3 |
| `mlp_32x32_no_ln` | 1.5213 +/- 0.1979 | 1.1248 +/- 0.3013 | 0.0 | 0.0 |
| `mlp_64x64_no_ln` | 1.5401 +/- 0.1325 | 1.2133 +/- 0.2671 | 0.0 | 0.0 |
| `single_mechanism` | 0.5023 +/- 0.1104 | 0.3544 +/- 0.0990 | 0.0 | 0.0 |
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `geometric`: -0.1655; `geometric` wins 0/3 seeds.
Paired delta `single_mechanism` minus `geometric`: -1.1845; `geometric` wins 0/3 seeds.

### rare

| Method | Final MSE | Heldout MSE | Active centers | Insertions |
|---|---:|---:|---:|---:|
| `geometric` | 0.2062 +/- 0.0037 | 0.6141 +/- 0.0910 | 48.0 | 662.7 |
| `mlp_32x32_no_ln` | 0.1177 +/- 0.0039 | 0.4915 +/- 0.0635 | 0.0 | 0.0 |
| `mlp_64x64_no_ln` | 0.1375 +/- 0.0011 | 0.5704 +/- 0.0906 | 0.0 | 0.0 |
| `single_mechanism` | 0.1208 +/- 0.0219 | 0.1412 +/- 0.0197 | 0.0 | 0.0 |
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `geometric`: -0.0884; `geometric` wins 0/3 seeds.
Paired delta `single_mechanism` minus `geometric`: -0.0853; `geometric` wins 0/3 seeds.

### triple

| Method | Final MSE | Heldout MSE | Active centers | Insertions |
|---|---:|---:|---:|---:|
| `geometric` | 1.3588 +/- 0.1553 | 1.2571 +/- 0.1260 | 48.0 | 916.0 |
| `mlp_32x32_no_ln` | 0.9047 +/- 0.1161 | 0.8598 +/- 0.2475 | 0.0 | 0.0 |
| `mlp_64x64_no_ln` | 0.9196 +/- 0.1179 | 0.7675 +/- 0.2209 | 0.0 | 0.0 |
| `single_mechanism` | 0.1120 +/- 0.0290 | 0.1046 +/- 0.0548 | 0.0 | 0.0 |
Best fair MLP: `mlp_32x32_no_ln`. Paired delta best MLP minus `geometric`: -0.4541; `geometric` wins 0/3 seeds.
Paired delta `single_mechanism` minus `geometric`: -1.2468; `geometric` wins 0/3 seeds.

## Decision

Status: **rejected**.

Promotion rule: nonlinear must beat the best fair MLP and the geometric dictionary must not lose most algebraic probes against `single_mechanism`.

## Failure notes

The dictionary filled its center budget on every task and then churned hundreds of replacements. That indicates the novelty/residual gate is finding local residual patches, but the patches are not reusable enough to compete with gradient-shaped MLP features on nonlinear or with recursive product construction on algebraic probes.

This is a local interpolation dictionary, not a recursive algebraic constructor. It has no exact mechanism for persistent products, triples, rare-task retention, or sinusoidal identities, and the 3-seed suite reflects that limitation.
