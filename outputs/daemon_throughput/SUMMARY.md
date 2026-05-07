# Daemon End-to-End Throughput

Closes the daemon end-to-end throughput boundary identified in DoD-10. Measures the production `AlbertaPipeline` driven by a synthetic JSON-line transport with parse, predict, update, serialize, and periodic checkpoint phases.

Generated: 2026-05-06T20:29:19.073268

Acceptance threshold: >=500 steps/sec.


## Results

| features | n_demons | hidden | feat_dim | steps/sec | parse ms | predict ms | update ms | serialize ms | ckpt ms | slowest | gate |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|---|
| no_features | 5 | () | 8 | 399.2 | 0.242 | 0.134 | 0.638 | 0.194 | 129.534 | checkpoint | FAIL |
| no_features | 5 | 32 | 8 | 343.6 | 0.283 | 0.228 | 0.939 | 0.220 | 123.863 | checkpoint | FAIL |
| no_features | 25 | () | 8 | 192.9 | 0.337 | 0.282 | 1.215 | 0.465 | 288.346 | checkpoint | FAIL |
| no_features | 25 | 32 | 8 | 148.7 | 0.381 | 0.662 | 2.125 | 0.526 | 302.915 | checkpoint | FAIL |
| features | 5 | () | 28 | 181.8 | 0.549 | 0.290 | 1.810 | 0.411 | 243.736 | checkpoint | FAIL |
| features | 5 | 32 | 28 | 223.5 | 0.375 | 0.405 | 1.672 | 0.326 | 169.437 | checkpoint | FAIL |
| features | 25 | () | 28 | 144.3 | 0.448 | 0.634 | 1.831 | 0.633 | 337.716 | checkpoint | FAIL |
| features | 25 | 32 | 28 | 104.0 | 0.488 | 1.125 | 3.321 | 0.683 | 399.612 | checkpoint | FAIL |

**Gate summary**: 0/8 configurations clear the >=500 steps/sec target.

**Most common slowest phase**: `checkpoint`.

**Slowest configuration**: features=features, n_demons=25, hidden_sizes=32 at 104.0 steps/sec; bottleneck phase `checkpoint`.

## Failing configurations

- features=no_features, n_demons=5, hidden_sizes=(): 399.2 steps/sec, slowest phase `checkpoint` (6.48s across 5000 steps).
- features=no_features, n_demons=5, hidden_sizes=32: 343.6 steps/sec, slowest phase `checkpoint` (6.19s across 5000 steps).
- features=no_features, n_demons=25, hidden_sizes=(): 192.9 steps/sec, slowest phase `checkpoint` (14.42s across 5000 steps).
- features=no_features, n_demons=25, hidden_sizes=32: 148.7 steps/sec, slowest phase `checkpoint` (15.15s across 5000 steps).
- features=features, n_demons=5, hidden_sizes=(): 181.8 steps/sec, slowest phase `checkpoint` (12.19s across 5000 steps).
- features=features, n_demons=5, hidden_sizes=32: 223.5 steps/sec, slowest phase `checkpoint` (8.47s across 5000 steps).
- features=features, n_demons=25, hidden_sizes=(): 144.3 steps/sec, slowest phase `checkpoint` (16.89s across 5000 steps).
- features=features, n_demons=25, hidden_sizes=32: 104.0 steps/sec, slowest phase `checkpoint` (19.98s across 5000 steps).
