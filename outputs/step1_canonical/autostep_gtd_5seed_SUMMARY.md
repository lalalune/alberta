# Autostep-for-GTD(lambda) — 5-seed Preliminary Probe

Preliminary 5-seed evidence that the new ``AutostepGTDLambda`` optimizer (Kearney et al. 2019, supervised limit) operates correctly on the canonical Alberta Plan Step 1 stream. The supervised limit with ``gamma=lambda=0`` reduces algebraically to Autostep, so the two columns should match within numerical noise.

## Configuration

- Seeds: [0, 1, 2, 3, 4]
- Burn-in steps: 1,000
- Measurement steps: 4,000
- Stream: AlbertaPlanStep1Stream(20, num_relevant=5)
- Normalizer: EMA(decay=0.99)

## Results (mean MSE on tail, mean ± stderr over 5 seeds)

| Optimizer | mean MSE | stderr |
|---|---|---|
| LMS | 1.1501 | 0.0088 |
| Autostep | 1.2477 | 0.0122 |
| AutostepGTDLambda | 1.2477 | 0.0122 |

Wall-clock: 2.92 s

**Status:** 5-seed preliminary evidence. For paper-grade headline claims, run the 30-seed, joint normalizer/optimizer-grid sweep in ``step1_full_baselines.py``.