# Alberta Plan Step 1 — Multi-Baseline Results

Replication of the Alberta Plan Step 1 supervised-learning task across all optimizers named in Sutton, Bowling & Pilarski 2022 footnote 11 (LMS, IDBD, Autostep, Adam, RMSprop, NADALINE), plus public AdaGain (Jacobsen et al. 2019), evaluated on four non-stationary streams.

## Configuration

- Seeds: 30
- Burn-in steps: 20,000
- Measurement steps: 10,000
- Total wall-clock: 4.55 minutes

Optimizer hyperparameter grids:

- **LMS**: ['step_size=0.001', 'step_size=0.005', 'step_size=0.01', 'step_size=0.02', 'step_size=0.05', 'step_size=0.1']
- **IDBD**: ['initial_step_size=0.05, meta_step_size=0.001', 'initial_step_size=0.05, meta_step_size=0.005', 'initial_step_size=0.05, meta_step_size=0.01', 'initial_step_size=0.05, meta_step_size=0.02']
- **Autostep**: ['initial_step_size=0.05, meta_step_size=0.001, tau=10000', 'initial_step_size=0.05, meta_step_size=0.005, tau=10000', 'initial_step_size=0.05, meta_step_size=0.01, tau=10000', 'initial_step_size=0.05, meta_step_size=0.05, tau=10000']
- **AdaGain**: ['initial_step_size=0.05, meta_step_size=0.0001, forgetting_rate=0.1', 'initial_step_size=0.05, meta_step_size=0.001, forgetting_rate=0.1', 'initial_step_size=0.05, meta_step_size=0.01, forgetting_rate=0.1', 'initial_step_size=0.1, meta_step_size=0.0001, forgetting_rate=0.1', 'initial_step_size=0.1, meta_step_size=0.001, forgetting_rate=0.1', 'initial_step_size=0.1, meta_step_size=0.01, forgetting_rate=0.1']
- **Adam**: ['step_size=0.0005', 'step_size=0.001', 'step_size=0.005', 'step_size=0.01']
- **RMSprop**: ['step_size=0.0005', 'step_size=0.001', 'step_size=0.005', 'step_size=0.01']
- **NADALINE**: ['step_size=0.005', 'step_size=0.01', 'step_size=0.05', 'step_size=0.1']

## Best-tuned MSE per (optimizer, stream)

| Optimizer | Sutton1992_noiseless | Sutton1992_noisy | AlbertaPlanStep1 | XDistShift |
|---|---|---|---|---|
| AdaGain | 1.6640 ± 0.0079 | 3.1689 ± 0.0114 | 1.1457 ± 0.0033 | inf ± 0.0000 |
| Adam | 4.8440 ± 0.0194 | 5.8844 ± 0.0236 | 1.0093 ± 0.0024 | 0.0132 ± 0.0001 |
| Autostep | 2.0780 ± 0.0142 | 3.2004 ± 0.0112 | 1.0400 ± 0.0037 | 0.0108 ± 0.0001 |
| IDBD | 1.5583 ± 0.0087 | 3.0144 ± 0.0091 | 1.0642 ± 0.0026 | inf ± 0.0000 |
| LMS | 3.6267 ± 0.0165 | 4.8894 ± 0.0201 | 1.0115 ± 0.0024 | 0.0160 ± 0.0002 |
| NADALINE | 3.8108 ± 0.0163 | 5.1889 ± 0.0228 | 1.0532 ± 0.0026 | 0.0106 ± 0.0000 |
| RMSprop | 4.6132 ± 0.0182 | 5.6692 ± 0.0226 | 1.0093 ± 0.0024 | 0.0130 ± 0.0000 |

## Selected hyperparameters (best per cell)

| Optimizer | Sutton1992_noiseless | Sutton1992_noisy | AlbertaPlanStep1 | XDistShift |
|---|---|---|---|---|
| AdaGain | initial_step_size=0.05, meta_step_size=0.01, forgetting_rate=0.1 | initial_step_size=0.05, meta_step_size=0.01, forgetting_rate=0.1 | initial_step_size=0.1, meta_step_size=0.01, forgetting_rate=0.1 | initial_step_size=0.05, meta_step_size=0.0001, forgetting_rate=0.1 |
| Adam | step_size=0.01 | step_size=0.01 | step_size=0.0005 | step_size=0.0005 |
| Autostep | initial_step_size=0.05, meta_step_size=0.05, tau=10000 | initial_step_size=0.05, meta_step_size=0.05, tau=10000 | initial_step_size=0.05, meta_step_size=0.05, tau=10000 | initial_step_size=0.05, meta_step_size=0.05, tau=10000 |
| IDBD | initial_step_size=0.05, meta_step_size=0.01 | initial_step_size=0.05, meta_step_size=0.01 | initial_step_size=0.05, meta_step_size=0.02 | initial_step_size=0.05, meta_step_size=0.001 |
| LMS | step_size=0.02 | step_size=0.02 | step_size=0.001 | step_size=0.001 |
| NADALINE | step_size=0.05 | step_size=0.01 | step_size=0.005 | step_size=0.005 |
| RMSprop | step_size=0.01 | step_size=0.01 | step_size=0.0005 | step_size=0.0005 |

## Paired differences vs. best-tuned LMS

Each cell shows ``mean(LMS − other) ± stderr`` over paired seeds, with wins-out-of-N (positive = the alternative beat LMS on that seed) and Cohen's d on the paired differences.

### Sutton1992_noiseless

| Optimizer | mean diff | stderr | wins | n | Cohen's d |
|---|---|---|---|---|---|
| AdaGain | 1.9669 | 0.0167 | 27/27 | 27 | 22.721 |
| Adam | -1.2173 | 0.0112 | 0/30 | 30 | -19.791 |
| Autostep | 1.5487 | 0.0198 | 30/30 | 30 | 14.252 |
| IDBD | 2.0684 | 0.0168 | 30/30 | 30 | 22.537 |
| NADALINE | -0.1841 | 0.0102 | 0/30 | 30 | -3.294 |
| RMSprop | -0.9864 | 0.0103 | 0/30 | 30 | -17.465 |

### Sutton1992_noisy

| Optimizer | mean diff | stderr | wins | n | Cohen's d |
|---|---|---|---|---|---|
| AdaGain | 1.7252 | 0.0151 | 29/29 | 29 | 21.234 |
| Adam | -0.9949 | 0.0118 | 0/30 | 30 | -15.354 |
| Autostep | 1.6891 | 0.0186 | 30/30 | 30 | 16.554 |
| IDBD | 1.8751 | 0.0143 | 30/30 | 30 | 23.885 |
| NADALINE | -0.2995 | 0.0064 | 0/30 | 30 | -8.586 |
| RMSprop | -0.7798 | 0.0111 | 0/30 | 30 | -12.788 |

### AlbertaPlanStep1

| Optimizer | mean diff | stderr | wins | n | Cohen's d |
|---|---|---|---|---|---|
| AdaGain | -0.1347 | 0.0012 | 0/25 | 25 | -23.381 |
| Adam | 0.0021 | 0.0002 | 29/30 | 30 | 1.749 |
| Autostep | -0.0285 | 0.0019 | 0/30 | 30 | -2.670 |
| IDBD | -0.0527 | 0.0003 | 0/30 | 30 | -29.143 |
| NADALINE | -0.0418 | 0.0003 | 0/30 | 30 | -23.611 |
| RMSprop | 0.0021 | 0.0002 | 29/30 | 30 | 1.651 |

### XDistShift

| Optimizer | mean diff | stderr | wins | n | Cohen's d |
|---|---|---|---|---|---|
| AdaGain | NaN | NaN | 0/0 | 0 | 0.000 |
| Adam | 0.0028 | 0.0001 | 30/30 | 30 | 3.896 |
| Autostep | 0.0052 | 0.0002 | 30/30 | 30 | 5.110 |
| IDBD | NaN | NaN | 0/0 | 0 | 0.000 |
| NADALINE | 0.0053 | 0.0002 | 30/30 | 30 | 5.712 |
| RMSprop | 0.0030 | 0.0001 | 30/30 | 30 | 4.115 |

## Headline

- **Sutton1992_noiseless**: IDBD beat best-tuned LMS by 2.0684 MSE (57.0% relative, 30/30 seed wins, Cohen's d = 22.537).
- **Sutton1992_noisy**: IDBD beat best-tuned LMS by 1.8751 MSE (38.3% relative, 30/30 seed wins, Cohen's d = 23.885).
- **AlbertaPlanStep1**: Adam beat best-tuned LMS by 0.0021 MSE (0.2% relative, 29/30 seed wins, Cohen's d = 1.749).
- **XDistShift**: NADALINE beat best-tuned LMS by 0.0053 MSE (33.4% relative, 30/30 seed wins, Cohen's d = 5.712).
