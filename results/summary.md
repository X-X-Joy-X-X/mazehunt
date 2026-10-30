# MazeHunt results summary

## Pretraining: final capture rate vs the training bots

| team | capture rate |
|---|---|
| MAPPO | 0.65 ± 0.08 |
| IPPO | 0.62 ± 0.07 |

## RQ3: team evaluation, 200 games per bot (mean ± std over seeds)

| team | bots | capture | escape | exit cover | spread |
|---|---|---|---|---|---|
| GREEDY | training | 0.47 ± 0.00 | 0.44 ± 0.00 | 0.00 | 6.3 |
| GREEDY | held-out | 0.37 ± 0.00 | 0.52 ± 0.00 | 0.00 | 5.6 |
| MAPPO | training | 0.55 ± 0.08 | 0.31 ± 0.13 | 0.14 | 8.9 |
| MAPPO | held-out | 0.41 ± 0.16 | 0.44 ± 0.18 | 0.16 | 8.2 |
| IPPO | training | 0.48 ± 0.03 | 0.41 ± 0.01 | 0.00 | 6.4 |
| IPPO | held-out | 0.35 ± 0.03 | 0.53 ± 0.02 | 0.00 | 5.8 |

## RQ1/RQ2: adaptation vs held-out bots (capture rate, mean ± std over seeds)

| bot | variant | pretrained | after L1 | after L4 | more pretraining (control) | oracle | one-shot 1 lvl | one-shot 3 lvl | Δ L4 − pretrained (paired) | Δ L4 − control (paired) |
|---|---|---|---|---|---|---|---|---|---|---|
| camper_top_middle | MAPPO ghost only | 0.59 ± 0.03 | 0.64 ± 0.07 | 0.66 ± 0.05 | 0.60 ± 0.07 | 0.64 ± 0.06 | 0.60 ± 0.09 | 0.61 ± 0.10 | 0.07 ± 0.02 | 0.06 ± 0.04 |
| camper_top_middle | MAPPO ghost + 50% rehearsal | 0.59 ± 0.03 | 0.59 ± 0.08 | 0.67 ± 0.02 | 0.62 ± 0.07 | 0.66 ± 0.08 | 0.64 ± 0.03 | 0.61 ± 0.13 | 0.08 ± 0.01 | 0.05 ± 0.05 |
| runner_bottom_left | MAPPO ghost only | 0.36 ± 0.26 | 0.21 ± 0.05 | 0.42 ± 0.29 | 0.40 ± 0.23 | 0.49 ± 0.22 | 0.29 ± 0.14 | 0.32 ± 0.19 | 0.05 ± 0.09 | 0.01 ± 0.10 |
| runner_bottom_left | MAPPO ghost + 50% rehearsal | 0.36 ± 0.26 | 0.30 ± 0.14 | 0.32 ± 0.17 | 0.40 ± 0.23 | 0.53 ± 0.24 | 0.36 ± 0.25 | 0.39 ± 0.21 | -0.04 ± 0.11 | -0.08 ± 0.07 |
| waiter_left | MAPPO ghost only | 0.35 ± 0.26 | 0.31 ± 0.32 | 0.30 ± 0.30 | 0.38 ± 0.30 | 0.37 ± 0.28 | 0.12 ± 0.03 | 0.34 ± 0.30 | -0.05 ± 0.06 | -0.08 ± 0.06 |
| waiter_left | MAPPO ghost + 50% rehearsal | 0.35 ± 0.26 | 0.35 ± 0.28 | 0.33 ± 0.24 | 0.40 ± 0.27 | 0.40 ± 0.28 | 0.37 ± 0.28 | 0.33 ± 0.28 | -0.03 ± 0.01 | -0.07 ± 0.03 |

## RQ4: fine-tuning budget (capture rate, mean ± std over seeds)

| bot | 0 s | 10 s | 30 s | 60 s | 120 s |
|---|---|---|---|---|---|
| runner_bottom_left | 0.36 ± 0.26 | 0.34 ± 0.15 | 0.26 ± 0.09 | 0.18 ± 0.02 | 0.19 ± 0.00 |
