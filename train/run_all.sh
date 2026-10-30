#!/usr/bin/env bash
# Reproduces every result in the README, from scratch, on one machine (CPU only).
#   bash train/run_all.sh            (Git Bash on Windows, or any Linux/macOS shell)
# Wall time on a 12-thread, 8 GB laptop: ~50 min pretraining + ~1.5 h experiments.
# Keep at most 2-3 PyTorch processes at once on 8 GB RAM (each takes ~400 MB).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=.venv/bin/python
mkdir -p results/logs

# 1. Pretrain: MAPPO and IPPO, 3 seeds each, in parallel.
for m in mappo ippo; do for s in 0 1 2; do
  "$PY" train/pretrain.py --mode $m --seed $s --steps 6000000 > results/logs/stdout_${m}_s${s}.txt 2>&1 &
done; done; wait

# 2. RQ3: every team vs every bot.
"$PY" train/eval_teams.py

# 3. RQ1 + RQ2: adaptation against each held-out bot, 3 seeds.
for s in 0 1 2; do for b in runner_bottom_left waiter_left camper_top_middle; do
  echo "--mode mappo --seed $s --bot $b"
done; done | xargs -P 2 -L 1 "$PY" train/exp_adaptation.py
# Same, keeping 50% of fine-tuning games on the pretraining bots (rehearsal).
for s in 0 1 2; do for b in runner_bottom_left waiter_left camper_top_middle; do
  echo "--mode mappo --seed $s --bot $b --mix 0.5"
done; done | xargs -P 2 -L 1 "$PY" train/exp_adaptation.py

# 4. RQ4: fine-tuning budget sweep.
for s in 0 1 2; do echo "--mode mappo --seed $s --bot runner_bottom_left"; done \
  | xargs -P 2 -L 1 "$PY" train/exp_budget.py

# 5. Figures, summary table, demo GIF.
"$PY" train/make_figures.py
"$PY" train/make_team_gif.py --left greedy --right mappo_s1 --bot runner_bottom_left --episode-seed 2
"$PY" train/make_demo_gif.py --mode mappo --seed 0 --bot camper_top_middle --max-ticks 240
