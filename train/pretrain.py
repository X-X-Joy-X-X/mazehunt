"""Pretrain a monster team against the scripted player-bot population.

    python train/pretrain.py --mode mappo --seed 0
    python train/pretrain.py --mode ippo --seed 0 --steps 3000000

Writes results/models/<mode>_s<seed>.pt and a JSON-lines training log in results/logs/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazehunt.agents.mappo import PPOConfig, Team, train  # noqa: E402
from mazehunt.bots.player_bots import training_population  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mappo", "ippo"], default="mappo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (1 lets seeds run in parallel)")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    tag = f"{args.mode}_s{args.seed}"
    log_path = RESULTS / "logs" / f"pretrain_{tag}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")

    def log(row: dict) -> None:
        print(f"[{tag}] {row['steps']:>9,} steps {row['seconds']:>6}s  capture {row['capture_rate']:.2f}"
              f"  escape {row['escape_rate']:.2f}  {row['per_opponent']}", flush=True)
        log_file.write(json.dumps(row) + "\n")
        log_file.flush()

    team = Team(args.mode, n_agents=3)
    # Each env gets its own bot objects (bots keep per-level memory); seeds differ per env.
    summary = train(team, lambda i: training_population(seed=args.seed * 100 + i),
                    total_steps=args.steps, cfg=PPOConfig(), seed=args.seed, log=log)
    team.save(RESULTS / "models" / f"{tag}.pt", trained_steps=summary["steps"], seed=args.seed)
    print(f"[{tag}] done: {summary['steps']:,} steps in {summary['seconds']:.0f}s")


if __name__ == "__main__":
    main()
