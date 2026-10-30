"""RQ4: how much does the fine-tuning budget matter?

One recorded level of a held-out bot → ghost → fine-tune for 10 / 30 / 60 / 120 s (each from
the pretrained team) → capture rate against the real bot.

    python train/exp_budget.py --mode mappo --seed 0 --bot runner_bottom_left
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazehunt.adapt.pipeline import adapt, controller, evaluate, play_level  # noqa: E402
from mazehunt.agents.mappo import Team  # noqa: E402
from mazehunt.bots.player_bots import all_bots, held_out_population  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mappo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bot", default="runner_bottom_left", choices=[b.name for b in held_out_population()])
    ap.add_argument("--budgets", default="10,30,60,120")
    ap.add_argument("--episodes", type=int, default=200)
    args = ap.parse_args()
    torch.set_num_threads(1)
    seed, tag = args.seed, f"{args.mode}_s{args.seed}_{args.bot}"

    def ev(team: Team) -> dict:
        return evaluate(controller(team, seed=seed + 12345), all_bots(seed=seed * 7919 + 99)[args.bot],
                        episodes=args.episodes)

    pretrained = Team.load(RESULTS / "models" / f"{args.mode}_s{seed}.pt")
    level = play_level(controller(pretrained, seed=seed + 1), all_bots(seed=seed * 7919 + 1)[args.bot])
    rows = [{"budget_s": 0, **ev(pretrained)}]
    for b in map(float, args.budgets.split(",")):
        team, info = adapt(pretrained, [level], budget_s=b, seed=seed)
        rows.append({"budget_s": b, **info, **ev(team)})
        print(f"[{tag}] budget {b:>5.0f}s  capture {rows[-1]['capture_rate']:.3f}", flush=True)
    out = RESULTS / "budget" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"mode": args.mode, "seed": seed, "bot": args.bot, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
