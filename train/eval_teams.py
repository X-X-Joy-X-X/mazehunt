"""RQ3: evaluate every pretrained team (and the scripted baselines) against every player bot.

Besides capture/escape rates it measures two coordination signals per game:
  exit_cover  fraction of ticks with at least one monster within 4 tiles of the exit
  spread      mean maze distance between pairs of live monsters (do they split up?)

    python train/eval_teams.py            → results/eval_teams.json
"""

from __future__ import annotations

import json
import sys
from itertools import combinations
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazehunt.agents.mappo import LearnedMonsters, Team  # noqa: E402
from mazehunt.agents.scripted_monsters import GreedyChasers  # noqa: E402
from mazehunt.bots.player_bots import all_bots, held_out_population, training_population  # noqa: E402
from mazehunt.core.game import Game  # noqa: E402
from mazehunt.env.hunt_env import shared_maze  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
EPISODES = 200


def run(job: tuple[str, str]) -> dict:
    team_name, bot_name = job
    if team_name == "greedy":
        monsters = GreedyChasers(seed=3)
    else:
        monsters = LearnedMonsters(Team.load(RESULTS / "models" / f"{team_name}.pt"), seed=3)
    bot = all_bots(seed=11)[bot_name]
    maze = shared_maze("arena")
    to_exit = maze.distances_to(maze.exit)
    game = Game(maze)
    outcomes, cover, spread = [], [], []
    for _ in range(EPISODES):
        game.reset()
        bot.reset()
        near = ticks = 0
        while not game.done:
            game.step(bot.act(game), monsters.act(game))
            alive = [m.pos for m in game.monsters if m.alive]
            near += any(to_exit[y][x] <= 4 for x, y in alive)
            ticks += 1
            if len(alive) > 1:
                spread.append(np.mean([maze.distances_to(a)[b[1]][b[0]] for a, b in combinations(alive, 2)]))
        outcomes.append(game.outcome.value)
        cover.append(near / ticks)
    return {"team": team_name, "bot": bot_name,
            "capture_rate": outcomes.count("caught") / EPISODES,
            "escape_rate": outcomes.count("escaped") / EPISODES,
            "exit_cover": float(np.mean(cover)), "spread": float(np.mean(spread))}


def main() -> None:
    teams = ["greedy"] + sorted(p.stem for p in (RESULTS / "models").glob("*_s*.pt"))
    bots = [b.name for b in training_population()] + [b.name for b in held_out_population()]
    with Pool(processes=3) as pool:  # each worker loads PyTorch (~400 MB); more can exhaust RAM
        rows = pool.map(run, [(t, b) for t in teams for b in bots])
    held = {b.name for b in held_out_population()}
    for r in rows:
        r["held_out"] = r["bot"] in held
    out = RESULTS / "eval_teams.json"
    out.write_text(json.dumps(rows, indent=1))
    # Mean ± std over seeds, per team type and bot split.
    print(f"{'team':<8} {'split':<9} {'capture':>14} {'escape':>14} {'exit_cover':>12} {'spread':>8}")
    for kind in ["greedy", "mappo", "ippo"]:
        for split in (False, True):
            sel = [r for r in rows if r["team"].split("_")[0] == kind and r["held_out"] == split]
            if not sel:
                continue
            by_seed = {}
            for r in sel:
                by_seed.setdefault(r["team"], []).append(r)
            cap = [np.mean([r["capture_rate"] for r in v]) for v in by_seed.values()]
            esc = [np.mean([r["escape_rate"] for r in v]) for v in by_seed.values()]
            cov = np.mean([r["exit_cover"] for r in sel])
            spr = np.mean([r["spread"] for r in sel])
            print(f"{kind:<8} {'held-out' if split else 'train':<9} {np.mean(cap):.3f}±{np.std(cap):.3f}"
                  f"   {np.mean(esc):.3f}±{np.std(esc):.3f}   {cov:>10.3f}   {spr:>6.1f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
