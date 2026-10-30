"""RQ1 + RQ2: does between-level adaptation help against a player the team never trained on?

For one pretrained team (mode, seed) and one held-out player bot:

  level loop (RQ1):  level k is played and recorded → ghost cloned from all levels so far →
                     team fine-tuned against the ghost for --budget seconds → evaluated
                     against the *real* bot (never trained on directly).
  controls, same budget, from the pretrained team:
     more_pretraining  fine-tune against the pretraining bots   (is it just extra training?)
     oracle            fine-tune against the real bot           (best case for this budget)
  data amount (RQ2): one-shot adaptation from 1 vs 3 recorded levels.

    python train/exp_adaptation.py --mode mappo --seed 0 --bot runner_bottom_left
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazehunt.adapt.pipeline import adapt, controller, evaluate, play_level  # noqa: E402
from mazehunt.agents.mappo import Team  # noqa: E402
from mazehunt.bots.player_bots import all_bots, held_out_population, training_population  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mappo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bot", default="runner_bottom_left", choices=[b.name for b in held_out_population()])
    ap.add_argument("--budget", type=float, default=45.0, help="fine-tuning seconds per adaptation")
    ap.add_argument("--levels", type=int, default=4, help="adaptations in the level loop")
    ap.add_argument("--episodes", type=int, default=200, help="evaluation games per checkpoint")
    ap.add_argument("--mix", type=float, default=0.0,
                    help="fraction of fine-tuning envs that keep playing the pretraining bots (rehearsal)")
    args = ap.parse_args()
    torch.set_num_threads(1)

    tag = f"{args.mode}_s{args.seed}_{args.bot}" + (f"_mix{round(args.mix * 100)}" if args.mix else "")
    out = RESULTS / "adaptation" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    def bot(offset: int = 0):
        return all_bots(seed=seed * 7919 + offset)[args.bot]

    def ev(team: Team, label: str) -> dict:
        r = evaluate(controller(team, seed=seed + 12345), bot(99), episodes=args.episodes)
        print(f"[{tag}] {label:<22} capture {r['capture_rate']:.3f}  escape {r['escape_rate']:.3f}", flush=True)
        return r

    t0 = time.perf_counter()
    pretrained = Team.load(RESULTS / "models" / f"{args.mode}_s{seed}.pt")
    result: dict = {"mode": args.mode, "seed": seed, "bot": args.bot, "budget_s": args.budget, "mix": args.mix,
                    "episodes": args.episodes, "pretrained": ev(pretrained, "pretrained"), "levels": []}

    # RQ1: the level loop, exactly what happens in the game.
    team, recordings = pretrained, []
    for level in range(1, args.levels + 1):
        recordings.append(play_level(controller(team, seed=seed + level), bot(level), level=level))
        team, info = adapt(team, recordings, budget_s=args.budget, seed=seed * 100 + level, mix=args.mix)
        result["levels"].append({"after_level": level, "recorded_ticks": sum(r["ticks_played"] for r in recordings),
                                 "level_outcome": recordings[-1]["outcome"], **info,
                                 **ev(team, f"adapted after L{level}")})

    # Controls from the pretrained team, same budget.
    first = recordings[:1]
    mp, _ = adapt(pretrained, first, budget_s=args.budget, seed=seed,
                  opponents=lambda i: training_population(seed=seed * 100 + i))
    result["more_pretraining"] = ev(mp, "more pretraining")
    oracle, _ = adapt(pretrained, first, budget_s=args.budget, seed=seed,
                      opponents=lambda i: [bot(1000 + i)])
    result["oracle"] = ev(oracle, "oracle (real bot)")

    # RQ2: one-shot adaptation with 1 vs 3 levels of data.
    for n in (1, 3):
        t, info = adapt(pretrained, recordings[:n], budget_s=args.budget, seed=seed + 7, mix=args.mix)
        result[f"one_shot_{n}_levels"] = {**info, **ev(t, f"one-shot, {n} level(s)")}

    result["wall_seconds"] = round(time.perf_counter() - t0, 1)
    out.write_text(json.dumps(result, indent=2))
    print(f"[{tag}] wrote {out} ({result['wall_seconds']}s)")


if __name__ == "__main__":
    main()
