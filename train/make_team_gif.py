"""Side-by-side GIF of two monster teams hunting the same player bot, e.g. the scripted
greedy chasers vs a trained MAPPO team.

    python train/make_team_gif.py --left greedy --right mappo_s1 --bot runner_bottom_left

Each side is captioned with its measured capture rate over 200 games.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazehunt.adapt.pipeline import evaluate  # noqa: E402
from mazehunt.agents.mappo import LearnedMonsters, Team  # noqa: E402
from mazehunt.agents.scripted_monsters import GreedyChasers  # noqa: E402
from mazehunt.bots.player_bots import all_bots  # noqa: E402
from mazehunt.config import GameConfig  # noqa: E402
from mazehunt.env.hunt_env import shared_maze  # noqa: E402
from mazehunt.render.renderer import Renderer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_demo_gif import episode_frames  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"


def team(name: str, seed: int):
    if name == "greedy":
        return GreedyChasers(seed=seed)
    return LearnedMonsters(Team.load(RESULTS / "models" / f"{name}.pt"), name=name, seed=seed)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--left", default="greedy")
    ap.add_argument("--right", default="mappo_s1")
    ap.add_argument("--bot", default="runner_bottom_left")
    ap.add_argument("--episode-seed", type=int, default=0)
    ap.add_argument("--max-ticks", type=int, default=200)
    ap.add_argument("--out", type=Path, default=RESULTS / "demo_teams.gif")
    args = ap.parse_args()

    pygame.init()
    renderer = Renderer(shared_maze("arena"), GameConfig(), tile=24)
    clips = []
    for name in (args.left, args.right):
        rate = evaluate(team(name, 5), all_bots(seed=99)[args.bot], episodes=200)["capture_rate"]
        label = "greedy chasers (scripted)" if name == "greedy" else f"{name.split('_')[0].upper()} team (learned)"
        clips.append(episode_frames(renderer, team(name, args.episode_seed), all_bots(seed=args.episode_seed)[args.bot],
                                    label, args.max_ticks, sub=2, caption=f"{label}: catches this player {rate:.0%} of 200 games"))
        print(name, rate)

    n = max(len(c) for c in clips)
    w, h = clips[0][0].get_size()
    images = []
    for i in range(n):
        canvas = pygame.Surface((w * 2 + 8, h))
        canvas.fill((0, 0, 0))
        for j, clip in enumerate(clips):
            canvas.blit(clip[min(i, len(clip) - 1)], (j * (w + 8), 0))
        images.append(Image.frombytes("RGB", canvas.get_size(), pygame.image.tobytes(canvas, "RGB")).quantize(colors=64))
    images[0].save(args.out, save_all=True, append_images=images[1:], duration=62, loop=0, optimize=True)
    print(f"wrote {args.out} ({len(images)} frames, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
