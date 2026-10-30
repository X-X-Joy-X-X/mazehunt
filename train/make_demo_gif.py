"""Side-by-side demo GIF: the pretrained team vs the same team after one between-level
adaptation, both hunting the same held-out player bot.

    python train/make_demo_gif.py --mode mappo --seed 0 --bot runner_bottom_left

The adapted team is fine-tuned exactly as in the game: one recorded level → ghost → budget
seconds of fine-tuning. The episode shown is the first one (bot seed --episode-seed) and is
labelled with the measured capture rates of both teams, so one lucky clip can't mislead.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mazehunt.adapt.pipeline import adapt, controller, evaluate, play_level  # noqa: E402
from mazehunt.agents.mappo import Team  # noqa: E402
from mazehunt.bots.player_bots import all_bots, held_out_population  # noqa: E402
from mazehunt.config import GameConfig  # noqa: E402
from mazehunt.core.game import Game  # noqa: E402
from mazehunt.env.hunt_env import shared_maze  # noqa: E402
from mazehunt.render.renderer import COLORS, HUD_HEIGHT, Renderer  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
OUTCOME_TEXT = {"caught": "CAUGHT", "escaped": "ESCAPED", "survived": "SURVIVED"}


def episode_frames(renderer: Renderer, monsters, player, label: str, max_ticks: int,
                   sub: int, caption: str) -> list[pygame.Surface]:
    game = Game(shared_maze("arena"), GameConfig())
    player.reset()
    frames = []
    font = pygame.font.SysFont("consolas", 16, bold=True)

    def snap(alpha: float, banner: str | None = None) -> pygame.Surface:
        surf = pygame.Surface((renderer.size[0], renderer.size[1] + 28))
        surf.fill(COLORS["bg"])
        screen = surf.subsurface(pygame.Rect(0, 0, *renderer.size))
        renderer.draw(screen, game, alpha, 1, 1, label)
        screen.fill(COLORS["bg"], pygame.Rect(0, 0, renderer.size[0], HUD_HEIGHT))  # compact header
        head = font.render(f"{label}   t = {game.tick / game.config.ticks_per_second:4.1f} s", True, COLORS["text"])
        screen.blit(head, (10, (HUD_HEIGHT - head.get_height()) // 2))
        text = font.render(banner or caption, True, COLORS["good"] if banner == "CAUGHT" else COLORS["text"])
        surf.blit(text, ((surf.get_width() - text.get_width()) // 2, renderer.size[1] + 5))
        return surf

    while not game.done and game.tick < max_ticks:
        game.step(player.act(game), monsters.act(game))
        frames += [snap((k + 1) / sub) for k in range(sub)]
    end = OUTCOME_TEXT.get(game.outcome.value, "TIME UP (clip)")
    frames += [snap(1.0, end)] * 12
    return frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="mappo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bot", default="runner_bottom_left", choices=[b.name for b in held_out_population()])
    ap.add_argument("--budget", type=float, default=45.0)
    ap.add_argument("--episode-seed", type=int, default=0)
    ap.add_argument("--max-ticks", type=int, default=200)
    ap.add_argument("--out", type=Path, default=RESULTS / "demo_adaptation.gif")
    args = ap.parse_args()
    torch.set_num_threads(2)

    pretrained = Team.load(RESULTS / "models" / f"{args.mode}_s{args.seed}.pt")
    level1 = play_level(controller(pretrained, seed=1), all_bots(seed=1)[args.bot], level=1)
    adapted, info = adapt(pretrained, [level1], budget_s=args.budget, seed=args.seed)
    print("adaptation:", info)
    rates = {}
    for name, team in (("before", pretrained), ("after", adapted)):
        r = evaluate(controller(team, seed=5), all_bots(seed=99)[args.bot], episodes=200)
        rates[name] = r["capture_rate"]
        print(name, r)

    pygame.init()
    renderer = Renderer(shared_maze("arena"), GameConfig(), tile=24)
    clips = []
    for name, team, label in (("before", pretrained, "pretrained"),
                              ("after", adapted, "adapted after 1 level")):
        bot = all_bots(seed=args.episode_seed)[args.bot]
        caption = f"{label}: catches this player {rates[name]:.0%} of 200 games"
        clips.append(episode_frames(renderer, controller(team, seed=args.episode_seed), bot, label,
                                    args.max_ticks, sub=2, caption=caption))

    n = max(len(c) for c in clips)
    w, h = clips[0][0].get_size()
    images = []
    for i in range(n):
        canvas = pygame.Surface((w * 2 + 8, h))
        canvas.fill((0, 0, 0))
        for j, clip in enumerate(clips):
            canvas.blit(clip[min(i, len(clip) - 1)], (j * (w + 8), 0))
        raw = pygame.image.tobytes(canvas, "RGB")
        images.append(Image.frombytes("RGB", canvas.get_size(), raw).quantize(colors=64))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(args.out, save_all=True, append_images=images[1:], duration=62, loop=0, optimize=True)
    print(f"wrote {args.out} ({len(images)} frames, {args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
