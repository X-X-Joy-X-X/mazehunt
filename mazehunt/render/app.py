"""Playable MazeHunt: keyboard → Game rules → Renderer, one level after another.

Each level is one round on the same map. Win by reaching the EXIT or surviving the timer;
getting caught costs a life. Every round is saved to results/recordings/ for the ghost player.
"""

from __future__ import annotations

import argparse
import threading
from pathlib import Path

import pygame

from mazehunt.agents.scripted_monsters import CONTROLLERS
from mazehunt.config import GameConfig
from mazehunt.core.actions import (DOWN, LEFT, RIGHT, SHOOT_DOWN, SHOOT_LEFT, SHOOT_RIGHT, SHOOT_UP,
                                   STAY, UP)
from mazehunt.core.game import Game, Outcome
from mazehunt.core.maze import Maze
from mazehunt.record import RECORDINGS_DIR, LevelRecorder, new_session_dir
from mazehunt.render.renderer import COLORS, Renderer

MOVE_KEYS = {pygame.K_w: UP, pygame.K_s: DOWN, pygame.K_a: LEFT, pygame.K_d: RIGHT}
SHOOT_KEYS = {pygame.K_UP: SHOOT_UP, pygame.K_DOWN: SHOOT_DOWN,
              pygame.K_LEFT: SHOOT_LEFT, pygame.K_RIGHT: SHOOT_RIGHT}
CONFIRM_KEYS = (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE)


class HumanInput:
    """Turns the keyboard into exactly one action per tick.

    The game ticks 8 times a second, so a quick tap can land between ticks. Taps are
    remembered until the next tick so they're never lost. A shot wins over a move only when
    the gun is actually ready; otherwise holding an arrow key wouldn't let you walk.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tapped_shot: int | None = None
        self.tapped_move: int | None = None
        self.last_move_key: int | None = None

    def on_keydown(self, key: int) -> None:
        if key in SHOOT_KEYS:
            self.tapped_shot = SHOOT_KEYS[key]
        elif key in MOVE_KEYS:
            self.tapped_move = MOVE_KEYS[key]
            self.last_move_key = key

    def next_action(self, game: Game) -> int:
        held = pygame.key.get_pressed()
        shot = self.tapped_shot or next((a for k, a in SHOOT_KEYS.items() if held[k]), None)
        if self.last_move_key is not None and held[self.last_move_key]:
            move = MOVE_KEYS[self.last_move_key]
        else:
            move = self.tapped_move or next((a for k, a in MOVE_KEYS.items() if held[k]), None)
        self.tapped_shot = self.tapped_move = None

        if shot is not None and (game.player.can_shoot or move is None):
            return shot
        return move if move is not None else STAY


MODELS_DIR = RECORDINGS_DIR.parent / "models"


class Adapter:
    """Runs between-level adaptation on a background thread so the window stays responsive.
    Imported lazily: plain scripted play shouldn't need PyTorch."""

    def __init__(self, model: Path, budget_s: float, seed: int | None):
        from mazehunt.agents.mappo import Team
        self.team = Team.load(model)
        self.budget_s, self.seed = budget_s, seed or 0
        self.recordings: list[dict] = []
        self.thread: threading.Thread | None = None
        self.info: dict = {}
        self.adaptations = 0

    def controller(self):
        from mazehunt.agents.mappo import LearnedMonsters
        tag = f"{self.team.mode}, adapted x{self.adaptations}" if self.adaptations else self.team.mode
        return LearnedMonsters(self.team, name=tag, seed=self.seed)

    def start(self, record: dict) -> None:
        self.recordings = (self.recordings + [record])[-3:]  # the player's last 3 levels
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        from mazehunt.adapt.pipeline import adapt
        self.team, self.info = adapt(self.team, self.recordings, budget_s=self.budget_s,
                                     seed=self.seed + len(self.recordings))
        self.adaptations += 1

    @property
    def busy(self) -> bool:
        return self.thread is not None and self.thread.is_alive()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play MazeHunt.")
    parser.add_argument("--monsters", choices=sorted(CONTROLLERS) + ["learned"], default="greedy",
                        help="monster controller (default: greedy); 'learned' loads --model")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "mappo_s0.pt",
                        help="trained team for --monsters learned")
    parser.add_argument("--adapt", type=float, default=0.0, metavar="SECONDS",
                        help="with --monsters learned: fine-tune against your ghost between levels")
    parser.add_argument("--seed", type=int, default=None, help="seed for the monsters' randomness")
    parser.add_argument("--no-record", action="store_true", help="don't save gameplay recordings")
    args = parser.parse_args(argv)
    if args.adapt and args.monsters != "learned":
        parser.error("--adapt needs --monsters learned")

    config = GameConfig()
    maze = Maze.load(config.map_name)
    game = Game(maze, config)
    adapter = None
    if args.monsters == "learned":
        adapter = Adapter(args.model, args.adapt, args.seed)
        monsters = adapter.controller()
    else:
        monsters = CONTROLLERS[args.monsters](seed=args.seed)

    pygame.init()
    pygame.display.set_caption("MazeHunt")
    renderer = Renderer(maze, config)
    screen = pygame.display.set_mode(renderer.size)
    clock = pygame.time.Clock()
    human = HumanInput()
    recorder = LevelRecorder()

    session_dir = new_session_dir()
    level, lives = 1, config.lives
    state = "title"  # title → playing ⇄ paused → level_end → playing … → game_over
    tick_seconds = 1.0 / config.ticks_per_second
    since_tick = 0.0
    last_save = None
    last_record = None

    def start_level() -> None:
        nonlocal since_tick
        game.reset()
        recorder.clear()
        human.reset()
        since_tick = 0.0

    while True:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            if event.type != pygame.KEYDOWN:
                continue
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                return
            if state == "playing" and event.key == pygame.K_p:
                state = "paused"
            elif state == "paused" and event.key in (pygame.K_p, *CONFIRM_KEYS):
                state = "playing"
            elif state in ("title", "level_end", "game_over") and event.key in CONFIRM_KEYS:
                if state == "level_end" and adapter is not None and args.adapt and last_record:
                    adapter.start(last_record)
                    state = "adapting"
                    continue
                if state == "level_end":
                    level += 1
                elif state == "game_over":
                    session_dir, level, lives = new_session_dir(), 1, config.lives
                start_level()
                state = "playing"
            elif state == "playing":
                human.on_keydown(event.key)

        if state == "adapting" and not adapter.busy:
            monsters = adapter.controller()
            level += 1
            start_level()
            state = "playing"

        if state == "playing":
            since_tick += dt
            while since_tick >= tick_seconds and not game.done:
                since_tick -= tick_seconds
                action = human.next_action(game)
                monster_actions = monsters.act(game)
                if not args.no_record:
                    recorder.log(game, action, monster_actions)
                game.step(action, monster_actions)
            if game.done:
                last_record = recorder.as_record(level, game, monsters.name)
                if not args.no_record:
                    last_save = recorder.save(session_dir, level, game, monsters.name)
                if game.outcome is Outcome.CAUGHT:
                    lives -= 1
                state = "game_over" if lives == 0 else "level_end"

        alpha = min(since_tick / tick_seconds, 1.0) if state == "playing" else 1.0
        label = monsters.name if adapter is not None else f"{monsters.name} (scripted)"
        renderer.draw(screen, game, alpha, level, lives, label)

        if state == "title":
            renderer.draw_overlay(screen, "MAZEHUNT", [
                "Reach the green EXIT, or survive until the timer runs out.",
                "",
                "WASD  move     ARROWS  shoot     P  pause     ESC  quit",
                "",
                "Press ENTER to start",
            ])
        elif state == "paused":
            renderer.draw_overlay(screen, "PAUSED", ["Press P to resume"])
        elif state == "level_end":
            title, color = {
                Outcome.ESCAPED: ("ESCAPED!", COLORS["good"]),
                Outcome.SURVIVED: ("SURVIVED!", COLORS["good"]),
                Outcome.CAUGHT: ("CAUGHT", COLORS["bad"]),
            }[game.outcome]
            renderer.draw_overlay(screen, f"LEVEL {level}: {title}", [
                f"Lives left: {lives}",
                "",
                ("The monsters will study this round before the next one." if adapter is not None and args.adapt
                 else "Play with --monsters learned --adapt 30 to make the monsters learn from you."),
                "",
                f"Press ENTER for level {level + 1}",
            ], color)
        elif state == "adapting":
            renderer.draw_overlay(screen, "THE MONSTERS ARE STUDYING YOU", [
                f"Cloning how you played your last {len(adapter.recordings)} level(s),",
                f"then practising against your ghost for about {args.adapt:.0f} s.",
            ], COLORS["bad"])
        elif state == "game_over":
            renderer.draw_overlay(screen, "GAME OVER", [
                f"You reached level {level}.",
                f"Recordings saved to {last_save.parent}" if last_save else "Recording was off.",
                "",
                "ENTER  play again     ESC  quit",
            ], COLORS["bad"])

        pygame.display.flip()
