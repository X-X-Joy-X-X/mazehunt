"""Between-level adaptation: recorded levels → ghost player → fine-tuned monster team.

    recordings = [play_level(team, player, level=1)]        # or a human's saved levels
    team, info = adapt(team, recordings, budget_s=45)       # the "loading screen"

Also holds `evaluate`, used by every experiment.
"""

from __future__ import annotations

import copy
import time
from collections import Counter

import numpy as np

from mazehunt.adapt.ghost import GhostPlayer, dataset, train_ghost
from mazehunt.agents.mappo import LearnedMonsters, PPOConfig, Team, train
from mazehunt.bots.player_bots import training_population
from mazehunt.config import GameConfig
from mazehunt.core.game import Game, Outcome
from mazehunt.env.hunt_env import shared_maze
from mazehunt.record import LevelRecorder

# Fine-tuning settings: fewer envs and a lower, constant learning rate, so a short budget
# still gives many small updates without wrecking the pretrained behaviour.
FINETUNE = PPOConfig(num_envs=16, num_steps=128, lr=1e-4, anneal_lr=False, ent_coef=0.005)


def play_level(monsters, player, level: int = 1, config: GameConfig | None = None) -> dict:
    """One level, recorded in the same format as human play."""
    config = config or GameConfig()
    game = Game(shared_maze(config.map_name), config)
    if hasattr(player, "reset"):
        player.reset()
    rec = LevelRecorder()
    while not game.done:
        action = player.act(game)
        monster_actions = monsters.act(game)
        rec.log(game, action, monster_actions)
        game.step(action, monster_actions)
    return rec.as_record(level, game, getattr(monsters, "name", "?"))


def evaluate(monsters, player, episodes: int = 200, config: GameConfig | None = None) -> dict:
    """Capture rate and time-to-capture of `monsters` against `player`."""
    config = config or GameConfig()
    game = Game(shared_maze(config.map_name), config)
    outcomes, capture_ticks = Counter(), []
    for _ in range(episodes):
        game.reset()
        if hasattr(player, "reset"):
            player.reset()
        while not game.done:
            game.step(player.act(game), monsters.act(game))
        outcomes[game.outcome.value] += 1
        if game.outcome is Outcome.CAUGHT:
            capture_ticks.append(game.tick)
    return {"episodes": episodes,
            "capture_rate": outcomes["caught"] / episodes,
            "escape_rate": outcomes["escaped"] / episodes,
            "survive_rate": outcomes["survived"] / episodes,
            "mean_capture_ticks": float(np.mean(capture_ticks)) if capture_ticks else None}


def adapt(team: Team, recordings: list[dict], budget_s: float = 45.0, seed: int = 0,
          opponents=None, config: GameConfig | None = None, mix: float = 0.0) -> tuple[Team, dict]:
    """Clone the player from `recordings`, then fine-tune a copy of `team` against the clone
    for `budget_s` seconds. Pass `opponents` (a list factory) to fine-tune against something
    else instead, e.g. the real player (oracle) or the pretraining bots (control).

    `mix` > 0 keeps that fraction of the fine-tuning envs playing the pretraining bots
    (rehearsal), so the team learns this player without forgetting everyone else."""
    config = config or GameConfig()
    start = time.perf_counter()
    info: dict = {}
    if opponents is None:
        x, y = dataset(recordings, shared_maze(config.map_name), config)
        net, acc = train_ghost(x, y, seed=seed)
        info.update(ghost_examples=len(y), ghost_accuracy=round(acc, 3),
                    ghost_seconds=round(time.perf_counter() - start, 2))
        n_rehearse = round(mix * FINETUNE.num_envs)

        def opponents(i: int):
            if i < n_rehearse:
                return training_population(seed=seed * 1000 + i)
            return [GhostPlayer(net, seed=seed * 1000 + i)]
    new = copy.deepcopy(team)
    left = max(1.0, budget_s - (time.perf_counter() - start))
    summary = train(new, opponents, total_steps=10**9, cfg=FINETUNE, seed=seed, time_budget=left)
    info.update(finetune_steps=summary["steps"], total_seconds=round(time.perf_counter() - start, 1))
    return new, info


def controller(team: Team, seed: int | None = None, name: str | None = None) -> LearnedMonsters:
    return LearnedMonsters(team, name=name, seed=seed)
