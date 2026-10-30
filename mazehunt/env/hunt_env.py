"""Multi-agent RL environment: the monster team is the learner, the player is the opponent.

    env = HuntEnv(opponents=[...])
    obs = env.reset()                       # (n_monsters, MONSTER_OBS_DIM)
    obs, reward, done, info = env.step(actions)

Parallel API (all monsters act at once, PettingZoo-parallel style) with one shared team
reward, the standard cooperative MARL setting MAPPO was designed for. Any object with
`act(game)` (and optionally `reset()`) can be the player: scripted bots, the ghost, a
recording replay.
"""

from __future__ import annotations

import random
from functools import lru_cache
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from mazehunt.config import GameConfig
from mazehunt.core.game import Game, Outcome
from mazehunt.core.maze import Maze
from mazehunt.env.features import DIST_SCALE, global_state, monster_obs


@lru_cache(maxsize=None)
def shared_maze(name: str) -> Maze:
    """One Maze per map per process, so every env shares the same BFS distance cache."""
    return Maze.load(name)


@dataclass(frozen=True)
class RewardConfig:
    capture: float = 10.0
    escape: float = -10.0     # the player reached the exit
    timeout: float = -10.0    # the player survived the clock: also a loss for the monsters
    monster_killed: float = 0.0  # dying already costs 5 s out of the hunt
    # Difference shaping: the per-tick terms telescope to (end - start) over a level, so they
    # guide early learning without being farmable.
    chase: float = 2.0        # closest monster's maze distance to the player shrinks
    block: float = 0.5        # the player's maze distance to the exit grows


class HuntEnv:
    def __init__(self, opponents: Sequence, config: GameConfig | None = None,
                 rewards: RewardConfig | None = None, seed: int | None = None):
        if not opponents:
            raise ValueError("need at least one opponent")
        self.opponents = list(opponents)
        self.config = config or GameConfig()
        self.rewards = rewards or RewardConfig()
        self.game = Game(shared_maze(self.config.map_name), self.config)
        self.rng = random.Random(seed)
        self.n_agents = len(self.game.monsters)
        self.opponent = self.opponents[0]

    @property
    def obs_dim(self) -> int:
        return self._obs().shape[1]

    def reset(self) -> np.ndarray:
        self.game.reset()
        self.opponent = self.rng.choice(self.opponents)
        if hasattr(self.opponent, "reset"):
            self.opponent.reset()
        self.player_move = (0, 0)
        self._potential = self._phi()
        return self._obs()

    def step(self, actions: Sequence[int]) -> tuple[np.ndarray, float, bool, dict]:
        g, r = self.game, self.rewards
        events = g.step(self.opponent.act(g), [int(a) for a in actions])
        p = g.player
        self.player_move = (p.pos[0] - p.prev_pos[0], p.pos[1] - p.prev_pos[1])

        reward = r.monster_killed * len(events.monsters_killed)
        if g.outcome is Outcome.CAUGHT:
            reward += r.capture
        elif g.outcome is Outcome.ESCAPED:
            reward += r.escape
        elif g.outcome is Outcome.SURVIVED:
            reward += r.timeout
        phi = self._phi()
        reward += phi - self._potential
        self._potential = phi

        info = {}
        if g.done:
            info = {"outcome": g.outcome.value, "ticks": g.tick,
                    "opponent": getattr(self.opponent, "name", "?")}
        return self._obs(), reward, g.done, info

    def state(self) -> np.ndarray:
        return global_state(self._obs())

    def _obs(self) -> np.ndarray:
        return monster_obs(self.game, self.player_move)

    def _phi(self) -> float:
        g, r = self.game, self.rewards
        to_player = g.maze.distances_to(g.player.pos)
        alive = [to_player[m.pos[1]][m.pos[0]] for m in g.monsters if m.alive]
        chase = min(alive) if alive else DIST_SCALE
        to_exit = g.maze.distances_to(g.maze.exit)[g.player.pos[1]][g.player.pos[0]]
        return (-r.chase * chase + r.block * to_exit) / DIST_SCALE
