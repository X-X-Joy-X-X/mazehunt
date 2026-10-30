"""The ghost player: a small network that imitates one player from their recorded levels.

Behavior cloning = supervised learning on (what the player saw → what the player pressed).
A few levels of play are a few hundred examples, so it trains in seconds on a CPU. The
monsters then practise against the ghost thousands of times instead of waiting for the real
player to play thousands of levels.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from mazehunt.agents.mappo import NumpyMLP, mlp, sample
from mazehunt.config import GameConfig
from mazehunt.core.actions import NUM_PLAYER_ACTIONS, STAY
from mazehunt.core.game import Game
from mazehunt.core.maze import Maze
from mazehunt.env.features import PLAYER_OBS_DIM, PlayerView, player_obs


def dataset(recordings: list[dict], maze: Maze, config: GameConfig) -> tuple[np.ndarray, np.ndarray]:
    """(observations, player actions) from recorded levels, in the format record.py writes."""
    xs, ys = [], []
    for rec in recordings:
        prev = STAY
        for step in rec["steps"]:
            view = PlayerView.from_step(step, prev)
            xs.append(player_obs(maze, view, config.max_ammo, config.max_ticks))
            ys.append(step["player_action"])
            prev = step["player_action"]
    if not xs:
        return np.zeros((0, PLAYER_OBS_DIM), np.float32), np.zeros(0, np.int64)
    return np.stack(xs), np.asarray(ys, np.int64)


def train_ghost(x: np.ndarray, y: np.ndarray, grad_steps: int = 600, batch: int = 256,
                seed: int = 0) -> tuple[nn.Sequential, float]:
    """Returns the cloned policy network and its training accuracy."""
    torch.manual_seed(seed)
    net = mlp(PLAYER_OBS_DIM, NUM_PLAYER_ACTIONS, out_std=0.01)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(grad_steps):
        idx = torch.randint(len(xt), (min(batch, len(xt)),), generator=gen)
        loss = nn.functional.cross_entropy(net(xt[idx]), yt[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (net(xt).argmax(-1) == yt).float().mean().item()
    return net, acc


class GhostPlayer:
    """Plays like the cloned player. Samples from the cloned action distribution, so it
    keeps some of the player's variety instead of one rigid path. A temperature below 1
    sharpens it toward the player's most likely choice (0.5 matched the bots best)."""

    name = "ghost"

    def __init__(self, net: nn.Sequential, seed: int | None = None, temperature: float = 0.5):
        self.policy = NumpyMLP(net)
        self.temperature = temperature
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self) -> None:
        self.prev = STAY

    def act(self, game: Game) -> int:
        view = PlayerView.from_game(game, self.prev)
        obs = player_obs(game.maze, view, game.config.max_ammo, game.config.max_ticks)
        self.prev = int(sample(self.policy(obs) / self.temperature, self.rng))
        return self.prev
