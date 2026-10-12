"""Hand-written monster teams. These are the baselines the RL team has to beat."""

from __future__ import annotations

import random

from mazehunt.core.actions import MOVES, STAY, step_pos
from mazehunt.core.game import Game


class RandomMonsters:
    """Each monster wanders randomly. The weakest possible baseline."""

    name = "random"

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)

    def act(self, game: Game) -> list[int]:
        return [self.rng.choice(MOVES) if m.alive else STAY for m in game.monsters]


class GreedyChasers:
    """Each monster walks its own shortest path to the player.

    There's no teamwork: they usually queue up behind each other in the same corridor, so a
    player can kite them around a loop. A learned team should beat this by splitting up,
    flanking and guarding the exit.
    """

    name = "greedy"

    def __init__(self, epsilon: float = 0.1, seed: int | None = None):
        self.epsilon = epsilon  # chance of a random move, so they aren't perfectly predictable
        self.rng = random.Random(seed)

    def act(self, game: Game) -> list[int]:
        dist = game.maze.distances_to(game.player.pos)
        actions = []
        for m in game.monsters:
            if not m.alive:
                actions.append(STAY)
                continue
            if self.rng.random() < self.epsilon:
                actions.append(self.rng.choice(MOVES))
                continue
            best, best_dist = [STAY], dist[m.pos[1]][m.pos[0]]
            for move in MOVES:
                x, y = step_pos(m.pos, move)
                if game.maze.is_wall((x, y)):
                    continue
                if dist[y][x] < best_dist:
                    best, best_dist = [move], dist[y][x]
                elif dist[y][x] == best_dist:
                    best.append(move)
            actions.append(self.rng.choice(best))
        return actions


CONTROLLERS = {"greedy": GreedyChasers, "random": RandomMonsters}
