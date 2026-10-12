"""Discrete action spaces, shared by the rules, human input, scripted bots and (later) the RL environment."""

from __future__ import annotations

Pos = tuple[int, int]

STAY, UP, DOWN, LEFT, RIGHT = 0, 1, 2, 3, 4
SHOOT_UP, SHOOT_DOWN, SHOOT_LEFT, SHOOT_RIGHT = 5, 6, 7, 8

MOVES = (UP, DOWN, LEFT, RIGHT)
NUM_MONSTER_ACTIONS = 5  # monsters can only move (or stay)
NUM_PLAYER_ACTIONS = 9   # the player can move, or shoot in one of 4 directions

# (dx, dy) per direction; y grows downward, like screen coordinates.
DELTAS = {STAY: (0, 0), UP: (0, -1), DOWN: (0, 1), LEFT: (-1, 0), RIGHT: (1, 0)}
SHOOT_TO_DIR = {SHOOT_UP: UP, SHOOT_DOWN: DOWN, SHOOT_LEFT: LEFT, SHOOT_RIGHT: RIGHT}


def step_pos(pos: Pos, direction: int) -> Pos:
    dx, dy = DELTAS[direction]
    return (pos[0] + dx, pos[1] + dy)
