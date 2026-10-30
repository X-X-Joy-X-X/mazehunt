"""Turns game state into the fixed-size number vectors that the networks read.

Two views:
  * monster_obs: what one monster sees. Used by the MAPPO/IPPO monster team.
  * player_obs:  what the player sees. Used by the ghost player (behavior cloning), so it
    can be built from a live Game *or* from a saved recording step.

Maze distances (BFS) do the heavy lifting: "does this move bring me closer to the player /
the exit?" is one table lookup, so small MLPs learn quickly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mazehunt.core.actions import DELTAS, MOVES, NUM_PLAYER_ACTIONS, Pos, step_pos
from mazehunt.core.game import Game
from mazehunt.core.maze import UNREACHABLE, Maze

DIST_SCALE = 40.0  # longest shortest-path in the arena is ~40 tiles
LOOKAHEAD = 6      # tiles a monster scans for incoming bullets


def _line_toward(maze: Maze, a: Pos, b: Pos) -> int | None:
    """Tiles from a to b along a clear row/column, else None."""
    if a[0] != b[0] and a[1] != b[1]:
        return None
    dx, dy = (b[0] > a[0]) - (b[0] < a[0]), (b[1] > a[1]) - (b[1] < a[1])
    x, y, n = a[0], a[1], 0
    while (x, y) != b:
        x, y, n = x + dx, y + dy, n + 1
        if maze.is_wall((x, y)):
            return None
    return n


def _d(table: list[list[int]], pos: Pos) -> float:
    v = table[pos[1]][pos[0]]
    return min(v, UNREACHABLE) / DIST_SCALE if v < UNREACHABLE else 2.0


# --- monster view ---------------------------------------------------------------------------

MONSTER_OBS_DIM = 3 + 4 + 12 + 2 + 2 + 2 + 3 + 8 + 4


def monster_obs(game: Game, player_move: tuple[int, int]) -> np.ndarray:
    """One row per monster. `player_move` is the player's last (dx, dy), so monsters can
    tell which way the player is heading."""
    maze, p, cfg = game.maze, game.player, game.config
    n = len(game.monsters)
    to_player = maze.distances_to(p.pos)
    to_exit = maze.distances_to(maze.exit)
    out = np.zeros((n, MONSTER_OBS_DIM), dtype=np.float32)
    time_left = game.ticks_left / cfg.max_ticks
    moves_next = float(game.tick + 1 > cfg.monster_start_delay_ticks
                       and game._monster_move_budget + cfg.monster_speed >= 1.0)
    player_exit = _d(to_exit, p.pos)

    for i, m in enumerate(game.monsters):
        row = out[i]
        row[i] = 1.0                                                           # which monster am I
        k = 3
        row[k:k + 4] = (float(m.alive), m.respawn_timer / cfg.monster_respawn_ticks,
                        m.pos[0] / maze.width, m.pos[1] / maze.height)
        k += 4
        here_p, here_e = to_player[m.pos[1]][m.pos[0]], to_exit[m.pos[1]][m.pos[0]]
        for move in MOVES:                                                     # what each step does
            t = step_pos(m.pos, move)
            if maze.is_wall(t):
                row[k:k + 3] = (1.0, 0.0, 0.0)
            else:
                row[k:k + 3] = (0.0, to_player[t[1]][t[0]] - here_p, to_exit[t[1]][t[0]] - here_e)
            k += 3
        row[k:k + 2] = (_d(to_player, m.pos), _d(to_exit, m.pos))
        k += 2
        row[k:k + 2] = ((p.pos[0] - m.pos[0]) / maze.width, (p.pos[1] - m.pos[1]) / maze.height)
        k += 2
        row[k:k + 2] = player_move
        k += 2
        line = _line_toward(maze, p.pos, m.pos) if m.alive else None           # danger
        threat = 0.0
        for b in game.bullets:
            n_tiles = _line_toward(maze, b.pos, m.pos)
            if n_tiles is not None and 0 < n_tiles <= LOOKAHEAD:
                dx, dy = DELTAS[b.direction]
                if (m.pos[0] - b.pos[0]) * dx + (m.pos[1] - b.pos[1]) * dy > 0:
                    threat = 1.0
        row[k:k + 3] = (float(line is not None), (line or 0) / DIST_SCALE, threat)
        k += 3
        for j, o in enumerate(x for x in game.monsters if x.index != i):       # teammates
            row[k:k + 4] = (float(o.alive), (o.pos[0] - m.pos[0]) / maze.width,
                            (o.pos[1] - m.pos[1]) / maze.height, _d(to_player, o.pos))
            k += 4
        row[k:k + 4] = (p.ammo / cfg.max_ammo, time_left, moves_next, player_exit)
    return out


def global_state(obs: np.ndarray) -> np.ndarray:
    """MAPPO's centralized critic sees every monster's view at once."""
    return obs.reshape(-1)


# --- player view (ghost) --------------------------------------------------------------------

@dataclass(slots=True)
class PlayerView:
    """The part of the game the player reacts to. Built from a Game or a recording step."""
    pos: Pos
    ammo: int
    cooldown: int
    monsters: list[Pos | None]  # None = dead
    tick: int
    prev_action: int

    @classmethod
    def from_game(cls, game: Game, prev_action: int) -> PlayerView:
        return cls(game.player.pos, game.player.ammo, game.player.cooldown,
                   [m.pos if m.alive else None for m in game.monsters], game.tick, prev_action)

    @classmethod
    def from_step(cls, step: dict, prev_action: int) -> PlayerView:
        return cls(tuple(step["player"]), step["ammo"], step["cooldown"],
                   [tuple(m[:2]) if m is not None else None for m in step["monsters"]],
                   step["tick"], prev_action)


PLAYER_OBS_DIM = 2 + 2 + 12 + 1 + 15 + 4 + NUM_PLAYER_ACTIONS + 1


def player_obs(maze: Maze, view: PlayerView, max_ammo: int, max_ticks: int) -> np.ndarray:
    out = np.zeros(PLAYER_OBS_DIM, dtype=np.float32)
    pos = view.pos
    to_exit = maze.distances_to(maze.exit)
    alive = [m for m in view.monsters if m is not None]
    monster_tables = [maze.distances_to(m) for m in alive]

    def nearest_monster(t: Pos) -> float:
        if not monster_tables:
            return 1.0
        return min(min(tab[t[1]][t[0]] for tab in monster_tables) / 10.0, 1.0)

    out[0:4] = (pos[0] / maze.width, pos[1] / maze.height, view.ammo / max_ammo, float(view.cooldown > 0))
    k = 4
    for move in MOVES:
        t = step_pos(pos, move)
        if maze.is_wall(t):
            out[k:k + 3] = (1.0, 0.0, 0.0)
        else:
            out[k:k + 3] = (0.0, to_exit[t[1]][t[0]] - to_exit[pos[1]][pos[0]], nearest_monster(t))
        k += 3
    out[k] = _d(to_exit, pos)
    k += 1
    for m in view.monsters:
        if m is None:
            k += 5
            continue
        line = _line_toward(maze, pos, m)
        out[k:k + 5] = (1.0, (m[0] - pos[0]) / maze.width, (m[1] - pos[1]) / maze.height,
                        _d(maze.distances_to(m), pos), float(line is not None))
        k += 5
    k = 4 + 12 + 1 + 15
    for d, move in enumerate(MOVES):  # nearest monster in a clear line in each shooting direction
        dx, dy = DELTAS[move]
        best = 0.0
        for m in alive:
            line = _line_toward(maze, pos, m)
            if line and (m[0] - pos[0]) * dx + (m[1] - pos[1]) * dy > 0:
                best = max(best, 1.0 - line / 12.0)
        out[k + d] = best
    k += 4
    out[k + view.prev_action] = 1.0
    k += NUM_PLAYER_ACTIONS
    out[k] = view.tick / max_ticks
    return out
