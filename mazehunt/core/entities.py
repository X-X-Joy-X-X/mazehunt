"""Plain data for everything that moves. The rules that change them live in game.py.

`prev_pos` is where the entity was at the start of the current tick. The renderer uses it
to animate smoothly between tiles, and the capture check uses it to catch swaps.
"""

from __future__ import annotations

from dataclasses import dataclass

from mazehunt.core.actions import DOWN, Pos


@dataclass(slots=True)
class Player:
    pos: Pos
    prev_pos: Pos
    ammo: int
    facing: int = DOWN
    cooldown: int = 0    # ticks until the next shot is allowed
    ammo_timer: int = 0  # ticks toward the next ammo refill

    @property
    def can_shoot(self) -> bool:
        return self.cooldown == 0 and self.ammo > 0


@dataclass(slots=True)
class Monster:
    index: int
    spawn: Pos
    pos: Pos
    prev_pos: Pos
    hp: int
    facing: int = DOWN
    alive: bool = True
    respawn_timer: int = 0
    last_hit_tick: int = -1_000  # lets the renderer flash hits and show deaths


@dataclass(slots=True)
class Bullet:
    pos: Pos
    prev_pos: Pos
    direction: int
