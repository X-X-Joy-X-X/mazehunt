"""All game-balance numbers in one place. Tune them here, never inside the rules."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GameConfig:
    map_name: str = "arena"
    max_ticks: int = 480              # level length: 60 s at 8 ticks/s
    ticks_per_second: int = 8         # human play speed only; training runs as fast as the CPU allows
    monster_speed: float = 0.75           # moves per tick; the player moves 1.0
    monster_start_delay_ticks: int = 0    # ticks monsters stay frozen at level start (0 = off)
    monster_hp: int = 1
    monster_respawn_ticks: int = 40       # ticks a killed monster stays dead: 5 s at 8 ticks/s
    bullet_speed: int = 2             # tiles per tick (everyone else moves 1)
    max_ammo: int = 3
    ammo_regen_ticks: int = 12        # +1 ammo every 1.5 s
    shoot_cooldown_ticks: int = 2     # ticks between shots
    lives: int = 3                    # human play sessions only
