"""Saves every level of human play as JSON. These files train the ghost player in week 4."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from mazehunt.core.game import Game

RECORDINGS_DIR = Path(__file__).resolve().parents[1] / "results" / "recordings"


def new_session_dir() -> Path:
    """A fresh folder per play session. Created on the first save, not here."""
    return RECORDINGS_DIR / time.strftime("%Y%m%d-%H%M%S")


class LevelRecorder:
    def __init__(self) -> None:
        self.steps: list[dict] = []

    def clear(self) -> None:
        self.steps = []

    def log(self, game: Game, player_action: int, monster_actions: Sequence[int]) -> None:
        """Call before game.step(): records what the player saw and what they chose."""
        p = game.player
        self.steps.append({
            "tick": game.tick,
            "player": list(p.pos),
            "ammo": p.ammo,
            "cooldown": p.cooldown,
            "monsters": [[*m.pos, m.hp] if m.alive else None for m in game.monsters],
            "bullets": [[*b.pos, b.direction] for b in game.bullets],
            "player_action": player_action,
            "monster_actions": list(monster_actions),
        })

    def save(self, session_dir: Path, level: int, game: Game, monster_controller: str) -> Path:
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"level_{level:02d}.json"
        path.write_text(json.dumps({
            "level": level,
            "map": game.config.map_name,
            "monster_controller": monster_controller,
            "outcome": game.outcome.value,
            "ticks_played": game.tick,
            "config": asdict(game.config),
            "steps": self.steps,
        }))
        return path
