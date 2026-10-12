"""Tile maze: loading, validation and shortest-path distances.

Map legend:  #  wall    .  floor    P  player spawn    M  monster spawn    E  exit
Positions are (x, y) = (column, row), with y growing downward.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

from mazehunt.core.actions import MOVES, Pos, step_pos

MAPS_DIR = Path(__file__).resolve().parent.parent / "maps"
UNREACHABLE = 1_000_000
_LEGEND = set("#.PME")


class Maze:
    def __init__(self, rows: list[str]):
        rows = [row.rstrip() for row in rows if row.strip()]
        if not rows:
            raise ValueError("map is empty")
        self.width, self.height = len(rows[0]), len(rows)
        for y, row in enumerate(rows):
            if len(row) != self.width:
                raise ValueError(f"row {y} has {len(row)} tiles, expected {self.width}")
            if set(row) - _LEGEND:
                raise ValueError(f"row {y} has unknown tiles {set(row) - _LEGEND}")

        def find(ch: str) -> list[Pos]:
            return [(x, y) for y, row in enumerate(rows) for x, c in enumerate(row) if c == ch]

        players, exits, monsters = find("P"), find("E"), find("M")
        if len(players) != 1 or len(exits) != 1:
            raise ValueError("map needs exactly one P (player spawn) and one E (exit)")
        if not monsters:
            raise ValueError("map needs at least one M (monster spawn)")

        # A solid border means the rules never have to bounds-check.
        border = [rows[0], rows[-1], *(row[0] + row[-1] for row in rows)]
        if any(set(edge) != {"#"} for edge in border):
            raise ValueError("map border must be all walls")

        self._walls = [[c == "#" for c in row] for row in rows]
        self.player_spawn: Pos = players[0]
        self.exit: Pos = exits[0]
        self.monster_spawns: list[Pos] = monsters
        self.open_tiles: list[Pos] = [(x, y) for y in range(self.height) for x in range(self.width)
                                      if not self._walls[y][x]]
        self._distance_cache: dict[Pos, list[list[int]]] = {}

        dist = self.distances_to(self.player_spawn)
        if any(dist[y][x] == UNREACHABLE for x, y in [self.exit, *self.monster_spawns]):
            raise ValueError("exit and every monster spawn must be reachable from the player spawn")

    @classmethod
    def load(cls, name: str) -> Maze:
        return cls((MAPS_DIR / f"{name}.txt").read_text().splitlines())

    def is_wall(self, pos: Pos) -> bool:
        return self._walls[pos[1]][pos[0]]

    def distances_to(self, target: Pos) -> list[list[int]]:
        """BFS distance (in moves) from every tile to `target`, indexed [y][x]. Cached per target."""
        cached = self._distance_cache.get(target)
        if cached is not None:
            return cached
        dist = [[UNREACHABLE] * self.width for _ in range(self.height)]
        dist[target[1]][target[0]] = 0
        queue = deque([target])
        while queue:
            pos = queue.popleft()
            d = dist[pos[1]][pos[0]] + 1
            for move in MOVES:
                nx, ny = step_pos(pos, move)
                if not self._walls[ny][nx] and dist[ny][nx] == UNREACHABLE:
                    dist[ny][nx] = d
                    queue.append((nx, ny))
        self._distance_cache[target] = dist
        return dist
