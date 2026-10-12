"""MazeHunt rules for one level.

Deterministic: the same state and actions always give the same next state. The playable
game and RL training both drive this class, so monsters train on exactly the game you play.

One call to `step` is one tick. Order within a tick:
  1. dead monsters whose timer ran out respawn
  2. the player moves or fires
  3. bullets fly `bullet_speed` tiles and stop at the first wall or monster
  4. monsters move (a monster can't step into a wall or another monster). They're slower
     than the player: on ticks where their speed budget runs short, their actions are ignored.
  5. bullets hit any monster that stepped onto them
  6. capture check: a monster on the player's tile, or the two swapped tiles
  7. escape and timeout checks
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum

from mazehunt.config import GameConfig
from mazehunt.core.actions import DELTAS, MOVES, SHOOT_TO_DIR, STAY, Pos, step_pos
from mazehunt.core.entities import Bullet, Monster, Player
from mazehunt.core.maze import Maze


class Outcome(Enum):
    RUNNING = "running"
    ESCAPED = "escaped"    # player reached the exit
    SURVIVED = "survived"  # time ran out with the player still alive
    CAUGHT = "caught"      # a monster reached the player


@dataclass
class StepEvents:
    """What happened during one tick. The RL environment turns these into rewards."""
    shot_fired: bool = False
    monsters_hit: list[int] = field(default_factory=list)
    monsters_killed: list[int] = field(default_factory=list)
    caught_by: int | None = None


class Game:
    def __init__(self, maze: Maze, config: GameConfig | None = None):
        self.maze = maze
        self.config = config or GameConfig()
        self.reset()

    def reset(self) -> None:
        spawn = self.maze.player_spawn
        self.tick = 0
        self.outcome = Outcome.RUNNING
        self.player = Player(pos=spawn, prev_pos=spawn, ammo=self.config.max_ammo)
        self.monsters = [Monster(index=i, spawn=s, pos=s, prev_pos=s, hp=self.config.monster_hp)
                         for i, s in enumerate(self.maze.monster_spawns)]
        self.bullets: list[Bullet] = []
        self._monster_move_budget = 0.0

    @property
    def done(self) -> bool:
        return self.outcome is not Outcome.RUNNING

    @property
    def ticks_left(self) -> int:
        return self.config.max_ticks - self.tick

    def step(self, player_action: int, monster_actions: Sequence[int]) -> StepEvents:
        if self.done:
            raise RuntimeError("level is over; call reset()")
        if len(monster_actions) != len(self.monsters):
            raise ValueError(f"expected {len(self.monsters)} monster actions, got {len(monster_actions)}")

        events = StepEvents()
        self.tick += 1
        self._remember_positions()
        self._respawn_phase()
        self._player_phase(player_action, events)
        self._bullet_phase(events)
        if self._monsters_move_this_tick():
            self._monster_phase(monster_actions)
        self._bullets_hit_monsters_standing_on_them(events)
        self._capture_check(events)
        if self.done:
            return events
        if self.player.pos == self.maze.exit:
            self.outcome = Outcome.ESCAPED
        elif self.tick >= self.config.max_ticks:
            self.outcome = Outcome.SURVIVED
        return events

    # --- phases -------------------------------------------------------------------------

    def _remember_positions(self) -> None:
        self.player.prev_pos = self.player.pos
        for m in self.monsters:
            m.prev_pos = m.pos
        for b in self.bullets:
            b.prev_pos = b.pos

    def _player_phase(self, action: int, events: StepEvents) -> None:
        p, cfg = self.player, self.config
        if action in SHOOT_TO_DIR:
            p.facing = SHOOT_TO_DIR[action]
            if p.can_shoot:
                p.ammo -= 1
                p.cooldown = cfg.shoot_cooldown_ticks
                self.bullets.append(Bullet(pos=p.pos, prev_pos=p.pos, direction=p.facing))
                events.shot_fired = True
        elif action in MOVES:
            p.facing = action
            target = step_pos(p.pos, action)
            if not self.maze.is_wall(target):
                p.pos = target
        elif action != STAY:
            raise ValueError(f"invalid player action {action}")

        if p.cooldown > 0:
            p.cooldown -= 1
        if p.ammo < cfg.max_ammo:
            p.ammo_timer += 1
            if p.ammo_timer >= cfg.ammo_regen_ticks:
                p.ammo += 1
                p.ammo_timer = 0
        else:
            p.ammo_timer = 0

    def _bullet_phase(self, events: StepEvents) -> None:
        self.bullets = [b for b in self.bullets if self._fly(b, events)]

    def _fly(self, bullet: Bullet, events: StepEvents) -> bool:
        """Advance one bullet tile by tile. Returns False once it hits something."""
        for _ in range(self.config.bullet_speed):
            target = step_pos(bullet.pos, bullet.direction)
            if self.maze.is_wall(target):
                return False
            bullet.pos = target
            monster = self._monster_at(target)
            if monster is not None:
                self._hit(monster, events)
                return False
        return True

    def _monsters_move_this_tick(self) -> bool:
        """Monsters wait out the start delay, then move on a `monster_speed` fraction of ticks."""
        if self.tick <= self.config.monster_start_delay_ticks:
            return False
        self._monster_move_budget += self.config.monster_speed
        if self._monster_move_budget >= 1.0:
            self._monster_move_budget -= 1.0
            return True
        return False

    def _monster_phase(self, actions: Sequence[int]) -> None:
        occupied = {m.pos for m in self.monsters if m.alive}
        for m, action in zip(self.monsters, actions):
            if action not in DELTAS:
                raise ValueError(f"invalid monster action {action}")
            if not m.alive or action == STAY:
                continue
            m.facing = action
            target = step_pos(m.pos, action)
            if self.maze.is_wall(target) or target in occupied:
                continue
            occupied.discard(m.pos)
            occupied.add(target)
            m.pos = target

    def _bullets_hit_monsters_standing_on_them(self, events: StepEvents) -> None:
        remaining = []
        for b in self.bullets:
            monster = self._monster_at(b.pos)
            if monster is None:
                remaining.append(b)
            else:
                self._hit(monster, events)
        self.bullets = remaining

    def _capture_check(self, events: StepEvents) -> None:
        p = self.player
        for m in self.monsters:
            if not m.alive:
                continue
            swapped = m.pos == p.prev_pos and m.prev_pos == p.pos
            if m.pos == p.pos or swapped:
                events.caught_by = m.index
                self.outcome = Outcome.CAUGHT
                return

    def _respawn_phase(self) -> None:
        for m in self.monsters:
            if m.alive:
                continue
            m.respawn_timer -= 1
            blocked = m.spawn == self.player.pos or self._monster_at(m.spawn) is not None
            if m.respawn_timer <= 0 and not blocked:
                m.alive = True
                m.hp = self.config.monster_hp
                m.pos = m.prev_pos = m.spawn

    # --- helpers ------------------------------------------------------------------------

    def _monster_at(self, pos: Pos) -> Monster | None:
        for m in self.monsters:
            if m.alive and m.pos == pos:
                return m
        return None

    def _hit(self, monster: Monster, events: StepEvents) -> None:
        monster.hp -= 1
        monster.last_hit_tick = self.tick
        events.monsters_hit.append(monster.index)
        if monster.hp <= 0:
            monster.alive = False
            monster.respawn_timer = self.config.monster_respawn_ticks
            events.monsters_killed.append(monster.index)
