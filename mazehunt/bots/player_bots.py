"""Scripted player bots: the opponents monsters pretrain against, and the held-out players
they're evaluated against.

Every bot has the same interface as a human at the keyboard:

    action = bot.act(game)   # one player action per tick

Each bot plays a recognisable *style* (rush the exit by a favourite route, camp a corner,
kite, hunt monsters...). Training uses one population of styles; evaluation uses held-out
styles the monsters never saw, so adaptation has to come from watching the player.
"""

from __future__ import annotations

import random

from mazehunt.core.actions import MOVES, SHOOT_DOWN, SHOOT_LEFT, SHOOT_RIGHT, SHOOT_UP, STAY, Pos, step_pos
from mazehunt.core.game import Game
from mazehunt.core.maze import UNREACHABLE

SHOOT_TOWARD = {(0, -1): SHOOT_UP, (0, 1): SHOOT_DOWN, (-1, 0): SHOOT_LEFT, (1, 0): SHOOT_RIGHT}


# --- shared helpers -------------------------------------------------------------------------

def clear_line(game: Game, a: Pos, b: Pos) -> int | None:
    """Tiles from a to b if they share a row/column with no wall between, else None."""
    if a[0] != b[0] and a[1] != b[1]:
        return None
    dx, dy = (b[0] > a[0]) - (b[0] < a[0]), (b[1] > a[1]) - (b[1] < a[1])
    x, y, n = a[0], a[1], 0
    while (x, y) != b:
        x, y, n = x + dx, y + dy, n + 1
        if game.maze.is_wall((x, y)):
            return None
    return n


def shot_at_monster(game: Game, max_range: int) -> int | None:
    """A shoot action that hits the nearest monster in a clear line, if the gun is ready."""
    if not game.player.can_shoot:
        return None
    p = game.player.pos
    best, best_n = None, max_range + 1
    for m in game.monsters:
        if not m.alive:
            continue
        n = clear_line(game, p, m.pos)
        if n is not None and 0 < n < best_n:
            d = ((m.pos[0] > p[0]) - (m.pos[0] < p[0]), (m.pos[1] > p[1]) - (m.pos[1] < p[1]))
            best, best_n = SHOOT_TOWARD[d], n
    return best


def monster_distance(game: Game, pos: Pos) -> int:
    """Maze distance from `pos` to the nearest live monster."""
    best = UNREACHABLE
    for m in game.monsters:
        if m.alive:
            best = min(best, game.maze.distances_to(m.pos)[pos[1]][pos[0]])
    return best


def open_moves(game: Game, pos: Pos) -> list[tuple[int, Pos]]:
    return [(a, t) for a in MOVES if not game.maze.is_wall(t := step_pos(pos, a))]


def flee(game: Game) -> int:
    """Step (or stay) to maximise distance from the nearest monster."""
    pos = game.player.pos
    options = [(STAY, pos), *open_moves(game, pos)]
    return max(options, key=lambda o: monster_distance(game, o[1]))[0]


def move_toward(game: Game, target: Pos, safety: int) -> int | None:
    """Shortest-path step toward `target` that keeps at least `safety` tiles from monsters.

    None means every step toward the target is unsafe.
    """
    pos = game.player.pos
    dist = game.maze.distances_to(target)
    here = dist[pos[1]][pos[0]]
    for action, tile in sorted(open_moves(game, pos), key=lambda o: dist[o[1][1]][o[1][0]]):
        if dist[tile[1]][tile[0]] >= here:
            break
        if monster_distance(game, tile) >= safety:
            return action
    return None


# --- bots -----------------------------------------------------------------------------------

class PlayerBot:
    """Base: an epsilon of random moves keeps every bot a little unpredictable, and
    `reaction` is the chance per tick of noticing a monster lined up for a shot. A bot that
    fires the same tick every time is superhuman; people react in a few hundred ms."""

    name = "bot"

    def __init__(self, epsilon: float = 0.05, reaction: float = 0.35,
                 reaction_range: tuple[float, float] | None = None, seed: int | None = None):
        self.epsilon, self.reaction = epsilon, reaction
        self.reaction_range = reaction_range  # if set, a new reaction is drawn every level
        self.rng = random.Random(seed)

    def start_level(self) -> None:
        if self.reaction_range is not None:
            self.reaction = self.rng.uniform(*self.reaction_range)

    def shoot(self, game: Game, max_range: int) -> int | None:
        if self.rng.random() >= self.reaction:
            return None
        return shot_at_monster(game, max_range)

    def reset(self) -> None:
        """Called at the start of every level, for bots that keep memory."""
        self.start_level()

    def act(self, game: Game) -> int:
        if self.rng.random() < self.epsilon:
            return self.rng.choice(MOVES)
        return self.decide(game)

    def decide(self, game: Game) -> int:
        raise NotImplementedError


class ExitRunner(PlayerBot):
    """Heads for the exit, optionally through favourite waypoints first; shoots or flees when
    a monster gets close. With no waypoints this is the bot that escaped 300/300 games against
    the greedy chasers."""

    def __init__(self, waypoints: tuple[Pos, ...] = (), shoot_range: int = 6, flee_radius: int = 3,
                 name: str = "exit_runner", **kw):
        super().__init__(**kw)
        self.waypoints, self.shoot_range, self.flee_radius = waypoints, shoot_range, flee_radius
        self.name = name
        self.reset()

    def reset(self) -> None:
        self.next_wp = 0
        self.start_level()

    def target(self, game: Game) -> Pos:
        while self.next_wp < len(self.waypoints) and game.player.pos == self.waypoints[self.next_wp]:
            self.next_wp += 1
        return self.waypoints[self.next_wp] if self.next_wp < len(self.waypoints) else game.maze.exit

    def decide(self, game: Game) -> int:
        shot = self.shoot(game, self.shoot_range)
        if shot is not None:
            return shot
        step = move_toward(game, self.target(game), safety=2)
        if step is not None and monster_distance(game, game.player.pos) > 1:
            return step
        if monster_distance(game, game.player.pos) <= self.flee_radius:
            return flee(game)
        return step if step is not None else STAY


class Camper(PlayerBot):
    """Walks to a favourite spot and holds it, shooting anything that lines up. Never exits."""

    def __init__(self, spot: Pos, shoot_range: int = 8, name: str = "camper", **kw):
        super().__init__(**kw)
        self.spot, self.shoot_range, self.name = spot, shoot_range, name

    def decide(self, game: Game) -> int:
        shot = self.shoot(game, self.shoot_range)
        if shot is not None:
            return shot
        if monster_distance(game, game.player.pos) <= 2:
            return flee(game)
        if game.player.pos == self.spot:
            return STAY
        step = move_toward(game, self.spot, safety=2)
        return step if step is not None else flee(game)


class Kiter(PlayerBot):
    """Never stops moving away from the monsters; shoots when it can. Runs out the clock."""

    def __init__(self, shoot_range: int = 5, name: str = "kiter", **kw):
        super().__init__(**kw)
        self.shoot_range, self.name = shoot_range, name

    def decide(self, game: Game) -> int:
        shot = self.shoot(game, self.shoot_range)
        return shot if shot is not None else flee(game)


class Hunter(PlayerBot):
    """Goes looking for monsters to shoot. Backs off while reloading; exits once all are dead."""

    def __init__(self, name: str = "hunter", **kw):
        super().__init__(**kw)
        self.name = name

    def decide(self, game: Game) -> int:
        shot = self.shoot(game, max_range=12)
        if shot is not None:
            return shot
        alive = [m for m in game.monsters if m.alive]
        if not alive:
            step = move_toward(game, game.maze.exit, safety=0)
            return step if step is not None else STAY
        if game.player.ammo == 0 or monster_distance(game, game.player.pos) <= 2:
            return flee(game)
        nearest = min(alive, key=lambda m: game.maze.distances_to(m.pos)[game.player.pos[1]][game.player.pos[0]])
        step = move_toward(game, nearest.pos, safety=3)
        return step if step is not None else STAY


class Waiter(PlayerBot):
    """Holds a hiding spot until no monster is within `trigger` tiles of the exit route, then
    dashes for the exit. A patient player: punishes teams that all leave the exit unguarded."""

    def __init__(self, spot: Pos, trigger: int = 8, name: str = "waiter", **kw):
        super().__init__(**kw)
        self.spot, self.trigger, self.name = spot, trigger, name
        self.reset()

    def reset(self) -> None:
        self.dashing = False
        self.start_level()

    def decide(self, game: Game) -> int:
        shot = self.shoot(game, 6)
        if shot is not None:
            return shot
        exit_dist = game.maze.distances_to(game.maze.exit)
        guards = [m for m in game.monsters if m.alive and exit_dist[m.pos[1]][m.pos[0]] < self.trigger]
        if not guards or game.tick > game.config.max_ticks // 2:
            self.dashing = True
        if self.dashing:
            step = move_toward(game, game.maze.exit, safety=2)
            if step is not None:
                return step
            return flee(game) if monster_distance(game, game.player.pos) <= 3 else STAY
        if monster_distance(game, game.player.pos) <= 2:
            return flee(game)
        if game.player.pos == self.spot:
            return STAY
        step = move_toward(game, self.spot, safety=2)
        return step if step is not None else flee(game)


class RandomBot(PlayerBot):
    def __init__(self, name: str = "random", **kw):
        super().__init__(epsilon=1.0, **kw)
        self.name = name

    def decide(self, game: Game) -> int:  # never reached: epsilon is 1
        return STAY


# --- populations ----------------------------------------------------------------------------

TRAIN_REACTION = (0.0, 0.5)  # skill randomization: easy levels teach chasing, hard ones teach caution


def training_population(seed: int | None = None, reaction_range=TRAIN_REACTION) -> list[PlayerBot]:
    """Styles the monsters pretrain against. Each level draws the bot's reaction from
    `reaction_range` (None = the fixed default)."""
    kw = {"seed": seed, "reaction_range": reaction_range}
    return [
        ExitRunner(**kw),
        ExitRunner(waypoints=((21, 1),), name="runner_top_right", **kw),
        ExitRunner(waypoints=((11, 9),), name="runner_middle", **kw),
        Camper((1, 13), name="camper_bottom_left", **kw),
        Camper((21, 1), name="camper_top_right", **kw),
        Kiter(**kw),
        Hunter(**kw),
        RandomBot(seed=seed),
    ]


def held_out_population(seed: int | None = None) -> list[PlayerBot]:
    """Styles never seen in pretraining. Evaluation and adaptation experiments use these."""
    return [
        ExitRunner(waypoints=((1, 13), (11, 11)), name="runner_bottom_left", seed=seed),
        Waiter((3, 7), name="waiter_left", seed=seed),
        Camper((11, 1), name="camper_top_middle", seed=seed),
    ]


def all_bots(seed: int | None = None) -> dict[str, PlayerBot]:
    """Every style at the fixed evaluation reaction (no randomization)."""
    return {b.name: b for b in [*training_population(seed, reaction_range=None), *held_out_population(seed)]}
