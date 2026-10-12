"""Headless rule tests. From the RL Game folder:  python -m pytest"""

import random
import time

import pytest

from mazehunt.agents.scripted_monsters import GreedyChasers, RandomMonsters
from mazehunt.config import GameConfig
from mazehunt.core.actions import (LEFT, NUM_PLAYER_ACTIONS, RIGHT, SHOOT_LEFT, SHOOT_RIGHT, STAY,
                                   UP)
from mazehunt.core.game import Game, Outcome
from mazehunt.core.maze import UNREACHABLE, Maze

CORRIDOR = [
    "#########",
    "#P....ME#",
    "#########",
]


def make(rows, **config) -> Game:
    """Rule tests use full-speed monsters with no start delay unless a test says otherwise."""
    config = {"monster_speed": 1.0, "monster_start_delay_ticks": 0, **config}
    return Game(Maze(rows), GameConfig(**config))


def test_arena_map_is_valid():
    maze = Maze.load("arena")
    assert (maze.width, maze.height) == (23, 15)
    assert len(maze.monster_spawns) == 3
    dist = maze.distances_to(maze.player_spawn)
    assert dist[maze.exit[1]][maze.exit[0]] < UNREACHABLE


@pytest.mark.parametrize("rows", [
    ["#####", "#P.E", "#####"],                 # ragged row
    ["#####", "#P.M#", "#####"],                # no exit
    ["#######", "#P#.ME#", "#######"],          # exit walled off from the player
    ["#####", "#PME.", "#####"],                # hole in the border
])
def test_bad_maps_are_rejected(rows):
    with pytest.raises(ValueError):
        Maze(rows)


def test_walls_block_movement():
    game = make(CORRIDOR)
    game.step(UP, [STAY])
    assert game.player.pos == (1, 1)


def test_monster_catches_player_on_same_tile():
    game = make(CORRIDOR)
    for _ in range(4):
        game.step(STAY, [LEFT])
        assert not game.done
    events = game.step(STAY, [LEFT])
    assert game.outcome is Outcome.CAUGHT and events.caught_by == 0


def test_monster_catches_player_when_they_swap_tiles():
    game = make(CORRIDOR)
    game.player.pos, game.monsters[0].pos = (3, 1), (4, 1)
    game.step(RIGHT, [LEFT])
    assert game.outcome is Outcome.CAUGHT


def test_shooting_kills_monster_which_then_respawns():
    game = make(CORRIDOR, monster_hp=2, monster_respawn_ticks=5)
    killed_at = None
    for tick in range(1, 20):
        events = game.step(SHOOT_RIGHT, [STAY])
        if events.monsters_killed:
            killed_at = tick
            break
    m = game.monsters[0]
    assert killed_at is not None and not m.alive

    # A dead monster can't catch anyone, even with the player standing on its tile.
    game.player.pos = m.pos
    game.step(LEFT, [STAY])
    assert not game.done

    # Dead for exactly `monster_respawn_ticks` ticks, counting the tick it died.
    for _ in range(3):
        game.step(STAY, [STAY])
        assert not m.alive
    game.step(STAY, [STAY])
    assert m.alive and m.hp == 2 and m.pos == m.spawn


def test_reaching_exit_ends_level():
    game = make(["#######", "#P.E.M#", "#######"])
    game.step(RIGHT, [STAY])
    game.step(RIGHT, [STAY])
    assert game.outcome is Outcome.ESCAPED


def test_surviving_the_timer_ends_level():
    game = make(CORRIDOR, max_ticks=3)
    for _ in range(3):
        game.step(STAY, [STAY])
    assert game.outcome is Outcome.SURVIVED


def test_ammo_limits_shots():
    game = make(CORRIDOR, max_ammo=2, shoot_cooldown_ticks=0, ammo_regen_ticks=100)
    shots = sum(game.step(SHOOT_LEFT, [STAY]).shot_fired for _ in range(6))
    assert shots == 2


def test_cooldown_spaces_out_shots():
    game = make(CORRIDOR, max_ammo=99, shoot_cooldown_ticks=2)
    shots = sum(game.step(SHOOT_LEFT, [STAY]).shot_fired for _ in range(6))
    assert shots == 3


def test_monsters_cannot_share_a_tile():
    game = make(["##########", "#P...MM.E#", "##########"])
    game.step(STAY, [STAY, LEFT])
    assert game.monsters[1].pos == (6, 1)


def test_monsters_wait_out_the_start_delay():
    game = make(CORRIDOR, monster_start_delay_ticks=3)
    for _ in range(3):
        game.step(STAY, [LEFT])
    assert game.monsters[0].pos == (6, 1)
    game.step(STAY, [LEFT])
    assert game.monsters[0].pos == (5, 1)


def test_slow_monsters_move_on_a_fraction_of_ticks():
    game = make(CORRIDOR, monster_speed=0.5, max_ticks=100)
    game.monsters[0].pos = (6, 1)
    for _ in range(4):
        game.step(STAY, [LEFT])
    assert game.monsters[0].pos == (4, 1)  # 2 moves in 4 ticks


def test_greedy_chasers_close_in():
    game = Game(Maze.load("arena"), GameConfig(monster_speed=1.0, monster_start_delay_ticks=0))
    chasers = GreedyChasers(epsilon=0.0, seed=0)
    dist = game.maze.distances_to(game.player.pos)
    before = [dist[m.pos[1]][m.pos[0]] for m in game.monsters]
    for _ in range(5):
        game.step(STAY, chasers.act(game))
    after = [dist[m.pos[1]][m.pos[0]] for m in game.monsters]
    assert all(a < b for a, b in zip(after, before))


def _play_random(seed: int) -> tuple:
    rng = random.Random(seed)
    game = Game(Maze.load("arena"))
    monsters = RandomMonsters(seed=seed)
    while not game.done:
        game.step(rng.randrange(NUM_PLAYER_ACTIONS), monsters.act(game))
    return game.tick, game.outcome, game.player.pos, [m.pos for m in game.monsters]


def test_rules_are_deterministic():
    assert _play_random(7) == _play_random(7)


def test_headless_speed():
    """Training needs millions of steps, so the rules must be fast without graphics."""
    rng = random.Random(0)
    game = Game(Maze.load("arena"))
    monsters = GreedyChasers(seed=0)
    steps, start = 0, time.perf_counter()
    while steps < 20_000:
        if game.done:
            game.reset()
        game.step(rng.randrange(NUM_PLAYER_ACTIONS), monsters.act(game))
        steps += 1
    rate = steps / (time.perf_counter() - start)
    print(f"\nheadless speed: {rate:,.0f} steps/s (greedy monsters included)")
    assert rate > 2_000
