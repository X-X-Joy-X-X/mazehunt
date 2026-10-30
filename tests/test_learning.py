"""Tests for the learning side: bots, RL environment, MAPPO/IPPO, ghost player, adaptation."""

import numpy as np
import pytest
import torch

from mazehunt.adapt.ghost import GhostPlayer, dataset, train_ghost
from mazehunt.adapt.pipeline import adapt, evaluate, play_level
from mazehunt.agents.mappo import LearnedMonsters, NumpyMLP, PPOConfig, Team, train
from mazehunt.agents.scripted_monsters import GreedyChasers
from mazehunt.bots.player_bots import all_bots, held_out_population, training_population
from mazehunt.config import GameConfig
from mazehunt.core.actions import LEFT, NUM_PLAYER_ACTIONS, STAY
from mazehunt.core.game import Game
from mazehunt.env.features import MONSTER_OBS_DIM, PLAYER_OBS_DIM, PlayerView, player_obs
from mazehunt.env.hunt_env import HuntEnv, shared_maze

TINY = PPOConfig(num_envs=2, num_steps=16, update_epochs=1, num_minibatches=2)


def test_bot_populations_dont_overlap():
    train_names = {b.name for b in training_population()}
    held_names = {b.name for b in held_out_population()}
    assert train_names.isdisjoint(held_names)
    assert len(all_bots()) == len(train_names) + len(held_names)


@pytest.mark.parametrize("name", sorted(all_bots()))
def test_every_bot_plays_valid_actions(name):
    bot = all_bots(seed=0)[name]
    game = Game(shared_maze("arena"))
    bot.reset()
    while not game.done:
        action = bot.act(game)
        assert 0 <= action < NUM_PLAYER_ACTIONS
        game.step(action, GreedyChasers(seed=0).act(game))


def test_exit_runner_beats_greedy_chasers():
    """The baseline gap the learned team has to close."""
    r = evaluate(GreedyChasers(seed=1), all_bots(seed=2)["exit_runner"], episodes=50)
    assert r["escape_rate"] > 0.7


def test_env_shapes_and_episode_end():
    env = HuntEnv(training_population(seed=0), seed=0)
    obs = env.reset()
    assert obs.shape == (3, MONSTER_OBS_DIM) and env.state().shape == (3 * MONSTER_OBS_DIM,)
    for _ in range(GameConfig().max_ticks):
        obs, reward, done, info = env.step([0, 0, 0])
        assert np.isfinite(obs).all() and np.isfinite(reward)
        if done:
            assert info["outcome"] in ("caught", "escaped", "survived")
            break
    assert done


class _Statue:
    name = "statue"

    def act(self, game):
        return STAY


def test_capture_is_rewarded():
    env = HuntEnv([_Statue()], config=GameConfig(monster_speed=1.0), seed=0)
    env.reset()
    g = env.game
    g.player.pos = g.player.prev_pos = (g.monsters[0].pos[0] - 1, g.monsters[0].pos[1])
    env._potential = env._phi()
    _, reward, done, info = env.step([LEFT, STAY, STAY])  # monster 0 steps onto the player
    assert done and info["outcome"] == "caught" and reward > 9


@pytest.mark.parametrize("mode", ["mappo", "ippo"])
def test_ppo_runs_and_checkpoint_roundtrips(mode, tmp_path):
    team = Team(mode, n_agents=3)
    summary = train(team, lambda i: training_population(seed=i), total_steps=64, cfg=TINY)
    assert summary["updates"] >= 1
    team.save(tmp_path / "t.pt")
    again = Team.load(tmp_path / "t.pt")
    x = torch.randn(4, 3, MONSTER_OBS_DIM)
    assert torch.allclose(team.actor(x), again.actor(x))
    assert team.value(x).shape == ((4,) if mode == "mappo" else (4, 3))


def test_numpy_inference_matches_torch():
    team = Team("mappo", 3)
    x = np.random.default_rng(0).standard_normal((3, MONSTER_OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        ref = team.actor(torch.from_numpy(x)).numpy()
    assert np.allclose(NumpyMLP(team.actor)(x), ref, atol=1e-5)


def test_learned_monsters_play_a_level():
    rec = play_level(LearnedMonsters(Team("ippo", 3), seed=0), all_bots(seed=0)["kiter"])
    assert rec["outcome"] in ("caught", "escaped", "survived") and rec["steps"]


def test_player_view_same_from_game_and_recording():
    """The ghost must see identical features live and from a saved recording."""
    bot = all_bots(seed=0)["runner_middle"]
    rec = play_level(GreedyChasers(seed=0), bot)
    game = Game(shared_maze("arena"))
    cfg, prev = game.config, 0
    bot = all_bots(seed=0)["runner_middle"]
    bot.reset()
    mon = GreedyChasers(seed=0)
    for step in rec["steps"][:20]:
        live = player_obs(game.maze, PlayerView.from_game(game, prev), cfg.max_ammo, cfg.max_ticks)
        saved = player_obs(game.maze, PlayerView.from_step(step, prev), cfg.max_ammo, cfg.max_ticks)
        assert np.allclose(live, saved)
        game.step(step["player_action"], step["monster_actions"])
        prev = step["player_action"]
        if game.done:
            break


def test_ghost_clones_a_bot():
    bot = all_bots(seed=0)["runner_bottom_left"]
    recs = [play_level(GreedyChasers(seed=i), bot, level=i) for i in range(2)]
    x, y = dataset(recs, shared_maze("arena"), GameConfig())
    assert x.shape == (len(y), PLAYER_OBS_DIM)
    net, acc = train_ghost(x, y, grad_steps=300)
    assert acc > 0.8
    ghost = GhostPlayer(net, seed=0)
    assert play_level(GreedyChasers(seed=0), ghost)["steps"]


def test_adapt_returns_a_new_team():
    team = Team("mappo", 3)
    recs = [play_level(LearnedMonsters(team, seed=0), all_bots(seed=0)["camper_top_middle"])]
    new, info = adapt(team, recs, budget_s=2.0)
    assert new is not team and info["ghost_examples"] == len(recs[0]["steps"])
    changed = any(not torch.equal(a, b) for a, b in zip(team.parameters(), new.parameters()))
    assert changed


def test_adapt_with_rehearsal_runs():
    team = Team("mappo", 3)
    recs = [play_level(LearnedMonsters(team, seed=0), all_bots(seed=0)["runner_bottom_left"])]
    new, info = adapt(team, recs, budget_s=2.0, mix=0.5)
    assert info["finetune_steps"] > 0 and new is not team
