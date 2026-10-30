"""MAPPO and IPPO for the monster team, in one readable file (CleanRL style).

Both variants share one actor network across the monsters (each monster's observation
includes its index). The only difference is the critic used during training:

  * MAPPO: one centralized critic reads every monster's view and values the *team* state.
  * IPPO:  each monster's critic reads only its own view.

At play time both are decentralized: each monster acts from its own view alone.

References: Yu et al. 2022 (MAPPO); de Witt et al. 2020 (IPPO); CleanRL's ppo.py.
"""

from __future__ import annotations

import time
from collections import Counter, deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from mazehunt.core.actions import NUM_MONSTER_ACTIONS
from mazehunt.core.game import Game
from mazehunt.env.features import MONSTER_OBS_DIM, monster_obs
from mazehunt.env.hunt_env import HuntEnv

HIDDEN = 128


def layer(n_in: int, n_out: int, std: float = np.sqrt(2)) -> nn.Linear:
    lin = nn.Linear(n_in, n_out)
    nn.init.orthogonal_(lin.weight, std)
    nn.init.constant_(lin.bias, 0.0)
    return lin


def mlp(n_in: int, n_out: int, out_std: float) -> nn.Sequential:
    return nn.Sequential(layer(n_in, HIDDEN), nn.Tanh(), layer(HIDDEN, HIDDEN), nn.Tanh(),
                         layer(HIDDEN, n_out, out_std))


class Team(nn.Module):
    def __init__(self, mode: str, n_agents: int, obs_dim: int = MONSTER_OBS_DIM):
        super().__init__()
        if mode not in ("mappo", "ippo"):
            raise ValueError(mode)
        self.mode, self.n_agents, self.obs_dim = mode, n_agents, obs_dim
        self.actor = mlp(obs_dim, NUM_MONSTER_ACTIONS, out_std=0.01)
        critic_in = obs_dim * n_agents if mode == "mappo" else obs_dim
        self.critic = mlp(critic_in, 1, out_std=1.0)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        """obs (B, N, D) → MAPPO (B,) one team value; IPPO (B, N) one value per monster."""
        if self.mode == "mappo":
            return self.critic(obs.reshape(obs.shape[0], -1)).squeeze(-1)
        return self.critic(obs).squeeze(-1)

    def dist(self, obs: torch.Tensor) -> torch.distributions.Categorical:
        return torch.distributions.Categorical(logits=self.actor(obs))

    def save(self, path: Path, **meta) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"mode": self.mode, "n_agents": self.n_agents, "obs_dim": self.obs_dim,
                    "state_dict": self.state_dict(), **meta}, path)

    @classmethod
    def load(cls, path: Path) -> Team:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        team = cls(ckpt["mode"], ckpt["n_agents"], ckpt["obs_dim"])
        team.load_state_dict(ckpt["state_dict"])
        return team


class NumpyMLP:
    """The same Tanh MLP evaluated with numpy. One-sample inference in a game loop is ~10x
    faster this way than through torch."""

    def __init__(self, seq: nn.Sequential):
        self.layers = [(m.weight.detach().numpy().T.copy(), m.bias.detach().numpy().copy())
                       for m in seq if isinstance(m, nn.Linear)]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                x = np.tanh(x)
        return x


def sample(logits: np.ndarray, rng: np.random.Generator, greedy: bool = False) -> np.ndarray:
    if greedy:
        return logits.argmax(-1)
    z = logits - logits.max(-1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(-1, keepdims=True)
    u = rng.random((*p.shape[:-1], 1))
    return (p.cumsum(-1) < u).sum(-1).clip(0, p.shape[-1] - 1)


class LearnedMonsters:
    """Game-loop controller for a trained team: same `act(game)` interface as the scripted ones."""

    def __init__(self, team: Team, name: str | None = None, greedy: bool = False, seed: int | None = None):
        self.policy = NumpyMLP(team.actor)
        self.name = name or team.mode
        self.greedy = greedy
        self.rng = np.random.default_rng(seed)

    def act(self, game: Game) -> list[int]:
        p = game.player
        move = (p.pos[0] - p.prev_pos[0], p.pos[1] - p.prev_pos[1])
        return sample(self.policy(monster_obs(game, move)), self.rng, self.greedy).tolist()


@dataclass
class PPOConfig:
    num_envs: int = 32
    num_steps: int = 128
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.995
    gae_lambda: float = 0.95
    update_epochs: int = 4
    num_minibatches: int = 4
    clip: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


def train(team: Team, make_opponents: Callable[[int], Sequence], total_steps: int,
          cfg: PPOConfig | None = None, seed: int = 0, time_budget: float | None = None,
          log: Callable[[dict], None] | None = None, log_every: int = 10) -> dict:
    """Train `team` in place with PPO. Stops at `total_steps` env steps or `time_budget`
    seconds, whichever comes first. `make_opponents(env_index)` builds each env's player pool."""
    cfg = cfg or PPOConfig()
    torch.manual_seed(seed)
    np_rng = np.random.default_rng(seed)
    envs = [HuntEnv(make_opponents(i), seed=seed * 1000 + i) for i in range(cfg.num_envs)]
    E, T, N, D = cfg.num_envs, cfg.num_steps, team.n_agents, team.obs_dim
    opt = torch.optim.Adam(team.parameters(), lr=cfg.lr, eps=1e-5)
    mappo = team.mode == "mappo"
    vshape = (T, E) if mappo else (T, E, N)

    obs_buf = np.zeros((T, E, N, D), np.float32)
    act_buf = np.zeros((T, E, N), np.int64)
    logp_buf = np.zeros((T, E, N), np.float32)
    alive_buf = np.zeros((T, E, N), np.float32)
    rew_buf = np.zeros((T, E), np.float32)
    done_buf = np.zeros((T, E), np.float32)
    val_buf = np.zeros(vshape, np.float32)

    obs = np.stack([e.reset() for e in envs])
    next_done = np.zeros(E, np.float32)
    recent = deque(maxlen=200)
    by_opponent: dict[str, deque] = {}
    steps, updates, start = 0, 0, time.perf_counter()
    num_updates = max(1, total_steps // (E * T))

    while steps < total_steps and (time_budget is None or time.perf_counter() - start < time_budget):
        if cfg.anneal_lr and time_budget is None:
            opt.param_groups[0]["lr"] = cfg.lr * max(0.05, 1.0 - updates / num_updates)
        for t in range(T):
            obs_buf[t] = obs
            done_buf[t] = next_done
            alive_buf[t] = obs[:, :, 3]
            with torch.no_grad():
                o = torch.from_numpy(obs)
                d = team.dist(o)
                a = d.sample()
                logp_buf[t] = d.log_prob(a).numpy()
                val_buf[t] = team.value(o).numpy()
            act_buf[t] = a.numpy()
            for i, env in enumerate(envs):
                o_i, r_i, done_i, info = env.step(act_buf[t, i])
                rew_buf[t, i] = r_i
                next_done[i] = float(done_i)
                if done_i:
                    recent.append(info)
                    by_opponent.setdefault(info["opponent"], deque(maxlen=100)).append(info["outcome"])
                    o_i = env.reset()
                obs[i] = o_i
            steps += E

        # --- generalized advantage estimation ---
        with torch.no_grad():
            next_val = team.value(torch.from_numpy(obs)).numpy()
        adv = np.zeros(vshape, np.float32)
        last = 0.0
        for t in reversed(range(T)):
            nv = next_val if t == T - 1 else val_buf[t + 1]
            nonterm = 1.0 - (next_done if t == T - 1 else done_buf[t + 1])
            r = rew_buf[t] if mappo else rew_buf[t][:, None]
            if not mappo:
                nonterm = nonterm[:, None]
            delta = r + cfg.gamma * nv * nonterm - val_buf[t]
            last = delta + cfg.gamma * cfg.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + val_buf

        # --- PPO update over (time, env) samples; each sample carries all N monsters ---
        b_obs = torch.from_numpy(obs_buf.reshape(T * E, N, D))
        b_act = torch.from_numpy(act_buf.reshape(T * E, N))
        b_logp = torch.from_numpy(logp_buf.reshape(T * E, N))
        b_alive = torch.from_numpy(alive_buf.reshape(T * E, N))
        b_adv = torch.from_numpy(adv.reshape(T * E, *vshape[2:]))
        b_ret = torch.from_numpy(ret.reshape(T * E, *vshape[2:]))
        mb = (T * E) // cfg.num_minibatches
        for _ in range(cfg.update_epochs):
            perm = torch.from_numpy(np_rng.permutation(T * E))
            for s in range(0, T * E, mb):
                idx = perm[s:s + mb]
                d = team.dist(b_obs[idx])
                logp = d.log_prob(b_act[idx])
                ratio = (logp - b_logp[idx]).exp()
                a_mb = b_adv[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                if mappo:
                    a_mb = a_mb[:, None].expand_as(ratio)
                mask = b_alive[idx]
                pg = torch.max(-a_mb * ratio, -a_mb * ratio.clamp(1 - cfg.clip, 1 + cfg.clip))
                denom = mask.sum().clamp(min=1.0)
                pg_loss = (pg * mask).sum() / denom
                ent = (d.entropy() * mask).sum() / denom
                v_loss = 0.5 * ((team.value(b_obs[idx]) - b_ret[idx]) ** 2).mean()
                loss = pg_loss - cfg.ent_coef * ent + cfg.vf_coef * v_loss
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(team.parameters(), cfg.max_grad_norm)
                opt.step()
        updates += 1

        if log and (updates % log_every == 0) and recent:
            outcomes = Counter(i["outcome"] for i in recent)
            log({"steps": steps, "seconds": round(time.perf_counter() - start, 1),
                 "capture_rate": round(outcomes["caught"] / len(recent), 3),
                 "escape_rate": round(outcomes["escaped"] / len(recent), 3),
                 "mean_ticks": round(float(np.mean([i["ticks"] for i in recent])), 1),
                 "per_opponent": {k: round(sum(o == "caught" for o in v) / len(v), 2)
                                  for k, v in sorted(by_opponent.items())}})

    return {"steps": steps, "updates": updates, "seconds": time.perf_counter() - start,
            "config": asdict(cfg)}
