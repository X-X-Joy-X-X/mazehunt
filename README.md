# MazeHunt: Monsters That Learn How You Play

**Rapid between-level adaptation of cooperative multi-agent pursuers to individual players.**

A 2D top-down maze shooter used as a testbed for multi-agent reinforcement learning (MARL). A team of RL-controlled monsters hunts the player. After every level, the monsters retrain against a learned copy of how *you* play, so on the next level they cut off your favorite escape routes, flank you, and anticipate your habits.

Underneath the game is **pursuit-evasion**, a classic multi-robot problem (security robots, drone interception, search-and-rescue). The game is the testbed; the learning method is the contribution.

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt   # Windows; on Linux/macOS use .venv/bin/python
.venv\Scripts\python play.py                              # play
.venv\Scripts\python -m pytest -s                         # run the rule tests
```

For GPU training, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org).

**Controls:** `WASD` move · `Arrow keys` shoot · `P` pause · `Esc` quit
**Options:** `play.py --monsters random` (default `greedy`), `--seed N`, `--no-record`

Each level you play is saved locally to `results/recordings/<session>/level_XX.json` (git-ignored). These recordings are the training data for the ghost player.

---

## How it works

RL normally needs millions of practice rounds, and a human player provides a few minutes of data. The monsters therefore **can't learn from scratch in real time**, so learning is split into two phases:

```
 ┌──────────────── OFFLINE ─────────────────────────────────────────┐
 │ 1. PRETRAIN: monsters learn general hunting with MARL (MAPPO)    │
 │    against a population of scripted player bots with different  │
 │    styles: exit rusher, camper, wall-hugger, aggressive shooter  │
 └──────────────────────────────────────────────────────────────────┘
                                │
 ┌──────────────── ONLINE (while a person plays) ───────────────────┐
 │ 2. PLAY level k      → record the player's positions + actions    │
 │ 3. CLONE the player  → train a small "ghost" network that imitates│
 │                        the player (behavior cloning, seconds)     │
 │ 4. ADAPT monsters    → fine-tune monsters against the ghost in a  │
 │                        fast headless sim during a short break     │
 │ 5. PLAY level k+1    → monsters now counter this player's habits  │
 └──────────────────────────────────────────────────────────────────┘
```

- **Same map every level**, so any improvement comes from learning the *player*, not the map.
- **Ghost player:** monsters practice thousands of rounds against a copy of the player in seconds, instead of waiting for the player to play thousands of rounds.

| Game concept | Research concept |
|---|---|
| Monsters cooperating to trap the player | Multi-agent coordination (MARL) |
| Monsters splitting up to cover exits | Macro-scale coordination (team strategy, role allocation) |
| Each monster chasing and dodging bullets | Micro-scale control (individual behavior) |
| Learning *this* player's habits | Opponent modeling / online adaptation |
| Adapting during a short break | Learning under data and compute constraints |

---

## Research questions

| # | Question | Measured by |
|---|---|---|
| RQ1 | Does between-level adaptation beat a fixed pretrained team? | Capture rate and time-to-capture over levels 1→5, adaptive vs non-adaptive, against **held-out** player bots |
| RQ2 | How much player data is enough? | Adaptation gain from 1 vs 2 vs 3 levels of recorded play |
| RQ3 | Do the monsters actually coordinate? | MAPPO (shared training critic) vs IPPO (independent learners); encirclement and exit-coverage metrics |
| RQ4 | How much does the compute budget matter? | Adaptation gain vs fine-tuning time (10 / 30 / 60 / 120 s) on a laptop GPU |

All results use ≥3 random seeds (mean ± std). Evaluation uses scripted held-out player bots so it's reproducible.

---

## Game design

| Element | Design |
|---|---|
| Map | One fixed 23×15 tile maze ([`arena.txt`](mazehunt/maps/arena.txt)) |
| Movement | Tile-based, **discrete actions** (stay, up, down, left, right); smooth animation in rendering only |
| Player | Moves 1 tile/tick; shoots straight-line bullets (2 tiles/tick), 3 ammo with regeneration and a cooldown |
| Monsters | 3 monsters at 75% of player speed; catch the player on contact; die in one hit and respawn after 5 s |
| Level win | Reach the exit, or survive 60 s |
| Level loss | A monster catches the player |

Every balance number lives in [`mazehunt/config.py`](mazehunt/config.py).

**Architecture:** game **rules** (`mazehunt/core`) are headless and deterministic, and fully separate from **rendering** (`mazehunt/render`). Training runs the rules at about 70k steps/s on one CPU core with no graphics; the playable game wraps the same rules with a pygame renderer.

**One tick** ([`game.py`](mazehunt/core/game.py)): respawns → player moves or fires → bullets fly → monsters move → bullet hits → capture check (same tile *or* swapped tiles) → exit/timeout check.

---

## Status

- [x] **Game core:** headless rules, arena map, shooting/ammo/respawn, scripted monster baselines (greedy chasers, random), pygame renderer, human play with level/lives loop, gameplay recording, 19 rule tests.
- [ ] Scripted player bots and the multi-agent RL environment
- [ ] MAPPO / IPPO pretraining
- [ ] Ghost player + between-level adaptation
- [ ] Experiments (RQ1–RQ4) and technical report

**First baseline finding:** a simple scripted bot ("head for the exit; shoot or flee when a monster is close") escapes **300/300** games against the greedy-chaser monsters. Scripted pursuers never guard the exit or avoid firing lines. That's the gap the learned team has to close.

---

## Repo structure

```
├── play.py              ← run the game
├── mazehunt/
│   ├── config.py        ← all balance numbers
│   ├── maps/            ← tile maps (# wall, . floor, P player, M monster, E exit)
│   ├── core/            ← headless rules: maze, entities, actions, game
│   ├── agents/          ← monster controllers (scripted baselines; learned teams next)
│   ├── record.py        ← gameplay recording
│   ├── render/          ← pygame renderer + playable game loop
│   ├── env/             ← multi-agent RL environment (in progress)
│   ├── bots/            ← scripted player bots (in progress)
│   └── adapt/           ← between-level adaptation (planned)
├── tests/               ← rule tests (pytest)
├── train/               ← training + experiment scripts
├── results/             ← figures and logs
└── report/              ← technical report
```

---

## Credits

Graphics are drawn with pygame shapes; see [CREDITS.md](CREDITS.md). MAPPO: Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games* (NeurIPS 2022).

Author: Joy Barno
