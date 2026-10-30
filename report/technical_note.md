# Between-level adaptation of a cooperative pursuer team to individual players

*Technical note, MazeHunt project. Joy Barno, October 2026.*

## 1. Problem

A team of three pursuers ("monsters") hunts one evader (the player) in a fixed 23×15 grid maze. The evader wins by reaching the exit or surviving 480 ticks (60 s); the pursuers win by touching it. The evader shoots straight-line bullets with limited ammo; a pursuer dies in one hit and respawns after 40 ticks. Pursuers move at 75% of the evader's speed, so neither pure chasing nor pure guarding wins on its own: the team has to split roles.

The question is whether a pretrained team can **adapt to one specific player within a few levels**, using only that player's recorded play and a fixed, short compute budget between levels, on a laptop CPU.

## 2. Method

**Pretraining (MARL).** The team is trained with PPO in a parallel multi-agent environment with a shared team reward: +10 for a capture, −10 for an escape or timeout, plus difference-based shaping on (i) the closest pursuer's maze distance to the evader and (ii) the evader's maze distance to the exit. Shaping terms telescope over a level, so they guide early learning without being farmable. Each pursuer sees a 40-dimensional egocentric vector built from breadth-first-search distances (which neighbour tile is closer to the evader or the exit), teammates' relative positions, firing-line danger, ammo, and time left. One actor network is shared by all pursuers (with an index one-hot).

* **MAPPO** (Yu et al., 2022): a centralized critic reads all three pursuers' observations and estimates one team value.
* **IPPO** (de Witt et al., 2020): each pursuer's critic reads only its own observation.

Both act from local observations only at play time. Hyperparameters: 32 envs × 128 steps per update, 4 epochs, 4 minibatches, lr 3·10⁻⁴ annealed, γ = 0.995, λ = 0.95, clip 0.2, entropy 0.01, 2×128 tanh MLPs, 6 M environment steps.

**Opponents.** Eight scripted evader styles form the training population: three exit runners with different favourite routes, two corner campers, a kiter, a monster hunter and a random walker. To make the bots human-like they notice a lined-up pursuer with probability *r* per tick; pretraining draws *r* ~ U(0, 0.5) every level (a skill curriculum), evaluation fixes *r* = 0.35. Three **held-out** styles are never used in pretraining: a runner via the bottom-left corridor, a "waiter" that hides until no pursuer is near the exit and then dashes, and a top-middle camper.

**Adaptation.** After each level: (1) the level is recorded in the same format as human play; (2) a ghost policy is trained by behavior cloning on all levels recorded so far (46-dimensional player view → 9 actions, 600 Adam steps, about a second); (3) a copy of the team is fine-tuned with PPO against the ghost for a fixed wall-clock budget (default 45 s, including cloning; lr 10⁻⁴ constant, 16 envs). The team never trains against the real held-out bot.

**Controls.** Same budget, starting from the pretrained team: *more pretraining* (fine-tune against the training population, to separate player-specific learning from extra training) and *oracle* (fine-tune against the real held-out bot, the best case for this budget).

All evaluations are 200 games against bots at *r* = 0.35, reported as mean ± std over 3 seeds. Paired differences use the same seed's pretrained team.

## 3. Results

**Pretraining and coordination (RQ3).** After 6 M steps (43–50 min per run on one CPU core), MAPPO catches 0.55 ± 0.08 of training-style players and 0.41 ± 0.16 of held-out players, IPPO 0.48 ± 0.03 and 0.35 ± 0.03, and scripted greedy chasers 0.47 and 0.37. The MAPPO advantage comes mainly from seed 1, which learned a role split: one pursuer holds the single corridor into the exit (a pursuer is within 4 tiles of the exit on 43% of ticks), the others hunt, and mean pairwise maze distance rises from about 6 to 12 tiles. That team catches 0.65 of all players, including 0.735 of the held-out bottom-left runner versus 0.22 for greedy chasers. No IPPO seed and no other MAPPO seed found this split (exit-cover 0%). This is consistent with a centralized critic helping credit assignment for a "sacrificial" guarding role, but with three seeds it is suggestive, not conclusive.

**Adaptation (RQ1).** Four rounds of record → clone → 45 s fine-tune:

| Held-out player | Pretrained | After 4 | Paired gain | Control (+training) | Oracle |
|---|---|---|---|---|---|
| Camper | 0.59 ± 0.03 | 0.66 ± 0.05 | +0.07 ± 0.02 | 0.60 ± 0.07 | 0.64 ± 0.06 |
| Runner | 0.36 ± 0.26 | 0.42 ± 0.29 | +0.05 ± 0.09 | 0.40 ± 0.23 | 0.49 ± 0.22 |
| Waiter | 0.35 ± 0.26 | 0.30 ± 0.30 | −0.05 ± 0.06 | 0.38 ± 0.30 | 0.37 ± 0.28 |

Adaptation reliably helps against the camper: every seed improves, and the gain over the control (+0.06 ± 0.04) shows it is player-specific, not just extra training. Against the two mobile players it does not beat the control, and the per-level curve oscillates (e.g. seed 1 vs the runner: 0.735 → 0.235 → 0.785 → 0.20 → 0.805). The oracle *does* improve against the runner on the two weaker seeds (+0.165, +0.15), which isolates the failure to the player model, not the fine-tuning.

**Rehearsal.** Keeping half of the fine-tuning environments on the pretraining population reduced variance (waiter paired gain −0.03 ± 0.01) but produced no gain against mobile players.

**Data amount (RQ2).** One-shot adaptation from 3 recorded levels was at least as good as from 1 (waiter 0.12 → 0.34; runner 0.29 → 0.32; camper 0.60 → 0.61). More data mainly limits damage.

**Budget (RQ4).** From one recorded runner level, capture falls from 0.36 (pretrained) to 0.34, 0.26, 0.18 and 0.19 after 10, 30, 60 and 120 s (45 k – 700 k fine-tuning steps). With a weak ghost, more optimization overfits the team to the ghost's errors.

**Human data.** A ghost trained on 7 of the author's 10 recorded sessions (284 ticks) predicts held-out sessions' actions with 59% accuracy (majority-class baseline 46%).

## 4. Discussion and limitations

The method works when the player's behaviour is cheap to clone: a camper produces hundreds of nearly identical ticks per level. A moving player produces 50–370 ticks per level, mostly in states the clone never revisits, and behavior cloning's compounding errors (Ross et al., 2011) yield a ghost that escapes far less often than the real player. Fine-tuning against it teaches the team to beat a weaker opponent. Next steps: (1) aggregate data with the ghost in the loop and stop fine-tuning early on held-out recorded ticks; (2) an ensemble of ghosts, or the ghost mixed with the nearest bot styles, to avoid overfitting one bad model; (3) step-count budgets. Wall-clock budgets varied up to 10× with machine load in these runs (fine-tuning steps are logged per run). Other limitations: 3 seeds; one map; scripted, not human, held-out players; bots' "reaction" is a crude human model.

## 5. Reproducibility

`bash train/run_all.sh` reruns everything. Models, logs and per-run JSON are in `results/`. Hardware: a laptop with 12 CPU threads and 8 GB RAM (no GPU used). Run at most 2–3 PyTorch processes at once on 8 GB.

## References

Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022. · de Witt et al., *Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?*, 2020. · Schulman et al., *Proximal Policy Optimization Algorithms*, 2017. · Ng, Harada & Russell, *Policy Invariance under Reward Transformations*, ICML 1999. · Ross, Gordon & Bagnell, *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning* (DAgger), AISTATS 2011. · Chung, Hollinger & Isler, *Search and pursuit-evasion in mobile robotics*, Autonomous Robots 2011.
