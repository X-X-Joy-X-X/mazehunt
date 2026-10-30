"""Figures + a summary table from the saved results.

    python train/make_figures.py   → results/figures/*.png, results/summary.md
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"
FIGS = RESULTS / "figures"
COLORS = {"mappo": "#2a6fdb", "ippo": "#e07b39", "greedy": "#7a7a7a"}


def ms(xs) -> str:
    return f"{np.mean(xs):.2f} ± {np.std(xs):.2f}"


def pretraining_curves(lines: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for mode in ("mappo", "ippo"):
        runs = []
        for path in sorted((RESULTS / "logs").glob(f"pretrain_{mode}_s*.jsonl")):
            rows = [json.loads(line) for line in path.read_text().splitlines() if line]
            runs.append(([r["steps"] for r in rows], [r["capture_rate"] for r in rows]))
        if not runs:
            continue
        n = min(len(s) for s, _ in runs)
        steps = np.array(runs[0][0][:n]) / 1e6
        cap = np.array([c[:n] for _, c in runs])
        ax.plot(steps, cap.mean(0), color=COLORS[mode], label=f"{mode.upper()} ({len(runs)} seeds)")
        ax.fill_between(steps, cap.mean(0) - cap.std(0), cap.mean(0) + cap.std(0), color=COLORS[mode], alpha=0.2)
        lines.append(f"| {mode.upper()} | {ms(cap[:, -1])} |")
    ax.set_xlabel("environment steps (millions)")
    ax.set_ylabel("capture rate (last 200 games)")
    ax.set_title("Pretraining against the scripted bot population")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "pretraining.png", dpi=150)


def adaptation(lines: list[str]) -> None:
    files = sorted((RESULTS / "adaptation").glob("*.json"))
    if not files:
        return
    runs = [json.loads(f.read_text()) for f in files]
    bots = sorted({r["bot"] for r in runs})
    fig, axes = plt.subplots(1, len(bots), figsize=(4.2 * len(bots), 3.8), sharey=True, squeeze=False)
    lines += ["", "## RQ1/RQ2: adaptation vs held-out bots (capture rate, mean ± std over seeds)", "",
              "| bot | variant | pretrained | after L1 | after L4 | more pretraining (control) | oracle | one-shot 1 lvl | one-shot 3 lvl | Δ L4 − pretrained (paired) | Δ L4 − control (paired) |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
    variants = sorted({r.get("mix", 0.0) for r in runs})
    for ax, bot in zip(axes[0], bots):
        for mode in ("mappo", "ippo"):
            for mix in variants:
                sel = [r for r in runs if r["bot"] == bot and r["mode"] == mode and r.get("mix", 0.0) == mix]
                if not sel:
                    continue
                name = f"{mode.upper()} ghost only" if not mix else f"{mode.upper()} ghost + {mix:.0%} rehearsal"
                curve = np.array([[r["pretrained"]["capture_rate"]] + [lv["capture_rate"] for lv in r["levels"]]
                                  for r in sel])
                x = np.arange(curve.shape[1])
                ax.errorbar(x + 0.05 * variants.index(mix), curve.mean(0), yerr=curve.std(0), color=COLORS[mode],
                            marker="o" if not mix else "s", ls="-" if not mix else "--", capsize=3, label=name)
                ctrl = [r["more_pretraining"]["capture_rate"] for r in sel]
                orc = [r["oracle"]["capture_rate"] for r in sel]
                if variants.index(mix) == 0:
                    ax.axhline(np.mean(ctrl), color="#555555", ls=":", lw=1, label="+training control")
                    ax.axhline(np.mean(orc), color="#2e9e5b", ls=":", lw=1, label="oracle (real bot)")
                lines.append(f"| {bot} | {name} | {ms(curve[:, 0])} | {ms(curve[:, 1])} | {ms(curve[:, -1])} | "
                             f"{ms(ctrl)} | {ms(orc)} | {ms([r['one_shot_1_levels']['capture_rate'] for r in sel])} | "
                             f"{ms([r['one_shot_3_levels']['capture_rate'] for r in sel])} | "
                             f"{ms(curve[:, -1] - curve[:, 0])} | {ms(curve[:, -1] - np.array(ctrl))} |")
        ax.set_title(bot.replace("_", " "))
        ax.set_xlabel("adaptations (levels played)")
        ax.set_xticks(range(0, 5))
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("capture rate vs the real player")
    axes[0][-1].legend(fontsize=7, loc="lower right")
    fig.suptitle("Between-level adaptation against held-out players (≥3 seeds)")
    fig.tight_layout()
    fig.savefig(FIGS / "adaptation.png", dpi=150)


def budget(lines: list[str]) -> None:
    files = sorted((RESULTS / "budget").glob("*.json"))
    if not files:
        return
    runs = [json.loads(f.read_text()) for f in files]
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    lines += ["", "## RQ4: fine-tuning budget (capture rate, mean ± std over seeds)", "",
              "| bot | " + " | ".join(f"{r['budget_s']:.0f} s" for r in runs[0]["rows"]) + " |",
              "|---|" + "---|" * len(runs[0]["rows"])]
    for bot in sorted({r["bot"] for r in runs}):
        sel = [r for r in runs if r["bot"] == bot]
        x = [row["budget_s"] for row in sel[0]["rows"]]
        y = np.array([[row["capture_rate"] for row in r["rows"]] for r in sel])
        ax.errorbar(x, y.mean(0), yerr=y.std(0), marker="o", capsize=3, label=bot.replace("_", " "))
        lines.append(f"| {bot} | " + " | ".join(ms(y[:, i]) for i in range(y.shape[1])) + " |")
    ax.set_xlabel("fine-tuning seconds (0 = pretrained)")
    ax.set_ylabel("capture rate vs the real player")
    ax.set_title("Adaptation gain vs compute budget")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "budget.png", dpi=150)


def teams(lines: list[str]) -> None:
    path = RESULTS / "eval_teams.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())
    lines += ["", "## RQ3: team evaluation, 200 games per bot (mean ± std over seeds)", "",
              "| team | bots | capture | escape | exit cover | spread |", "|---|---|---|---|---|---|"]
    for kind in ("greedy", "mappo", "ippo"):
        for held in (False, True):
            sel = [r for r in rows if r["team"].split("_")[0] == kind and r["held_out"] == held]
            if not sel:
                continue
            seeds: dict[str, list] = {}
            for r in sel:
                seeds.setdefault(r["team"], []).append(r)
            cap = [np.mean([r["capture_rate"] for r in v]) for v in seeds.values()]
            esc = [np.mean([r["escape_rate"] for r in v]) for v in seeds.values()]
            lines.append(f"| {kind.upper()} | {'held-out' if held else 'training'} | {ms(cap)} | {ms(esc)} | "
                         f"{np.mean([r['exit_cover'] for r in sel]):.2f} | {np.mean([r['spread'] for r in sel]):.1f} |")


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    lines = ["# MazeHunt results summary", "", "## Pretraining: final capture rate vs the training bots", "",
             "| team | capture rate |", "|---|---|"]
    pretraining_curves(lines)
    teams(lines)
    adaptation(lines)
    budget(lines)
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
