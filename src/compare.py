"""Slide 4 figures for embed and pool, in the style of project 1's evaluation slide:

  1. grouped box plot of validation accuracy (15 runs per setting): 3 MLPs x 3 GloVe sizes
  2. Bayesian correlated t-test (Benavoli et al., 2017) for a few pairs, with a ROPE of 0.5 points

Usage (from the project root, after `python src/embed_pool.py combos`):
    python src/compare.py   # writes results/q3_box.png, results/q3_bayes.png and prints the numbers
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INK, MUTED, FAINT, GRID = "#15181d", "#616872", "#8a8f98", "#eceef1"
PINK, PURPLE, GOLD, GREY = "#c8467a", "#5b52c4", "#c9962a", "#6b7280"
FONT = "Helvetica Neue"
ROPE = 0.5          # percentage points
K = 5               # folds per seed: test share rho = 1 / K

combos = json.loads((ROOT / "results" / "q3_combos.json").read_text())
p1 = json.loads((ROOT / "results" / "p1_baseline.json").read_text())

MLPS = ["MLP 1×16", "MLP 2×64", "MLP 3×128"]
DIMS = [("100 dims", PURPLE), ("200 dims", PINK), ("300 dims", GOLD)]
MLP_LABEL = {"MLP 1×16": "MLP\n1 × 16", "MLP 2×64": "MLP\n2 × 64", "MLP 3×128": "MLP\n3 × 128"}


def runs(mlp, dims):
    """Accuracies (%) keyed by (seed, fold), so pairs are compared fold by fold; mlp=None is project 1."""
    src = p1["runs"] if mlp is None else combos[mlp][dims]["runs"]
    return {(r["seed"], r["fold"]): 100 * r["val_acc"] for r in src}


PAIRS = [  # (first label, mlp, dims), (second label, mlp, dims), sub-label
    (("100 dims", "MLP 1×16", "100 dims"), ("300 dims", "MLP 1×16", "300 dims"), "both with MLP 1×16"),
    (("200 dims", "MLP 1×16", "200 dims"), ("300 dims", "MLP 1×16", "300 dims"), "both with MLP 1×16"),
    (("MLP 1×16", "MLP 1×16", "300 dims"), ("MLP 3×128", "MLP 3×128", "300 dims"), "both with 300 dims"),
    (("MLP 3×128, 300 dims", "MLP 3×128", "300 dims"), ("bpe-logcounts-mlp", None, None), "best embed and pool vs project 1"),
]


def posterior(a, b, n_samples=400_000, seed=0):
    """Bayesian correlated t-test: posterior of the mean gain of b over a (percentage points)."""
    keys = sorted(a)
    d = np.array([b[k] - a[k] for k in keys])
    n, rho = len(d), 1 / K
    scale = np.sqrt((1 / n + rho / (1 - rho)) * d.var(ddof=1))
    mu = d.mean() + scale * np.random.default_rng(seed).standard_t(n - 1, n_samples)
    return mu, {"worse": float((mu < -ROPE).mean()), "same": float((abs(mu) <= ROPE).mean()),
                "better": float((mu > ROPE).mean()), "mean": float(d.mean())}


def setup_fonts():
    plt.rcParams.update({"font.family": [FONT, "DejaVu Sans"], "font.size": 15})


def box_figure(out, size=(8.6, 5.6)):
    setup_fonts()
    fig, ax = plt.subplots(figsize=size, dpi=200)
    rng = np.random.default_rng(0)
    w = 0.27
    for g, mlp in enumerate(MLPS):
        for j, (dims, c) in enumerate(DIMS):
            x = g + (j - 1) * (w + 0.03)
            acc = np.array(list(runs(mlp, dims).values()))
            ax.boxplot([acc], positions=[x], widths=w, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=c + "33", edgecolor="#b9bcc2", linewidth=1.2),
                       medianprops=dict(color=INK, linewidth=2.4), whiskerprops=dict(color=MUTED, linewidth=1.2),
                       capprops=dict(color=MUTED, linewidth=1.2))
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(acc)), acc, s=22, color=c, edgecolors="white",
                       linewidths=0.6, zorder=3)
            print(f"{mlp:10s} {dims:9s} {acc.mean():.2f} ± {acc.std():.2f}")
    p1_mean = float(np.mean(list(runs(None, None).values())))
    ax.axhline(p1_mean, color=GREY, linestyle=(0, (1.5, 2.5)), linewidth=1.8, zorder=1)
    ax.annotate(f"bpe-logcounts-mlp ({p1_mean:.1f}%)", (len(MLPS) - 0.5, p1_mean), xytext=(0, 6),
                textcoords="offset points", ha="right", va="bottom", fontsize=14, color=GREY)
    ax.set_xticks(range(len(MLPS)))
    ax.set_xticklabels([MLP_LABEL[m] for m in MLPS], fontsize=15, color=INK, linespacing=1.2)
    ax.set_xlim(-0.55, len(MLPS) - 0.45)
    ax.set_ylim(82.3, 89)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("validation accuracy", color=MUTED, fontsize=16, labelpad=8)
    ax.grid(axis="y", color=GRID, linewidth=1.1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c6cc")
    ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=15)
    ax.tick_params(axis="x", length=0, pad=8)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=c) for _, c in DIMS]
    leg = ax.legend(handles, [f"GloVe {d}" for d, _ in DIMS], loc="upper left", frameon=False, fontsize=15,
                    handletextpad=0.3, borderaxespad=0.2, bbox_to_anchor=(0, 0.8))
    for t, (_, c) in zip(leg.get_texts(), DIMS):
        t.set_color(c)
        t.set_fontweight("bold")
    fig.patch.set_alpha(0)
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    print("saved", out)


def bayes_figure(out, size=(5.0, 6.4)):
    setup_fonts()
    fig, ax = plt.subplots(figsize=size, dpi=200)
    lo, hi = -1.5, 4
    ax.axvspan(-ROPE, ROPE, color="#e9eaed", zorder=0)
    ax.axvline(0, color="#b9bcc2", linewidth=1.3, zorder=1)
    ax.text(0, len(PAIRS) + 0.02, f"ROPE ±{ROPE}%", ha="center", va="bottom", fontsize=15, color=MUTED)
    for row, ((na, ma, da), (nb, mb, db), sub) in enumerate(PAIRS):
        y = len(PAIRS) - 1 - row
        mu, p = posterior(runs(ma, da), runs(mb, db))
        verdict = max(("better", "same", "worse"), key=lambda k: p[k])
        c = {"better": PINK, "same": PURPLE, "worse": GOLD}[verdict]
        dens, edges = np.histogram(mu, bins=300, range=(lo, hi), density=True)
        xs = (edges[:-1] + edges[1:]) / 2
        dens = np.convolve(dens, np.ones(5) / 5, mode="same")
        keep = dens > dens.max() * 0.02
        xs, dens = xs[keep], dens[keep]
        h = 0.72 * dens / dens.max()
        ax.fill_between(xs, y, y + h, color=c, alpha=0.18, linewidth=0, zorder=2)
        ax.plot(xs, y + h, color=c, linewidth=2, zorder=3)
        ax.text(lo - 0.25, y + 0.42, f"{na}\n→ {nb}", ha="right", va="center", fontsize=15, color=INK,
                weight="bold", linespacing=1.25)
        ax.text(lo - 0.25, y + 0.02, sub, ha="right", va="center", fontsize=13.5, color=MUTED)
        ax.text(hi + 0.2, y + 0.2, f"P({verdict}) = {p[verdict]:.2f}", ha="left", va="center", fontsize=15,
                color=c, weight="bold")
        print(f"{na} -> {nb}: gain {p['mean']:+.2f}  worse {p['worse']:.2f} same {p['same']:.2f} better {p['better']:.2f}")
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.15, len(PAIRS) + 0.25)
    ax.set_yticks([])
    ax.set_xticks([-1, 0, 1, 2, 3, 4])
    ax.set_xticklabels(["−1", "0", "+1", "+2", "+3", "+4"])
    ax.set_xlabel("gain of 2nd system, %", color=MUTED, fontsize=15, labelpad=6)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c6cc")
    ax.tick_params(axis="x", colors=INK, labelsize=15)
    fig.patch.set_alpha(0)
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    print("saved", out)


REG_GROUPS = [  # group in q3_regularization.json, tick label, settings (name, short label), colour
    ("Dropout", "dropout", [("0", "0"), ("0.1 (base)", "0.1"), ("0.3", "0.3")], PURPLE),
    ("Weight decay", "weight decay", [("0", "0"), ("1e-4 (base)", "1e-4"), ("1e-2", "1e-2")], PINK),
    ("Batch size", "batch size", [("32", "32"), ("64 (base)", "64"), ("128", "128")], GOLD),
]


def reg_figure(out, size=(6.4, 6.0)):
    """Regularization and batch size, one change at a time from the best combination (300 dims, MLP 3x128)."""
    reg = json.loads((ROOT / "results" / "q3_regularization.json").read_text())
    setup_fonts()
    fig, ax = plt.subplots(figsize=size, dpi=200)
    rng = np.random.default_rng(0)
    x, ticks, labels = 0, [], []
    for g, (group, name, settings, c) in enumerate(REG_GROUPS):
        for setting, short in settings:
            acc = np.array([100 * r["val_acc"] for r in reg[group][setting]["runs"]])
            base = "base" in setting
            ax.boxplot([acc], positions=[x], widths=0.6, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=c + ("55" if base else "22"), edgecolor=INK if base else "#b9bcc2",
                                     linewidth=2.2 if base else 1.2),
                       medianprops=dict(color=INK, linewidth=2.4), whiskerprops=dict(color=MUTED, linewidth=1.2),
                       capprops=dict(color=MUTED, linewidth=1.2))
            ax.scatter(x + rng.uniform(-0.13, 0.13, len(acc)), acc, s=18, color=c, edgecolors="white",
                       linewidths=0.5, zorder=3)
            if base:  # the shared base model: best of the 3 x 3 grid
                ax.annotate("base", (x, acc.max()), xytext=(0, 8), textcoords="offset points", ha="center",
                            va="bottom", fontsize=13, color=INK, weight="bold")
            ticks.append(x)
            labels.append(short)
            print(f"{group:13s} {setting:12s} {acc.mean():.2f} ± {acc.std():.2f}")
            x += 1
        ax.annotate(name, ((x - 2) , 0), xycoords=("data", "axes fraction"), xytext=(0, -38),
                    textcoords="offset points", ha="center", va="top", fontsize=15, color=c, weight="bold")
        x += 0.6
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=14, color=INK)
    ax.set_xlim(-0.7, x - 0.9)
    ax.set_ylim(82.3, 89)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("validation accuracy", color=MUTED, fontsize=16, labelpad=8)
    ax.grid(axis="y", color=GRID, linewidth=1.1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c6cc")
    ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=15)
    ax.tick_params(axis="x", length=0, pad=6)
    p1_mean = float(np.mean(list(runs(None, None).values())))
    ax.axhline(p1_mean, color=GREY, linestyle=(0, (1.5, 2.5)), linewidth=1.8, zorder=1)
    ax.annotate("bpe-logcounts-mlp", (x - 0.9, p1_mean), xytext=(0, 6), textcoords="offset points",
                ha="right", va="bottom", fontsize=14, color=GREY)
    fig.patch.set_alpha(0)
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    print("saved", out)


def reg_grid_figure(out, size=(6.4, 6.0)):
    """2 dropout x 2 weight decay x 2 batch sizes (8 settings), MLP 3x128 with 300 dims."""
    reg = json.loads((ROOT / "results" / "q3_reg_grid.json").read_text())
    batches = [("batch 32", PINK), ("batch 64", GOLD)]
    setup_fonts()
    fig, ax = plt.subplots(figsize=size, dpi=200)
    rng = np.random.default_rng(0)
    w = 0.34
    groups = list(reg)
    for g, group in enumerate(groups):
        for j, (b, c) in enumerate(batches):
            x = g + (j - 0.5) * (w + 0.05)
            acc = np.array([100 * r["val_acc"] for r in reg[group][b]["runs"]])
            ax.boxplot([acc], positions=[x], widths=w, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=c + "33", edgecolor="#b9bcc2", linewidth=1.2),
                       medianprops=dict(color=INK, linewidth=2.4), whiskerprops=dict(color=MUTED, linewidth=1.2),
                       capprops=dict(color=MUTED, linewidth=1.2))
            ax.scatter(x + rng.uniform(-0.08, 0.08, len(acc)), acc, s=18, color=c, edgecolors="white",
                       linewidths=0.5, zorder=3)
            print(f"{group:28s} {b:9s} {acc.mean():.2f} ± {acc.std():.2f}")
    labels = []
    for group in groups:
        d, wd = group.replace("dropout ", "").split(" · decay ")
        labels.append(f"dropout {d}\ndecay {float(wd):.0e}".replace("e-0", "e-"))
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels(labels, fontsize=13.5, color=INK, linespacing=1.3)
    ax.set_xlim(-0.55, len(groups) - 0.45)
    ax.set_ylim(82.3, 89)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("validation accuracy", color=MUTED, fontsize=16, labelpad=8)
    ax.grid(axis="y", color=GRID, linewidth=1.1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c6cc")
    ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=15)
    ax.tick_params(axis="x", length=0, pad=8)
    p1_mean = float(np.mean(list(runs(None, None).values())))
    ax.axhline(p1_mean, color=GREY, linestyle=(0, (1.5, 2.5)), linewidth=1.8, zorder=1)
    ax.annotate("bpe-logcounts-mlp", (len(groups) - 0.45, p1_mean), xytext=(0, 6), textcoords="offset points",
                ha="right", va="bottom", fontsize=14, color=GREY)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=c) for _, c in batches]
    leg = ax.legend(handles, [b for b, _ in batches], loc="upper left", frameon=False, fontsize=15,
                    handletextpad=0.3, borderaxespad=0.2, bbox_to_anchor=(0, 0.8))
    for t, (_, c) in zip(leg.get_texts(), batches):
        t.set_color(c)
        t.set_fontweight("bold")
    fig.patch.set_alpha(0)
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    print("saved", out)


def simple_boxes(out, items, xlabel, title_color, mark=None, size=(4.6, 6.0), ylim=(82.3, 89)):
    """One box per setting (15 runs): items = [(tick label, accuracies %, colour)]; `mark` = index labelled 'best'."""
    setup_fonts()
    fig, ax = plt.subplots(figsize=size, dpi=200)
    rng = np.random.default_rng(0)
    for i, (label, acc, c) in enumerate(items):
        strong = i == mark
        ax.boxplot([acc], positions=[i], widths=0.55, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=c + ("55" if strong else "26"), edgecolor=INK if strong else "#b9bcc2",
                                 linewidth=2.2 if strong else 1.2),
                   medianprops=dict(color=INK, linewidth=2.4), whiskerprops=dict(color=MUTED, linewidth=1.2),
                   capprops=dict(color=MUTED, linewidth=1.2))
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(acc)), acc, s=20, color=c, edgecolors="white",
                   linewidths=0.5, zorder=3)
        ax.annotate(f"{acc.mean():.1f}%", (i, acc.max()), xytext=(0, 8), textcoords="offset points", ha="center",
                    va="bottom", fontsize=14, color=INK, weight="bold" if strong else "normal")
        print(f"{xlabel:14s} {label:10s} {acc.mean():.2f} ± {acc.std():.2f}")
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([it[0] for it in items], fontsize=15, color=INK)
    ax.set_xlim(-0.6, len(items) - 0.4)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("validation accuracy", color=MUTED, fontsize=16, labelpad=8)
    ax.set_xlabel(xlabel, color=title_color, fontsize=16, weight="bold", labelpad=10)
    ax.grid(axis="y", color=GRID, linewidth=1.1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c6cc")
    ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=15)
    ax.tick_params(axis="x", length=0, pad=8)
    p1_mean = float(np.mean(list(runs(None, None).values())))
    ax.axhline(p1_mean, color=GREY, linestyle=(0, (1.5, 2.5)), linewidth=1.8, zorder=1)
    ax.annotate("bpe-logcounts-mlp", (len(items) - 0.4, p1_mean), xytext=(0, 6), textcoords="offset points",
                ha="right", va="bottom", fontsize=14, color=GREY)
    fig.patch.set_alpha(0)
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    print("saved", out)


def acc_of(mlp, dims):
    return np.array([100 * r["val_acc"] for r in combos[mlp][dims]["runs"]])


def arch_figure(out):
    """Step 1: the three MLPs, all with GloVe 300 dims."""
    simple_boxes(out, [(m.replace("MLP ", ""), acc_of(m, "300 dims"), c) for m, c in zip(MLPS, (PURPLE, PINK, GOLD))],
                 "MLP layers × units", PURPLE, mark=2)


def dims_figure(out):
    """Step 3: the best MLP (3x128) with the best hyperparameters (dropout 0.1, decay 1e-4, batch 64) per GloVe size."""
    simple_boxes(out, [(d.replace(" dims", ""), acc_of("MLP 3×128", d), c) for d, c in DIMS],
                 "embedding dimensions", PINK, mark=2)


def lstm_units_figure(out):
    """Slide 5, step 1: LSTM hidden units (frozen GloVe 300d, linear head), 15 runs each."""
    d = json.loads((ROOT / "results" / "q4_lstm.json").read_text())["15 folds"]
    items = [(str(u), np.array([100 * r["val_acc"] for r in d[f"linear · {u} · frozen"]["runs"]]), c)
             for u, c in ((32, PURPLE), (64, PINK), (128, GOLD))]
    best = max(range(3), key=lambda i: items[i][1].mean())
    simple_boxes(out, items, "LSTM units", INK, mark=best, ylim=(86.5, 90.5))


def grouped_boxes(out, groups, size=(4.6, 6.0), ylim=(86.5, 90.5)):
    """Groups of boxes side by side: groups = [(name, colour, [(tick, accuracies %, is_base)])]."""
    setup_fonts()
    fig, ax = plt.subplots(figsize=size, dpi=200)
    rng = np.random.default_rng(0)
    x, ticks, labels, ends = 0.0, [], [], []
    for name, c, items in groups:
        start = x
        for tick, acc, base in items:
            ax.boxplot([acc], positions=[x], widths=0.5, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=c + ("55" if base else "22"), edgecolor=INK if base else "#b9bcc2",
                                     linewidth=2.2 if base else 1.2),
                       medianprops=dict(color=INK, linewidth=2.4), whiskerprops=dict(color=MUTED, linewidth=1.2),
                       capprops=dict(color=MUTED, linewidth=1.2))
            ax.scatter(x + rng.uniform(-0.1, 0.1, len(acc)), acc, s=18, color=c, edgecolors="white",
                       linewidths=0.5, zorder=3)
            ax.annotate(f"{acc.mean():.1f}%", (x, acc.max()), xytext=(0, 8), textcoords="offset points",
                        ha="center", va="bottom", fontsize=13, color=INK, weight="bold" if base else "normal")
            print(f"{name:14s} {tick:6s} {acc.mean():.2f} ± {acc.std():.2f}")
            ticks.append(x)
            labels.append(tick)
            x += 0.7
        ax.annotate(name, ((start + x - 0.7) / 2, 0), xycoords=("data", "axes fraction"), xytext=(0, -34),
                    textcoords="offset points", ha="center", va="top", fontsize=15, color=c, weight="bold")
        ends.append(x - 0.7)
        x += 0.5
    for e in ends[:-1]:
        ax.axvline(e + 0.6, color=GRID, linewidth=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=14, color=INK)
    ax.set_xlim(-0.45, x - 0.85)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("validation accuracy", color=MUTED, fontsize=16, labelpad=8)
    ax.grid(axis="y", color=GRID, linewidth=1.1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c6cc")
    ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=15)
    ax.tick_params(axis="x", length=0, pad=8)
    p1_mean = float(np.mean(list(runs(None, None).values())))
    ax.axhline(p1_mean, color=GREY, linestyle=(0, (1.5, 2.5)), linewidth=1.8, zorder=1)
    ax.annotate("bpe-logcounts-mlp", (x - 0.85, p1_mean), xytext=(0, -18), textcoords="offset points",
                ha="right", fontsize=13, color=GREY)
    fig.patch.set_alpha(0)
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    print("saved", out)


def lstm_step2_figure(out):
    """Slide 5, step 2 on LSTM 64 (frozen): bidirectional or not, last hidden state or mean of all states."""
    d = json.loads((ROOT / "results" / "q4_lstm.json").read_text())["15 folds"]
    acc = lambda name: np.array([100 * r["val_acc"] for r in d[name]["runs"]])
    base = "linear · 64 · frozen"
    grouped_boxes(out, [("bidirectional", PINK, [("no", acc(base), True), ("yes", acc(base + " · bidirectional"), False)]),
                        ("readout", "#3b5b8c", [("last", acc(base), True), ("mean", acc(base + " · mean of hidden states"), False)])])


def lstm_step3_figure(out, ylim=(85, 91.8)):
    """Slide 5, step 3 on LSTM 64 (frozen): words read (maxlen) x word dropout, 15 runs per box; base = 400 words, no dropout."""
    d = json.loads((ROOT / "results" / "q4_lstm.json").read_text())["15 folds"]
    base = "linear · 64 · frozen"
    name = lambda p, n: base if (p, n) == ("0.0", 400) else f"{base} · word dropout {p} · {n} words"
    drops = [(p, c) for p, c in (("0.0", PURPLE), ("0.1", PINK), ("0.25", GOLD)) if name(p, 200) in d]
    setup_fonts()
    fig, ax = plt.subplots(figsize=(4.6, 6.0), dpi=200)
    rng = np.random.default_rng(0)
    w, k = 0.34, len(drops)
    for g, n in enumerate((200, 400, 800)):
        for j, (p, c) in enumerate(drops):
            if name(p, n) not in d:
                continue
            x = g + (j - (k - 1) / 2) * (w + 0.04)
            acc = np.array([100 * r["val_acc"] for r in d[name(p, n)]["runs"]])
            is_base = (p, n) == ("0.0", 400)
            ax.boxplot([acc], positions=[x], widths=w, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=c + ("55" if is_base else "26"), edgecolor=INK if is_base else "#b9bcc2",
                                     linewidth=2.2 if is_base else 1.2),
                       medianprops=dict(color=INK, linewidth=2.2), whiskerprops=dict(color=MUTED, linewidth=1.1),
                       capprops=dict(color=MUTED, linewidth=1.1))
            ax.scatter(x + rng.uniform(-0.07, 0.07, len(acc)), acc, s=14, color=c, edgecolors="white",
                       linewidths=0.4, zorder=3)
            ax.annotate(f"{acc.mean():.1f}", (x, min(acc.max(), ylim[1] - 0.3)), xytext=(0, 6),
                        textcoords="offset points", ha="center", va="bottom", fontsize=12.5, color=INK,
                        weight="bold" if is_base else "normal")
            low = acc[acc < ylim[0]]
            print(f"words {n:4d} dropout {p:5s} {acc.mean():.2f} ± {acc.std():.2f}"
                  + (f"  (without runs < {ylim[0]}: {acc[acc >= ylim[0]].mean():.2f})" if len(low) else ""))
    p1_mean = float(np.mean(list(runs(None, None).values())))
    ax.axhline(p1_mean, color=GREY, linestyle=(0, (1.5, 2.5)), linewidth=1.8, zorder=1)
    ax.annotate("bpe-logcounts-mlp", (2.5, p1_mean), xytext=(0, -18), textcoords="offset points", ha="right",
                fontsize=13, color=GREY)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["200", "400", "800"], fontsize=15, color=INK)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(1))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("words read", color=INK, fontsize=16, weight="bold", labelpad=10)
    ax.set_ylabel("validation accuracy", color=MUTED, fontsize=16, labelpad=8)
    ax.grid(axis="y", color=GRID, linewidth=1.1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c6cc")
    ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=15)
    ax.tick_params(axis="x", length=0, pad=8)
    handles = [Line2D([], [], marker="o", linestyle="", markersize=8, color=c) for _, c in drops]
    leg = ax.legend(handles, [f"word dropout {float(p):g}" for p, _ in drops], loc="upper left", frameon=False,
                    fontsize=13, handletextpad=0.3, borderaxespad=0.2)
    for t, (_, c) in zip(leg.get_texts(), drops):
        t.set_color(c)
        t.set_fontweight("bold")
    fig.patch.set_alpha(0)
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    print("saved", out)


def step4_runs(name):
    """Validation accuracies (%) of every finished run of a setting, read from its run.json files (also mid-setting)."""
    d = ROOT / "results" / "models" / "lstm" / name
    return np.array([100 * json.loads(f.read_text())["val_acc"] for f in sorted(d.glob("seed-*/fold-*/run.json"))])


def lstm_step4_figure(out, size=(4.6, 6.0), wide=False, box=None, baseline=True, grid=True, show_means=True, legend=True):
    # box = (width, height) slide pixels: draw to fill it exactly, text at the size of the loss plots
    """Slide 5, step 4 on the best of step 3: training reviews (1k / 5k / all) x frozen or fine-tuned GloVe.

    Every run is a dot (frozen left, fine-tuned right of each size); the line joins the means.
    """
    setup_fonts()
    base = "linear-64-frozen-word-dropout-0-1-800-words"
    sizes = [("1k", "1k-reviews"), ("5k", "5k-reviews"), ("all", "all-reviews")]
    kinds = [("frozen", "frozen", GOLD, -0.09), ("fine-tuned", "fine-tuned", PINK, 0.09)]
    k = 0.7 if box else 1.0  # scale of text and marks
    if box:
        fig, ax = plt.subplots(figsize=(box[0] / 100, box[1] / 100), dpi=200, layout="constrained")
        fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    else:
        fig, ax = plt.subplots(figsize=size, dpi=200)
    rng = np.random.default_rng(0)
    for label, kind, c, dx in kinds:
        means = []
        for i, (tick, size) in enumerate(sizes):
            name = base if (kind, size) == ("frozen", "all-reviews") else f"{base}-{kind}-{size}"
            acc = step4_runs(name)
            means.append(acc.mean())
            ax.scatter(i + dx + rng.uniform(-0.035, 0.035, len(acc)), acc, s=14 * k, color=c, alpha=0.55,
                       edgecolors="white", linewidths=0.4, zorder=3)
            print(f"{kind:10s} {tick:4s} {acc.mean():.2f} ± {acc.std():.2f} ({len(acc)} runs)")
            if len(acc) < 15:
                ax.annotate(f"{len(acc)}/15 runs", (i + dx, acc.max()), xytext=(0, 6), textcoords="offset points",
                            ha="center", va="bottom", fontsize=11.5 * k, color=c)
        if show_means:  # the line through the means
            ax.plot(np.arange(3) + dx, means, color=c, linewidth=2.6 * k, marker="o", markersize=8 * k, markeredgecolor="white",
                    markeredgewidth=1.4, zorder=4, label=label)
    if baseline:  # project 1's bpe-logcounts-mlp
        p1_mean = float(np.mean(list(runs(None, None).values())))
        ax.axhline(p1_mean, color=GREY, linestyle=(0, (1.5, 2.5)), linewidth=1.8, zorder=1)
        ax.annotate("bpe-logcounts-mlp", (-0.4, p1_mean), xytext=(0, 6), textcoords="offset points", ha="left",
                    va="bottom", fontsize=13 * k, color=GREY)
    ax.set_xticks(range(3))
    ax.set_xticklabels([t for t, _ in sizes], fontsize=15 * k, color=INK)
    ax.set_xlim(-0.45, 2.45)
    ax.set_ylim(68, 92.5)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(4))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("training reviews", color=MUTED, fontsize=16 * k, labelpad=10 * k)
    ax.set_ylabel("validation accuracy", color=MUTED, fontsize=16 * k, labelpad=8 * k)
    if grid:
        ax.grid(axis="y", color=GRID, linewidth=1.1)
    ax.set_axisbelow(True)
    for side in ("top", "right") + (() if grid else ("left",)):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c6cc")
    ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=15 * k)
    ax.tick_params(axis="x", length=0, pad=8 * k)
    handles = None if show_means else [Line2D([], [], marker="o", linestyle="", markersize=6 * k, color=c) for _, _, c, _ in kinds]
    labels = None if show_means else [label for label, *_ in kinds]
    args = () if show_means else (handles, labels)
    leg = (ax.legend(*args, loc="lower right", frameon=False, fontsize=13 * k, handlelength=1.4) if wide else
           ax.legend(*args, loc="upper left", bbox_to_anchor=(0.0, 0.82), frameon=False, fontsize=13 * k,
                     handlelength=1.4 if show_means else 0.8, handletextpad=0.4))
    for t, (_, _, c, _) in zip(leg.get_texts(), kinds):
        t.set_color(c)
        t.set_fontweight("bold")
    if not legend:  # the colours are explained elsewhere (e.g. the table on the slide)
        leg.remove()
    fig.patch.set_alpha(0)
    if box:
        fig.savefig(out, transparent=True)
    else:
        fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.05)
    print("saved", out)


# ---- slide 4: loss curves under each table (mean over the 15 runs; an epoch is drawn while >= 8 runs still train)
LOSS_CARDS = {
    "arch": [("1×16", "mlp-1-16/300-dims"), ("2×64", "mlp-2-64/300-dims"), ("3×128", "mlp-3-128/300-dims")],
    "dropout": [("0", "dropout/0"), ("0.1", "mlp-3-128/300-dims"), ("0.3", "dropout/0-3")],
    "decay": [("0", "weight-decay/0"), ("1e-4", "mlp-3-128/300-dims"), ("1e-2", "weight-decay/1e-2")],
    "batch": [("32", "batch-size/32"), ("64", "mlp-3-128/300-dims"), ("128", "batch-size/128")],
    "dims": [("100", "mlp-3-128/100-dims"), ("200", "mlp-3-128/200-dims"), ("300", "mlp-3-128/300-dims")],
}
LOSS_COLOURS = (PURPLE, PINK, GOLD)


def mean_curve(histories, key, min_runs=8):
    n = max(len(h) for h in histories)
    out = []
    for e in range(n):
        vals = [h[e][key] for h in histories if len(h) > e]
        if len(vals) < min_runs:
            break
        out.append(np.mean(vals))
    return np.array(out)


def loss_figure(out, items, ylim=(0.3, 0.7), size=(3.2, 3.0), colours=LOSS_COLOURS,
                keys=("train_loss", "val_loss"), legend_labels=("validation", "training"), legend=True, font=14, ystep=0.1):
    """Validation loss (solid) and training loss (dashed) per epoch, one colour per setting."""
    setup_fonts()
    fig, ax = plt.subplots(figsize=size, dpi=200)
    for (label, folder), c in zip(items, colours):
        H = [json.loads(f.read_text())["history"] for f in sorted((ROOT / "results" / "models" / folder).glob("seed-*/fold-*/run.json"))]
        assert len(H) == 15, folder
        for key, style, w in (("train_loss", (0, (3, 2)), 1.6), ("val_loss", "-", 2.4)):
            if key not in keys:
                continue
            y = mean_curve(H, key)
            ax.plot(np.arange(1, len(y) + 1), y, color=c, linestyle=style, linewidth=w)
    ax.set_ylim(*ylim)
    ax.set_xlim(left=1)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ystep))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
    ax.set_xlabel("epoch", color=MUTED, fontsize=font, labelpad=3)
    ax.set_ylabel("loss", color=MUTED, fontsize=font, labelpad=4)
    ax.grid(axis="y", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c6cc")
    ax.tick_params(colors=MUTED, length=0, labelsize=font - 1, pad=4)
    handles = [Line2D([], [], color=INK, linewidth=2.2), Line2D([], [], color=INK, linewidth=1.5, linestyle=(0, (3, 2)))]
    if legend == "row":  # one row, for the flat plots
        ax.legend(handles, legend_labels, loc="upper right", ncol=2, frameon=False, fontsize=font - 2,
                  handlelength=1.8, columnspacing=1.2, borderaxespad=0.1)
    elif legend:
        ax.legend(handles, legend_labels, loc="upper right", frameon=False, fontsize=12, handlelength=1.8)
    fig.patch.set_alpha(0)
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved", out)


# ---- slide 5: LSTM loss curves. Cards 1-2 like slide 4; cards 3-4 validation loss only:
# colour = words read / training reviews, solid vs dashed = the second factor
LSTM_LOSS_CARDS = {
    "units": [("32", "lstm/linear-32-frozen"), ("64", "lstm/linear-64-frozen"), ("128", "lstm/linear-128-frozen")],
    "readout": [("one, last", "lstm/linear-64-frozen"), ("both", "lstm/linear-64-frozen-bidirectional"),
                ("average", "lstm/linear-64-frozen-mean-of-hidden-states")],
}
_S3 = "lstm/linear-64-frozen-word-dropout-"
LSTM_VAL_CARDS = {
    "words": ([(_S3 + "0-0-200-words", PURPLE, "-"), ("lstm/linear-64-frozen", PINK, "-"), (_S3 + "0-0-800-words", GOLD, "-"),
               (_S3 + "0-1-200-words", PURPLE, (0, (3, 2))), (_S3 + "0-1-400-words", PINK, (0, (3, 2))),
               (_S3 + "0-1-800-words", GOLD, (0, (3, 2)))], ("dropout 0", "dropout 0.1")),
    "data": ([(_S3 + "0-1-800-words-frozen-1k-reviews", PURPLE, "-"), (_S3 + "0-1-800-words-frozen-5k-reviews", PINK, "-"),
              (_S3 + "0-1-800-words", GOLD, "-"),
              (_S3 + "0-1-800-words-fine-tuned-1k-reviews", PURPLE, (0, (3, 2))),
              (_S3 + "0-1-800-words-fine-tuned-5k-reviews", PINK, (0, (3, 2))),
              (_S3 + "0-1-800-words-fine-tuned-all-reviews", GOLD, (0, (3, 2)))], ("frozen", "fine-tuned")),
}


def val_loss_figure(out, items, legend_labels, ylim=(0.2, 0.7), size=(3.2, 3.0)):
    """Validation loss per epoch (mean of 15 runs) for several settings: items = [(folder, colour, linestyle)]."""
    setup_fonts()
    fig, ax = plt.subplots(figsize=size, dpi=200)
    for folder, c, style in items:
        H = [json.loads(f.read_text())["history"] for f in sorted((ROOT / "results" / "models" / folder).glob("seed-*/fold-*/run.json"))]
        assert len(H) == 15, folder
        y = mean_curve(H, "val_loss")
        ax.plot(np.arange(1, len(y) + 1), y, color=c, linestyle=style, linewidth=2.2 if style == "-" else 1.8)
    ax.set_ylim(*ylim)
    ax.set_xlim(left=1)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.1))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4, integer=True))
    ax.set_xlabel("epoch", color=MUTED, fontsize=14, labelpad=4)
    ax.set_ylabel("validation loss", color=MUTED, fontsize=14, labelpad=4)
    ax.grid(axis="y", color=GRID, linewidth=1.0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c6cc")
    ax.tick_params(colors=MUTED, length=0, labelsize=13, pad=5)
    handles = [Line2D([], [], color=INK, linewidth=2.2), Line2D([], [], color=INK, linewidth=1.5, linestyle=(0, (3, 2)))]
    ax.legend(handles, legend_labels, loc="upper right", frameon=False, fontsize=12, handlelength=1.8)
    fig.patch.set_alpha(0)
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved", out)


def loss_figure_fit(out, items, box, ylim=(0.1, 0.8), colours=LOSS_COLOURS):
    """Loss curves drawn to fill a box of `box` = (width, height) slide pixels exactly, so text has the same size in
    every card: the figure is box/100 inches at 200 dpi (1 pt = 1.39 px on the slide)."""
    setup_fonts()
    w, h = box
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=200, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    for (label, folder), c in zip(items, colours):
        H = [json.loads(f.read_text())["history"] for f in sorted((ROOT / "results" / "models" / folder).glob("seed-*/fold-*/run.json"))]
        assert len(H) == 15, folder
        for key, style, lw in (("train_loss", (0, (3, 2)), 1.3), ("val_loss", "-", 2.0)):
            y = mean_curve(H, key)
            ax.plot(np.arange(1, len(y) + 1), y, color=c, linestyle=style, linewidth=lw)
    ax.set_ylim(*ylim)
    ax.set_xlim(left=1)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.2))
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5, integer=True))
    ax.set_xlabel("epoch", color=MUTED, fontsize=10.5, labelpad=2)
    ax.set_ylabel("loss", color=MUTED, fontsize=10.5, labelpad=3)
    ax.grid(axis="y", color=GRID, linewidth=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c6cc")
    ax.tick_params(colors=MUTED, length=0, labelsize=10, pad=3)
    handles = [Line2D([], [], color=INK, linewidth=1.8), Line2D([], [], color=INK, linewidth=1.2, linestyle=(0, (3, 2)))]
    ax.legend(handles, ["validation", "training"], loc="upper right", ncol=2, frameon=False, fontsize=9.5,
              handlelength=1.8, columnspacing=1.2, borderaxespad=0.1)
    fig.patch.set_alpha(0)
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print("saved", out)


def runs_figure_fit(out, items, box, ylim, ystep=1.0):
    """Accuracy of the 15 runs per setting, drawn to fill `box` = (width, height) slide pixels:
    every run a dot. items = [(label, accuracies %, colour)]."""
    setup_fonts()
    w, h = box
    fig, ax = plt.subplots(figsize=(w / 100, h / 100), dpi=200, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02)
    rng = np.random.default_rng(0)
    for i, (label, acc, c) in enumerate(items):
        acc = np.asarray(acc)
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(acc)), acc, s=12, color=c, alpha=0.45, linewidths=0, zorder=2)
    ax.set_xticks(range(len(items)))
    ax.set_xticklabels([it[0] for it in items], fontsize=10.5, color=INK)
    ax.set_xlim(-0.6, len(items) - 0.4)
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(ystep))
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_ylabel("accuracy", color=MUTED, fontsize=10.5, labelpad=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c6cc")
    ax.tick_params(axis="y", colors=MUTED, length=0, labelsize=10, pad=3)
    ax.tick_params(axis="x", length=0, pad=4)
    fig.patch.set_alpha(0)
    fig.savefig(out, transparent=True)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    arch_figure(ROOT / "results" / "q3_arch.png")
    if (ROOT / "results" / "q4_lstm.json").exists():
        lstm_units_figure(ROOT / "results" / "q4_units.png")
        if "linear · 64 · frozen · bidirectional" in json.loads((ROOT / "results" / "q4_lstm.json").read_text())["15 folds"]:
            lstm_step2_figure(ROOT / "results" / "q4_step2.png")
        if "linear · 64 · frozen · word dropout 0.0 · 200 words" in json.loads((ROOT / "results" / "q4_lstm.json").read_text())["15 folds"]:
            lstm_step3_figure(ROOT / "results" / "q4_step3.png")
        if (ROOT / "results" / "models" / "lstm" / "linear-64-frozen-word-dropout-0-1-800-words-fine-tuned-all-reviews").exists():
            lstm_step4_figure(ROOT / "results" / "q4_step4.png")
    dims_figure(ROOT / "results" / "q3_dims.png")
    for key, items in LOSS_CARDS.items():
        loss_figure(ROOT / "results" / f"q3_loss_{key}.png", items)
    for key, items in LSTM_LOSS_CARDS.items():
        loss_figure(ROOT / "results" / f"q4_loss_{key}.png", items, ylim=(0.1, 0.7))
    for key, (items, labels) in LSTM_VAL_CARDS.items():
        val_loss_figure(ROOT / "results" / f"q4_loss_{key}.png", items, labels,
                        ylim=(0.2, 0.9) if key == "data" else (0.2, 0.7))
    box_figure(ROOT / "results" / "q3_box.png")
    bayes_figure(ROOT / "results" / "q3_bayes.png")
    if (ROOT / "results" / "q3_reg_grid.json").exists():  # slide 4, right: 2 x 2 x 2 combinations
        reg_grid_figure(ROOT / "results" / "q3_reg.png")
    elif (ROOT / "results" / "q3_regularization.json").exists():  # earlier one-change-at-a-time version
        reg_figure(ROOT / "results" / "q3_reg.png")
