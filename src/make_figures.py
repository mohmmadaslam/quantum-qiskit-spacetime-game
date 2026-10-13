"""Figures for the report.  Reads results/*.csv, writes figures/*.png."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from board import draw_board  # noqa: E402
from spacetime import Level  # noqa: E402

R, F = ROOT / "results", ROOT / "figures"
F.mkdir(exist_ok=True)
BLUE, GREEN, AMBER, RED, GREY, INK = "#2563eb", "#059669", "#d97706", "#dc2626", "#94a3b8", "#0f172a"
plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.25, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#475569", "text.color": INK, "axes.labelcolor": INK,
                     "xtick.color": INK, "ytick.color": INK})
rd = lambda n: pd.read_csv(R / n, dtype={"targets": str})  # noqa: E731
L = rd("landscape.csv")
L0 = L[(L.t_count == 0) & (L.p2 == 0.01) & (L.depth == 5)]


def save(fig, name):
    fig.savefig(F / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# 1. pipeline diagram ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 2.3))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 2.3)
boxes = [("1  Payload", "echo circuit\nX → C → C†", BLUE),
         ("2  Spacetime checks", "controlled-Pauli pairs\non check ancillas", GREEN),
         ("3  Ancilla readout", "measure ancillas\n→ syndrome bits", AMBER),
         ("4  Post-selection", "syndrome ≠ 0 → discard\nthen score F, kept", RED)]
for i, (t, s, c) in enumerate(boxes):
    x = 0.15 + i * 2.5
    ax.add_patch(FancyBboxPatch((x, 0.3), 2.1, 1.7, boxstyle="round,pad=0.03", fc="white", ec=c, lw=2))
    ax.text(x + 1.05, 1.55, t, ha="center", fontweight="bold", color=c, fontsize=11)
    ax.text(x + 1.05, 0.9, s, ha="center", va="center", fontsize=9.5)
    if i < 3:
        ax.add_patch(FancyArrowPatch((x + 2.12, 1.15), (x + 2.5, 1.15), arrowstyle="-|>", mutation_scale=16, color=INK))
save(fig, "pipeline.png")

# 2. board with checks (light) ----------------------------------------------------------
fig = draw_board(Level(t_count=4).payload, [0, 2, 4], dark=False, figsize=(11, 3.6))
save(fig, "board_T4.png")
fig = draw_board(Level(t_count=0).payload, [2, 3], dark=False, figsize=(11, 3.6))
save(fig, "board_clifford.png")

# 3. trade-off cloud (headline) ---------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.4, 4.4))
xs = np.linspace(0.45, 1.0, 100)
for lvl in np.linspace(0.45, 0.7, 6):
    ys = np.sqrt(lvl / xs)
    m = ys <= 1
    ax.plot(xs[m], ys[m], "--", color="#cbd5e1", lw=0.9, zorder=1)
sc = ax.scatter(L0.kept, L0.fidelity_post, c=L0.checks, cmap="viridis", s=34, zorder=3)
z = L0[L0.targets == "-"].iloc[0]; b = L0.loc[L0.efficiency.idxmax()]
ax.scatter([z.kept], [z.fidelity_post], marker="s", s=90, c=BLUE, zorder=4, label="no checks")
ax.scatter([b.kept], [b.fidelity_post], marker="*", s=300, c=AMBER, ec="k", zorder=5,
           label=f"best: wires {','.join(b.targets)}")
ax.set(xlabel="fraction of shots kept", ylabel="post-selected fidelity",
       title="All 64 check placements, 6-qubit Clifford echo, p₂ = 1%")
fig.colorbar(sc, ax=ax, label="number of checks")
ax.legend(loc="lower left", fontsize=8)
ax.text(0.99, 0.02, "dashed: equal kept×F²", transform=ax.transAxes, ha="right", fontsize=7, color="#64748b")
save(fig, "tradeoff_cloud.png")

# 4. placement spread -------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.2, 4))
jit = np.random.default_rng(0).uniform(-.14, .14, len(L0))
ax.scatter(L0.checks + jit, L0.efficiency, s=16, alpha=.6, color=BLUE)
g = L0.groupby("checks").efficiency.max()
ax.plot(g.index, g.values, "-o", color=GREEN, label="best placement")
ax.axhline(z.efficiency, color=AMBER, ls="--", label="no checks")
ax.set(xlabel="number of checks", ylabel="efficiency  kept×F²",
       title="Same number of checks, different wires")
ax.legend(fontsize=8)
save(fig, "placement.png")

# 5. noise sweep ------------------------------------------------------------------------
n = rd("noise_sweep.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
cm = plt.cm.plasma(np.linspace(0.05, 0.85, n.p2.nunique()))
rows = []
for c, (p2, gp) in zip(cm, n.groupby("p2")):
    ax[0].plot(gp.checks, gp.efficiency / gp.efficiency.iloc[0], "-o", ms=4, color=c, label=f"p₂={p2:.2%}")
    bb = gp.loc[gp.efficiency.idxmax()]
    rows.append((p2, bb.efficiency / gp.efficiency.iloc[0], int(bb.checks)))
ax[0].axhline(1, color="k", lw=.8)
ax[0].set(xlabel="checks (wires 0..k-1)", ylabel="efficiency relative to no checks",
          title="Noisier hardware benefits more")
ax[0].legend(fontsize=7)
ax[1].bar([f"{p:.2%}" for p, _, _ in rows], [g_ - 1 for _, g_, _ in rows], color=cm)
for i, (_, g_, k) in enumerate(rows):
    ax[1].text(i, g_ - 1 + 0.003, f"k={k}", ha="center", fontsize=8)
ax[1].set(xlabel="two-qubit gate error", ylabel="best gain over no checks", title="Gain at the best number of checks")
save(fig, "noise_sweep.png")

# 6. depth sweep ------------------------------------------------------------------------
d = rd("depth_sweep.csv")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
dep, gain, kbest, f0, fb = [], [], [], [], []
for dd, gp in d.groupby("depth"):
    bb = gp.loc[gp.efficiency.idxmax()]
    dep.append(dd); gain.append(bb.efficiency / gp.efficiency.iloc[0]); kbest.append(int(bb.checks))
    f0.append(gp.fidelity_post.iloc[0]); fb.append(bb.fidelity_post)
ax[0].plot(dep, f0, "-o", color=RED, label="no checks")
ax[0].plot(dep, fb, "-o", color=GREEN, label="best configuration")
ax[0].set(xlabel="depth of C (echo = 2×)", ylabel="post-selected fidelity", title="Deeper circuits lose more fidelity")
ax[0].legend(fontsize=8)
ax[1].bar([str(x) for x in dep], np.array(gain) - 1, color=BLUE)
for i, k in enumerate(kbest):
    ax[1].text(i, gain[i] - 1 + 0.01, f"k={k}", ha="center", fontsize=8)
ax[1].set(xlabel="depth of C", ylabel="best gain over no checks", title="Checks pay off only when there is enough to catch")
save(fig, "depth_sweep.png")

# 7. T count ----------------------------------------------------------------------------
t = rd("t_sweep.csv")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
cm = plt.cm.cividis(np.linspace(0.05, 0.9, t.t_count.nunique()))
for c, (tc, gp) in zip(cm, t.groupby("t_count")):
    ax[0].plot(gp.checks, gp.fidelity_post, "-o", ms=4, color=c, label=f"{tc} T/T†")
    ax[1].plot(gp.checks, gp.kept, "-o", ms=4, color=c)
pool = t.drop_duplicates("t_count").set_index("t_count").pool
ax[0].set(xlabel="requested checks", ylabel="post-selected fidelity", title="T gates cap the fidelity checks can reach")
ax[0].legend(fontsize=7)
ax[1].set(xlabel="requested checks", ylabel="kept fraction", title="kept fraction")
ax[2].bar([str(k) for k in pool.index], pool.values, color=cm)
ax[2].set_yscale("log")
ax[2].set(xlabel="T/T† gates in the circuit", ylabel="valid checks (of 4095)", title="Valid checks halve per T/T† pair")
save(fig, "t_sweep.png")

# 8. paulice vs hand --------------------------------------------------------------------
p = rd("paulice_vs_hand.csv")
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
for m, c in (("paulice", BLUE), ("hand", GREEN)):
    gp = p[p.method == m]
    ax[0].plot(gp.checks, gp.fidelity_post, "-o", ms=4, color=c, label=m)
    ax[1].plot(gp.checks, gp.two_qubit_gates, "-o", ms=4, color=c)
    ax[2].plot(gp.checks, gp.efficiency, "-o", ms=4, color=c)
ax[0].set(xlabel="checks", ylabel="post-selected fidelity", title="paulice reaches slightly higher fidelity")
ax[0].legend()
ax[1].set(xlabel="checks", ylabel="two-qubit gates", title="...with more two-qubit gates")
ax[2].set(xlabel="checks", ylabel="efficiency kept×F²", title="...so hand-made checks win on efficiency")
save(fig, "paulice_vs_hand.png")

# 9. Clifford vs T-doped at equal checks ------------------------------------------------
fig, ax = plt.subplots(figsize=(5.6, 3.8))
for c, (tc, gp) in zip(cm, t.groupby("t_count")):
    ax.plot(gp.checks, gp.efficiency, "-o", ms=4, color=c, label=f"{tc} T/T†")
ax.set(xlabel="requested checks", ylabel="efficiency kept×F²", title="Efficiency vs checks for each T count")
ax.legend(fontsize=7)
save(fig, "t_efficiency.png")
print("figures:", sorted(p.name for p in F.iterdir()))
