"""Draw the payload circuit as a game board (matplotlib)."""
from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
from qiskit import QuantumCircuit

PALETTE = dict(
    wire="#9aa5b1", cx="#3b82f6", one="#cbd5e1", t="#f59e0b", check="#10b981",
    bad="#ef4444", text="#0f172a",
)


def schedule(qc: QuantumCircuit):
    """ASAP column for every instruction -> list of (name, [qubits], column)."""
    free = [0] * qc.num_qubits
    ops = []
    for inst in qc.data:
        qs = [qc.find_bit(q).index for q in inst.qubits]
        col = max(free[q] for q in qs)
        for q in qs:
            free[q] = col + 1
        ops.append((inst.operation.name, qs, col))
    return ops, max(free) if free else 0


def draw_board(payload, selected: list[int], used: list[int] | None = None,
               dark: bool = True, figsize=(11, 3.6)):
    """Wires are rows, time runs left to right.  Selected wires get a check lane below.

    `selected` = wires the player picked, `used` = wires that actually got a valid check
    (T gates can make a check impossible; those wires are drawn in red).
    """
    used = selected if used is None else used
    # `payload` is a QuantumCircuit or an already-computed (ops, ncol, n) tuple (plain data,
    # safe to cache and share between Streamlit threads; Qiskit circuits are not)
    ops, ncol, n = payload if isinstance(payload, tuple) else (*schedule(payload),
                                                                payload.num_qubits)
    fg = "#e2e8f0" if dark else "#0f172a"
    bg = "#0b1220" if dark else "#ffffff"
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.set_xlim(-1.6, ncol + 0.6)
    ax.set_ylim(-(n + 0.2) - 0.9, 0.9)
    ax.axis("off")
    # check lanes (ancillas) under the data wires
    for i, q in enumerate(sorted(selected)):
        y_lane = -(n + 0.2) + 0.0 - i * 0.0
        ok = q in used
        col = PALETTE["check"] if ok else PALETTE["bad"]
        # the check spans the whole circuit in time: shaded band on the wire, bracket below
        ax.add_patch(Rectangle((-0.5, -q - 0.38), ncol + 0.9, 0.76, fc=col, alpha=0.13, ec=col,
                               lw=1.2, ls="-" if ok else "--"))
    for q in range(n):
        ax.plot([-0.6, ncol + 0.2], [-q, -q], color=PALETTE["wire"], lw=1.2, zorder=1)
        sel = q in selected
        ok = q in used
        label_col = (PALETTE["check"] if ok else PALETTE["bad"]) if sel else fg
        ax.text(-1.0, -q, f"q{q}", color=label_col, ha="center", va="center", fontsize=11,
                fontweight="bold")
        if sel:
            ax.text(-1.5, -q, "A" if ok else "x", color=label_col, ha="center", va="center",
                    fontsize=10, fontweight="bold")
    for name, qs, col in ops:
        if len(qs) == 2:
            a, b = qs
            ax.plot([col, col], [-a, -b], color=PALETTE["cx"], lw=2, zorder=2)
            ax.plot(col, -a, "o", color=PALETTE["cx"], ms=6, zorder=3)
            ax.plot(col, -b, "o", mfc=bg, mec=PALETTE["cx"], ms=10, mew=2, zorder=3)
            ax.plot([col, col], [-b - 0.12, -b + 0.12], color=PALETTE["cx"], lw=2, zorder=4)
            ax.plot([col - 0.12, col + 0.12], [-b, -b], color=PALETTE["cx"], lw=2, zorder=4)
        else:
            q = qs[0]
            if name in ("t", "tdg"):
                ax.add_patch(FancyBboxPatch((col - 0.3, -q - 0.3), 0.6, 0.6,
                                            boxstyle="round,pad=0.02", fc=PALETTE["t"], ec="none",
                                            zorder=3))
                ax.text(col, -q, "T" if name == "t" else "T†", ha="center", va="center",
                        fontsize=8, color="#1f2937", fontweight="bold", zorder=4)
            elif name == "x":
                ax.add_patch(Rectangle((col - 0.28, -q - 0.28), 0.56, 0.56, fc="#a78bfa",
                                       ec="none", zorder=3))
                ax.text(col, -q, "X", ha="center", va="center", fontsize=7, color="#1f2937",
                        zorder=4)
            else:
                ax.add_patch(Rectangle((col - 0.18, -q - 0.18), 0.36, 0.36,
                                       fc=PALETTE["one"], ec="none", alpha=0.85, zorder=3))
    ax.text(ncol / 2, 0.62, "time  →", color=fg, ha="center", fontsize=9, alpha=0.7)
    ax.text(ncol / 2, -(n - 0.3) - 0.45,
            "green band = a spacetime check riding on that wire for the whole circuit   |   "
            "orange = T / T† gate   |   blue = CNOT", color=fg, ha="center", fontsize=8,
            alpha=0.75)
    fig.tight_layout(pad=0.3)
    return fig
