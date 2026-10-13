"""Detect or Discard: a game of spacetime checks.

Run:  streamlit run game/app.py

You protect a quantum circuit by attaching check ancillas to wires.  Every number on screen
(fidelity, kept shots, gate cost, score) comes from Qiskit / Aer / qiskit-paulice results.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from board import draw_board  # noqa: E402
from levels import LEVELS, best_under_budget, level_rows, load_landscape, stars  # noqa: E402
from spacetime import Level  # noqa: E402

st.set_page_config(page_title="Detect or Discard", page_icon="⚛️", layout="wide")

BG, FG, GRID = "#0b1220", "#e2e8f0", "#334155"
GREEN, BLUE, AMBER, RED, GREY = "#10b981", "#3b82f6", "#f59e0b", "#ef4444", "#64748b"


@st.cache_data(show_spinner=False)
def landscape() -> pd.DataFrame:
    return load_landscape()


@st.cache_data(show_spinner=False)
def board_data(key: str):
    """Gate schedule of the payload as plain data (precomputed by src/make_boards.py)."""
    d = json.loads((ROOT / "results" / "boards.json").read_text())[key]
    return [tuple(o) for o in d["ops"]], d["ncol"], d["n"]


def key_of(targets: list[int]) -> str:
    return "".join(map(str, sorted(targets))) or "-"


@st.cache_data(show_spinner="Running the circuit on Aer...")
def live_eval(params: dict, targets: tuple) -> dict:
    """Live mode: evaluate in a fresh process (Qiskit circuits stay out of web-server threads)."""
    arg = json.dumps({**params, "targets": list(targets)})
    out = subprocess.run([sys.executable, "-I", str(ROOT / "src" / "live_eval.py"), arg],
                         capture_output=True, text=True, check=True, timeout=600)
    return json.loads(out.stdout.strip().splitlines()[-1])


def level_params(lv: Level) -> dict:
    return dict(n=lv.n, depth=lv.depth, t_count=lv.t_count, p2=lv.p2, pro=lv.pro, seed=lv.seed,
                t_seed=lv.t_seed)


# ----------------------------------------------------------------------------- state
if "attempts" not in st.session_state:
    st.session_state.attempts = {}  # level key -> list of dicts
if "stars" not in st.session_state:
    st.session_state.stars = {}

st.markdown(
    f"""<style>
    .stApp {{ background:{BG}; color:{FG}; }}
    h1,h2,h3,h4,p,li,label,span {{ color:{FG}; }}
    .card {{ background:#111a2e; border:1px solid {GRID}; border-radius:12px; padding:14px 16px; }}
    .big {{ font-size:28px; font-weight:700; line-height:1.1; }}
    .small {{ font-size:12px; opacity:.75; }}
    .stars {{ font-size:22px; letter-spacing:2px; color:{AMBER}; }}
    </style>""", unsafe_allow_html=True)

df_all = landscape()

# ----------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.markdown("## ⚛️ Detect or Discard")
    st.caption("A game of spacetime error-detection checks, scored by real Qiskit runs.")
    names = [f"{l.title}  {'★' * st.session_state.stars.get(l.key, 0)}" for l in LEVELS]
    qp = st.query_params  # deep link for demos / screenshots: ?level=0&wires=013&run=1
    if "level_idx" not in st.session_state and "level" in qp:
        st.session_state["level_idx"] = int(qp["level"])
    idx = st.radio("Level", range(len(LEVELS)), format_func=lambda i: names[i], key="level_idx")
    live = st.toggle("Re-run live on Aer instead of using the precomputed table", value=False,
                     help="Both give the same numbers up to shot noise (same seeds). "
                          "Live mode is slower but shows the simulation really runs.")
    if st.button("Reset progress"):
        st.session_state.attempts = {}
        st.session_state.stars = {}
        st.rerun()
    st.markdown("---")
    st.caption("Score = 100 × (kept × F²) / best possible under the budget. "
               "kept × F² is the inverse of the raw shots needed for a fixed precision.")

game = LEVELS[idx]
phys = game.physics
rows = level_rows(df_all, phys)
best = best_under_budget(rows, game.budget)
base = rows[rows.targets == "-"].iloc[0]

tab_play, tab_rules, tab_explore = st.tabs(
    ["\U0001f3ae Play", "\U0001f4d6 Rules ↔ Physics", "\U0001f50e Explore every placement"])

# ----------------------------------------------------------------------------- PLAY
with tab_play:
    st.markdown(f"### {game.title}")
    c1, c2 = st.columns([3, 2])
    with c1:
        st.write(game.story)
    with c2:
        st.markdown(
            f"<div class='card'><span class='small'>Ancilla budget</span><br>"
            f"<span class='big'>{game.budget}</span> "
            f"<span class='small'>of {phys.n} wires · 2-qubit gate error "
            f"{phys.p2:.1%} · {phys.t_count} T/T† gates · "
            f"{'qiskit-paulice' if phys.t_count == 0 else 'hand-made checks'}</span></div>",
            unsafe_allow_html=True)

    st.markdown("**Choose the wires that get a check ancilla**")
    demo_wires = qp.get("wires") if "wires" in qp else None
    if demo_wires is not None and not st.session_state.get(f"_demo_set_{game.key}"):
        for q in range(phys.n):
            st.session_state[f"{game.key}_q{q}"] = str(q) in demo_wires
        st.session_state[f"_demo_set_{game.key}"] = True
    cols = st.columns(phys.n)
    chosen = []
    for q in range(phys.n):
        with cols[q]:
            if st.checkbox(f"q{q}", key=f"{game.key}_q{q}"):
                chosen.append(q)
    over = len(chosen) > game.budget

    # Which wires would really get a valid check (matters for T-doped levels)?
    hist = st.session_state.attempts.setdefault(game.key, [])
    last = hist[-1] if hist else None
    st.pyplot(draw_board(board_data(game.key), chosen,
                         used=[q for q in chosen]  # validity is shown after the run
                         if not last or last["targets_req"] != key_of(chosen)
                         else [int(c) for c in last["targets"] if c.isdigit()] or []),
              clear_figure=True)

    demo_run = "run" in qp and not st.session_state.get(f"_demo_run_{game.key}")
    run = st.button("▶  Run on Qiskit", type="primary", disabled=over) or demo_run
    if demo_run:
        st.session_state[f"_demo_run_{game.key}"] = True
    if over:
        st.error(f"Over budget: you picked {len(chosen)} wires but only {game.budget} ancillas "
                 "are available.")

    if run:
        k = key_of(chosen)
        if live:
            r = live_eval(level_params(phys), tuple(sorted(chosen)))
        else:
            r = rows[rows.targets == k].iloc[0].to_dict()
        rel = r["efficiency"] / best["efficiency"]
        hist.append(dict(targets_req=k, targets=r["targets"], F=r["fidelity_post"],
                         kept=r["kept"], eff=r["efficiency"], rel=rel,
                         checks=int(r["checks"]), gates=int(r["two_qubit_gates"]),
                         depth=int(r["depth_total"]), anc=int(r["ancillas"]),
                         F_raw=r["fidelity_raw"], F_err=r["fidelity_post_err"]))
        st.session_state.stars[game.key] = max(st.session_state.stars.get(game.key, 0),
                                               stars(rel))
        st.rerun()

    last = hist[-1] if hist else None
    if last:
        s = stars(last["rel"])
        st.markdown("---")
        a, b, c, d = st.columns(4)
        a.markdown(f"<div class='card'><span class='small'>Fidelity after post-selection</span><br>"
                   f"<span class='big'>{last['F']:.3f}</span><br><span class='small'>"
                   f"no checks: {base['fidelity_post']:.3f} · ±{last['F_err']:.3f}"
                   f"</span></div>", unsafe_allow_html=True)
        b.markdown(f"<div class='card'><span class='small'>Shots kept</span><br>"
                   f"<span class='big'>{last['kept']:.1%}</span><br><span class='small'>"
                   f"{(1 - last['kept']):.1%} discarded</span></div>", unsafe_allow_html=True)
        c.markdown(f"<div class='card'><span class='small'>Score</span><br>"
                   f"<span class='big'>{100 * last['rel']:.0f}</span><span class='small'> / 100"
                   f"</span><br><span class='stars'>{'★' * s}{'☆' * (3 - s)}</span></div>",
                   unsafe_allow_html=True)
        d.markdown(f"<div class='card'><span class='small'>Cost added</span><br>"
                   f"<span class='big'>+{last['anc']}</span><span class='small'> ancillas</span>"
                   f"<br><span class='small'>{last['gates']} two-qubit gates "
                   f"(was {int(base['two_qubit_gates'])}) · depth {last['depth']} "
                   f"(was {int(base['depth_total'])})</span></div>", unsafe_allow_html=True)

        # ---- explanation, generated from the numbers
        req = [int(ch) for ch in last["targets_req"] if ch.isdigit()]
        got = [int(ch) for ch in last["targets"] if ch.isdigit()] if last["targets"] != "-" else []
        lines = []
        if not req:
            lines.append("No checks: every shot is kept, but nothing catches the errors the "
                         "CNOTs introduce.")
        else:
            dF = last["F"] - base["fidelity_post"]
            lines.append(
                f"The {last['checks']} check(s) flagged and discarded {1 - last['kept']:.1%} of "
                f"shots. The shots that remained are {'cleaner' if dF > 0 else 'no cleaner'}: "
                f"fidelity {base['fidelity_post']:.3f} → {last['F']:.3f}.")
            extra_gates = last["gates"] - int(base["two_qubit_gates"])
            lines.append(
                f"The checks themselves added {extra_gates} two-qubit gates, and those gates are "
                f"noisy too. Fidelity of ALL shots (before discarding) went "
                f"{base['fidelity_raw']:.3f} → {last['F_raw']:.3f}: a check can also inject "
                "errors, which is why undetected errors remain.")
            missing = [q for q in req if q not in got and phys.t_count > 0]
            if missing and len(got) < len(req):
                lines.append(
                    f"⚠ Wire(s) {', '.join(f'q{q}' for q in missing)} could not get an "
                    "independent valid check: the T gates block every Pauli that would be "
                    "usable there (a check may cross a T only as I or Z).")
        if last["rel"] >= 0.985:
            lines.append("\U0001f3c6 That is the best placement available under this budget.")
        elif last["rel"] < 1:
            lines.append("There is a better placement under this budget. Try a different set "
                         "of wires.")
        for ln in lines:
            st.markdown(f"- {ln}")

        # ---- trade-off plot
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        fig.patch.set_facecolor(BG)
        ax.set_facecolor(BG)
        sub = rows[rows.requested <= game.budget]
        # iso-efficiency curves  kept * F^2 = const
        xs = np.linspace(0.4, 1.0, 100)
        for lvl in np.linspace(0.45, 0.95, 6):
            ys = np.sqrt(lvl / xs)
            m = ys <= 1.0
            ax.plot(xs[m], ys[m], color=GRID, lw=0.8, ls="--", zorder=1)
        ax.scatter(sub.kept, sub.fidelity_post, s=18 + 16 * sub.checks, c=GREY, alpha=0.35,
                   zorder=2, label="every placement within budget")
        ax.scatter([base.kept], [base.fidelity_post], marker="s", s=70, c=BLUE, zorder=4,
                   label="no checks")
        ax.scatter([best.kept], [best.fidelity_post], marker="*", s=220, c=GREEN, zorder=5,
                   label="best available")
        pa = pd.DataFrame(hist)
        ax.plot(pa.kept, pa.F, "-o", color=AMBER, ms=5, lw=1.2, zorder=6, label="your attempts")
        ax.scatter([last["kept"]], [last["F"]], s=120, facecolors="none", edgecolors=AMBER,
                   lw=2, zorder=7)
        ax.set_xlabel("fraction of shots kept  →", color=FG)
        ax.set_ylabel("fidelity after post-selection  →", color=FG)
        ax.set_title("The trade-off: cleaner shots vs. fewer shots", color=FG, fontsize=11)
        ax.tick_params(colors=FG)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8, loc="lower left")
        ax.text(0.99, 0.02, "dashed: equal kept×F² (equal score)", transform=ax.transAxes,
                ha="right", color=FG, alpha=0.6, fontsize=7)
        st.pyplot(fig, clear_figure=True)

        with st.expander("Show the best placement (spoiler)"):
            bt = [int(ch) for ch in best["targets"] if ch.isdigit()]
            st.write(f"Best under budget {game.budget}: wires "
                     f"{', '.join(f'q{q}' for q in bt) or 'none'}  →  "
                     f"fidelity {best['fidelity_post']:.3f}, kept {best['kept']:.1%}, "
                     f"efficiency {best['efficiency']:.3f} "
                     f"(no checks: {base['efficiency']:.3f}).")
        st.info(f"**Lesson.** {game.lesson}")

        with st.expander("Your attempts"):
            h = pd.DataFrame(hist)[["targets_req", "checks", "F", "kept", "rel", "gates"]]
            h.columns = ["wires", "valid checks", "fidelity", "kept", "score", "2q gates"]
            h["score"] = (100 * h["score"]).round(0).astype(int)
            h["fidelity"] = h["fidelity"].round(3)
            h["kept"] = (100 * h["kept"]).round(1).astype(str) + "%"
            st.dataframe(h, hide_index=True, width="stretch")

# ----------------------------------------------------------------------------- RULES
with tab_rules:
    st.markdown("""
### Every rule is a physics fact

| Game rule | What it means in the physics |
|---|---|
| **The board** is the payload circuit (wires = qubits, columns = time). | The *payload*: the data qubits and the logic being run. It is an echo circuit: encode a bitstring, run a random Clifford circuit, run its inverse. The ideal answer is one known bitstring, so every wrong bit is an error. |
| **A check ancilla** on a wire. | A *spacetime Pauli check*: an ancilla in \\|+\\> is entangled (controlled-Pauli) with the data at the start, again at the end, then read out in the X basis. If a single Pauli error happened in between and it does not commute with the check, the ancilla flips. |
| **A flagged shot is thrown away.** | *Post-selection*: error **detection**, not correction. Nothing is repaired; shots whose syndrome is non-trivial are discarded. Errors the checks cannot see go through undetected. |
| **Shots kept** | Checks (and their own noisy CNOTs) flag shots that would otherwise have been counted. More checks flag more shots, so fewer remain. |
| **Fidelity** | Hellinger fidelity of the (post-selected) output distribution to the exact ideal one, computed from stabilizer (Clifford) or statevector (T-doped) simulation. |
| **Cost added** | Every check adds ancillas, two-qubit gates, and depth. Those gates are noisy too: *checks are not free*. |
| **Score = kept × F²** | Precision per raw shot. To estimate an observable damped by about F, the variance scales like 1/(kept·N·F²). Throwing away shots hurts exactly as much as noise does. |
| **Quiet hardware level** | Detection only pays when there are errors worth catching. |
| **T-gate levels** | `qiskit-paulice` accepts Clifford circuits only. A hand-made check can cross a T gate only if its Pauli is I or Z there (T is diagonal, so it commutes with Z but turns X into a non-Pauli). T gates remove valid checks. |
| **Ancilla budget** | Qubits are the scarce resource on real devices. |

### What is *not* in this game
* **Not error correction.** Nothing is decoded or repaired. One logical qubit is not encoded in many.
* **Not error mitigation.** No estimate is corrected in post-processing (no ZNE/PEC).
* It is **detection**: flag what the checks can see, discard those shots, accept what they cannot see.

### Honest limitations
* Noise: 2-qubit depolarizing + readout flips (the brief's example `NoiseModel(gate_noise=0.01)`); no idle or 1-qubit noise, T gates are noiseless. Real hardware differs.
* The payload is small (6 qubits) and one random circuit family. Absolute numbers will change with the circuit; the trade-off pattern is the point.
* Hand-made checks are picked greedily by cost, not optimised like paulice's.
""")

# ----------------------------------------------------------------------------- EXPLORE
with tab_explore:
    st.markdown(f"### All {len(rows)} placements on {game.title}")
    st.caption("Each dot is one choice of wires, run through Qiskit. Color = number of checks.")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    fig.patch.set_facecolor(BG)
    for a in ax:
        a.set_facecolor(BG)
        a.tick_params(colors=FG)
        for sp in a.spines.values():
            sp.set_color(GRID)
    sc = ax[0].scatter(rows.kept, rows.fidelity_post, c=rows.checks, cmap="viridis", s=36)
    ax[0].set_xlabel("kept fraction", color=FG)
    ax[0].set_ylabel("post-selected fidelity", color=FG)
    ax[0].set_title("Pareto picture", color=FG)
    cb = fig.colorbar(sc, ax=ax[0])
    cb.ax.tick_params(colors=FG)
    cb.set_label("valid checks", color=FG)
    g = rows.groupby("checks").efficiency.agg(["min", "max"])
    ax[1].fill_between(g.index, g["min"], g["max"], color=BLUE, alpha=0.3,
                       label="worst to best placement")
    ax[1].plot(g.index, g["max"], "-o", color=GREEN, label="best placement")
    ax[1].axhline(base.efficiency, color=AMBER, ls="--", label="no checks")
    ax[1].set_xlabel("number of checks", color=FG)
    ax[1].set_ylabel("efficiency  kept×F²", color=FG)
    ax[1].set_title("Placement matters as much as count", color=FG)
    ax[1].legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, fontsize=8)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
    top = rows.sort_values("efficiency", ascending=False).head(8)[
        ["targets", "checks", "fidelity_post", "kept", "efficiency", "two_qubit_gates"]]
    top.columns = ["wires", "valid checks", "fidelity", "kept", "efficiency", "2q gates"]
    st.dataframe(top.round(3), hide_index=True, width="stretch")
