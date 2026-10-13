"""Game levels.  A level is just a payload + noise setting + ancilla budget.

Star ratings are NOT hand-set numbers per level: they are fractions of the best achievable
score under the level's budget, computed from the precomputed Qiskit results (landscape.csv).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from spacetime import Level

DATA = Path(__file__).resolve().parent.parent / "results" / "landscape.csv"


@dataclass(frozen=True)
class GameLevel:
    key: str
    title: str
    story: str
    lesson: str
    physics: Level
    budget: int


LEVELS = [
    GameLevel(
        "L1", "1 · First Contact",
        "A 6-qubit echo circuit runs on typical hardware (1% two-qubit gate error). "
        "You may attach up to 3 check ancillas. Pick the wires to guard.",
        "Each check catches errors but costs extra CNOTs and throws shots away. "
        "Where you place checks, not just how many, changes the result.",
        Level(t_count=0), 3),
    GameLevel(
        "L2", "2 \u00b7 Short Sprint",
        "A very shallow circuit (depth 2 each way, 10 CNOTs) on the same 1% hardware. "
        "Budget: 3 ancillas.",
        "When a circuit has few errors to begin with, checks have little to catch but still add "
        "noisy gates and discard shots. The best move can be to place no check at all.",
        Level(t_count=0, depth=2), 3),
    GameLevel(
        "L3", "3 · Storm",
        "A noisy chip: 2% two-qubit error. Budget: 4 ancillas.",
        "The noisier the hardware, the more a check pays for itself.",
        Level(t_count=0, p2=0.02), 4),
    GameLevel(
        "L4", "4 · Deep Dive",
        "The circuit is much deeper (depth 8 each way). Budget: 4 ancillas.",
        "Deeper circuits collect more errors, so detection is worth more, but each check "
        "is also longer and more exposed.",
        Level(t_count=0, depth=8), 4),
    GameLevel(
        "L5", "5 · The T-Gate Trap",
        "Magic gates appear: 4 T/T† gates are hidden in the circuit. "
        "qiskit-paulice refuses non-Clifford circuits, so your checks are hand-made. Budget: 4.",
        "A check may cross a T gate only if its Pauli is I or Z there (T commutes with Z, not X/Y). "
        "Each T/T\u2020 pair halves the number of valid checks, so the protection per ancilla is "
        "weaker than on the Clifford levels.",
        Level(t_count=4), 4),
    GameLevel(
        "L6", "6 · Magic Overload",
        "8 T/T† gates. Very few valid checks survive. Budget: 4.",
        "With many T gates the pool of valid checks collapses (255 of 4095 left), so the best "
        "reachable fidelity is capped and the best number of checks drops. Detection gets harder "
        "as circuits get more non-Clifford, which is part of why fault tolerance needs more than checks.",
        Level(t_count=8), 4),
]


def load_landscape() -> pd.DataFrame:
    df = pd.read_csv(DATA, dtype={"targets": str})
    return df


def level_rows(df: pd.DataFrame, lv: Level) -> pd.DataFrame:
    p = lv
    m = (df.t_count == p.t_count) & ((df.p2 - p.p2).abs() < 1e-9) & (df.depth == p.depth) \
        & (df.n == p.n)
    return df[m].copy()


def best_under_budget(rows: pd.DataFrame, budget: int) -> pd.Series:
    sub = rows[rows.requested <= budget]
    return sub.loc[sub.efficiency.idxmax()]


def stars(rel: float) -> int:
    """rel = player's efficiency / best achievable under the budget."""
    return 3 if rel >= 0.985 else 2 if rel >= 0.95 else 1 if rel >= 0.85 else 0
