"""Run every experiment behind the report, the notebook and the game.

Usage:  python src/run_experiments.py          (about 10-15 minutes on a laptop)

Writes CSV files to results/.  Everything is seeded, so reruns reproduce the same numbers.
"""
from __future__ import annotations

import sys
import time
from itertools import combinations
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from spacetime import Level, run_checked  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "results"
OUT.mkdir(exist_ok=True)
N = 6
ALL_SUBSETS = [list(c) for k in range(N + 1) for c in combinations(range(N), k)]  # 64 subsets


def row(level: Level, targets: list[int], res, **extra) -> dict:
    return dict(
        n=level.n, depth=level.depth, t_count=level.t_count, p2=level.p2, pro=level.pro,
        targets="".join(map(str, targets)) or "-", requested=len(targets), checks=res.n_checks,
        fidelity_raw=res.fidelity_raw, fidelity_post=res.fidelity_post,
        fidelity_post_err=res.fidelity_post_err, kept=res.kept, kept_err=res.kept_err,
        efficiency=res.efficiency, ancillas=res.ancillas, two_qubit_gates=res.two_qubit_gates,
        depth_total=res.depth, source=res.source, paulice_gamma=res.predicted_cost, **extra,
    )


def timed(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    # ------------------------------------------------------------------------------
    # E1  Placement landscape: every subset of target wires, per level.  The game reads this.
    #     5 T counts x 64 subsets at the standard noise p2=1%, plus noise 0.5% and 2% for the
    #     Clifford payload.
    # ------------------------------------------------------------------------------
    rows = []
    levels = [Level(t_count=t) for t in (0, 2, 4, 8)]
    levels += [Level(t_count=0, p2=p) for p in (0.005, 0.02)]
    levels += [Level(t_count=0, depth=d) for d in (2, 8)]
    for lv in levels:
        timed(f"E1 level T={lv.t_count} p2={lv.p2} depth={lv.depth}")
        for tg in ALL_SUBSETS:
            rows.append(row(lv, tg, lv.evaluate(tg)))
    pd.DataFrame(rows).to_csv(OUT / "landscape.csv", index=False)

    # ------------------------------------------------------------------------------
    # E2  Paulice vs hand-made checks on the SAME Clifford payload (sanity check of the
    #     hand-made check machinery, and a fair comparison).
    # ------------------------------------------------------------------------------
    rows = []
    lv = Level(t_count=0)
    for k in range(N + 1):
        tg = list(range(k))
        for method in ("paulice", "hand"):
            rows.append(row(lv, tg, lv.evaluate(tg, method=method), method=method))
    pd.DataFrame(rows).to_csv(OUT / "paulice_vs_hand.csv", index=False)

    # ------------------------------------------------------------------------------
    # E3  Noise-strength sweep: when do checks pay for themselves?  Prefix targets 0..k-1.
    # ------------------------------------------------------------------------------
    rows = []
    for p2 in (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03):
        lv = Level(t_count=0, p2=p2)
        timed(f"E3 p2={p2}")
        for k in range(N + 1):
            tg = list(range(k))
            rows.append(row(lv, tg, lv.evaluate(tg)))
    pd.DataFrame(rows).to_csv(OUT / "noise_sweep.csv", index=False)

    # ------------------------------------------------------------------------------
    # E4  Depth sweep: deeper circuits accumulate more errors -> checks matter more.
    # ------------------------------------------------------------------------------
    rows = []
    for d in (1, 2, 3, 5, 8, 12):
        lv = Level(t_count=0, depth=d)
        timed(f"E4 depth={d}")
        for k in range(N + 1):
            tg = list(range(k))
            rows.append(row(lv, tg, lv.evaluate(tg)))
    pd.DataFrame(rows).to_csv(OUT / "depth_sweep.csv", index=False)

    # ------------------------------------------------------------------------------
    # E5  T-count sweep (prefix targets, hand-made checks) incl. how many wires can be
    #     covered at all.
    # ------------------------------------------------------------------------------
    rows = []
    for t in (0, 2, 4, 6, 8, 10):
        lv = Level(t_count=t)
        timed(f"E5 T={t}")
        method = "hand"  # use the same hand-made machinery for every T so T=0 is comparable
        for k in range(N + 1):
            tg = list(range(k))
            rows.append(row(lv, tg, lv.evaluate(tg, method=method), pool=len(lv.pool)))
    pd.DataFrame(rows).to_csv(OUT / "t_sweep.csv", index=False)

    # ------------------------------------------------------------------------------
    # E6  Reproducibility: the headline configuration with 5 different simulator seeds.
    # ------------------------------------------------------------------------------
    rows = []
    lv = Level(t_count=0)
    for seed in range(5):
        for tg in ([], [0, 1, 2, 3]):
            rows.append(row(lv, tg, lv.evaluate(tg, seed=seed), sim_seed=seed))
    pd.DataFrame(rows).to_csv(OUT / "seed_repeat.csv", index=False)

    # ------------------------------------------------------------------------------
    # E7  Noiseless validity check: every hand-made check must flag nothing without noise.
    # ------------------------------------------------------------------------------
    rows = []
    for t in (0, 2, 4, 8):
        lv = Level(t_count=t)
        for tg in ALL_SUBSETS:
            chk = lv.build(tg, method="hand")
            r0 = run_checked(chk, lv.ideal, 0.0, 0.0, shots=500)
            rows.append(dict(t_count=t, targets="".join(map(str, tg)) or "-",
                             checks=chk.n_checks, kept_noiseless=r0.kept,
                             fidelity_noiseless=r0.fidelity_post))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "validity_noiseless.csv", index=False)
    assert (df.kept_noiseless == 1.0).all(), "a hand-made check flagged a noiseless shot!"
    timed("done")


if __name__ == "__main__":
    main()
