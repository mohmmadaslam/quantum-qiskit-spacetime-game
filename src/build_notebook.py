"""Build notebook.ipynb (then execute it with nbconvert).  Keeps the notebook source in one
readable Python file instead of hand-editing JSON."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []


def md(s):
    C.append(nbf.v4.new_markdown_cell(s.strip("\n")))


def code(s):
    C.append(nbf.v4.new_code_cell(s.strip("\n")))


md("""
# Detect or Discard: error detection with spacetime codes, as a game

**Qiskit Fall Fest 2026 Hackathon, Advanced Track.**
This notebook contains the physics behind the game. It follows the six required steps of the brief
and reproduces every number used in the game and the report.

| Brief step | Section |
|---|---|
| 1. Build the payloads (Clifford + T-doped family) | §1 |
| 2. Fix the reference and the fidelity measure | §2 |
| 3. Add checks with qiskit-paulice | §3 |
| 4. Run under noise | §4 |
| 5. Measure | §5 |
| 6. Compare, find the best trade-off | §6 – §8 |

The game itself is in `game/app.py` (`streamlit run game/app.py`). It reads the same numbers.

**Reproducibility.** Everything is seeded. Section 5 recomputes live (about a minute); the large
sweeps in §6 are read from `results/*.csv`, written by `python src/run_experiments.py`.
""")

code("""
import sys, platform
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import qiskit, qiskit_aer, qiskit_paulice
from importlib.metadata import version
from spacetime import *
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .25})
print("python", platform.python_version(), "| qiskit", qiskit.__version__,
      "| aer", qiskit_aer.__version__, "| qiskit-paulice", version("qiskit-paulice"))
RES = ROOT / "results"
""")

md("""
## 1 · Payload circuits

**Design choice: an echo (mirror) circuit.** The payload encodes a bitstring, applies a random
Clifford circuit `C`, then its inverse `C†`. In the ideal case the output is *one known bitstring*.
That makes the fidelity sensitive to noise. (A random Clifford spreads its output uniformly over all
bitstrings, so a noisy and an ideal run look the same under a distribution fidelity; we tried it and
it gave F = 0.999 at 1% noise.)

* **Width** 6 data qubits. **Depth**: `C` has 5 brickwork layers, so the echo is 10 layers; 26 CNOTs.
* **T-doped family.** `t_count` T gates are inserted in the middle of `C`; the mirror half contains
  the matching T†. The circuit has `t_count` non-Clifford gates (`t_count` = 0, 2, 4, 8, ...) but its
  ideal output is *still* the same bitstring. So the **fidelity stays comparable across T counts**,
  and T gates act on the problem the way the brief describes: they limit which checks are valid.
""")

code("""
rows = []
for t in (0, 2, 4, 8):
    lv = Level(t_count=t)
    ops = lv.payload.count_ops()
    rows.append(dict(t_count=t, width=lv.n, depth=lv.payload.depth(),
                     cnots=ops.get("cx", 0), T=ops.get("t", 0), Tdg=ops.get("tdg", 0),
                     ideal_outputs=len(lv.ideal), ideal_bitstring=max(lv.ideal, key=lv.ideal.get)))
pd.DataFrame(rows)
""")

code("""
from board import draw_board
fig = draw_board(Level(t_count=4).payload, [], dark=False, figsize=(11, 3.2))
plt.show()
""")

md("""
## 2 · Reference and fidelity measure

* **Clifford payload:** the ideal distribution is computed by **stabilizer simulation**
  (`StabilizerState(Clifford(circuit)).probabilities_dict()`).
* **T-doped payload:** by **statevector simulation** (`Statevector(circuit)`), fine for 6 qubits.
* **Fidelity:** the classical (Hellinger) fidelity `F = (Σ_x √(p(x) q(x)))²` between the measured
  distribution `q` over the data bits and the ideal `p`.
  *Justification:* the ideal output is a single bitstring, so `F` is the probability of getting it;
  the same formula also covers the T-doped case without changes. Post-selection compares only
  the kept shots. Uncertainty: bootstrap (multinomial resampling of the kept counts), 100 resamples.
* **Shots:** 20 000 per configuration. A post-selected fidelity is always reported with its
  kept-shot fraction.
""")

code("""
lv = Level(t_count=4)
print("ideal (T-doped, statevector):", {k: round(v, 6) for k, v in lv.ideal.items()})
print("ideal (Clifford, stabilizer):", Level(t_count=0).ideal)
""")

md("""
## 3 · Checks with qiskit-paulice

`add_pauli_checks(circuit, target_qubits, noise_model, ...)` returns one circuit per number of
checks: the bare circuit, then one check per target qubit, in order. The checks are chosen to minimise
a cost (the default, `gamma`, is the inverse of the fraction of noise left undetected, so *lower is
better*). It needs a **Clifford** circuit. Given the T-doped payload it raises an error, shown below.
""")

code("""
lv0 = Level(t_count=0)
res = add_pauli_checks(with_measure(lv0.clifford), list(range(6)), paulice_noise(0.01, 0.005),
                       cost="gamma", seed=1)
pd.DataFrame([dict(checks=len(r.check_qubits), two_qubit_gates=sum(len(i.qubits) == 2 for i in r.circuit.data),
                   depth=r.circuit.depth(), qubits=r.circuit.num_qubits,
                   gamma=r.cost, support=r.check_support) for r in res])
""")

code("""
try:
    add_pauli_checks(with_measure(Level(t_count=4).payload), [0, 1], paulice_noise(), seed=1)
except Exception as e:
    print(type(e).__name__, "->", str(e)[:400])
""")

md("""
### Hand-made checks for the T-doped circuits

A spacetime check is a pair of Paulis `(L, R)` with `U L U† = ±R`: apply controlled-`L` before the
payload, controlled-`R` after it, read the ancilla in the X basis. For a Clifford `U` every `L`
works. A **T gate** is diagonal, so it commutes with Z but turns X into `(X+Y)/√2`, which is not a
Pauli. A check is valid across a T gate only if its Pauli acts as **I or Z** there. We enumerate
all `4⁶` left Paulis, push each through the circuit (`propagate_pauli`), and keep those that never meet
X/Y at a T gate. Then each target wire gets the cheapest valid check that touches it, is independent of
the others, and **commutes** with them (anticommuting checks disturb each other even without noise;
we found this bug the hard way).

**Self-test:** with noise switched off, every check we build must flag zero shots. We verify this on all
256 (4 T-counts × 64 target subsets) configurations.
""")

code("""
pool_sizes = {t: len(Level(t_count=t).pool) for t in (0, 2, 4, 8)}
print("valid left-Paulis (out of 4^6-1 = 4095):", pool_sizes)
v = pd.read_csv(RES / "validity_noiseless.csv")
print("noiseless self-test: configurations =", len(v), "| all shots kept:", bool((v.kept_noiseless == 1).all()),
      "| min noiseless fidelity:", round(v.fidelity_noiseless.min(), 4))
""")

md("""
## 4 · Noise

The brief says to use the organizers' model; its example is `NoiseModel(gate_noise=0.01)`. We use
exactly that **for both paulice (to pick checks) and Aer (to simulate)**:

* every CNOT: a Pauli error with total probability `p2 = 1%`, spread uniformly over the 15 non-identity
  two-qubit Paulis (`depolarizing_error(p2·16/15, 2)` in Aer);
* every measurement flips with probability 0.5%;
* single-qubit gates (incl. T) are noiseless, as in paulice's own model.

> If the organizers publish a different model, change `P2_DEFAULT` / `PRO_DEFAULT` in `src/spacetime.py`
> and rerun `python src/run_experiments.py`.

All the controlled-Pauli gates of the checks are built from the *same noisy CNOT*, so checks pay for
their own gates.
""")

md("""
## 5 · Measure (recomputed live)

For each configuration: fidelity without checks, post-selected fidelity, kept fraction and cost
(ancillas, two-qubit gates, depth). Targets are wires `0..k-1`.
""")

code("""
def sweep(level, method=None, ks=range(7)):
    out = []
    for k in ks:
        r = level.evaluate(list(range(k)), method=method)
        out.append(dict(T=level.t_count, checks=r.n_checks, F_raw=r.fidelity_raw,
                        F_post=r.fidelity_post, F_err=r.fidelity_post_err, kept=r.kept,
                        efficiency=r.efficiency, ancillas=r.ancillas, cx=r.two_qubit_gates, depth=r.depth))
    return pd.DataFrame(out)

live0 = sweep(Level(t_count=0)); live0.round(4)
""")

code("""
live4 = sweep(Level(t_count=4)); live4.round(4)
""")

md("""
**Clifford vs T-doped, at equal check count.** (Prefix targets, hand-made checks for both so that
the machinery is identical.)
""")

code("""
t = pd.read_csv(RES / "t_sweep.csv", dtype={"targets": str})
piv = t.pivot_table(index="checks", columns="t_count", values="fidelity_post")
piv.round(3)
""")

md("""
## 6 · Compare

### 6.1 Fidelity and kept fraction vs number of checks
""")

code("""
n = pd.read_csv(RES / "noise_sweep.csv", dtype={"targets": str})
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for p2, g in n.groupby("p2"):
    ax[0].plot(g.checks, g.fidelity_post, "-o", ms=4, label=f"p2={p2:.2%}")
    ax[1].plot(g.checks, g.kept, "-o", ms=4)
    ax[2].plot(g.checks, g.efficiency, "-o", ms=4)
ax[0].set(xlabel="checks", ylabel="post-selected fidelity", title="cleaner...")
ax[1].set(xlabel="checks", ylabel="kept fraction", title="...but fewer shots")
ax[2].set(xlabel="checks", ylabel="efficiency kept·F²", title="trade-off")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.show()
""")

md("""
**Reading it.** Post-selected fidelity rises with every check and saturates; the kept fraction falls
steadily, because every check adds noisy CNOTs that flag extra shots. Their product has an interior
maximum: that is the best trade-off. The noisier the hardware, the more checks are worth it and the more
the optimum moves right. On quiet hardware it can be zero checks.
""")

md("""
### 6.2 Depth
""")

code("""
d = pd.read_csv(RES / "depth_sweep.csv", dtype={"targets": str})
base = d[d.checks == 0].set_index("depth")
best = d.loc[d.groupby("depth").efficiency.idxmax()].set_index("depth")
pd.DataFrame({"2q gates": base.two_qubit_gates, "F no checks": base.fidelity_post,
              "best #checks": best.checks, "F best": best.fidelity_post, "kept best": best.kept,
              "eff no checks": base.efficiency, "eff best": best.efficiency,
              "gain": best.efficiency / base.efficiency}).round(3)
""")

md("""
### 6.3 T count
""")

code("""
fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
for tc, g in t.groupby("t_count"):
    ax[0].plot(g.checks, g.fidelity_post, "-o", ms=4, label=f"{tc} T/T† gates")
    ax[1].plot(g.checks, g.kept, "-o", ms=4)
    ax[2].plot(g.checks, g.efficiency, "-o", ms=4)
ax[0].set(xlabel="requested checks (wires 0..k-1)", ylabel="post-selected fidelity")
ax[1].set(xlabel="requested checks", ylabel="kept fraction")
ax[2].set(xlabel="requested checks", ylabel="efficiency")
ax[0].legend(fontsize=8); plt.tight_layout(); plt.show()
print("valid left Paulis in the pool:", t.drop_duplicates("t_count")[["t_count", "pool"]].set_index("t_count").pool.to_dict())
""")

md("""
**Reading it.** Without checks the fidelity is the same for every T count (the T gates are noiseless in
the model, and mirrored). With checks, more T gates means fewer valid checks (the pool of valid Paulis
halves with every extra T/T† pair), so the best reachable fidelity falls and the best number of checks
falls too.

### 6.4 Check location
""")

code("""
L = pd.read_csv(RES / "landscape.csv", dtype={"targets": str})
L0 = L[(L.t_count == 0) & (L.p2 == 0.01) & (L.depth == 5)]
g = L0.groupby("checks").efficiency.agg(["min", "median", "max"])
g["spread"] = g["max"] - g["min"]
print(g.round(3))
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.scatter(L0.checks + np.random.default_rng(0).uniform(-.12, .12, len(L0)), L0.efficiency, s=14, alpha=.6)
ax.set(xlabel="checks", ylabel="efficiency kept·F²", title="Same number of checks, different wires")
plt.show()
""")

code("""
k3 = L0[L0.checks == 3].sort_values("efficiency")
print("worst 3-check placement:", k3.iloc[0][["targets", "fidelity_post", "kept", "efficiency"]].to_dict())
print("best  3-check placement:", k3.iloc[-1][["targets", "fidelity_post", "kept", "efficiency"]].to_dict())
""")

md("""
## 7 · Best trade-off
""")

code("""
rows = []
for (t_, p2, dep), g in L.groupby(["t_count", "p2", "depth"]):
    b = g.loc[g.efficiency.idxmax()]; z = g[g.targets == "-"].iloc[0]
    rows.append(dict(T=t_, p2=p2, depth=dep, best_wires=b.targets, checks=int(b.checks),
                     F_none=z.fidelity_post, F_best=b.fidelity_post, kept_best=b.kept,
                     eff_none=z.efficiency, eff_best=b.efficiency, gain=b.efficiency / z.efficiency))
best_tbl = pd.DataFrame(rows).round(3); best_tbl
""")

md("""
## 8 · Reproducibility and statistical honesty
""")

code("""
s = pd.read_csv(RES / "seed_repeat.csv", dtype={"targets": str})
s.groupby("targets")[["fidelity_post", "kept", "efficiency"]].agg(["mean", "std"]).round(4)
""")

md("""
The simulator-seed to seed spread (above) is the honest error on every number in the tables; the
`F_err` columns are bootstrap errors for a *fixed* run. Differences smaller than about 0.005 in efficiency
should not be read as real.

### Limitations
* One payload family (6 qubits, one random Clifford circuit, one seed) and one noise family. The shape of
  the trade-off is the result, not the exact numbers.
* The T gates are noiseless and mirrored by design. Real T-doped circuits do not echo, but then the
  ideal distribution is broad and hard to separate from noise with any distribution fidelity.
* Hand-made checks are selected greedily by gate cost, not optimised like paulice's.
* We use `kept·F²` as the score; that is a modelling choice (inverse raw-shot cost for a fixed precision),
  explained in `report.pdf`.
""")

nb["cells"] = C
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
out = Path(__file__).resolve().parent.parent / "notebook.ipynb"
nbf.write(nb, out)
print("wrote", out)
