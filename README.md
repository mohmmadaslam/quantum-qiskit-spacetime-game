# Detect or Discard


Live: https://quantum-qiskit-spacetime-game.streamlit.app/

**Qiskit Fall Fest 2026 Hackathon, Advanced Track: gamifying error detection with spacetime codes.**

A game in which you protect a 6-qubit circuit by attaching spacetime-Pauli-check ancillas to chosen wires.
Each placement is run through the real pipeline (payload, checks from `qiskit-paulice`, ancilla readout,
post-selection) on Qiskit Aer, and the score comes from the resulting fidelity and kept-shot fraction.
Nothing is hand-set.

| Deliverable | File |
|---|---|
| Code (physics core, experiments, game) | `src/`, `game/` |
| Recorded presentation (7 min) | `presentation_v1.mp4` |
| Written report | `report.pdf` (source: `report/report.html`) |
| Reproducible notebook (the six steps of the brief) | `notebook.ipynb` |

## Headline result (Clifford echo circuit, 1% two-qubit error)

| | fidelity (post-selected) | shots kept | efficiency (kept x F^2) |
|---|---|---|---|
| no checks | 0.771 | 100% | 0.595 |
| **best placement: wires 2, 3** | **0.902** | **81.0%** | **0.658 (+10.7%)** |
| worst two-check placement | 0.823 | 88.0% | 0.596 |
| checks on all six wires | 0.946 | 65.9% | 0.590 |

* Placement matters as much as the number of checks; more checks is not always better.
* Gain grows with noise (+2% at 0.25% error to +12% at 3%) and depth (none at depth <= 3, +36% at depth 12).
* Every T/T-dagger pair halves the pool of valid checks (4095, 2047, 1023, 255, ...): detection gets harder as circuits become less Clifford.

See `report.pdf` for method, all results, validity checks and limitations.

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run game/app.py                 # play the game (uses precomputed Qiskit results; toggle "live" to rerun on Aer)
jupyter nbconvert --to notebook --execute --inplace notebook.ipynb   # ~25 s, reproduces the numbers
python src/run_experiments.py             # regenerate every results/*.csv (~5 min)
python src/make_boards.py && python src/make_figures.py              # board layouts + report figures
python -m pytest -q                       # tests
```

Tested with Python 3.13, Qiskit 2.5.2, Qiskit Aer 0.17.2, qiskit-paulice 0.2.1. All randomness is seeded.

## Layout

```
src/spacetime.py        payloads, ideal reference, paulice + hand-made checks, noise, run & score
src/run_experiments.py  all sweeps -> results/*.csv (landscape of all 64 placements x 8 levels, noise, depth, T count, ...)
src/levels.py, board.py game levels and circuit drawing
src/live_eval.py        live Aer evaluation (run in a subprocess by the game)
game/app.py             Streamlit game
notebook.ipynb          the six required steps, executed
report/report.html      report source -> report.pdf
tests/                  physics sanity tests (including: hand-made checks flag nothing without noise)
```

## Honest notes

* Noise is the brief's example model (two-qubit depolarising 1% + 0.5% readout; single-qubit gates noiseless).
  If the organizers announce a different model, change `P2_DEFAULT` / `PRO_DEFAULT` in `src/spacetime.py` and rerun the experiments.
* The T-doped circuits are *echo* circuits (T in the first half, T-dagger in the mirror) so that the ideal output stays one known bitstring and fidelity is comparable across T counts.
* paulice only accepts Clifford circuits, so the T levels use hand-made checks (valid across a T gate only where their Pauli is I or Z). In a like-for-like comparison on the Clifford circuit, cheap hand-made checks beat paulice's *default* checks on efficiency (fewer CNOTs); we did not tune paulice.
* `kept x F^2` as the score is our modelling choice (inverse raw-shot cost for a fixed precision). Fidelity and kept fraction are always shown separately.
* Small, single payload family, simulation only; absolute numbers will change with circuit and noise, the pattern is the result.
