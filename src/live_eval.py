"""Evaluate one placement live on Aer in a fresh process.  Used by the game's "live" mode.

Usage: python src/live_eval.py '{"n":6,"depth":5,"t_count":0,"p2":0.01,"pro":0.005,"seed":7,"t_seed":1,"targets":[2,3]}'
Prints one JSON line.  (Run in a subprocess so the web server never builds Qiskit circuits in
its worker threads.)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from spacetime import Level  # noqa: E402

a = json.loads(sys.argv[1])
targets = a.pop("targets")
lv = Level(**a)
r = lv.evaluate(targets)
print(json.dumps(dict(
    fidelity_raw=r.fidelity_raw, fidelity_post=r.fidelity_post, fidelity_post_err=r.fidelity_post_err,
    kept=r.kept, kept_err=r.kept_err, efficiency=r.efficiency, ancillas=r.ancillas,
    two_qubit_gates=r.two_qubit_gates, depth_total=r.depth, checks=r.n_checks, source=r.source,
    targets="".join(map(str, r.targets)) if r.targets else "-")))
