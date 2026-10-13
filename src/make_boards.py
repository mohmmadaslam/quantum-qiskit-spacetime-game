"""Precompute the gate layout of every game level (plain JSON), so the web app never has to
build Qiskit circuits itself."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from board import schedule  # noqa: E402
from levels import LEVELS  # noqa: E402

out = {}
for g in LEVELS:
    lv = g.physics
    ops, ncol = schedule(lv.payload)
    out[g.key] = dict(ops=ops, ncol=ncol, n=lv.n)
path = Path(__file__).resolve().parent.parent / "results" / "boards.json"
path.write_text(json.dumps(out))
print("wrote", path)
