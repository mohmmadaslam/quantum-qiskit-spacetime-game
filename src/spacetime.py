"""Core physics for the Spacetime Checks game.

Everything the game shows (fidelity, kept-shot fraction, cost) is computed here from
Qiskit / Aer / qiskit-paulice results.  Nothing is hand-set.

Pipeline (matches the challenge brief):
    payload circuit -> spacetime Pauli checks -> ancilla readout -> post-selection
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from itertools import product

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, transpile
from qiskit.quantum_info import Clifford, Pauli, StabilizerState, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel as AerNoiseModel
from qiskit_aer.noise import ReadoutError, depolarizing_error
from qiskit_paulice import NoiseModel as PauliceNoiseModel
from qiskit_paulice import add_pauli_checks

# --------------------------------------------------------------------------------------
# Shared noise assumption.  The brief says "use the noise model announced by the
# organizers" and gives NoiseModel(gate_noise=0.01) as an example, so that is our default:
# every two-qubit gate suffers a Pauli error with total probability P2, spread uniformly
# over the 15 non-identity two-qubit Paulis, and every measurement flips with prob PRO.
# The SAME numbers are given to qiskit-paulice (to pick checks) and to Aer (to simulate).
# --------------------------------------------------------------------------------------
P2_DEFAULT = 0.01
PRO_DEFAULT = 0.005
SHOTS_DEFAULT = 20000


def aer_noise(p2: float = P2_DEFAULT, p_readout: float = PRO_DEFAULT) -> AerNoiseModel:
    nm = AerNoiseModel()
    # Aer's depolarizing parameter lambda gives total Pauli-error prob = lambda * 15/16
    nm.add_all_qubit_quantum_error(depolarizing_error(p2 * 16 / 15, 2), ["cx"])
    if p_readout > 0:
        nm.add_all_qubit_readout_error(
            ReadoutError([[1 - p_readout, p_readout], [p_readout, 1 - p_readout]])
        )
    return nm


def paulice_noise(p2: float = P2_DEFAULT, p_readout: float = PRO_DEFAULT) -> PauliceNoiseModel:
    return PauliceNoiseModel(gate_noise=p2, readout_noise=p_readout)


# --------------------------------------------------------------------------------------
# Payload circuits
# --------------------------------------------------------------------------------------
def clifford_payload(n: int, depth: int, seed: int = 7) -> QuantumCircuit:
    """Random brickwork Clifford circuit (h, s, sx singles + cx), no measurement."""
    rng = np.random.default_rng(seed)
    qc = QuantumCircuit(n)
    for layer in range(depth):
        for q in range(n):
            [qc.h, qc.s, qc.sx][int(rng.integers(3))](q)
        for q in range(layer % 2, n - 1, 2):
            qc.cx(q, q + 1)
    return qc


def echo_payload(n: int, depth: int, seed: int = 7, flip: tuple[int, ...] | None = None,
                 t_pairs: int = 0, t_seed: int = 1) -> QuantumCircuit:
    """Echo (mirror) payload  X-pattern -> C -> C^dagger  (encode a bitstring, compute, uncompute).

    The ideal output is ONE known bitstring (the X pattern), so any deviation is noise and the
    fidelity is sensitive to it.  (A random Clifford spreads its output uniformly and would hide
    the noise.)  Total two-qubit-gate layers = 2 * depth.

    With t_pairs > 0 the first half C contains that many T gates and the mirror half contains
    the matching T^dagger gates, so the circuit has 2 * t_pairs non-Clifford gates but its ideal
    output is STILL the same deterministic bitstring.  This keeps the fidelity comparable
    across T counts: T gates then change which checks are *valid*, not the reference.
    """
    c = clifford_payload(n, depth, seed)
    if t_pairs:
        c = t_dope(c, t_pairs, t_seed)
    if flip is None:
        flip = tuple(range(0, n, 2))  # encode input bitstring '...010101'
    qc = QuantumCircuit(n)
    for q in flip:
        qc.x(q)
    qc.compose(c, inplace=True)
    qc.compose(c.inverse(), inplace=True)
    return qc


def t_dope(clifford: QuantumCircuit, t_count: int, seed: int = 1) -> QuantumCircuit:
    """Insert `t_count` T gates after single-qubit gates in the middle 60% of the circuit.

    (A T gate right after |0> does nothing, so we place them where they actually act.)
    The T positions are nested: the T=k circuit's T gates are a subset of the T=k+1 circuit's.
    """
    if t_count == 0:
        return clifford.copy()
    rng = np.random.default_rng(seed)
    n_ops = len(clifford.data)
    lo, hi = int(0.2 * n_ops), int(0.8 * n_ops)
    cands = [i for i in range(lo, hi) if len(clifford.data[i].qubits) == 1
             and clifford.data[i].operation.name in ("h", "sx", "sxdg")]
    slots = {int(i) for i in rng.permutation(cands)[:t_count]}
    out = QuantumCircuit(clifford.num_qubits)
    for i, inst in enumerate(clifford.data):
        out.append(inst.operation, inst.qubits)
        if i in slots:
            out.t(clifford.find_bit(inst.qubits[0]).index)
    return out


def with_measure(qc: QuantumCircuit) -> QuantumCircuit:
    out = qc.copy()
    out.measure_all()
    return out


# --------------------------------------------------------------------------------------
# Ideal reference + fidelity
# --------------------------------------------------------------------------------------
def ideal_distribution(payload: QuantumCircuit) -> dict[str, float]:
    """Exact ideal output distribution of the payload.

    Clifford circuit -> stabilizer simulation.  T-doped circuit -> statevector simulation
    (fine for the small widths used here).
    """
    if not ({"t", "tdg"} & set(payload.count_ops())):
        probs = StabilizerState(Clifford(payload)).probabilities_dict()
    else:
        probs = Statevector(payload).probabilities_dict()
    return {str(k): float(v) for k, v in probs.items() if v > 1e-12}


def hellinger_fidelity(p: dict[str, float], q: dict[str, float]) -> float:
    """Classical (Bhattacharyya / Hellinger) fidelity F = (sum_x sqrt(p(x) q(x)))^2."""
    keys = set(p) | set(q)
    return float(sum(np.sqrt(p.get(k, 0.0) * q.get(k, 0.0)) for k in keys) ** 2)


# --------------------------------------------------------------------------------------
# Hand-made checks (needed for T-doped payloads: qiskit-paulice only accepts Cliffords)
# --------------------------------------------------------------------------------------
def _segments(payload: QuantumCircuit):
    """Split payload at T gates -> list of ('cliff', QuantumCircuit) / ('t', qubit)."""
    n = payload.num_qubits
    segs, cur = [], QuantumCircuit(n)
    for inst in payload.data:
        if inst.operation.name in ("t", "tdg"):
            segs.append(("cliff", cur))
            segs.append(("t", payload.find_bit(inst.qubits[0]).index))
            cur = QuantumCircuit(n)
        else:
            cur.append(inst.operation, inst.qubits)
    segs.append(("cliff", cur))
    return segs


def propagate_pauli(payload: QuantumCircuit, left: Pauli):
    """Push `left` through the payload:  R = U L U^dagger.

    Returns None when a T gate sees X or Y on its qubit: T X T^dagger is not a Pauli, so the
    check would not close.  A Pauli that acts as I or Z there passes through (T is diagonal).
    """
    p = left
    for kind, item in _segments(payload):
        if kind == "cliff":
            if len(item.data):
                p = p.evolve(Clifford(item), frame="s")
        elif p.to_label().lstrip("-i")[::-1][item] in "XY":  # label is little-endian
            return None
    return p


@dataclass(frozen=True)
class HandCheck:
    left: str  # Pauli applied (controlled) before the payload
    right: str  # Pauli applied (controlled) after the payload
    sign: int  # +1 / -1 : U L U^dagger = sign * R

    @property
    def cost(self) -> int:  # number of controlled-Pauli gates it needs
        wl = 0 if set(self.left) <= {"I", "Z"} else sum(c != "I" for c in self.left)
        return wl + sum(c != "I" for c in self.right)

    def support(self) -> set[int]:
        """Qubits that the check touches (either end)."""
        s = {q for q, c in enumerate(self.right[::-1]) if c != "I"}
        s |= {q for q, c in enumerate(self.left[::-1]) if c != "I"}
        return s


def check_pool(payload: QuantumCircuit) -> list[HandCheck]:
    """All valid hand checks (any left Pauli whose propagation never meets X/Y at a T gate)."""
    n = payload.num_qubits
    pool = []
    for labels in product("IXYZ", repeat=n):
        lab = "".join(labels)
        if set(lab) == {"I"}:
            continue
        R = propagate_pauli(payload, Pauli(lab))
        if R is None:
            continue
        rl = R.to_label()
        pool.append(HandCheck(lab, rl.lstrip("-"), -1 if rl.startswith("-") else 1))
    pool.sort(key=lambda c: (c.cost, c.left, c.right))
    return pool


def _symplectic(lab: str, rl: str) -> np.ndarray:
    return np.array(
        [int(c in "XY") for c in lab[::-1]] + [int(c in "ZY") for c in lab[::-1]]
        + [int(c in "XY") for c in rl[::-1]] + [int(c in "ZY") for c in rl[::-1]],
        dtype=np.uint8,
    )


def _gf2_rank(rows: list[np.ndarray]) -> int:
    m = np.array(rows, dtype=np.uint8)
    rank = 0
    for c in range(m.shape[1]):
        piv = next((r for r in range(rank, m.shape[0]) if m[r, c]), None)
        if piv is None:
            continue
        m[[rank, piv]] = m[[piv, rank]]
        for r in range(m.shape[0]):
            if r != rank and m[r, c]:
                m[r] ^= m[rank]
        rank += 1
    return rank


def pick_hand_checks(pool: list[HandCheck], targets: list[int]) -> list[HandCheck | None]:
    """For each target wire, pick the cheapest valid check that touches that wire, commutes with
    the checks already picked and is independent of them.  None if no such check exists."""
    picked: list[HandCheck | None] = []
    rows: list[np.ndarray] = []
    for q in targets:
        chosen = None
        for c in pool:
            if q not in c.support():
                continue
            if any(p is not None and not Pauli(c.left).commutes(Pauli(p.left)) for p in picked):
                continue
            v = _symplectic(c.left, c.right)
            if not v.any() or _gf2_rank(rows + [v]) != len(rows) + 1:
                continue
            chosen = c
            rows.append(v)
            break
        picked.append(chosen)
    return picked


def _controlled_pauli(qc: QuantumCircuit, anc: int, label: str):
    """Controlled-P with the ancilla as control, built only from the noisy native cx."""
    for q, c in enumerate(label[::-1]):
        if c == "X":
            qc.cx(anc, q)
        elif c == "Z":
            qc.h(q)
            qc.cx(anc, q)
            qc.h(q)
        elif c == "Y":
            qc.sdg(q)
            qc.cx(anc, q)
            qc.s(q)


@dataclass
class Checked:
    """A payload with checks: measured circuit + what is needed for post-selection."""

    circuit: QuantumCircuit  # clbits: payload bits first, then check bits
    n_payload: int
    n_checks: int
    source: str = "none"  # 'paulice' | 'hand' | 'none'
    predicted_cost: float | None = None  # paulice gamma (lower is better)
    syndrome_fn: object = None
    targets: tuple[int, ...] = ()

    def syndrome(self, bits: str) -> np.ndarray:
        if self.syndrome_fn is not None:
            return self.syndrome_fn(bits)
        s = bits.replace(" ", "")
        return np.array([int(b) for b in s[: len(s) - self.n_payload]], dtype=np.uint8)


def build_hand_checked(payload: QuantumCircuit, checks: list[HandCheck]) -> Checked:
    n, k = payload.num_qubits, len(checks)
    qc = QuantumCircuit(n + k)
    qc.add_register(ClassicalRegister(n, "meas"))
    if k:
        qc.add_register(ClassicalRegister(k, "checks_c"))
    for i, c in enumerate(checks):  # ancilla in |+>, controlled-L before the payload
        a = n + i
        qc.h(a)
        if c.sign == -1:
            qc.z(a)
        if not set(c.left) <= {"I", "Z"}:  # Z-only L acts trivially on |0...0>
            _controlled_pauli(qc, a, c.left)
    qc.compose(payload, qubits=list(range(n)), inplace=True)
    for i, c in enumerate(checks):  # controlled-R after the payload, then read the ancilla in X
        a = n + i
        _controlled_pauli(qc, a, c.right)
        qc.h(a)
    qc.measure(range(n), range(n))
    for i in range(k):
        qc.measure(n + i, n + i)
    return Checked(qc, n, k, "hand" if k else "none")


def build_paulice_checked(clifford: QuantumCircuit, targets: list[int], p2: float, pro: float,
                          seed: int = 1) -> Checked:
    """qiskit-paulice on a Clifford payload, one check per chosen target wire."""
    n = clifford.num_qubits
    if not targets:
        return Checked(with_measure(clifford), n, 0, "none")
    res = add_pauli_checks(with_measure(clifford), list(targets), paulice_noise(p2, pro),
                           cost="gamma", seed=seed)
    cc = res[-1]
    return Checked(cc.circuit, n, len(cc.check_qubits), "paulice", cc.cost,
                   cc.get_postselection_method(), tuple(targets))


# --------------------------------------------------------------------------------------
# Running and scoring
# --------------------------------------------------------------------------------------
@dataclass
class Result:
    fidelity_raw: float  # fidelity of ALL shots of the (checked) circuit, no post-selection
    fidelity_post: float  # fidelity after discarding flagged shots
    fidelity_post_err: float  # bootstrap 1-sigma on fidelity_post
    kept: float  # kept-shot fraction
    kept_err: float  # binomial 1-sigma
    ancillas: int
    two_qubit_gates: int
    depth: int
    shots: int
    n_checks: int = 0
    predicted_cost: float | None = None
    source: str = "none"
    targets: tuple = ()
    extra: dict = field(default_factory=dict)

    @property
    def efficiency(self) -> float:
        """Precision per raw shot  =  kept * F_post^2.

        To estimate an observable whose noisy value is attenuated by roughly F, the variance
        of the estimate scales like 1/(kept * N * F^2).  So kept*F^2 is (up to a constant) the
        inverse of the number of raw shots you need for a fixed precision.  It rewards both
        cleaner shots and not throwing too many shots away.
        """
        return self.fidelity_post ** 2 * self.kept


def _two_q_count(qc: QuantumCircuit) -> int:
    return sum(1 for i in qc.data if len(i.qubits) == 2)


def run_checked(chk: Checked, ideal: dict[str, float], p2=P2_DEFAULT, pro=PRO_DEFAULT,
                shots: int = SHOTS_DEFAULT, seed: int = 123, n_boot: int = 100) -> Result:
    """Run noisy on Aer, post-select on the syndrome, score against the ideal distribution."""
    sim = AerSimulator(noise_model=aer_noise(p2, pro), seed_simulator=seed)
    tqc = transpile(chk.circuit, sim, optimization_level=0, seed_transpiler=0)
    counts = sim.run(tqc, shots=shots).result().get_counts()
    n = chk.n_payload
    kept_c: dict[str, int] = {}
    all_c: dict[str, int] = {}
    for bits, c in counts.items():
        pb = bits.replace(" ", "")[-n:]
        all_c[pb] = all_c.get(pb, 0) + c
        if chk.n_checks == 0 or not chk.syndrome(bits).any():
            kept_c[pb] = kept_c.get(pb, 0) + c
    n_kept = sum(kept_c.values())
    kept = n_kept / shots
    f_raw = hellinger_fidelity({k: v / shots for k, v in all_c.items()}, ideal)
    f_post, f_err = 0.0, 0.0
    if n_kept:
        keys = list(kept_c)
        pk = np.array([kept_c[k] for k in keys], float) / n_kept
        f_post = hellinger_fidelity(dict(zip(keys, pk)), ideal)
        rng = np.random.default_rng(seed)
        boots = []
        for _ in range(n_boot):
            s = rng.multinomial(n_kept, pk) / n_kept
            boots.append(hellinger_fidelity(dict(zip(keys, s)), ideal))
        f_err = float(np.std(boots))
    return Result(f_raw, f_post, f_err, kept, float(np.sqrt(kept * (1 - kept) / shots)),
                  chk.n_checks, _two_q_count(chk.circuit), chk.circuit.depth(), shots,
                  chk.n_checks, chk.predicted_cost, chk.source, chk.targets)


# --------------------------------------------------------------------------------------
# A game level = payload + noise.  evaluate(targets) is the single entry point the game,
# the notebook and the experiments all use.
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Level:
    n: int = 6
    depth: int = 5
    t_count: int = 0  # total non-Clifford gates (T and T^dagger); must be even
    p2: float = P2_DEFAULT
    pro: float = PRO_DEFAULT
    seed: int = 7
    t_seed: int = 1

    @cached_property
    def clifford(self) -> QuantumCircuit:
        return echo_payload(self.n, self.depth, self.seed)

    @cached_property
    def payload(self) -> QuantumCircuit:
        return echo_payload(self.n, self.depth, self.seed, t_pairs=self.t_count // 2,
                            t_seed=self.t_seed)

    @cached_property
    def ideal(self) -> dict[str, float]:
        return ideal_distribution(self.payload)

    @cached_property
    def pool(self) -> list[HandCheck]:
        return check_pool(self.payload)

    @property
    def method(self) -> str:
        return "paulice" if self.t_count == 0 else "hand"

    def build(self, targets: list[int], method: str | None = None) -> Checked:
        targets = sorted(targets)
        method = method or self.method
        if method == "paulice":
            return build_paulice_checked(self.clifford, targets, self.p2, self.pro)
        picks = pick_hand_checks(self.pool, targets)
        chk = build_hand_checked(self.payload, [c for c in picks if c is not None])
        chk.targets = tuple(t for t, c in zip(targets, picks) if c is not None)
        return chk

    def evaluate(self, targets: list[int], method: str | None = None,
                 shots: int = SHOTS_DEFAULT, seed: int = 123) -> Result:
        chk = self.build(targets, method)
        return run_checked(chk, self.ideal, self.p2, self.pro, shots, seed)
