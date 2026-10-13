"""Fast sanity tests for the physics core.  Run:  python -m pytest -q"""
import sys
from pathlib import Path

import pytest
from qiskit.quantum_info import Pauli

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from spacetime import (Level, add_pauli_checks, build_hand_checked, check_pool,  # noqa: E402
                       hellinger_fidelity, paulice_noise, pick_hand_checks, propagate_pauli,
                       run_checked, with_measure)


def test_echo_payload_has_single_ideal_bitstring():
    for t in (0, 2, 4):
        ideal = Level(t_count=t).ideal
        assert list(ideal) == ["010101"] and ideal["010101"] == pytest.approx(1.0)


def test_t_gates_restrict_the_check_pool():
    sizes = [len(Level(t_count=t).pool) for t in (0, 2, 4)]
    assert sizes == [4095, 2047, 1023]


def test_pauli_through_t_gate_rule():
    from qiskit import QuantumCircuit
    qc = QuantumCircuit(1)
    qc.t(0)
    assert propagate_pauli(qc, Pauli("Z")) is not None   # Z commutes with T
    assert propagate_pauli(qc, Pauli("X")) is None       # X does not pass through T
    assert propagate_pauli(qc, Pauli("Y")) is None


def test_paulice_rejects_non_clifford():
    with pytest.raises(Exception):
        add_pauli_checks(with_measure(Level(t_count=2).payload), [0], paulice_noise())


@pytest.mark.parametrize("t", [0, 2, 4])
def test_hand_checks_flag_nothing_without_noise(t):
    lv = Level(t_count=t)
    for targets in ([0], [0, 1], [1, 3, 5], [0, 1, 2, 3]):
        chk = lv.build(targets, method="hand")
        r = run_checked(chk, lv.ideal, 0.0, 0.0, shots=300)
        assert r.kept == 1.0 and r.fidelity_post == pytest.approx(1.0, abs=1e-6)


def test_picked_checks_commute():
    lv = Level(t_count=4)
    picks = [c for c in pick_hand_checks(lv.pool, [0, 1, 2, 3, 4, 5]) if c]
    for a in picks:
        for b in picks:
            assert Pauli(a.left).commutes(Pauli(b.left))


def test_noise_lowers_fidelity_and_checks_raise_post_selected_fidelity():
    lv = Level(t_count=0)
    base = lv.evaluate([], shots=4000)
    chk = lv.evaluate([2, 3], shots=4000)
    assert base.fidelity_post < 0.9
    assert chk.fidelity_post > base.fidelity_post and chk.kept < 1.0


def test_hellinger_fidelity_basic():
    assert hellinger_fidelity({"0": 1.0}, {"0": 1.0}) == pytest.approx(1.0)
    assert hellinger_fidelity({"0": 1.0}, {"1": 1.0}) == pytest.approx(0.0)
