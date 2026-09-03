"""Independent Clifford/Pauli propagation, implemented from conjugation RULES.

This module deliberately uses NO dense matrices.  A Pauli string is stored as
a pair of bit vectors (x, z) plus a sign exponent, and CNOT/H act on it via
their textbook conjugation rules.  It therefore provides a verification route
that is logically independent of the dense simulator in
:mod:`pqec_distill.gates` / :mod:`pqec_distill.circuit`.

Conjugation rules used (U P U^dagger convention is stated per function):

    CNOT(c,t):  X_c -> X_c X_t ,  X_t -> X_t ,  Z_c -> Z_c ,  Z_t -> Z_c Z_t
    H(q):       X_q -> Z_q ,  Z_q -> X_q ,  Y_q -> -Y_q

Qubit indices follow the package convention: q1->0 .. q4->3.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["PauliString", "conjugate_by_circuit", "heisenberg_measured_observables"]


@dataclass
class PauliString:
    """A signed n-qubit Pauli string:  (-1)^sign * i^0 * prod_k X^x_k Z^z_k."""

    n: int
    x: list[int] = field(default_factory=list)
    z: list[int] = field(default_factory=list)
    sign: int = 0  # exponent of -1

    def __post_init__(self):
        if not self.x:
            self.x = [0] * self.n
        if not self.z:
            self.z = [0] * self.n

    @classmethod
    def single(cls, n: int, qubit: int, letter: str) -> "PauliString":
        ps = cls(n)
        if letter == "X":
            ps.x[qubit] = 1
        elif letter == "Z":
            ps.z[qubit] = 1
        elif letter == "Y":
            ps.x[qubit] = 1
            ps.z[qubit] = 1
        elif letter != "I":
            raise ValueError(f"bad Pauli letter {letter!r}")
        return ps

    def letters(self) -> str:
        out = []
        for k in range(self.n):
            xk, zk = self.x[k], self.z[k]
            out.append({(0, 0): "I", (1, 0): "X", (0, 1): "Z", (1, 1): "Y"}[(xk, zk)])
        return "".join(out)

    def __str__(self) -> str:
        return ("-" if self.sign % 2 else "+") + self.letters()

    def copy(self) -> "PauliString":
        return PauliString(self.n, list(self.x), list(self.z), self.sign)


def _cnot_conjugate(ps: PauliString, control: int, target: int) -> PauliString:
    """Return  CNOT P CNOT  (CNOT is its own inverse, so direction is moot)."""
    out = ps.copy()
    # X on control propagates X onto target; Z on target propagates Z onto control.
    out.x[target] ^= ps.x[control]
    out.z[control] ^= ps.z[target]
    return out


def _h_conjugate(ps: PauliString, qubit: int) -> PauliString:
    """Return  H P H : swaps X and Z on ``qubit``; Y picks up a minus sign."""
    out = ps.copy()
    if ps.x[qubit] == 1 and ps.z[qubit] == 1:  # Y -> -Y
        out.sign ^= 1
    out.x[qubit], out.z[qubit] = ps.z[qubit], ps.x[qubit]
    return out


def conjugate_by_circuit(ps: PauliString, cnot_sequence, h_qubit: int | None) -> PauliString:
    """Compute V^dagger P V for V = H_{h_qubit} @ C_last @ ... @ C_first.

    Since V^dag P V peels gates off from the last one inwards, the final H is
    undone first and the CNOTs are then traversed in reverse circuit order.
    All gates here are self-inverse, so each step is a plain conjugation.
    """
    out = ps.copy()
    if h_qubit is not None:
        out = _h_conjugate(out, h_qubit)
    for control, target in reversed(list(cnot_sequence)):
        out = _cnot_conjugate(out, control, target)
    return out


def heisenberg_measured_observables(cnot_sequence, h_qubit: int, m_qubits=(2, 3)):
    """V^dag Z_q V for each measured qubit q, as Pauli strings."""
    n = 4
    return {
        q: conjugate_by_circuit(PauliString.single(n, q, "Z"), cnot_sequence, h_qubit)
        for q in m_qubits
    }
