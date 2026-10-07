"""Ideal six-qubit distributed (LOCC) realisation of the two-copy SWAP-test /
virtual-distillation estimator, and the ordinary five-qubit SWAP test it is
supposed to reproduce.

BLIND BY CONSTRUCTION
---------------------
This module builds both circuits from gates only.  It does not import, derive
or use any analytic matrix-square expression, any Bell-diagonal recurrence, or
any previously frozen estimator formula.  Nothing here computes ``rho @ rho``.
The quantity reconstructed from the six-qubit correlators is called
``T_6Q(rho)`` and is deliberately left uninterpreted at this stage.

SETTING
-------
Two independently prepared identical bipartite states

    rho_(A1 B1)   and   rho_(A2 B2)

give the four-qubit data input ``rho (x) rho``.  Alice owns ``A1, A2``, Bob owns
``B1, B2``.  Each party holds one local ancilla, both initialised in ``|+>``.

SIX-QUBIT ORDERING (fixed throughout; wire 0 is most significant)

    [a, b, A1, B1, A2, B2]  ->  indices 0, 1, 2, 3, 4, 5

Alice applies only ``Fredkin(a; A1, A2)`` = ``Fredkin(0; 2, 4)``.
Bob applies only   ``Fredkin(b; B1, B2)`` = ``Fredkin(1; 3, 5)``.
No gate crosses the Alice/Bob cut.  The two Fredkins act on disjoint wires and
therefore commute; the simulator nevertheless applies them in the fixed order
Alice-then-Bob.

FIVE-QUBIT BASELINE ORDERING

    [c, A1, B1, A2, B2]  ->  indices 0, 1, 2, 3, 4

with the single ancilla ``c`` in ``|+>`` controlling BOTH local swaps,
``Fredkin(c; A1, A2)`` and ``Fredkin(c; B1, B2)``, i.e. the ordinary controlled
full-copy SWAP.

Everything is ideal: exact Fredkin gates, no noise, no CNOT decomposition, no
postselection, no twirling, no Bell-diagonal projection, no re-isotropisation.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "N_QUBITS_6Q", "N_QUBITS_5Q", "WIRES_6Q", "WIRES_5Q",
    "A", "B", "A1", "B1", "A2", "B2", "C5", "A1_5", "B1_5", "A2_5", "B2_5",
    "PAULI", "PAULI_LABELS", "pauli_product",
    "fredkin_matrix", "embed_three_qubit_gate", "swap_on",
    "s_a_6q", "s_b_6q", "unitary_6q_route_a", "unitary_6q_route_b",
    "unitary_5q", "unitary_5q_permutation",
    "plus_state_dm", "data_input", "final_state_6q", "final_state_5q",
    "correlator_6q", "delta_6q", "delta_5q", "delta_6q_all_paulis",
    "delta_5q_all_paulis", "reconstruct_operator",
    "bell_fidelity_numerator_from_deltas", "normalised_bell_fidelity",
]

# ---------------------------------------------------------------------------
# wire names
# ---------------------------------------------------------------------------

N_QUBITS_6Q = 6
WIRES_6Q = ("a", "b", "A1", "B1", "A2", "B2")
A, B, A1, B1, A2, B2 = 0, 1, 2, 3, 4, 5

N_QUBITS_5Q = 5
WIRES_5Q = ("c", "A1", "B1", "A2", "B2")
C5, A1_5, B1_5, A2_5, B2_5 = 0, 1, 2, 3, 4

#: Alice's quantum operations touch only these six-qubit wires.
ALICE_WIRES_6Q = (A, A1, A2)
#: Bob's quantum operations touch only these six-qubit wires.
BOB_WIRES_6Q = (B, B1, B2)

_I2 = np.eye(2, dtype=complex)
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)

PAULI = {
    "I": _I2,
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

#: the sixteen two-qubit Pauli products on (A1, B1), in the required order
PAULI_LABELS = tuple(p + q for p in "IXYZ" for q in "IXYZ")


def pauli_product(label: str) -> np.ndarray:
    """4x4 Pauli product P_1 (x) P_2 for a two-character label such as ``"XY"``."""
    if len(label) != 2 or any(ch not in PAULI for ch in label):
        raise ValueError(f"not a two-qubit Pauli label: {label!r}")
    return np.kron(PAULI[label[0]], PAULI[label[1]])


# ---------------------------------------------------------------------------
# Route A ingredients: explicit gate matrices embedded on chosen wires
# ---------------------------------------------------------------------------

def _kron_list(mats) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def fredkin_matrix() -> np.ndarray:
    """The 8x8 Fredkin (controlled-SWAP) on (control, t1, t2), control first.

    Built from the definition |0><0| (x) I_4 + |1><1| (x) SWAP, with SWAP
    written out explicitly rather than taken from another module.
    """
    swap = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)
    return np.kron(_P0, np.eye(4, dtype=complex)) + np.kron(_P1, swap)


def embed_three_qubit_gate(gate8: np.ndarray, wires, n_qubits: int) -> np.ndarray:
    """Embed an 8x8 three-qubit gate on arbitrary, possibly non-adjacent wires.

    The gate is first placed on the three leading wires and then the tensor
    legs are transposed into position, so no assumption of adjacency is made.
    ``wires`` is ordered: ``wires[k]`` receives leg ``k`` of ``gate8``.
    """
    wires = tuple(int(w) for w in wires)
    if len(set(wires)) != 3 or not all(0 <= w < n_qubits for w in wires):
        raise ValueError(f"bad wires {wires} for {n_qubits} qubits")
    dim = 2 ** n_qubits
    rest = [w for w in range(n_qubits) if w not in wires]
    # gate on the leading three legs, identity on the rest
    big = np.kron(gate8, np.eye(2 ** len(rest), dtype=complex))
    t = big.reshape([2] * (2 * n_qubits))
    # current leg order: wires[0], wires[1], wires[2], rest...   (rows then cols)
    order = list(wires) + rest
    perm = np.argsort(order)                      # where each wire must land
    axes = list(perm) + [n_qubits + p for p in perm]
    return t.transpose(axes).reshape(dim, dim)


def swap_on(i: int, j: int, n_qubits: int) -> np.ndarray:
    """SWAP of wires i and j, as a 2^n x 2^n permutation matrix."""
    dim = 2 ** n_qubits
    m = np.zeros((dim, dim), dtype=complex)
    for idx in range(dim):
        bits = [(idx >> (n_qubits - 1 - w)) & 1 for w in range(n_qubits)]
        bits[i], bits[j] = bits[j], bits[i]
        out = sum(bit << (n_qubits - 1 - w) for w, bit in enumerate(bits))
        m[out, idx] = 1.0
    return m


def s_a_6q() -> np.ndarray:
    """S_A = SWAP(A1, A2) on the six-qubit register."""
    return swap_on(A1, A2, N_QUBITS_6Q)


def s_b_6q() -> np.ndarray:
    """S_B = SWAP(B1, B2) on the six-qubit register."""
    return swap_on(B1, B2, N_QUBITS_6Q)


def unitary_6q_route_a() -> np.ndarray:
    """Route A: Alice's Fredkin then Bob's Fredkin, each an embedded 8x8 gate.

    Fixed order: Alice first, Bob second.  (They act on disjoint wires, so the
    order is immaterial; the test suite checks that numerically.)
    """
    g = fredkin_matrix()
    u_alice = embed_three_qubit_gate(g, (A, A1, A2), N_QUBITS_6Q)
    u_bob = embed_three_qubit_gate(g, (B, B1, B2), N_QUBITS_6Q)
    return u_bob @ u_alice


# ---------------------------------------------------------------------------
# Route B: the whole six-qubit controlled action as a basis permutation
# ---------------------------------------------------------------------------

def unitary_6q_route_b() -> np.ndarray:
    """Route B: the complete six-qubit action written directly as the
    permutation it induces on computational-basis states.

    For each basis index the six bits are read in the order
    ``[a, b, A1, B1, A2, B2]``; the A1/A2 bits are exchanged when ``a = 1`` and
    the B1/B2 bits are exchanged when ``b = 1``.  This uses no Fredkin matrix,
    no embedding helper and no gate product -- it is an independent route to
    the same claimed unitary.
    """
    n = N_QUBITS_6Q
    dim = 2 ** n
    m = np.zeros((dim, dim), dtype=complex)
    for idx in range(dim):
        bits = [(idx >> (n - 1 - w)) & 1 for w in range(n)]
        a_bit, b_bit = bits[0], bits[1]
        a1, b1, a2, b2 = bits[2], bits[3], bits[4], bits[5]
        if a_bit == 1:
            a1, a2 = a2, a1
        if b_bit == 1:
            b1, b2 = b2, b1
        out_bits = [a_bit, b_bit, a1, b1, a2, b2]
        out = sum(bit << (n - 1 - w) for w, bit in enumerate(out_bits))
        m[out, idx] = 1.0
    return m


# ---------------------------------------------------------------------------
# the five-qubit standard SWAP test, also built from gates
# ---------------------------------------------------------------------------

def unitary_5q() -> np.ndarray:
    """One ancilla c controlling both local swaps: Fredkin(c;A1,A2) then
    Fredkin(c;B1,B2) on the ordering [c, A1, B1, A2, B2]."""
    g = fredkin_matrix()
    u1 = embed_three_qubit_gate(g, (C5, A1_5, A2_5), N_QUBITS_5Q)
    u2 = embed_three_qubit_gate(g, (C5, B1_5, B2_5), N_QUBITS_5Q)
    return u2 @ u1


def unitary_5q_permutation() -> np.ndarray:
    """The same five-qubit action as a basis permutation, built independently."""
    n = N_QUBITS_5Q
    dim = 2 ** n
    m = np.zeros((dim, dim), dtype=complex)
    for idx in range(dim):
        bits = [(idx >> (n - 1 - w)) & 1 for w in range(n)]
        c_bit = bits[0]
        a1, b1, a2, b2 = bits[1], bits[2], bits[3], bits[4]
        if c_bit == 1:
            a1, a2 = a2, a1
            b1, b2 = b2, b1
        out_bits = [c_bit, a1, b1, a2, b2]
        out = sum(bit << (n - 1 - w) for w, bit in enumerate(out_bits))
        m[out, idx] = 1.0
    return m


# ---------------------------------------------------------------------------
# states
# ---------------------------------------------------------------------------

def plus_state_dm() -> np.ndarray:
    """|+><+| for one ancilla."""
    plus = np.array([1, 1], dtype=complex) / np.sqrt(2)
    return np.outer(plus, plus.conj())


def data_input(rho: np.ndarray) -> np.ndarray:
    """rho_(A1B1) (x) rho_(A2B2) on the data wires, in the order (A1,B1,A2,B2).

    Both circuits use the same data ordering, so the same 16x16 matrix feeds
    the six-qubit and the five-qubit register.
    """
    return np.kron(rho, rho)


def final_state_6q(rho: np.ndarray, unitary=None) -> np.ndarray:
    """U ( |+><+|_a (x) |+><+|_b (x) rho (x) rho ) U^dagger, 64x64."""
    u = unitary_6q_route_a() if unitary is None else unitary
    sigma = np.kron(plus_state_dm(), np.kron(plus_state_dm(), data_input(rho)))
    return u @ sigma @ u.conj().T


def final_state_5q(rho: np.ndarray, unitary=None) -> np.ndarray:
    """U ( |+><+|_c (x) rho (x) rho ) U^dagger, 32x32."""
    u = unitary_5q() if unitary is None else unitary
    sigma = np.kron(plus_state_dm(), data_input(rho))
    return u @ sigma @ u.conj().T


# ---------------------------------------------------------------------------
# observables and estimators
# ---------------------------------------------------------------------------

def _observable_6q(ancilla_pauli: str, o_a1b1: np.ndarray) -> np.ndarray:
    """P_a (x) P_b (x) O_(A1B1) (x) I_(A2B2) in the ordering [a,b,A1,B1,A2,B2]."""
    p = PAULI[ancilla_pauli]
    return _kron_list([p, p, o_a1b1, np.eye(4, dtype=complex)])


def _observable_5q(o_a1b1: np.ndarray) -> np.ndarray:
    """X_c (x) O_(A1B1) (x) I_(A2B2) in the ordering [c,A1,B1,A2,B2]."""
    return _kron_list([PAULI["X"], o_a1b1, np.eye(4, dtype=complex)])


def correlator_6q(sigma6: np.ndarray, ancilla_pauli: str,
                  o_a1b1: np.ndarray) -> float:
    """< P_a P_b (x) O > on the six-qubit final state, P in {X, Y}."""
    if ancilla_pauli not in ("X", "Y"):
        raise ValueError("ancilla_pauli must be 'X' or 'Y'")
    return float(np.real(np.trace(_observable_6q(ancilla_pauli, o_a1b1) @ sigma6)))


def delta_6q(sigma6: np.ndarray, o_a1b1: np.ndarray):
    """(Delta_6Q, C_XX, C_YY) with Delta_6Q = C_XX - C_YY."""
    c_xx = correlator_6q(sigma6, "X", o_a1b1)
    c_yy = correlator_6q(sigma6, "Y", o_a1b1)
    return c_xx - c_yy, c_xx, c_yy


def delta_5q(sigma5: np.ndarray, o_a1b1: np.ndarray) -> float:
    """Delta_5Q = < X_c (x) O >."""
    return float(np.real(np.trace(_observable_5q(o_a1b1) @ sigma5)))


def delta_6q_all_paulis(rho: np.ndarray, unitary=None) -> dict:
    """{label: (Delta, C_XX, C_YY)} over the sixteen two-qubit Pauli products."""
    sigma6 = final_state_6q(rho, unitary)
    return {lab: delta_6q(sigma6, pauli_product(lab)) for lab in PAULI_LABELS}


def delta_5q_all_paulis(rho: np.ndarray, unitary=None) -> dict:
    """{label: Delta_5Q} over the sixteen two-qubit Pauli products."""
    sigma5 = final_state_5q(rho, unitary)
    return {lab: delta_5q(sigma5, pauli_product(lab)) for lab in PAULI_LABELS}


def reconstruct_operator(deltas: dict) -> np.ndarray:
    """T = (1/4) sum_P d_P P over the sixteen two-qubit Pauli products.

    ``deltas`` maps a Pauli label to its real coefficient.  No interpretation of
    the result is made here.
    """
    t = np.zeros((4, 4), dtype=complex)
    for lab in PAULI_LABELS:
        t = t + deltas[lab] * pauli_product(lab)
    return t / 4.0


def bell_fidelity_numerator_from_deltas(deltas: dict) -> float:
    """Numerator of the Phi+ fidelity estimator from the locally measurable
    product Paulis only:  Phi+ = (II + XX - YY + ZZ)/4."""
    return 0.25 * (deltas["II"] + deltas["XX"] - deltas["YY"] + deltas["ZZ"])


def normalised_bell_fidelity(deltas: dict) -> float:
    """numerator / Delta(II)."""
    return bell_fidelity_numerator_from_deltas(deltas) / deltas["II"]
