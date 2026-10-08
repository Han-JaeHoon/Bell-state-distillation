"""Raw (UNNORMALIZED) noisy estimator of the five-qubit Type-3 SWAP-test circuit.

WHY THIS MODULE EXISTS
----------------------
The frozen Type-3 studies run the circuit through ``one_round()``-style helpers
that immediately produce a *normalised* effective two-qubit state.  For the
5Q <-> 6Q estimator comparison we need the raw, unnormalised ancilla signal

    Delta_5(q; O, R) = < X_c (x) O_(A1B1) (x) I_(A2B2) >

measured on the full five-qubit density matrix immediately before the final
ancilla Hadamard (``route = "A"``), or equivalently

    Delta_5(q; O, R) = < Z_c (x) O_(A1B1) (x) I_(A2B2) >

after that Hadamard (``route = "B"``).  Nothing is divided by anything here and
no partial trace is taken before the observable is evaluated.

WHAT IS GIVEN (not derived here)
--------------------------------
* the canonical Type-3 Fredkin decomposition ``_c2(b,a); _tof(q,a,b); _c2(b,a)``
  with the Clifford+T Toffoli, 8 CNOTs per Fredkin, as used by
  ``scripts/verify_type3_qabs_circuit.py``;
* the two-qubit REPLACEMENT depolarizing channel, imported from this
  repository's own :func:`pqec_distill.noise.replacement_depolarizing` so that
  the 5Q and 6Q simulations provably share one noise convention.

Everything else -- the dense gate matrices, the five-wire register, the
observables -- is built here from the definitions, independently of
:mod:`pqec_distill.locc_vd` and :mod:`pqec_distill.locc_vd_noisy`.

WHAT IS NOT USED
----------------
No analytic Type-3 recurrence, no Bell-diagonal parameterisation, no
closed-form noisy map, no previously derived threshold or fidelity formula.
The module imports nothing from :mod:`pqec_distill.analytic_exact_map`.

FIVE-QUBIT ORDERING (fixed throughout; wire 0 is most significant)

    [c, A1, B1, A2, B2]  ->  indices 0, 1, 2, 3, 4

so that ``Fredkin(c; A1, A2) = Fredkin(0; 1, 3)`` and
``Fredkin(c; B1, B2) = Fredkin(0; 2, 4)``, which is exactly the wire-index
gate list of the frozen canonical Type-3 circuit.
"""

from __future__ import annotations

import numpy as np

from .noise import replacement_depolarizing

__all__ = [
    "N_WIRES_5Q", "DIM_5Q", "WIRES_5Q_ORDER", "C5", "A1_5", "B1_5", "A2_5",
    "B2_5", "DATA_WIRES_5Q", "fredkin_ops", "TYPE3_5Q_OPS", "N_CNOT_5Q",
    "CIRCUIT_SOURCE", "NOISE_CONVENTION", "format_sequence", "cnot_wire_pairs",
    "circuit_unitary", "build_program", "PROGRAM", "FINAL_H",
    "initial_state_5q", "state_before_final_h", "state_after_final_h",
    "observable_5q_x", "observable_5q_z", "delta_from_state_x",
    "delta_from_state_z", "delta_5q_type3_noisy", "delta_5q_type3_noisy_general",
    "deltas_5q_all_paulis", "ancilla_01_block", "PAULI_1Q",
]

N_WIRES_5Q = 5
DIM_5Q = 2 ** N_WIRES_5Q
WIRES_5Q_ORDER = ("c", "A1", "B1", "A2", "B2")
C5, A1_5, B1_5, A2_5, B2_5 = 0, 1, 2, 3, 4
#: the four data wires, in the order the 16x16 input R is written in
DATA_WIRES_5Q = (A1_5, B1_5, A2_5, B2_5)

CIRCUIT_SOURCE = (
    "canonical Type-3 Fredkin decomposition _c2(b,a); _tof(q,a,b); _c2(b,a) "
    "with the Clifford+T Toffoli, 8 CNOTs per Fredkin; same wire-index gate "
    "list as scripts/verify_type3_qabs_circuit.py TYPE3_OPS")

NOISE_CONVENTION = (
    "two-qubit REPLACEMENT depolarizing applied after every CNOT on that "
    "CNOT's two wires, via pqec_distill.noise.replacement_depolarizing")


# ---------------------------------------------------------------------------
# dense gates on the five-wire register, built from the definitions
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_H1 = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_T1 = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
_PROJ0 = np.array([[1, 0], [0, 0]], dtype=complex)
_PROJ1 = np.array([[0, 0], [0, 1]], dtype=complex)
_X1 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y1 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z1 = np.array([[1, 0], [0, -1]], dtype=complex)
_ONE_QUBIT_5Q = {"H": _H1, "T": _T1, "Tdg": _T1.conj().T}

PAULI_1Q = {"I": _I2, "X": _X1, "Y": _Y1, "Z": _Z1}


def _kron_all(mats) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def _single_5q(name: str, wire: int) -> np.ndarray:
    factors = [_I2] * N_WIRES_5Q
    factors[wire] = _ONE_QUBIT_5Q[name]
    return _kron_all(factors)


def _cnot_5q(control: int, target: int) -> np.ndarray:
    """|0><0|_c (x) I + |1><1|_c (x) X_t, from the definition."""
    if control == target:
        raise ValueError("control and target must differ")
    a = [_I2] * N_WIRES_5Q
    a[control] = _PROJ0
    b = [_I2] * N_WIRES_5Q
    b[control] = _PROJ1
    b[target] = _X1
    return _kron_all(a) + _kron_all(b)


def _op_matrix_5q(kind, arg) -> np.ndarray:
    return _cnot_5q(*arg) if kind == "CNOT" else _single_5q(kind, arg)


# ---------------------------------------------------------------------------
# the canonical Type-3 gate sequence on the [c, A1, B1, A2, B2] ordering
# ---------------------------------------------------------------------------

def fredkin_ops(q: int, a: int, b: int) -> list:
    """``_fred(q, a, b) = _c2(b,a); _tof(q,a,b); _c2(b,a)``: eight CNOTs.

    The single-qubit gates are kept in their original positions, because the
    noise sees the decomposition and not merely the unitary it implements.
    """
    ops = [("CNOT", (b, a))]                                   # _c2(b, a)
    # ---- _tof(c1 = q, c2 = a, t = b) ----------------------------------
    ops += [("H", b)]
    ops += [("CNOT", (a, b)), ("Tdg", b)]
    ops += [("CNOT", (q, b)), ("T", b)]
    ops += [("CNOT", (a, b)), ("Tdg", b)]
    ops += [("CNOT", (q, b)), ("T", b), ("T", a)]
    ops += [("CNOT", (q, a)), ("H", b)]
    ops += [("T", q), ("Tdg", a)]
    ops += [("CNOT", (q, a))]
    # -------------------------------------------------------------------
    ops += [("CNOT", (b, a))]                                  # _c2(b, a)
    return ops


#: H_c -> Fredkin(c; A1, A2) -> Fredkin(c; B1, B2).  The FINAL H is deliberately
#: NOT part of this list: route A measures X_c before it, route B applies it and
#: measures Z_c.
TYPE3_5Q_OPS = ([("H", C5)]
                + fredkin_ops(C5, A1_5, A2_5)
                + fredkin_ops(C5, B1_5, B2_5))

N_CNOT_5Q = sum(1 for kind, _ in TYPE3_5Q_OPS if kind == "CNOT")


def cnot_wire_pairs(ops=None) -> list:
    return [tuple(arg) for kind, arg in (ops or TYPE3_5Q_OPS) if kind == "CNOT"]


def format_sequence(ops=None) -> str:
    lines, k = [], 0
    for kind, arg in (ops or TYPE3_5Q_OPS):
        if kind == "CNOT":
            k += 1
            lines.append(f"  {len(lines)+1:3d}.  CNOT({arg[0]} -> {arg[1]})"
                         f"   [CNOT #{k:2d}]  + D_q on ({arg[0]},{arg[1]})")
        else:
            lines.append(f"  {len(lines)+1:3d}.  {kind}({arg})")
    return "\n".join(lines)


def circuit_unitary(ops=None) -> np.ndarray:
    """Noiseless 32x32 unitary of the transcribed sequence (no final H)."""
    u = np.eye(DIM_5Q, dtype=complex)
    for kind, arg in (ops or TYPE3_5Q_OPS):
        u = _op_matrix_5q(kind, arg) @ u
    return u


def build_program(ops=None):
    """``[("U", M) | ("D", (i, j))]``: runs of unitaries merged into one matrix,
    channel locations left exactly where the CNOTs are."""
    program, acc = [], np.eye(DIM_5Q, dtype=complex)
    for kind, arg in (ops or TYPE3_5Q_OPS):
        acc = _op_matrix_5q(kind, arg) @ acc
        if kind == "CNOT":
            program.append(("U", acc))
            acc = np.eye(DIM_5Q, dtype=complex)
            program.append(("D", tuple(arg)))
    program.append(("U", acc))
    return program


PROGRAM = build_program()
#: the ideal final ancilla Hadamard, applied by route B only
FINAL_H = _single_5q("H", C5)


# ---------------------------------------------------------------------------
# states
# ---------------------------------------------------------------------------

def initial_state_5q(data_r: np.ndarray) -> np.ndarray:
    """``|0><0|_c (x) R`` on ``[c, A1, B1, A2, B2]``.

    The leading Hadamard of :data:`TYPE3_5Q_OPS` turns this into ``|+><+|_c``,
    exactly as the frozen Type-3 circuit does.  ``R`` is any 16x16 operator on
    the four data wires in the order ``(A1, B1, A2, B2)``.
    """
    data_r = np.asarray(data_r, dtype=complex)
    if data_r.shape != (16, 16):
        raise ValueError(f"R must be 16x16, got {data_r.shape}")
    return np.kron(_PROJ0, data_r)


def state_before_final_h(data_r: np.ndarray, q: float, program=None) -> np.ndarray:
    """Run ``H_c`` and both noisy Fredkins; return the 32x32 state (route A)."""
    sigma = initial_state_5q(data_r)
    for kind, payload in (program if program is not None else PROGRAM):
        if kind == "U":
            sigma = payload @ sigma @ payload.conj().T
        else:
            sigma = replacement_depolarizing(sigma, payload, float(q),
                                             n_qubits=N_WIRES_5Q)
    return sigma


def state_after_final_h(data_r: np.ndarray, q: float, program=None) -> np.ndarray:
    """Route B: the same run, with the ideal final ancilla Hadamard applied."""
    sigma = state_before_final_h(data_r, q, program)
    return FINAL_H @ sigma @ FINAL_H.conj().T


# ---------------------------------------------------------------------------
# observables and the raw estimator
# ---------------------------------------------------------------------------

def observable_5q_x(o_a1b1: np.ndarray) -> np.ndarray:
    """``X_c (x) O_(A1B1) (x) I_(A2B2)`` on ``[c, A1, B1, A2, B2]``."""
    return _kron_all([_X1, np.asarray(o_a1b1, dtype=complex),
                      np.eye(4, dtype=complex)])


def observable_5q_z(o_a1b1: np.ndarray) -> np.ndarray:
    """``Z_c (x) O_(A1B1) (x) I_(A2B2)`` on ``[c, A1, B1, A2, B2]``."""
    return _kron_all([_Z1, np.asarray(o_a1b1, dtype=complex),
                      np.eye(4, dtype=complex)])


def _expectation(obs: np.ndarray, sigma: np.ndarray) -> float:
    """Tr(obs @ sigma), returned as a real float (the imaginary part is
    reported by the caller's diagnostics, never silently dropped elsewhere)."""
    return float(np.real(np.sum(obs * sigma.T)))


def delta_from_state_x(sigma5: np.ndarray, o_a1b1: np.ndarray) -> float:
    """Route A read-out on a pre-final-H state."""
    return _expectation(observable_5q_x(o_a1b1), sigma5)


def delta_from_state_z(sigma5: np.ndarray, o_a1b1: np.ndarray) -> float:
    """Route B read-out on a post-final-H state."""
    return _expectation(observable_5q_z(o_a1b1), sigma5)


def delta_5q_type3_noisy_general(data_r: np.ndarray, o_a1b1: np.ndarray,
                                 q: float, route: str = "A") -> float:
    """Raw Delta_5 for an arbitrary 16x16 four-data-qubit input ``R``."""
    if route == "A":
        return delta_from_state_x(state_before_final_h(data_r, q), o_a1b1)
    if route == "B":
        return delta_from_state_z(state_after_final_h(data_r, q), o_a1b1)
    raise ValueError("route must be 'A' or 'B'")


def delta_5q_type3_noisy(rho: np.ndarray, o_a1b1: np.ndarray, q: float,
                         route: str = "A") -> float:
    """Raw Delta_5 for two identical copies, ``R = rho (x) rho``."""
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError(f"rho must be 4x4, got {rho.shape}")
    return delta_5q_type3_noisy_general(np.kron(rho, rho), o_a1b1, q, route)


def deltas_5q_all_paulis(data_r: np.ndarray, q: float,
                         labels=None, route: str = "A") -> dict:
    """``{label: Delta_5}`` over two-qubit Pauli products, one circuit run."""
    labels = labels or tuple(p + s for p in "IXYZ" for s in "IXYZ")
    sigma = (state_before_final_h(data_r, q) if route == "A"
             else state_after_final_h(data_r, q))
    read = delta_from_state_x if route == "A" else delta_from_state_z
    out = {}
    for lab in labels:
        o = np.kron(PAULI_1Q[lab[0]], PAULI_1Q[lab[1]])
        out[lab] = read(sigma, o)
    return out


# ---------------------------------------------------------------------------
# mechanism diagnostic (not a substitute for the observable-level comparison)
# ---------------------------------------------------------------------------

def ancilla_01_block(sigma5: np.ndarray) -> np.ndarray:
    """``<0|_c sigma |1>_c``: a 16x16 operator on ``(A1, B1, A2, B2)``."""
    t = np.asarray(sigma5, dtype=complex).reshape([2] * (2 * N_WIRES_5Q))
    return t[0, :, :, :, :, 1, :, :, :, :].reshape(16, 16)
