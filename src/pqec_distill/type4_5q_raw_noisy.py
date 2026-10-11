"""Raw (UNNORMALIZED) noisy estimator of the five-qubit Type-4 SWAP-test circuit.

Two gate orderings are offered and both are run, so that the comparison can
never degenerate into comparing one circuit with itself:

* ``"original"`` -- the frozen 36-operation, 14-CNOT Type-4 sequence exactly as
  transcribed in :data:`pqec_distill.type4_7cnot_fredkin.TYPE4_5Q_OPS`.  Its
  final ancilla Hadamard sits two operations before the end, so only the route-B
  read-out ``<Z_c (x) O (x) I>`` is defined on it;
* ``"regrouped"`` -- ``H_c -> G7(c;A1,A2) -> G7(c;B1,B2) -> H_c``, whose final
  Hadamard is last, so both read-outs are available:

      route A   Delta_5 = < X_c (x) O_(A1B1) (x) I_(A2B2) >   (before final H)
      route B   Delta_5 = < Z_c (x) O_(A1B1) (x) I_(A2B2) >   (after final H)

Nothing is normalised, and no partial trace is taken before the observable is
evaluated.

WIRE ORDERING
-------------
``[c, A1, B1, A2, B2]`` -> ``0, 1, 2, 3, 4``, the ordering of the 5Q/6Q
comparison studies.  The 16x16 data input ``R`` lives on ``(A1, B1, A2, B2)``.

NOISE
-----
The two-qubit REPLACEMENT depolarizing channel after every CNOT, taken from
this repository's own :func:`pqec_distill.noise.replacement_depolarizing`.  No
noise on U3 gates, the ancilla Hadamards, the read-out, or state preparation.

NOT USED
--------
No analytic Type-4 recurrence, no closed-form noisy map, no Bell-diagonal
parameterisation, no previously derived threshold.  Nothing from
:mod:`pqec_distill.analytic_exact_map` is imported.
"""

from __future__ import annotations

import numpy as np

from .noise import replacement_depolarizing
from .type4_7cnot_fredkin import (
    A1_5, A2_5, B1_5, B2_5, C5, N_WIRES_5Q, REGROUPED_5Q_OPS, TYPE4_5Q_OPS,
    WIRES_5Q, build_program, gate_matrix, ops_cnot_pairs, u3_matrix,
)

__all__ = [
    "N_WIRES_5Q", "DIM_5Q", "WIRES_5Q", "DATA_WIRES_5Q", "ORDERINGS",
    "OPS_ORIGINAL", "OPS_REGROUPED", "OPS_REGROUPED_NO_FINAL_H", "FINAL_H",
    "N_CNOT_5Q_T4", "NOISE_CONVENTION", "PROGRAMS", "initial_state_5q",
    "state_before_final_h", "state_after_final_h", "state_original",
    "observable_5q_x", "observable_5q_z", "delta_from_state_x",
    "delta_from_state_z", "delta_5q_type4_noisy_general",
    "delta_5q_type4_noisy", "ancilla_01_block", "PAULI_1Q",
]

DIM_5Q = 2 ** N_WIRES_5Q
#: data wires in the order the 16x16 input R is written in
DATA_WIRES_5Q = (A1_5, B1_5, A2_5, B2_5)
ORDERINGS = ("original", "regrouped")

OPS_ORIGINAL = list(TYPE4_5Q_OPS)
OPS_REGROUPED = list(REGROUPED_5Q_OPS)
#: the regrouped list with its trailing ancilla Hadamard removed (route A)
OPS_REGROUPED_NO_FINAL_H = OPS_REGROUPED[:-1]

FINAL_H = gate_matrix(*OPS_REGROUPED[-1], N_WIRES_5Q)

N_CNOT_5Q_T4 = len(ops_cnot_pairs(OPS_ORIGINAL))

NOISE_CONVENTION = (
    "two-qubit REPLACEMENT depolarizing applied after every CNOT on that "
    "CNOT's two wires, via pqec_distill.noise.replacement_depolarizing")

PROGRAMS = {
    "original": build_program(OPS_ORIGINAL, N_WIRES_5Q),
    "regrouped": build_program(OPS_REGROUPED, N_WIRES_5Q),
    "regrouped_no_final_h": build_program(OPS_REGROUPED_NO_FINAL_H, N_WIRES_5Q),
}

_I2 = np.eye(2, dtype=complex)
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_X1 = np.array([[0, 1], [1, 0]], dtype=complex)
_Y1 = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z1 = np.array([[1, 0], [0, -1]], dtype=complex)
PAULI_1Q = {"I": _I2, "X": _X1, "Y": _Y1, "Z": _Z1}


def _kron_all(mats) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


# ---------------------------------------------------------------------------
# states
# ---------------------------------------------------------------------------

def initial_state_5q(data_r: np.ndarray) -> np.ndarray:
    """``|0><0|_c (x) R``; the leading Hadamard of the sequence makes it |+>."""
    data_r = np.asarray(data_r, dtype=complex)
    if data_r.shape != (16, 16):
        raise ValueError(f"R must be 16x16, got {data_r.shape}")
    return np.kron(_P0, data_r)


def _run(data_r, q, program) -> np.ndarray:
    sigma = initial_state_5q(data_r)
    for kind, payload in program:
        if kind == "U":
            sigma = payload @ sigma @ payload.conj().T
        else:
            sigma = replacement_depolarizing(sigma, payload, float(q),
                                             n_qubits=N_WIRES_5Q)
    return sigma


def state_before_final_h(data_r: np.ndarray, q: float) -> np.ndarray:
    """Regrouped ordering, stopped just before the final ancilla H (route A)."""
    return _run(data_r, q, PROGRAMS["regrouped_no_final_h"])


def state_after_final_h(data_r: np.ndarray, q: float) -> np.ndarray:
    """Regrouped ordering, complete (route B)."""
    return _run(data_r, q, PROGRAMS["regrouped"])


def state_original(data_r: np.ndarray, q: float) -> np.ndarray:
    """The frozen ORIGINAL Type-4 gate ordering, complete (route B only)."""
    return _run(data_r, q, PROGRAMS["original"])


# ---------------------------------------------------------------------------
# observables and the raw estimator
# ---------------------------------------------------------------------------

def observable_5q_x(o_a1b1: np.ndarray) -> np.ndarray:
    """``X_c (x) O_(A1B1) (x) I_(A2B2)``."""
    return _kron_all([_X1, np.asarray(o_a1b1, dtype=complex),
                      np.eye(4, dtype=complex)])


def observable_5q_z(o_a1b1: np.ndarray) -> np.ndarray:
    """``Z_c (x) O_(A1B1) (x) I_(A2B2)``."""
    return _kron_all([_Z1, np.asarray(o_a1b1, dtype=complex),
                      np.eye(4, dtype=complex)])


def _expectation(obs: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.real(np.sum(obs * sigma.T)))


def delta_from_state_x(sigma5: np.ndarray, o_a1b1: np.ndarray) -> float:
    return _expectation(observable_5q_x(o_a1b1), sigma5)


def delta_from_state_z(sigma5: np.ndarray, o_a1b1: np.ndarray) -> float:
    return _expectation(observable_5q_z(o_a1b1), sigma5)


def delta_5q_type4_noisy_general(data_r: np.ndarray, o_a1b1: np.ndarray,
                                 q: float, route: str = "A",
                                 ordering: str = "regrouped") -> float:
    """Raw Delta_5 for an arbitrary 16x16 four-data-qubit input ``R``.

    ``route="A"`` requires ``ordering="regrouped"``; the frozen original
    ordering has two operations after its final Hadamard, so only its route-B
    read-out is defined.
    """
    if ordering == "original":
        if route != "B":
            raise ValueError("the original ordering supports route 'B' only")
        return delta_from_state_z(state_original(data_r, q), o_a1b1)
    if ordering != "regrouped":
        raise ValueError(f"unknown ordering {ordering!r}")
    if route == "A":
        return delta_from_state_x(state_before_final_h(data_r, q), o_a1b1)
    if route == "B":
        return delta_from_state_z(state_after_final_h(data_r, q), o_a1b1)
    raise ValueError("route must be 'A' or 'B'")


def delta_5q_type4_noisy(rho: np.ndarray, o_a1b1: np.ndarray, q: float,
                         route: str = "A",
                         ordering: str = "regrouped") -> float:
    """Raw Delta_5 for two identical copies, ``R = rho (x) rho``."""
    rho = np.asarray(rho, dtype=complex)
    if rho.shape != (4, 4):
        raise ValueError(f"rho must be 4x4, got {rho.shape}")
    return delta_5q_type4_noisy_general(np.kron(rho, rho), o_a1b1, q,
                                        route, ordering)


# ---------------------------------------------------------------------------
# mechanism diagnostic
# ---------------------------------------------------------------------------

def ancilla_01_block(sigma5: np.ndarray) -> np.ndarray:
    """``<0|_c sigma |1>_c``: a 16x16 operator on ``(A1, B1, A2, B2)``."""
    t = np.asarray(sigma5, dtype=complex).reshape([2] * (2 * N_WIRES_5Q))
    return t[0, :, :, :, :, 1, :, :, :, :].reshape(16, 16)
