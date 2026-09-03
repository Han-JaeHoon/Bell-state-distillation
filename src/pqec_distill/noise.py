"""Configurable two-qubit noise channels inserted after each CNOT.

TWO DISTINCT CONVENTIONS ARE SUPPORTED AND MUST NEVER BE MIXED.
Every result file and figure produced by this package records which one was
used, via the string returned by :func:`convention_name`.

A. ``"replace"`` -- two-qubit REPLACEMENT depolarizing channel::

       D_replace,p(rho) = (1-p) rho + p [ I_ct/4  (x)  Tr_ct(rho) ]

B. ``"pauli"``   -- two-qubit PAULI depolarizing channel::

       D_Pauli,p(rho) = (1-p) rho + (p/15) sum_{P != II} P rho P

Both reduce to the identity channel at p = 0.  They are *different* channels
at the same numerical p, so the value of p is only meaningful together with
the convention name.

EXACT RELATION BETWEEN THE TWO CONVENTIONS (verified in
``tests/test_noise_conventions.py``).  Using the Pauli-twirl identity
(1/16) sum_{P} P rho P = (I_ct/4) (x) Tr_ct(rho) for the two-qubit subsystem,

    sum_{P != II} P rho P = 16 [ (I_ct/4) (x) Tr_ct(rho) ] - rho

so that

    D_Pauli,p = (1 - 16p/15) rho + (16p/15) [ (I_ct/4) (x) Tr_ct(rho) ]
              = D_replace,(16p/15) .

The two conventions therefore trace out the SAME one-parameter family of
channels under the reparameterization p_replace = 16 p_pauli / 15; the Pauli
convention saturates (full replacement) already at p_pauli = 15/16.  This does
not make the labels interchangeable: a threshold quoted in one convention must
be rescaled before being compared with the other.
"""

from __future__ import annotations

import itertools

import numpy as np

from .circuit import CNOT_SEQUENCE, N_QUBITS, Q3, apply_unitary_dm
from .gates import HADAMARD, I2, PAULI_X, PAULI_Y, PAULI_Z, cnot, kron_list, single_qubit_gate
from .measurement import partial_trace

__all__ = [
    "NOISE_CONVENTIONS", "convention_name",
    "replacement_depolarizing", "pauli_depolarizing", "apply_two_qubit_noise",
    "noisy_full_channel_dm",
]

NOISE_CONVENTIONS = ("replace", "pauli")

_ONE_QUBIT_PAULIS = [I2, PAULI_X, PAULI_Y, PAULI_Z]


def convention_name(convention: str) -> str:
    """Human-readable label used in result files and plot annotations."""
    if convention == "replace":
        return "two-qubit REPLACEMENT depolarizing: (1-p)rho + p (I/4 (x) Tr_ct rho)"
    if convention == "pauli":
        return "two-qubit PAULI depolarizing: (1-p)rho + (p/15) sum_{P!=II} P rho P"
    raise ValueError(f"unknown noise convention {convention!r}")


def _reinsert_identity(reduced: np.ndarray, n_qubits: int, noisy_pair: tuple[int, int]) -> np.ndarray:
    """Build (I_pair/4) (x) reduced, placed back at the correct qubit slots.

    ``reduced`` is the state of the remaining qubits (in increasing index
    order).  The result is a full n-qubit operator.
    """
    c, t = noisy_pair
    others = [k for k in range(n_qubits) if k not in (c, t)]
    # Tensor (I/4 on the pair) with the reduced state, then permute axes back.
    pair_part = np.eye(4, dtype=complex) / 4.0
    combined = np.kron(pair_part, reduced)          # order: [c, t] + others
    current_order = [c, t] + others
    tensor = combined.reshape([2] * (2 * n_qubits))
    # Permutation sending current axis positions to true qubit positions.
    perm_row = [current_order.index(q) for q in range(n_qubits)]
    perm = perm_row + [n_qubits + p for p in perm_row]
    tensor = tensor.transpose(perm)
    d = 2 ** n_qubits
    return tensor.reshape(d, d)


def replacement_depolarizing(rho: np.ndarray, pair: tuple[int, int], p: float,
                             n_qubits: int = N_QUBITS) -> np.ndarray:
    """Convention A on the two qubits of ``pair``."""
    if p == 0.0:
        return rho
    keep = [k for k in range(n_qubits) if k not in pair]
    reduced = partial_trace(rho, n_qubits, keep=keep)
    replaced = _reinsert_identity(reduced, n_qubits, pair)
    return (1.0 - p) * rho + p * replaced


def _two_qubit_pauli_operators(pair: tuple[int, int], n_qubits: int) -> list[np.ndarray]:
    """The 15 non-identity two-qubit Paulis on ``pair``, embedded in n qubits."""
    c, t = pair
    ops = []
    for i, j in itertools.product(range(4), repeat=2):
        if i == 0 and j == 0:
            continue
        factors = [I2] * n_qubits
        factors[c] = _ONE_QUBIT_PAULIS[i]
        factors[t] = _ONE_QUBIT_PAULIS[j]
        ops.append(kron_list(factors))
    return ops


def pauli_depolarizing(rho: np.ndarray, pair: tuple[int, int], p: float,
                       n_qubits: int = N_QUBITS) -> np.ndarray:
    """Convention B on the two qubits of ``pair``."""
    if p == 0.0:
        return rho
    acc = np.zeros_like(rho)
    for op in _two_qubit_pauli_operators(pair, n_qubits):
        acc = acc + op @ rho @ op.conj().T
    return (1.0 - p) * rho + (p / 15.0) * acc


def apply_two_qubit_noise(rho: np.ndarray, pair: tuple[int, int], p: float,
                          convention: str, n_qubits: int = N_QUBITS) -> np.ndarray:
    if convention == "replace":
        return replacement_depolarizing(rho, pair, p, n_qubits)
    if convention == "pauli":
        return pauli_depolarizing(rho, pair, p, n_qubits)
    raise ValueError(f"unknown noise convention {convention!r}")


def noisy_full_channel_dm(rho4: np.ndarray, p: float, convention: str) -> np.ndarray:
    """Propagate a 4-qubit density matrix through the noisy 5-CNOT + H circuit.

    Noise is applied immediately after each CNOT, on the two qubits that CNOT
    acted on.  The final H on q3 is ideal.  At p = 0 this reduces exactly to
    the ideal circuit.
    """
    if convention not in NOISE_CONVENTIONS:
        raise ValueError(f"unknown noise convention {convention!r}")
    rho = rho4
    for control, target in CNOT_SEQUENCE:
        rho = apply_unitary_dm(rho, cnot(N_QUBITS, control, target))
        rho = apply_two_qubit_noise(rho, (control, target), p, convention)
    rho = apply_unitary_dm(rho, single_qubit_gate(N_QUBITS, Q3, HADAMARD))
    return rho
