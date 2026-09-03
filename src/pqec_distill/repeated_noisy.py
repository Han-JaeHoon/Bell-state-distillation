"""Repeated NOISY purification on the full two-qubit state: dense route.

The one-round effective map is

    tau(rho) = B(rho, rho),        B(A, C) := 00-branch retained operator of
                                             A (x) C through the noisy circuit
    M_p(rho) = tau(rho) / Tr tau(rho)

Because B is bilinear the Jacobian of M_p can be written EXACTLY (no finite
differences):

    d tau / d r_j = B(P_j/4, rho) + B(rho, P_j/4)

with rho = 1/4 (II + sum_j r_j P_j) in Pauli coordinates.  This module is the
dense counterpart of :mod:`pqec_distill.noisy_analytics`; the two are compared
in tests/test_repeated_noisy.py.
"""

from __future__ import annotations

import itertools

import numpy as np

from .gates import I2, PAULI_X, PAULI_Y, PAULI_Z, kron_list
from .measurement import postselect_branch
from .noise import noisy_full_channel_dm

__all__ = [
    "PAULI_BASIS_2Q", "PAULI_LABELS_2Q", "to_pauli_coords", "from_pauli_coords",
    "bilinear_map", "effective_map", "iterate", "fixed_point_dense",
    "full_jacobian", "off_bell_projection_norm",
]

_ONE = {"I": I2, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}
PAULI_LABELS_2Q = ["".join(t) for t in itertools.product("IXYZ", repeat=2)][1:]
PAULI_BASIS_2Q = [kron_list([_ONE[a], _ONE[b]]) for a, b in PAULI_LABELS_2Q]
BELL_SECTOR = {"XX", "YY", "ZZ"}


def to_pauli_coords(rho: np.ndarray) -> np.ndarray:
    """15 real coordinates r_j = Tr(P_j rho)."""
    return np.array([float(np.real(np.trace(P @ rho))) for P in PAULI_BASIS_2Q])


def from_pauli_coords(r: np.ndarray) -> np.ndarray:
    rho = np.eye(4, dtype=complex)
    for rj, P in zip(r, PAULI_BASIS_2Q):
        rho = rho + rj * P
    return rho / 4.0


def bilinear_map(a: np.ndarray, c: np.ndarray, p: float, convention: str) -> np.ndarray:
    """B(A, C): unnormalized 00-branch retained operator for input A (x) C."""
    out4 = noisy_full_channel_dm(np.kron(a, c), p, convention)
    rho_tilde, _ = postselect_branch(out4, 0, 0)
    return rho_tilde


def effective_map(rho: np.ndarray, p: float, convention: str) -> tuple[np.ndarray, float]:
    tau = bilinear_map(rho, rho, p, convention)
    q = float(np.real(np.trace(tau)))
    return tau / q, q


def iterate(rho0: np.ndarray, p: float, convention: str, n_rounds: int,
            hermitian_project: bool = True):
    """Return the list of states rho_0 .. rho_n.  Each iterate is projected onto
    the Hermitian manifold (an exact-arithmetic identity) so that floating-point
    anti-Hermitian residue is not amplified x2 per round by the quadratic map."""
    states = [rho0]
    rho = rho0
    for _ in range(n_rounds):
        rho, _ = effective_map(rho, p, convention)
        if hermitian_project:
            rho = (rho + rho.conj().T) / 2.0
        states.append(rho)
    return states


def fixed_point_dense(rho0: np.ndarray, p: float, convention: str,
                      tol: float = 1e-13, max_iter: int = 5000):
    """Iterate to convergence; returns (rho*, n_iter, residual)."""
    rho = rho0
    for n in range(1, max_iter + 1):
        nxt, _ = effective_map(rho, p, convention)
        nxt = (nxt + nxt.conj().T) / 2.0
        res = float(np.linalg.norm(nxt - rho, "fro"))
        rho = nxt
        if res < tol:
            return rho, n, res
    return rho, max_iter, res


def full_jacobian(rho_star: np.ndarray, p: float, convention: str) -> np.ndarray:
    """Exact 15x15 Jacobian of M_p in Pauli coordinates at rho_star."""
    tau_star = bilinear_map(rho_star, rho_star, p, convention)
    q_star = float(np.real(np.trace(tau_star)))
    jac = np.zeros((15, 15))
    for j, Pj in enumerate(PAULI_BASIS_2Q):
        dP = Pj / 4.0
        dtau = bilinear_map(dP, rho_star, p, convention) + bilinear_map(rho_star, dP, p, convention)
        dq = float(np.real(np.trace(dtau)))
        dM = dtau / q_star - tau_star * dq / q_star ** 2
        for i, Pi in enumerate(PAULI_BASIS_2Q):
            jac[i, j] = float(np.real(np.trace(Pi @ dM)))
    return jac


def off_bell_projection_norm(rho: np.ndarray) -> float:
    r = to_pauli_coords(rho)
    return float(np.linalg.norm([rj for rj, lab in zip(r, PAULI_LABELS_2Q)
                                 if lab not in BELL_SECTOR]))
