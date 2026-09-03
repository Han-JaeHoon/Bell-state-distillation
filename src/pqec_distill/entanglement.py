"""Entanglement measures for two-qubit states: concurrence and negativity."""

from __future__ import annotations

import numpy as np

from .gates import PAULI_Y, kron_list

__all__ = ["concurrence", "negativity", "bell_diagonal_concurrence", "partial_transpose_B"]

_YY = kron_list([PAULI_Y, PAULI_Y])


def concurrence(rho: np.ndarray) -> float:
    """Wootters concurrence of a general two-qubit density matrix."""
    rho_tilde = _YY @ rho.conj() @ _YY
    product = rho @ rho_tilde
    eigs = np.linalg.eigvals(product)
    # Numerical noise can give tiny negative real parts.
    lam = np.sqrt(np.clip(np.real(eigs), 0.0, None))
    lam = np.sort(lam)[::-1]
    return float(max(0.0, lam[0] - lam[1] - lam[2] - lam[3]))


def bell_diagonal_concurrence(p) -> float:
    """C = max(0, 2 F_max - 1) for a Bell-diagonal state with populations p."""
    p = np.asarray(p, dtype=float)
    return float(max(0.0, 2.0 * p.max() - 1.0))


def partial_transpose_B(rho: np.ndarray) -> np.ndarray:
    """Partial transpose on the second (right / less significant) qubit."""
    t = rho.reshape(2, 2, 2, 2)          # (i1, i2, j1, j2)
    t = t.transpose(0, 3, 2, 1)          # swap i2 <-> j2
    return t.reshape(4, 4)


def negativity(rho: np.ndarray) -> float:
    """N = (||rho^{T_B}||_1 - 1) / 2."""
    pt = partial_transpose_B(rho)
    singular = np.linalg.svd(pt, compute_uv=False)
    return float((singular.sum() - 1.0) / 2.0)
