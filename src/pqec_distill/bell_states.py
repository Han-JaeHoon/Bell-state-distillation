"""Bell-state definitions and Bell-basis transformations.

Two binary labels (a, b) index the Bell states via

    |B_ab> = 1/sqrt(2) * sum_{r=0,1} (-1)^(b*r) |r, r xor a>

with the naming convention

    B_00 = Phi+      B_01 = Phi-      B_10 = Psi+      B_11 = Psi-

These satisfy  ZZ|B_ab> = (-1)^a |B_ab>  and  XX|B_ab> = (-1)^b |B_ab>.

Bell states are ordered by ``index = 2*a + b`` throughout, i.e.

    0 -> Phi+ ,  1 -> Phi- ,  2 -> Psi+ ,  3 -> Psi-

Qubit ordering follows :mod:`pqec_distill.gates`: for a two-qubit Bell pair the
basis is |q_left q_right> with q_left most significant.
"""

from __future__ import annotations

import numpy as np

from .gates import kron_list

__all__ = [
    "BELL_LABELS", "BELL_NAMES", "bell_index", "bell_labels_from_index",
    "bell_state", "bell_projector", "bell_basis_matrix",
    "to_bell_basis", "from_bell_basis", "bell_diagonal_state",
    "bell_populations", "is_bell_diagonal", "bell_offdiagonal_norm",
]

BELL_LABELS = [(0, 0), (0, 1), (1, 0), (1, 1)]
BELL_NAMES = ["Phi+", "Phi-", "Psi+", "Psi-"]


def bell_index(a: int, b: int) -> int:
    """Map labels (a, b) to the canonical Bell index 2a + b."""
    return 2 * a + b


def bell_labels_from_index(i: int) -> tuple[int, int]:
    return (i >> 1) & 1, i & 1


def bell_state(a: int, b: int) -> np.ndarray:
    """|B_ab> as a length-4 complex vector, built from the defining formula."""
    if a not in (0, 1) or b not in (0, 1):
        raise ValueError("Bell labels must be 0 or 1")
    vec = np.zeros(4, dtype=complex)
    for r in (0, 1):
        left, right = r, r ^ a
        vec[2 * left + right] += ((-1.0) ** (b * r)) / np.sqrt(2.0)
    return vec


def bell_projector(a: int, b: int) -> np.ndarray:
    v = bell_state(a, b).reshape(4, 1)
    return v @ v.conj().T


def bell_basis_matrix() -> np.ndarray:
    """4x4 matrix whose columns are |B_ab> in canonical index order."""
    return np.column_stack([bell_state(a, b) for (a, b) in BELL_LABELS])


def to_bell_basis(rho: np.ndarray) -> np.ndarray:
    """Represent a two-qubit operator in the Bell basis: (U^dag rho U)."""
    u = bell_basis_matrix()
    return u.conj().T @ rho @ u


def from_bell_basis(rho_bell: np.ndarray) -> np.ndarray:
    u = bell_basis_matrix()
    return u @ rho_bell @ u.conj().T


def bell_diagonal_state(p) -> np.ndarray:
    """Bell-diagonal two-qubit state sum_i p_i B_i in the computational basis."""
    p = np.asarray(p, dtype=float)
    if p.shape != (4,):
        raise ValueError("need 4 Bell populations")
    rho = np.zeros((4, 4), dtype=complex)
    for i, (a, b) in enumerate(BELL_LABELS):
        rho = rho + p[i] * bell_projector(a, b)
    return rho


def bell_populations(rho: np.ndarray) -> np.ndarray:
    """Diagonal of rho in the Bell basis (real part)."""
    return np.real(np.diag(to_bell_basis(rho)))


def bell_offdiagonal_norm(rho: np.ndarray) -> float:
    """Frobenius norm of the off-diagonal part in the Bell basis ('leakage')."""
    rb = to_bell_basis(rho)
    off = rb - np.diag(np.diag(rb))
    return float(np.linalg.norm(off, "fro"))


def is_bell_diagonal(rho: np.ndarray, atol: float = 1e-12) -> bool:
    return bell_offdiagonal_norm(rho) < atol


def two_qubit_pauli(p_left: np.ndarray, p_right: np.ndarray) -> np.ndarray:
    return kron_list([p_left, p_right])
