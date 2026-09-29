"""The repeated-dynamics recurrences exactly as written in the report note.

Implemented here from the note's formulas ALONE -- this module deliberately
does not import ``pqec_distill.analytic_exact_map``.  Its only purpose is to be
compared against the repository's verified map and against the full
density-matrix simulation, so that a disagreement shows up instead of being
hidden by sharing one implementation.

Conventions (identical to the note):

    rho(x,y,z) = 1/4 ( II + x XX + y YY + z ZZ ),      s = 1 - q

    p_Phi+ = (1+x-y+z)/4      p_Phi- = (1-x+y+z)/4
    p_Psi+ = (1+x+y-z)/4      p_Psi- = (1-x-y-z)/4

Type 3 (textbook 16-CNOT):
    D3 = 1 + s^4 (x^2+y^2+z^2)
    x' = 2 s^6 (x - yz)/D3 ;  y' = 2 s^6 (y - xz)/D3 ;  z' = s^5(1+s)(z - xy)/D3

Type 4 (resynthesised 14-CNOT):
    D4 = 1 + s^2 (x^2+y^2+z^2)
    x' = 2 s^4 (x - yz)/D4 ;  y' = 2 s^4 (y - xz)/D4 ;  z' = s^3(1+s)(z - xy)/D4

4Q (postselected 4-qubit 5-CNOT):
    D4Q = 1 + s^5(x^2+y^2) + s^3 z^2 ;   P_succ = D4Q / 4
    x' = [(s^3+s^4) x - 2 s^5 y z]/D4Q
    y' = [(s^4+s^5) y - 2 s^5 x z]/D4Q
    z' = [(s^4+s^5) z - 2 s^5 x y]/D4Q

No twirling and no re-projection anywhere: each round's exact output is the next
round's input.  For a Bell-diagonal input these maps return a Bell-diagonal
output, which is a property of the maps -- verified against the full
density-matrix simulation, not assumed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BELL_ORDER", "pops_to_xyz", "xyz_to_pops", "isotropic_pops",
           "note_step", "note_success", "note_fidelity_in_plane", "NOTE_PROTOCOLS"]

BELL_ORDER = ("Phi+", "Phi-", "Psi+", "Psi-")
NOTE_PROTOCOLS = ("Type3", "Type4", "4Q")


def pops_to_xyz(p):
    a, b, c, d = (float(v) for v in p)
    return (a - b + c - d, -a + b + c - d, a + b - c - d)


def xyz_to_pops(x, y, z):
    return np.array([(1 + x - y + z) / 4, (1 - x + y + z) / 4,
                     (1 + x + y - z) / 4, (1 - x - y - z) / 4])


def isotropic_pops(eps: float) -> np.ndarray:
    e = float(eps)
    return np.array([1 - 3 * e / 4, e / 4, e / 4, e / 4])


def note_step(p, q: float, protocol: str) -> np.ndarray:
    """One round, from the note's recurrence, on Bell populations."""
    x, y, z = pops_to_xyz(p)
    s = 1.0 - float(q)
    if protocol == "Type3":
        d = 1 + s**4 * (x * x + y * y + z * z)
        xp = 2 * s**6 * (x - y * z) / d
        yp = 2 * s**6 * (y - x * z) / d
        zp = s**5 * (1 + s) * (z - x * y) / d
    elif protocol == "Type4":
        d = 1 + s**2 * (x * x + y * y + z * z)
        xp = 2 * s**4 * (x - y * z) / d
        yp = 2 * s**4 * (y - x * z) / d
        zp = s**3 * (1 + s) * (z - x * y) / d
    elif protocol == "4Q":
        d = 1 + s**5 * (x * x + y * y) + s**3 * z * z
        xp = ((s**3 + s**4) * x - 2 * s**5 * y * z) / d
        yp = ((s**4 + s**5) * y - 2 * s**5 * x * z) / d
        zp = ((s**4 + s**5) * z - 2 * s**5 * x * y) / d
    else:
        raise KeyError(f"the note gives no Bell-diagonal recurrence for {protocol!r}")
    return xyz_to_pops(xp, yp, zp)


def note_success(p, q: float) -> float:
    """4Q postselection probability, P_succ = D4Q / 4."""
    x, y, z = pops_to_xyz(p)
    s = 1.0 - float(q)
    return (1 + s**5 * (x * x + y * y) + s**3 * z * z) / 4


def note_fidelity_in_plane(p, protocol: str) -> float:
    """Target fidelity written in the protocol's invariant plane.

    Type 3/4 keep y = -x, where p_Phi+ = (1 + 2x + z)/4.
    4Q keeps y = -z,      where p_Phi+ = (1 + x + 2z)/4.
    Both must equal p_Phi+ itself; the difference is what the caller checks.
    """
    x, _y, z = pops_to_xyz(p)
    return (1 + x + 2 * z) / 4 if protocol == "4Q" else (1 + 2 * x + z) / 4
