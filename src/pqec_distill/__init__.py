"""4-qubit / 5-CNOT Bell-state purification: implementation and verification.

Qubit ordering convention (see :mod:`pqec_distill.gates`):
    |q1 q2 q3 q4>, q1 most significant; 0-based indices q1->0 .. q4->3.
    retained pair A = (q1, q2); measured pair B = (q3, q4).

Submodules are imported explicitly by the caller, e.g.::

    from pqec_distill.circuit import full_unitary
    from pqec_distill.analytics import eps_prime
"""

from __future__ import annotations

__version__ = "0.1.0"
