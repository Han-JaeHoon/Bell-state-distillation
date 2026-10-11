"""The 14-CNOT Type-4 circuit, its two local 7-CNOT blocks, and the regrouping.

WHAT IS GIVEN (quoted, not derived here)
----------------------------------------
:data:`TYPE4_5Q_OPS` is the canonical Type-4 gate sequence, transcribed
verbatim from ``PQEC-Operational-Threshold/pqec_resynth_noise.py`` (module-level
``GATES``: Qiskit transpile of ``H_a . CSWAP(a;.,.) . CSWAP(a;.,.) . H_a`` into
the basis ``{u, cx}``, ``optimization_level 3``, ``seed_transpiler 7``, global
phase ``pi/4``).  It is byte-for-byte the list already frozen in
``scripts/verify_type4_qabs_circuit.py`` and
``scripts/verify_type4_qrel_circuit.py``.

WHAT IS CLAIMED AND MEASURED (never assumed)
--------------------------------------------
That this 36-operation, 14-CNOT sequence can be regrouped into

    H_c  ->  G7(c; A1, A2)  ->  G7(c; B1, B2)  ->  H_c

with one single standalone 7-CNOT block ``G7`` used twice, by reordering only
operations with disjoint wire support.  Three separate things are therefore
checked by the verification script and by the tests, not by this module:

1. the two blocks extracted from the frozen sequence are gate-for-gate
   identical after relabelling their local wires to ``(c, u, v)``;
2. each block's 8x8 unitary equals the ideal Fredkin up to a global phase;
3. the regrouped sequence is a permutation of the frozen one in which every
   reordered pair of operations has disjoint support -- so the regrouping
   changes neither the ideal unitary nor, under per-CNOT two-qubit noise, the
   noisy channel.

WIRE CONVENTION
---------------
This module uses the ordering of the 5Q/6Q comparison studies,

    [c, A1, B1, A2, B2]  ->  0, 1, 2, 3, 4

so Alice's triple is ``(c, A1, A2) = (0, 1, 3)`` and Bob's is
``(c, B1, B2) = (0, 2, 4)``.  The frozen Type-4 scripts print the same wire
indices under the names ``(a, A1, A2, B1, B2)``; only the labels differ, the
gate list and every wire index are identical.

WHAT IS NOT USED
----------------
No analytic Type-4 Bell-diagonal recurrence, no ``analytic_exact_map``, no
closed-form noisy map and no previously derived Type-4 threshold.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "WIRES_5Q", "C5", "A1_5", "B1_5", "A2_5", "B2_5", "ALICE_TRIPLE_5Q",
    "BOB_TRIPLE_5Q", "N_WIRES_5Q", "TYPE4_5Q_OPS", "TYPE4_SOURCE", "H_C_OP",
    "G7_OPS", "N_CNOT_G7", "u3_matrix", "fredkin_matrix", "gate_matrix",
    "ops_unitary", "ops_cnot_pairs", "ops_support", "embed_ops",
    "split_type4_sequence", "regrouped_type4_ops", "REGROUPED_5Q_OPS",
    "block_unitary", "global_phase_fit", "commutation_certificate",
    "build_program", "run_noisy_ops",
]

# ---------------------------------------------------------------------------
# wires
# ---------------------------------------------------------------------------

N_WIRES_5Q = 5
WIRES_5Q = ("c", "A1", "B1", "A2", "B2")
C5, A1_5, B1_5, A2_5, B2_5 = 0, 1, 2, 3, 4
#: (c, A1, A2) -- the first local block's wires
ALICE_TRIPLE_5Q = (C5, A1_5, A2_5)        # (0, 1, 3)
#: (c, B1, B2) -- the second local block's wires
BOB_TRIPLE_5Q = (C5, B1_5, B2_5)          # (0, 2, 4)

TYPE4_SOURCE = (
    "PQEC-Operational-Threshold/pqec_resynth_noise.py module-level GATES "
    "(Qiskit transpile of H_a CSWAP CSWAP H_a into {u, cx}, "
    "optimization_level 3, seed_transpiler 7, global phase pi/4); identical to "
    "TYPE4_OPS in scripts/verify_type4_qabs_circuit.py and "
    "scripts/verify_type4_qrel_circuit.py")

#: the canonical Type-4 sequence, verbatim, in time order
TYPE4_5Q_OPS = [
    ("U3", (0, 1.5707963267948966, 0.0, 3.141592653589793)),
    ("U3", (1, 1.5707963267948966, -1.5707963267948966, 1.5707963267948966)),
    ("U3", (2, 1.5707963267948966, -1.5707963267948966, 1.5707963267948966)),
    ("U3", (3, 1.5707963267948968, -3.141592653589793, -2.356194490192345)),
    ("CNOT", (1, 3)),
    ("U3", (1, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966)),
    ("U3", (3, 0.7853981633974475, -2.356194490192345, -1.5707963267948966)),
    ("CNOT", (0, 3)),
    ("U3", (3, 0.0, 0.0, 0.7853981633974483)),
    ("CNOT", (1, 3)),
    ("U3", (1, 0.0, 0.0, 0.7853981633974483)),
    ("U3", (3, 0.0, 0.0, -0.7853981633974483)),
    ("CNOT", (0, 3)),
    ("CNOT", (0, 1)),
    ("U3", (0, 0.0, 0.0, 0.7853981633974483)),
    ("U3", (1, 0.0, 0.0, -0.7853981633974483)),
    ("CNOT", (0, 1)),
    ("U3", (3, 1.5707963267948966, 0.0, -2.3561944901923453)),
    ("CNOT", (3, 1)),
    ("U3", (4, 1.5707963267948968, -3.141592653589793, -2.356194490192345)),
    ("CNOT", (2, 4)),
    ("U3", (2, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966)),
    ("U3", (4, 0.7853981633974475, -2.356194490192345, -1.5707963267948966)),
    ("CNOT", (0, 4)),
    ("U3", (4, 0.0, 0.0, 0.7853981633974483)),
    ("CNOT", (2, 4)),
    ("U3", (2, 0.0, 0.0, 0.7853981633974483)),
    ("U3", (4, 0.0, 0.0, -0.7853981633974483)),
    ("CNOT", (0, 4)),
    ("CNOT", (0, 2)),
    ("U3", (0, 0.0, 0.0, 0.7853981633974483)),
    ("U3", (2, 0.0, 0.0, -0.7853981633974483)),
    ("CNOT", (0, 2)),
    ("U3", (0, 1.5707963267948966, 0.0, 3.141592653589793)),
    ("U3", (4, 1.5707963267948966, 0.0, -2.3561944901923453)),
    ("CNOT", (4, 2)),
]

#: the ancilla Hadamard as it appears in the frozen sequence (U3(pi/2, 0, pi))
H_C_OP = ("U3", (C5, 1.5707963267948966, 0.0, 3.141592653589793))

#: the candidate standalone local block on local wires (c, u, v) = (0, 1, 2).
#: Whether the frozen sequence really contains this block twice is CHECKED, not
#: assumed: :func:`split_type4_sequence` extracts the blocks from
#: :data:`TYPE4_5Q_OPS` and the verification compares them against this list.
G7_OPS = [
    ("U3", (1, 1.5707963267948966, -1.5707963267948966, 1.5707963267948966)),
    ("U3", (2, 1.5707963267948968, -3.141592653589793, -2.356194490192345)),
    ("CNOT", (1, 2)),
    ("U3", (1, 1.5707963267948966, 1.5707963267948966, 1.5707963267948966)),
    ("U3", (2, 0.7853981633974475, -2.356194490192345, -1.5707963267948966)),
    ("CNOT", (0, 2)),
    ("U3", (2, 0.0, 0.0, 0.7853981633974483)),
    ("CNOT", (1, 2)),
    ("U3", (1, 0.0, 0.0, 0.7853981633974483)),
    ("U3", (2, 0.0, 0.0, -0.7853981633974483)),
    ("CNOT", (0, 2)),
    ("CNOT", (0, 1)),
    ("U3", (0, 0.0, 0.0, 0.7853981633974483)),
    ("U3", (1, 0.0, 0.0, -0.7853981633974483)),
    ("CNOT", (0, 1)),
    ("U3", (2, 1.5707963267948966, 0.0, -2.3561944901923453)),
    ("CNOT", (2, 1)),
]

N_CNOT_G7 = sum(1 for kind, _ in G7_OPS if kind == "CNOT")


# ---------------------------------------------------------------------------
# gate primitives, built from the definitions
# ---------------------------------------------------------------------------

_I2 = np.eye(2, dtype=complex)
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)


def _kron_all(mats) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """Qiskit ``u`` / PennyLane ``U3`` convention::

        [[            cos(t/2) , -e^{i lam}       sin(t/2) ],
         [ e^{i phi}   sin(t/2) ,  e^{i(phi+lam)} cos(t/2) ]]
    """
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
                    dtype=complex)


def fredkin_matrix() -> np.ndarray:
    """``|0><0|_c (x) I_uv + |1><1|_c (x) SWAP_uv`` on (control, u, v)."""
    swap = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1]], dtype=complex)
    return np.kron(_P0, np.eye(4, dtype=complex)) + np.kron(_P1, swap)


def gate_matrix(kind, arg, n_qubits: int) -> np.ndarray:
    """Dense matrix of one operation on an ``n_qubits`` register."""
    if kind == "CNOT":
        control, target = arg
        if control == target:
            raise ValueError("control and target must differ")
        a = [_I2] * n_qubits
        a[control] = _P0
        b = [_I2] * n_qubits
        b[control] = _P1
        b[target] = _X
        return _kron_all(a) + _kron_all(b)
    if kind == "U3":
        wire, theta, phi, lam = arg
        factors = [_I2] * n_qubits
        factors[wire] = u3_matrix(theta, phi, lam)
        return _kron_all(factors)
    raise ValueError(f"unknown operation kind {kind!r}")


def ops_unitary(ops, n_qubits: int) -> np.ndarray:
    """Product of the operation list in time order (later gates on the left)."""
    u = np.eye(2 ** n_qubits, dtype=complex)
    for kind, arg in ops:
        u = gate_matrix(kind, arg, n_qubits) @ u
    return u


def ops_cnot_pairs(ops) -> list:
    return [tuple(arg) for kind, arg in ops if kind == "CNOT"]


def ops_support(kind, arg) -> frozenset:
    """The set of wires one operation acts on."""
    return frozenset(arg) if kind == "CNOT" else frozenset({arg[0]})


def block_unitary(ops_local=None) -> np.ndarray:
    """The 8x8 unitary of a three-wire local block on wires (c, u, v)."""
    return ops_unitary(ops_local if ops_local is not None else G7_OPS, 3)


def global_phase_fit(u: np.ndarray, target: np.ndarray) -> dict:
    """Fit ``u = e^{i phi} target`` and report every residual.

    ``phi = arg Tr(target^dagger u)``.  Nothing is assumed about the result.
    """
    u = np.asarray(u, dtype=complex)
    target = np.asarray(target, dtype=complex)
    overlap = complex(np.trace(target.conj().T @ u))
    phi = float(np.angle(overlap))
    resid = u - np.exp(1j * phi) * target
    dim = u.shape[0]
    return {
        "global_phase": phi,
        "global_phase_over_pi": phi / np.pi,
        "trace_overlap_abs": abs(overlap),
        "trace_overlap_abs_over_dim": abs(overlap) / dim,
        "max_entry_residual": float(np.max(np.abs(resid))),
        "frobenius_residual": float(np.linalg.norm(resid)),
        "unitarity_residual": float(np.max(np.abs(
            u.conj().T @ u - np.eye(dim)))),
        "residual_without_phase": float(np.max(np.abs(u - target))),
    }


# ---------------------------------------------------------------------------
# embedding and regrouping
# ---------------------------------------------------------------------------

def embed_ops(ops_local, wires) -> list:
    """Relabel a local operation list onto ``wires`` of a larger register.

    ``wires[k]`` receives local wire ``k``, so ``wires = (c, u, v)``.
    """
    wires = tuple(int(w) for w in wires)
    if len(set(wires)) != len(wires):
        raise ValueError(f"wires must be distinct: {wires}")
    out = []
    for kind, arg in ops_local:
        if kind == "CNOT":
            out.append((kind, (wires[arg[0]], wires[arg[1]])))
        else:
            out.append((kind, (wires[arg[0]],) + tuple(arg[1:])))
    return out


def _local_relabel(ops, wires) -> list:
    """Inverse of :func:`embed_ops`: express ops on ``wires`` in local indices."""
    index = {w: k for k, w in enumerate(tuple(int(w) for w in wires))}
    out = []
    for kind, arg in ops:
        if kind == "CNOT":
            out.append((kind, (index[arg[0]], index[arg[1]])))
        else:
            out.append((kind, (index[arg[0]],) + tuple(arg[1:])))
    return out


def split_type4_sequence(ops=None) -> dict:
    """Partition the frozen sequence into ``H_c``, block A, block B, ``H_c``.

    The rule is explicit and position-based, so that it can be audited:

    * the two ``U3(0; pi/2, 0, pi)`` operations on the ancilla alone are the
      leading and trailing Hadamards;
    * an operation whose support lies inside ``{A1, A2} = {1, 3}`` (or involves
      the ancilla together with one of them) belongs to block A; likewise
      ``{B1, B2} = {2, 4}`` for block B;
    * the two remaining ancilla-only operations ``U3(0; 0, 0, pi/4)`` are
      assigned by position: each lies strictly between the first and the last
      two-qubit gate of exactly one block.

    If any operation cannot be assigned this way, a ``ValueError`` is raised
    rather than the sequence being quietly reinterpreted.
    """
    ops = list(ops if ops is not None else TYPE4_5Q_OPS)
    a_data, b_data = {A1_5, A2_5}, {B1_5, B2_5}

    two_qubit_pos = {"A": [], "B": []}
    for i, (kind, arg) in enumerate(ops):
        if kind != "CNOT":
            continue
        sup = ops_support(kind, arg)
        if sup <= (a_data | {C5}):
            two_qubit_pos["A"].append(i)
        elif sup <= (b_data | {C5}):
            two_qubit_pos["B"].append(i)
        else:
            raise ValueError(f"two-qubit gate {arg} crosses both blocks")
    span = {k: (min(v), max(v)) for k, v in two_qubit_pos.items()}

    leading = trailing = None
    alice, bob = [], []
    for i, (kind, arg) in enumerate(ops):
        sup = ops_support(kind, arg)
        if sup == {C5}:
            is_h = (kind == "U3"
                    and abs(arg[1] - np.pi / 2) < 1e-12
                    and abs(arg[2]) < 1e-12
                    and abs(abs(arg[3]) - np.pi) < 1e-12)
            if is_h:
                if leading is None and i < span["A"][0]:
                    leading = (i, (kind, arg))
                elif trailing is None:
                    trailing = (i, (kind, arg))
                else:
                    raise ValueError("more ancilla Hadamards than expected")
                continue
            inside_a = span["A"][0] < i < span["A"][1]
            inside_b = span["B"][0] < i < span["B"][1]
            if inside_a and not inside_b:
                alice.append((i, (kind, arg)))
            elif inside_b and not inside_a:
                bob.append((i, (kind, arg)))
            else:
                raise ValueError(
                    f"ancilla-only op at position {i} is not inside exactly "
                    f"one block")
            continue
        if sup <= (a_data | {C5}):
            alice.append((i, (kind, arg)))
        elif sup <= (b_data | {C5}):
            bob.append((i, (kind, arg)))
        else:
            raise ValueError(f"operation {kind}{arg} belongs to neither block")

    if leading is None or trailing is None:
        raise ValueError("did not find both ancilla Hadamards")
    return {
        "leading_h": leading[1],
        "trailing_h": trailing[1],
        "leading_h_position": leading[0],
        "trailing_h_position": trailing[0],
        "alice_positions": [i for i, _ in alice],
        "bob_positions": [i for i, _ in bob],
        "alice_ops": [op for _, op in alice],
        "bob_ops": [op for _, op in bob],
        "alice_local": _local_relabel([op for _, op in alice], ALICE_TRIPLE_5Q),
        "bob_local": _local_relabel([op for _, op in bob], BOB_TRIPLE_5Q),
        "n_cnot_alice": sum(1 for _, (k, _a) in alice if k == "CNOT"),
        "n_cnot_bob": sum(1 for _, (k, _a) in bob if k == "CNOT"),
    }


def regrouped_type4_ops(block_local=None) -> list:
    """``H_c -> G7(c;A1,A2) -> G7(c;B1,B2) -> H_c`` as a 5-wire op list."""
    block = G7_OPS if block_local is None else block_local
    return ([H_C_OP]
            + embed_ops(block, ALICE_TRIPLE_5Q)
            + embed_ops(block, BOB_TRIPLE_5Q)
            + [H_C_OP])


REGROUPED_5Q_OPS = regrouped_type4_ops()


def _op_key(kind, arg):
    if kind == "CNOT":
        return (kind, tuple(int(x) for x in arg))
    return (kind, (int(arg[0]),) + tuple(float(x) for x in arg[1:]))


def commutation_certificate(original, reordered) -> dict:
    """Is ``reordered`` a reordering of ``original`` by disjoint-support moves?

    Operations are matched by exact key (kind, wires, angles), earliest unused
    first.  Then every pair whose relative order differs between the two lists
    is required to have disjoint wire support; such a pair commutes exactly, so
    the reordering changes neither the unitary nor -- because the per-CNOT
    channel acts on that CNOT's own wires -- the noisy channel.
    """
    original, reordered = list(original), list(reordered)
    out = {"same_length": len(original) == len(reordered),
           "same_multiset": False, "n_inversions": 0,
           "offending_pairs": [], "is_certified": False}
    if not out["same_length"]:
        return out
    used = [False] * len(original)
    mapping = []
    for kind, arg in reordered:
        key = _op_key(kind, arg)
        for i, (k0, a0) in enumerate(original):
            if not used[i] and _op_key(k0, a0) == key:
                used[i] = True
                mapping.append(i)
                break
        else:
            out["unmatched"] = str((kind, arg))
            return out
    out["same_multiset"] = all(used)
    out["permutation"] = mapping
    for p in range(len(mapping)):
        for r in range(p + 1, len(mapping)):
            if mapping[p] <= mapping[r]:
                continue                                  # order preserved
            out["n_inversions"] += 1
            k1, a1 = original[mapping[r]]
            k2, a2 = original[mapping[p]]
            if ops_support(k1, a1) & ops_support(k2, a2):
                out["offending_pairs"].append(
                    {"earlier_original_index": mapping[r],
                     "later_original_index": mapping[p],
                     "earlier": str((k1, a1)), "later": str((k2, a2))})
    out["is_certified"] = (out["same_multiset"]
                           and not out["offending_pairs"])
    return out


# ---------------------------------------------------------------------------
# noisy propagation helpers (channel supplied by the caller)
# ---------------------------------------------------------------------------

def build_program(ops, n_qubits: int):
    """``[("U", M) | ("D", (i, j))]``: unitary runs merged, one channel slot
    immediately after every CNOT, on that CNOT's own wires."""
    dim = 2 ** n_qubits
    program, acc = [], np.eye(dim, dtype=complex)
    for kind, arg in ops:
        acc = gate_matrix(kind, arg, n_qubits) @ acc
        if kind == "CNOT":
            program.append(("U", acc))
            acc = np.eye(dim, dtype=complex)
            program.append(("D", tuple(arg)))
    program.append(("U", acc))
    return program


def run_noisy_ops(sigma: np.ndarray, ops, q: float, n_qubits: int,
                  channel=None, program=None) -> np.ndarray:
    """Propagate ``sigma`` through ``ops`` with ``channel(sigma, pair, q, n)``
    after every CNOT.  The channel is injected, never chosen here."""
    if channel is None:
        from .noise import replacement_depolarizing

        def channel(s, pair, qq, n):
            return replacement_depolarizing(s, pair, qq, n_qubits=n)
    for kind, payload in (program if program is not None
                          else build_program(ops, n_qubits)):
        if kind == "U":
            sigma = payload @ sigma @ payload.conj().T
        else:
            sigma = channel(sigma, payload, float(q), n_qubits)
    return sigma
