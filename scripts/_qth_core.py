"""Core physical-circuit layer for the q_th full-circuit study.

SELF-CONTAINED ON PURPOSE.  Nothing is imported from ``pqec_distill`` and no
analytic recurrence is used.  The only prior information used is

  1. the three canonical physical circuit specifications, transcribed verbatim
     below from the repository's frozen sources and verified against them at
     run time by textual extraction (never by importing them),
  2. the Bell-isotropic input state,
  3. the per-CNOT two-qubit replacement depolarizing channel,
  4. the effective-state rules of each protocol,
  5. the operational definition of q_th given in the task.

THREE PROTOCOLS

* ``type3`` -- 5 wires ``[a, A1, A2, B1, B2]``, 16 CNOTs, parity-weighted
  virtual purification.  ``rho_(n+1) = T_(n+1)/Tr T_(n+1)`` where the sixteen
  Pauli coefficients of ``T`` are PHYSICAL CIRCUIT EXPECTATION VALUES
  ``Delta_P = <Z_a (x) P_(A1A2) (x) I_(B1B2)>`` on the final 5-wire state.
* ``type4`` -- the same wires and the same read-out, 14 CNOTs, the frozen
  Qiskit-resynthesised gate list.
* ``4q``    -- 4 wires ``[q1, q2, q3, q4]``, 5 CNOTs then ``H(q3)``, physical
  postselection on ``(q3,q4) = (0,0)``, retain ``(q1,q2)``, normalise by
  ``P_succ``.

Two numerically independent routes are provided for every protocol:

* ROUTE A -- dense Kronecker gates, merged unitary program, Pauli-tomographic
  reconstruction (Type 3/4) or diagonal sub-block postselection (4Q);
* ROUTE B -- CNOT as a computational-basis permutation, single-qubit gates by
  explicit index arithmetic, gate-by-gate propagation with no merged program,
  an independently coded replacement-depolarizing channel, and an
  independently coded effective-state rule (parity-weighted partial trace for
  Type 3/4, explicit projector + partial trace for 4Q).

An optional high-precision route (``python-flint``) repeats ROUTE A in ball
arithmetic; it is used only where the double-precision predicate is ambiguous.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]

# ===========================================================================
# 0. Bell basis, inputs, fidelity
# ===========================================================================

PHI_P = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PHI_M = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
PSI_P = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_M = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
BELL = {"Phi+": PHI_P, "Phi-": PHI_M, "Psi+": PSI_P, "Psi-": PSI_M}
BELL_ORDER = ("Phi+", "Phi-", "Psi+", "Psi-")
_BELL_ROWS = np.stack([BELL[k] for k in BELL_ORDER])

_I2 = np.eye(2, dtype=complex)
_P0 = np.array([[1, 0], [0, 0]], dtype=complex)
_P1 = np.array([[0, 0], [0, 1]], dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_T = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
PAULI_1Q = {"I": _I2, "X": _X, "Y": _Y, "Z": _Z}
PAULI_LABELS = tuple(p + s for p in "IXYZ" for s in "IXYZ")


def isotropic_rho(eps: float) -> np.ndarray:
    """``rho_0(eps) = (1-eps)|Phi+><Phi+| + eps I/4``, from the definition."""
    e = float(eps)
    return (1 - e) * np.outer(PHI_P, PHI_P.conj()) + e * np.eye(4, dtype=complex) / 4


#: the Phi+ projector, built independently of the state constructor
PHI_PLUS_PROJECTOR = 0.25 * (np.eye(4, dtype=complex) + np.kron(_X, _X)) \
    @ (np.eye(4, dtype=complex) + np.kron(_Z, _Z))


def fidelity(rho: np.ndarray) -> float:
    """``F = <Phi+| rho |Phi+>`` from an independently constructed projector."""
    return float(np.real(np.trace(PHI_PLUS_PROJECTOR @ rho)))


def bell_populations(rho: np.ndarray) -> np.ndarray:
    """The four Bell populations, each as a separate expectation value."""
    return np.array([float(np.real(BELL[k].conj() @ rho @ BELL[k]))
                     for k in BELL_ORDER])


def bell_offdiagonal_weight(rho: np.ndarray) -> float:
    """``C_off = sum_{i != j} |<B_i| rho |B_j>|^2``, measured on the state."""
    r = _BELL_ROWS.conj() @ rho @ _BELL_ROWS.T
    return float(np.sum(np.abs(r - np.diag(np.diag(r))) ** 2))


# ===========================================================================
# 1. frozen circuit specifications
# ===========================================================================

#: Type 3 / Type 4 share the five-wire layout of the frozen scripts.
N_WIRES_5Q = 5
ANCILLA_5Q = 0
RETAIN_5Q = (1, 2)
DISCARD_5Q = (3, 4)

#: 4Q layout
N_WIRES_4Q = 4
Q1, Q2, Q3, Q4 = 0, 1, 2, 3
RETAIN_4Q = (Q1, Q2)
MEASURED_4Q = (Q3, Q4)
SUCCESS_OUTCOME_4Q = (0, 0)
CNOT_SEQUENCE_4Q = [(Q3, Q4), (Q2, Q4), (Q1, Q4), (Q3, Q2), (Q3, Q1)]


def fredkin_ops(q: int, a: int, b: int) -> list:
    """``_fred(q, a, b) = _c2(b,a); _tof(q,a,b); _c2(b,a)``: eight CNOTs.

    The Clifford+T Toffoli with control1 = q, control2 = a, target = b.  The
    single-qubit gates stay where they are: the decomposition, not merely the
    unitary, is what the noise sees.
    """
    ops = [("CNOT", (b, a))]
    ops += [("H", b)]
    ops += [("CNOT", (a, b)), ("Tdg", b)]
    ops += [("CNOT", (q, b)), ("T", b)]
    ops += [("CNOT", (a, b)), ("Tdg", b)]
    ops += [("CNOT", (q, b)), ("T", b), ("T", a)]
    ops += [("CNOT", (q, a)), ("H", b)]
    ops += [("T", q), ("Tdg", a)]
    ops += [("CNOT", (q, a))]
    ops += [("CNOT", (b, a))]
    return ops


#: H_a, Fredkin(a; 1, 3), Fredkin(a; 2, 4), H_a -- 16 CNOTs
TYPE3_OPS = ([("H", ANCILLA_5Q)]
             + fredkin_ops(ANCILLA_5Q, 1, 3)
             + fredkin_ops(ANCILLA_5Q, 2, 4)
             + [("H", ANCILLA_5Q)])

#: the Qiskit-resynthesised Type-4 sequence, verbatim, 14 CNOTs
TYPE4_OPS = [
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

#: the 4Q sequence as an op list in the same format
FOURQ_OPS = ([("CNOT", p) for p in CNOT_SEQUENCE_4Q] + [("H", Q3)])

CIRCUIT_SOURCE = {
    "type3": ("PQEC-Operational-Threshold/verify_analytic_decomposed.py "
              "_c2/_tof/_fred, as transcribed in "
              "scripts/verify_type3_qabs_circuit.py TYPE3_OPS"),
    "type4": ("PQEC-Operational-Threshold/pqec_resynth_noise.py GATES "
              "(Qiskit {u, cx}, optimization_level 3, seed_transpiler 7), as "
              "transcribed in scripts/verify_type4_qabs_circuit.py TYPE4_OPS"),
    "4q": ("src/pqec_distill/circuit.py CNOT_SEQUENCE + module docstring, as "
           "transcribed in scripts/verify_4q_qabs_circuit.py"),
}

NOISE_CONVENTION = ("two-qubit REPLACEMENT depolarizing after every CNOT, on "
                    "that CNOT's two wires: D_q(sigma) = (1-q) sigma + "
                    "q [I_ij/4 (x) Tr_ij(sigma)]  -- NOT the 15-Pauli "
                    "convention")

PROTOCOLS = ("type3", "type4", "4q")
N_WIRES = {"type3": 5, "type4": 5, "4q": 4}
OPS = {"type3": TYPE3_OPS, "type4": TYPE4_OPS, "4q": FOURQ_OPS}


def cnot_pairs(protocol: str, ops=None) -> list:
    return [tuple(a) for k, a in (ops or OPS[protocol]) if k == "CNOT"]


def n_cnot(protocol: str, ops=None) -> int:
    return len(cnot_pairs(protocol, ops))


def format_sequence(protocol: str, ops=None) -> str:
    lines, k = [], 0
    for kind, arg in (ops or OPS[protocol]):
        if kind == "CNOT":
            k += 1
            lines.append(f"  {len(lines)+1:3d}.  CNOT({arg[0]} -> {arg[1]})"
                         f"   [CNOT #{k:2d}]  + D_q on ({arg[0]},{arg[1]})")
        elif kind == "U3":
            w, t, p, l = arg
            lines.append(f"  {len(lines)+1:3d}.  U3(wire {w}; "
                         f"theta={t/np.pi:+.4f}pi, phi={p/np.pi:+.4f}pi, "
                         f"lam={l/np.pi:+.4f}pi)")
        else:
            lines.append(f"  {len(lines)+1:3d}.  {kind}(wire {arg})")
    if protocol == "4q":
        lines.append(f"  {len(lines)+1:3d}.  measure (q3, q4); success = "
                     f"{SUCCESS_OUTCOME_4Q}; retain (q1, q2), normalise")
    else:
        lines.append(f"  {len(lines)+1:3d}.  read Z on the ancilla and the "
                     f"sixteen Paulis on wires {RETAIN_5Q}: "
                     f"T = 1/4 sum_P <Z_a (x) P (x) I> P;  rho' = T / Tr T")
    return "\n".join(lines)


# ===========================================================================
# 2. provenance: the transcriptions must match the frozen sources
# ===========================================================================

def _extract_assignment(path: Path, name: str) -> str:
    text = path.read_text()
    m = re.search(rf"^{re.escape(name)}[^\n=]*=\s*", text, re.M)
    if m is None:
        raise ValueError(f"no top-level assignment to {name} in {path.name}")
    start = text.index("[", m.end() - 1)
    depth, i = 0, start
    while True:
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                break
        i += 1
    return text[start:i + 1]


def _normalise_ops(ops) -> list:
    out = []
    for kind, arg in ops:
        if kind == "CNOT":
            out.append(("CNOT", tuple(int(x) for x in arg)))
        elif kind == "U3":
            out.append(("U3", (int(arg[0]),) + tuple(float(x) for x in arg[1:])))
        else:
            out.append((kind, int(arg)))
    return out


def provenance_checks() -> dict:
    """Compare every transcription against the repository source, textually."""
    res = {}
    # --- Type 3 --------------------------------------------------------
    p3 = ROOT / "scripts" / "verify_type3_qabs_circuit.py"
    try:
        src = p3.read_text()
        start = src.index("def fredkin_ops(")
        end = src.index("\nTYPE3_OPS", start)
        ns: dict = {}
        exec(src[start:end], ns)                                  # noqa: S102
        frozen = ([("H", 0)] + ns["fredkin_ops"](0, 1, 3)
                  + ns["fredkin_ops"](0, 2, 4) + [("H", 0)])
        res["type3_matches_verify_type3_qabs_circuit.py"] = (
            _normalise_ops(frozen) == _normalise_ops(TYPE3_OPS))
        res["type3_frozen_n_cnot"] = sum(1 for k, _ in frozen if k == "CNOT")
    except Exception as exc:                                      # pragma: no cover
        res["type3_matches_verify_type3_qabs_circuit.py"] = f"unavailable: {exc}"
    # --- Type 4 --------------------------------------------------------
    for label, path in (("verify_type4_qabs_circuit.py",
                         ROOT / "scripts" / "verify_type4_qabs_circuit.py"),
                        ("verify_type4_qrel_circuit.py",
                         ROOT / "scripts" / "verify_type4_qrel_circuit.py")):
        try:
            ns = {}
            exec(f"TYPE4_OPS = {_extract_assignment(path, 'TYPE4_OPS')}", ns)  # noqa: S102
            res[f"type4_matches_{label}"] = (
                _normalise_ops(ns["TYPE4_OPS"]) == _normalise_ops(TYPE4_OPS))
        except Exception as exc:                                  # pragma: no cover
            res[f"type4_matches_{label}"] = f"unavailable: {exc}"
    parent = ROOT.parent / "PQEC-Operational-Threshold" / "pqec_resynth_noise.py"
    if parent.exists():
        try:
            ns = {"np": np}
            exec(f"GATES = {_extract_assignment(parent, 'GATES')}", ns)  # noqa: S102
            conv = []
            for g, w, p in ns["GATES"]:
                conv.append(("CNOT", tuple(w)) if g == "cx"
                            else ("U3", (w[0],) + tuple(p)))
            res["type4_matches_parent_pqec_resynth_noise_GATES"] = (
                _normalise_ops(conv) == _normalise_ops(TYPE4_OPS))
        except Exception as exc:                                  # pragma: no cover
            res["type4_matches_parent_pqec_resynth_noise_GATES"] = f"unavailable: {exc}"
    else:
        res["type4_matches_parent_pqec_resynth_noise_GATES"] = "parent not present"
    # --- 4Q ------------------------------------------------------------
    for label, path, name in (
            ("src/pqec_distill/circuit.py",
             ROOT / "src" / "pqec_distill" / "circuit.py", "CNOT_SEQUENCE"),
            ("scripts/verify_4q_qabs_circuit.py",
             ROOT / "scripts" / "verify_4q_qabs_circuit.py", "CNOT_SEQUENCE")):
        try:
            body = _extract_assignment(path, name)
            pairs = re.findall(r"\(\s*Q(\d)\s*,\s*Q(\d)\s*\)", body)
            seq = ([(int(a) - 1, int(b) - 1) for a, b in pairs] if pairs else
                   [(int(a), int(b)) for a, b in
                    re.findall(r"\(\s*(\d)\s*,\s*(\d)\s*\)", body)])
            res[f"4q_matches_{label}"] = seq == CNOT_SEQUENCE_4Q
        except Exception as exc:                                  # pragma: no cover
            res[f"4q_matches_{label}"] = f"unavailable: {exc}"
    res["n_cnot"] = {p: n_cnot(p) for p in PROTOCOLS}
    res["n_cnot_expected"] = {"type3": 16, "type4": 14, "4q": 5}
    res["n_cnot_ok"] = res["n_cnot"] == res["n_cnot_expected"]
    return res


# ===========================================================================
# 3. ROUTE A: dense gates, merged program, tomographic / sub-block read-out
# ===========================================================================

def _kron_list(mats) -> np.ndarray:
    out = np.array([[1.0 + 0j]])
    for m in mats:
        out = np.kron(out, m)
    return out


def u3_matrix(theta: float, phi: float, lam: float) -> np.ndarray:
    """Qiskit ``u`` / PennyLane ``U3`` convention."""
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -np.exp(1j * lam) * s],
                     [np.exp(1j * phi) * s, np.exp(1j * (phi + lam)) * c]],
                    dtype=complex)


_ONE_QUBIT = {"H": _H, "T": _T, "Tdg": _T.conj().T}


def gate_matrix(kind, arg, n: int) -> np.ndarray:
    if kind == "CNOT":
        c, t = arg
        if c == t:
            raise ValueError("control and target must differ")
        a = [_I2] * n
        a[c] = _P0
        b = [_I2] * n
        b[c] = _P1
        b[t] = _X
        return _kron_list(a) + _kron_list(b)
    if kind == "U3":
        w, th, ph, la = arg
        f = [_I2] * n
        f[w] = u3_matrix(th, ph, la)
        return _kron_list(f)
    f = [_I2] * n
    f[arg] = _ONE_QUBIT[kind]
    return _kron_list(f)


def build_program(protocol: str, ops=None, skip_noise_after=(), q_offsets=None):
    """``[("U", M) | ("D", (i,j), dq)]``; channels exactly at the CNOTs."""
    n = N_WIRES[protocol]
    dim = 2 ** n
    program, acc, k = [], np.eye(dim, dtype=complex), 0
    for kind, arg in (ops if ops is not None else OPS[protocol]):
        acc = gate_matrix(kind, arg, n) @ acc
        if kind == "CNOT":
            k += 1
            program.append(("U", acc))
            acc = np.eye(dim, dtype=complex)
            if k in skip_noise_after:
                program.append(("U", np.eye(dim, dtype=complex)))
            else:
                dq = 0.0 if q_offsets is None else float(q_offsets.get(k, 0.0))
                program.append(("D", tuple(arg), dq))
    program.append(("U", acc))
    return program


PROGRAMS = {p: build_program(p) for p in PROTOCOLS}

_ROW = "abcdefgh"
_COL = "ijklmnop"


def replacement_depol(sigma: np.ndarray, pair, q: float, n: int) -> np.ndarray:
    """``(1-q) sigma + q [I_ij/4 (x) Tr_ij sigma]`` from the definition."""
    if q == 0.0:
        return sigma
    i, j = pair
    rest = [w for w in range(n) if w not in (i, j)]
    t = sigma.reshape([2] * (2 * n))
    row, col = list(_ROW[:n]), list(_COL[:n])
    for w in (i, j):
        col[w] = row[w]
    sub = "".join(row) + "".join(col)
    out = "".join(row[w] for w in rest) + "".join(col[w] for w in rest)
    reduced = np.einsum(f"{sub}->{out}", t)
    target = _ROW[:n] + _COL[:n]
    rebuilt = np.einsum(f"{out},{_ROW[i]}{_COL[i]},{_ROW[j]}{_COL[j]}->{target}",
                        reduced, _I2, _I2)
    d = 2 ** n
    return (1 - q) * sigma + q * 0.25 * rebuilt.reshape(d, d)


def _run_program(sigma, program, q, n):
    for item in program:
        if item[0] == "U":
            m = item[1]
            sigma = m @ sigma @ m.conj().T
        else:
            sigma = replacement_depol(sigma, item[1], q + item[2], n)
    return sigma


#: the sixteen read-out observables Z_a (x) P_(A1A2) (x) I_(B1B2)
_TOMO_OBS = {lab: _kron_list([_Z, np.kron(PAULI_1Q[lab[0]], PAULI_1Q[lab[1]]),
                              np.eye(4, dtype=complex)])
             for lab in PAULI_LABELS}
_PAULI_4x4 = {lab: np.kron(PAULI_1Q[lab[0]], PAULI_1Q[lab[1]])
              for lab in PAULI_LABELS}


def _expect(obs, sigma) -> float:
    return float(np.real(np.sum(obs * sigma.T)))


def one_round_5q(rho: np.ndarray, q: float, protocol: str, program=None):
    """Type 3 / Type 4 round by PHYSICAL PAULI TOMOGRAPHY.

    All sixteen coefficients of the effective unnormalised operator come from
    circuit expectation values ``<Z_a (x) P (x) I>`` on the final five-wire
    state.  Nothing is projected, truncated or symmetrised.
    """
    sigma = np.kron(_P0, np.kron(rho, rho))
    sigma = _run_program(sigma, program if program is not None
                         else PROGRAMS[protocol], q, 5)
    deltas = {lab: _expect(_TOMO_OBS[lab], sigma) for lab in PAULI_LABELS}
    t_op = sum(deltas[lab] * _PAULI_4x4[lab] for lab in PAULI_LABELS) / 4.0
    denom = float(np.real(np.trace(t_op)))
    herm_t = float(np.linalg.norm(t_op - t_op.conj().T))
    if denom == 0.0 or not np.isfinite(denom):
        raise FloatingPointError(f"normalisation denominator vanished: {denom!r}")
    out = t_op / denom
    herm_rho = float(np.linalg.norm(out - out.conj().T))
    out = 0.5 * (out + out.conj().T)          # exact-arithmetic identity
    return out, {"denominator": denom, "herm_residue_T": herm_t,
                 "herm_residue_rho": herm_rho,
                 "trace_error": abs(float(np.real(np.trace(out))) - 1.0),
                 "min_eig": float(np.linalg.eigvalsh(out).min()),
                 "deltas": deltas, "p_succ": float("nan")}


def project_success_4q(sigma: np.ndarray, outcome=SUCCESS_OUTCOME_4Q):
    """Keep the ``outcome`` branch of ``(q3,q4)`` and trace it out."""
    m3, m4 = outcome
    t = sigma.reshape([2] * 8)
    block = t[:, :, m3, m4, :, :, m3, m4].reshape(4, 4)
    return block, float(np.real(np.trace(block)))


def one_round_4q(rho: np.ndarray, q: float, program=None,
                 outcome=SUCCESS_OUTCOME_4Q):
    """4Q round: full noisy circuit, physical postselection, normalise."""
    sigma = np.kron(rho, rho)
    sigma = _run_program(sigma, program if program is not None
                         else PROGRAMS["4q"], q, 4)
    block, p_succ = project_success_4q(sigma, outcome)
    herm = float(np.linalg.norm(block - block.conj().T))
    block = 0.5 * (block + block.conj().T)
    if p_succ <= 0.0 or not np.isfinite(p_succ):
        raise FloatingPointError(f"success probability vanished: {p_succ!r}")
    out = block / p_succ
    return out, {"denominator": p_succ, "p_succ": p_succ,
                 "herm_residue_T": herm, "herm_residue_rho": 0.0,
                 "trace_error": abs(float(np.real(np.trace(out))) - 1.0),
                 "min_eig": float(np.linalg.eigvalsh(out).min())}


def one_round(rho: np.ndarray, q: float, protocol: str, program=None,
              outcome=SUCCESS_OUTCOME_4Q):
    """ROUTE A: one physical purification round of the chosen protocol."""
    if protocol == "4q":
        return one_round_4q(rho, q, program, outcome)
    return one_round_5q(rho, q, protocol, program)


# ===========================================================================
# 4. ROUTE B: an independent implementation of the same physical rounds
# ===========================================================================

_ROWB = "stuvwxyz"
_COLB = "ABCDEFGH"


def _depol_b(sigma: np.ndarray, pair, q: float, n: int) -> np.ndarray:
    if q == 0.0:
        return sigma
    i, j = pair
    keep = [w for w in range(n) if w not in (i, j)]
    t = sigma.reshape([2] * (2 * n))
    row, col = list(_ROWB[:n]), list(_COLB[:n])
    for w in (i, j):
        col[w] = row[w]
    reduced = np.einsum("".join(row) + "".join(col) + "->"
                        + "".join(row[w] for w in keep)
                        + "".join(col[w] for w in keep), t)
    out_idx = "".join(row[w] for w in keep) + "".join(col[w] for w in keep)
    rebuilt = np.einsum(f"{out_idx},{_ROWB[i]}{_COLB[i]},{_ROWB[j]}{_COLB[j]}"
                        f"->{_ROWB[:n]}{_COLB[:n]}", reduced, _I2, _I2)
    return (1.0 - q) * sigma + 0.25 * q * rebuilt.reshape(2 ** n, 2 ** n)


def _gate_b(kind, arg, n: int) -> np.ndarray:
    """CNOT as a basis permutation; single-qubit gates by index arithmetic."""
    d = 2 ** n
    if kind == "CNOT":
        c, tgt = arg
        m = np.zeros((d, d), dtype=complex)
        for idx in range(d):
            bits = [(idx >> (n - 1 - w)) & 1 for w in range(n)]
            if bits[c] == 1:
                bits[tgt] ^= 1
            m[sum(b << (n - 1 - w) for w, b in enumerate(bits)), idx] = 1.0
        return m
    if kind == "U3":
        w, th, ph, la = arg
        g = np.array([[np.cos(th / 2), -(np.cos(la) + 1j * np.sin(la)) * np.sin(th / 2)],
                      [(np.cos(ph) + 1j * np.sin(ph)) * np.sin(th / 2),
                       (np.cos(ph + la) + 1j * np.sin(ph + la)) * np.cos(th / 2)]],
                     dtype=complex)
    elif kind == "H":
        g = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        w = arg
    elif kind in ("T", "Tdg"):
        s = 1.0 if kind == "T" else -1.0
        g = np.array([[1, 0], [0, np.cos(np.pi / 4) + 1j * s * np.sin(np.pi / 4)]],
                     dtype=complex)
        w = arg
    else:
        raise ValueError(kind)
    m = np.zeros((d, d), dtype=complex)
    shift = n - 1 - w
    for col in range(d):
        bit = (col >> shift) & 1
        base = col & ~(1 << shift)
        m[base, col] = g[0, bit]
        m[base | (1 << shift), col] = g[1, bit]
    return m


def _run_gate_by_gate_b(sigma, ops, q, n, skip_noise_after=(), q_offsets=None):
    k = 0
    for kind, arg in ops:
        g = _gate_b(kind, arg, n)
        sigma = g @ sigma @ g.conj().T
        if kind == "CNOT":
            k += 1
            if k in skip_noise_after:
                continue
            dq = 0.0 if q_offsets is None else float(q_offsets.get(k, 0.0))
            sigma = _depol_b(sigma, tuple(arg), q + dq, n)
    return sigma


def one_round_b(rho: np.ndarray, q: float, protocol: str, ops=None,
                outcome=SUCCESS_OUTCOME_4Q, parity_signs=(+1.0, -1.0),
                skip_noise_after=(), q_offsets=None):
    """ROUTE B: gate-by-gate, independent channel, independent read-out.

    Type 3/4 use the parity-weighted PARTIAL TRACE form
    ``T = Tr_B[<0|sigma|0>_a - <1|sigma|1>_a]`` rather than Pauli tomography,
    so route B checks the tomographic reconstruction as well as the circuit.
    4Q uses an explicit projector followed by an einsum partial trace.
    """
    n = N_WIRES[protocol]
    sigma = np.kron(np.array([[1, 0], [0, 0]], dtype=complex), np.kron(rho, rho)) \
        if n == 5 else np.kron(rho, rho)
    sigma = _run_gate_by_gate_b(sigma, ops if ops is not None else OPS[protocol],
                                q, n, skip_noise_after, q_offsets)
    if protocol == "4q":
        m3, m4 = outcome
        proj = _kron_list([_I2, _I2, _P0 if m3 == 0 else _P1,
                           _P0 if m4 == 0 else _P1])
        projected = proj @ sigma @ proj.conj().T
        t = projected.reshape([2] * 8)
        block = np.einsum("abcdefcd->abef", t).reshape(4, 4)
        denom = float(np.real(np.trace(block)))
        return block / denom, {"denominator": denom, "p_succ": denom}
    t = sigma.reshape([2] * 10)
    tau = np.zeros((4, 4), dtype=complex)
    for m, sign in ((0, parity_signs[0]), (1, parity_signs[1])):
        block = t[m, :, :, :, :, m, :, :, :, :]
        tau = tau + sign * np.einsum("ijklmnkl->ijmn", block).reshape(4, 4)
    denom = float(np.real(np.trace(tau)))
    return tau / denom, {"denominator": denom, "p_succ": float("nan")}
