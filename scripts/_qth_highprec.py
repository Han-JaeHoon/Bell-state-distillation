"""Optional high-precision (ball-arithmetic) repeat of ROUTE A.

WHY THIS EXISTS
---------------
Type 3 and Type 4 amplify the Bell-OFF-diagonal (transverse) direction.  The
Bell-isotropic input is exactly Bell diagonal, so in double precision the
transverse sector starts at round-off and grows geometrically; after roughly
40-60 rounds it is no longer negligible and the simulated orbit stops being a
faithful realisation of the exact Bell-diagonal initial-value problem.  Near
the q_th boundary the fidelity has not yet settled by then, so the
double-precision predicate returns AMBIGUOUS rather than PASS or FAIL.

This module repeats exactly the same physical circuit in ``python-flint`` ball
arithmetic at a configurable precision (200 bits by default).  The transverse
seed then starts around ``1e-60`` and stays negligible for several hundred
rounds, which resolves the ambiguous band.

IMPORTANT MODELLING NOTE, stated explicitly because it is an assumption:
the frozen Type-4 gate list stores its U3 angles as float64 renderings of
multiples of ``pi/4``.  At double precision that is the circuit.  To make
high precision mean anything the angles are snapped to the exact multiple of
``pi/4``; the snap residual is measured and reported by
:func:`angle_snap_report` and is of order ``1e-16``.  Type 3 uses only ``H``,
``T`` and ``T^dagger``, which are exact at any precision.

``python-flint`` is an OPTIONAL dependency (``pip install python-flint``).
Everything degrades gracefully when it is absent: :data:`AVAILABLE` is False
and the double-precision study still runs, reporting the ambiguous band as
such.
"""

from __future__ import annotations

import numpy as np

try:
    import flint

    AVAILABLE = True
except Exception:                                                 # pragma: no cover
    flint = None
    AVAILABLE = False

from _qth_core import (                                           # noqa: E402
    N_WIRES, OPS, PROTOCOLS, SUCCESS_OUTCOME_4Q, isotropic_rho,
)

DEFAULT_PREC_BITS = 400


# ---------------------------------------------------------------------------
# angle handling
# ---------------------------------------------------------------------------

def angle_snap_report() -> dict:
    """How far the frozen Type-4 angles are from exact multiples of pi/4."""
    worst, rows = 0.0, []
    for kind, arg in OPS["type4"]:
        if kind != "U3":
            continue
        for a in arg[1:]:
            k = round(a / (np.pi / 4))
            d = abs(a - k * np.pi / 4)
            worst = max(worst, d)
            rows.append({"angle": float(a), "k_times_pi_over_4": int(k),
                         "abs_residual": float(d)})
    return {"max_abs_residual": worst, "n_angles": len(rows), "rows": rows}


def _snap_k(angle: float) -> int:
    return int(round(angle / (np.pi / 4)))


# ---------------------------------------------------------------------------
# high-precision gate primitives
# ---------------------------------------------------------------------------

def _ctx(prec_bits: int):
    flint.ctx.prec = prec_bits


def _acb(re, im=0):
    return flint.acb(re, im)


def _one_qubit_gate(kind, arg, prec_bits: int):
    """Return (wire, 2x2 nested list of acb)."""
    pi = flint.arb.pi()
    if kind == "H":
        s = 1 / flint.arb(2).sqrt()
        return arg, [[_acb(s), _acb(s)], [_acb(s), _acb(-s)]]
    if kind in ("T", "Tdg"):
        sgn = 1 if kind == "T" else -1
        th = pi / 4
        return arg, [[_acb(1), _acb(0)],
                     [_acb(0), flint.acb(th.cos(), sgn * th.sin())]]
    if kind == "U3":
        w, th_f, ph_f, la_f = arg
        th = pi * _snap_k(th_f) / 4
        ph = pi * _snap_k(ph_f) / 4
        la = pi * _snap_k(la_f) / 4
        c, s = (th / 2).cos(), (th / 2).sin()
        e_la = flint.acb(la.cos(), la.sin())
        e_ph = flint.acb(ph.cos(), ph.sin())
        e_pl = flint.acb((ph + la).cos(), (ph + la).sin())
        return w, [[_acb(c), -e_la * s], [e_ph * s, e_pl * c]]
    raise ValueError(kind)


def _gate_matrix_hp(kind, arg, n: int, prec_bits: int):
    d = 2 ** n
    rows = [[_acb(0)] * d for _ in range(d)]
    if kind == "CNOT":
        c, t = arg
        for idx in range(d):
            bits = [(idx >> (n - 1 - w)) & 1 for w in range(n)]
            if bits[c] == 1:
                bits[t] ^= 1
            rows[sum(b << (n - 1 - w) for w, b in enumerate(bits))][idx] = _acb(1)
        return flint.acb_mat(rows)
    w, g = _one_qubit_gate(kind, arg, prec_bits)
    shift = n - 1 - w
    for col in range(d):
        bit = (col >> shift) & 1
        base = col & ~(1 << shift)
        rows[base][col] = g[0][bit]
        rows[base | (1 << shift)][col] = g[1][bit]
    return flint.acb_mat(rows)


def _dagger(m):
    d = m.nrows()
    return flint.acb_mat([[m[j, i].conjugate() for j in range(d)]
                          for i in range(d)])


def build_program_hp(protocol: str, prec_bits: int = DEFAULT_PREC_BITS,
                     ops=None):
    """``[("U", M, Mdag) | ("D", (i,j))]`` with unitary runs merged."""
    _ctx(prec_bits)
    n = N_WIRES[protocol]
    d = 2 ** n
    eye = flint.acb_mat(d, d)
    for i in range(d):
        eye[i, i] = _acb(1)
    program, acc = [], eye
    for kind, arg in (ops if ops is not None else OPS[protocol]):
        acc = _gate_matrix_hp(kind, arg, n, prec_bits) * acc
        if kind == "CNOT":
            program.append(("U", acc, _dagger(acc)))
            acc = flint.acb_mat(d, d)
            for i in range(d):
                acc[i, i] = _acb(1)
            program.append(("D", tuple(arg)))
    program.append(("U", acc, _dagger(acc)))
    return program


# ---------------------------------------------------------------------------
# high-precision replacement depolarizing, from the definition
# ---------------------------------------------------------------------------

_INDEX_CACHE: dict = {}


def _index_maps(n: int, pair):
    key = (n, tuple(pair))
    if key in _INDEX_CACHE:
        return _INDEX_CACHE[key]
    i, j = pair
    rest = [w for w in range(n) if w not in (i, j)]
    d = 2 ** n
    pair_bits, rest_idx = [], []
    for idx in range(d):
        bits = [(idx >> (n - 1 - w)) & 1 for w in range(n)]
        pair_bits.append((bits[i] << 1) | bits[j])
        rest_idx.append(sum(bits[w] << (len(rest) - 1 - k)
                            for k, w in enumerate(rest)))
    # for each (reduced row, reduced col, pair value) the full index
    full = {}
    for idx in range(d):
        full[(rest_idx[idx], pair_bits[idx])] = idx
    _INDEX_CACHE[key] = (pair_bits, rest_idx, full, 2 ** len(rest))
    return _INDEX_CACHE[key]


def _depol_hp(sigma, pair, q, n):
    if q == 0:
        return sigma
    pair_bits, rest_idx, full, dr = _index_maps(n, pair)
    d = 2 ** n
    one_minus_q = flint.arb(1) - q
    quarter_q = q / 4
    reduced = [[_acb(0)] * dr for _ in range(dr)]
    for r in range(dr):
        for c in range(dr):
            acc = _acb(0)
            for b in range(4):
                acc = acc + sigma[full[(r, b)], full[(c, b)]]
            reduced[r][c] = acc
    out = flint.acb_mat(d, d)
    for r in range(d):
        pr, rr = pair_bits[r], rest_idx[r]
        for c in range(d):
            v = one_minus_q * sigma[r, c]
            if pair_bits[c] == pr:
                v = v + quarter_q * reduced[rr][rest_idx[c]]
            out[r, c] = v
    return out


# ---------------------------------------------------------------------------
# high-precision rounds
# ---------------------------------------------------------------------------

_PAULI_HP = None


def _pauli_hp():
    global _PAULI_HP
    if _PAULI_HP is None:
        i2 = [[_acb(1), _acb(0)], [_acb(0), _acb(1)]]
        x = [[_acb(0), _acb(1)], [_acb(1), _acb(0)]]
        y = [[_acb(0), _acb(0, -1)], [_acb(0, 1), _acb(0)]]
        z = [[_acb(1), _acb(0)], [_acb(0), _acb(-1)]]
        _PAULI_HP = {"I": i2, "X": x, "Y": y, "Z": z}
    return _PAULI_HP


def isotropic_rho_hp(eps_num: int, eps_den: int, prec_bits: int):
    """``rho_0`` built from an exact rational epsilon = eps_num / eps_den."""
    _ctx(prec_bits)
    e = flint.arb(eps_num) / eps_den
    one_m = flint.arb(1) - e
    rho = flint.acb_mat(4, 4)
    half = one_m / 2
    quarter = e / 4
    rho[0, 0] = _acb(half + quarter)
    rho[3, 3] = _acb(half + quarter)
    rho[0, 3] = _acb(half)
    rho[3, 0] = _acb(half)
    rho[1, 1] = _acb(quarter)
    rho[2, 2] = _acb(quarter)
    return rho


def _kron_hp(a, b):
    na, nb = a.nrows(), b.nrows()
    out = flint.acb_mat(na * nb, na * nb)
    for i in range(na):
        for j in range(na):
            aij = a[i, j]
            for k in range(nb):
                for l in range(nb):
                    out[i * nb + k, j * nb + l] = aij * b[k, l]
    return out


def one_round_hp(rho, q_num, q_den, protocol: str, program,
                 prec_bits: int = DEFAULT_PREC_BITS):
    """ROUTE A repeated in ball arithmetic.  ``q = q_num / q_den`` exactly."""
    _ctx(prec_bits)
    q = flint.arb(q_num) / q_den
    n = N_WIRES[protocol]
    r2 = _kron_hp(rho, rho)
    if n == 5:
        proj = flint.acb_mat(2, 2)
        proj[0, 0] = _acb(1)
        sigma = _kron_hp(proj, r2)
    else:
        sigma = r2
    for item in program:
        if item[0] == "U":
            sigma = item[1] * sigma * item[2]
        else:
            sigma = _depol_hp(sigma, item[1], q, n)
    if protocol == "4q":
        m3, m4 = SUCCESS_OUTCOME_4Q
        block = flint.acb_mat(4, 4)
        for r in range(4):
            for c in range(4):
                block[r, c] = sigma[(r << 2) | (m3 << 1) | m4,
                                    (c << 2) | (m3 << 1) | m4]
        denom = block[0, 0] + block[1, 1] + block[2, 2] + block[3, 3]
    else:
        # parity-weighted Pauli tomography, exactly as ROUTE A
        tau = flint.acb_mat(4, 4)
        for r in range(4):
            for c in range(4):
                acc = _acb(0)
                for b in range(4):
                    acc = acc + sigma[(0 << 4) | (r << 2) | b,
                                      (0 << 4) | (c << 2) | b]
                    acc = acc - sigma[(1 << 4) | (r << 2) | b,
                                      (1 << 4) | (c << 2) | b]
                tau[r, c] = acc
        block = tau
        denom = block[0, 0] + block[1, 1] + block[2, 2] + block[3, 3]
    out = flint.acb_mat(4, 4)
    for r in range(4):
        for c in range(4):
            out[r, c] = block[r, c] / denom
    return out, denom


def tighten(m):
    """Replace every ball by its midpoint (radius zero).

    Ball arithmetic propagates a PESSIMISTIC error bound: the enclosure of the
    5-wire state widens by about four orders of magnitude per round even
    though the true error does not, so after ~15 rounds at 200 bits the balls
    are useless.  Re-centring after each round turns the computation into
    ordinary high-precision arithmetic while keeping the per-round enclosure
    as a certified bound on that round, which is what
    ``max_abs_radius`` records.  The true accumulated error then behaves like
    round-off at the working precision, i.e. ~1e-120 at 400 bits, which is
    some sixty orders of magnitude below the double-precision seed that causes
    the transverse contamination this route exists to remove.
    """
    d = m.nrows()
    return flint.acb_mat([[flint.acb(m[i, j].real.mid(), m[i, j].imag.mid())
                           for j in range(d)] for i in range(d)])


def to_numpy(m) -> np.ndarray:
    d = m.nrows()
    return np.array([[complex(float(m[i, j].real), float(m[i, j].imag))
                      for j in range(d)] for i in range(d)], dtype=complex)


def fidelity_hp(rho):
    """``F = <Phi+| rho |Phi+> = (rho00 + rho03 + rho30 + rho33)/2``, in arb."""
    return ((rho[0, 0] + rho[0, 3] + rho[3, 0] + rho[3, 3]) / 2).real


def bell_offdiagonal_weight_hp(rho):
    """Squared Frobenius norm of the Bell off-diagonal block, in arb."""
    p = flint.acb_mat([[_acb(1), _acb(0), _acb(0), _acb(1)],
                       [_acb(1), _acb(0), _acb(0), _acb(-1)],
                       [_acb(0), _acb(1), _acb(1), _acb(0)],
                       [_acb(0), _acb(1), _acb(-1), _acb(0)]])
    s = flint.arb(2).sqrt()
    r = flint.acb_mat(4, 4)
    for i in range(4):
        for j in range(4):
            acc = _acb(0)
            for a in range(4):
                for b in range(4):
                    acc = acc + p[i, a].conjugate() * rho[a, b] * p[j, b]
            r[i, j] = acc / (s * s)
    tot = flint.arb(0)
    for i in range(4):
        for j in range(4):
            if i != j:
                tot = tot + r[i, j].abs_upper() ** 2
    return tot


def max_abs_radius(m) -> float:
    """Largest ball radius in the matrix: the arithmetic's own error bound."""
    d = m.nrows()
    worst = 0.0
    for i in range(d):
        for j in range(d):
            worst = max(worst, float(m[i, j].real.rad()),
                        float(m[i, j].imag.rad()))
    return worst
