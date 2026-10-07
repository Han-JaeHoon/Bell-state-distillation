"""Circuit-level numerical study of the noisy 6Q-L16 distributed LOCC
virtual-distillation circuit.

This script asks the circuit three questions and does not presuppose the
answers:

  A. if the input is an arbitrary Bell-diagonal state, is the reconstructed
     effective operator still Bell diagonal?
  B. if the input is Bell isotropic, is the output still Bell isotropic?
  C. is there a range of CNOT noise in which one noisy round raises the Phi+
     fidelity?

No analytic noisy recurrence, no Bell-diagonal projection, no twirling, no
assumed invariant plane and no threshold formula is used anywhere.  The
reconstructed operator is called T_q(rho); it is NOT called rho^2 at q > 0.
This is ONE ROUND ONLY -- the normalised operator is never fed back.

Run:
    python scripts/verify_locc_vd_noisy.py
"""

from __future__ import annotations

import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

from pqec_distill.locc_vd import (  # noqa: E402
    PAULI_LABELS, delta_6q_all_paulis, pauli_product, reconstruct_operator,
)
from pqec_distill.locc_vd_noisy import (  # noqa: E402
    ALICE_TRIPLE, BOB_TRIPLE, BELL_ORDER, BOB_WIRES, CIRCUIT_SOURCE, DIM,
    L16_OPS, N_CNOT, N_CNOT_ALICE, N_CNOT_BOB, N_WIRES, NOISE_CONVENTION,
    PROGRAM, bell_diagonal_rho, bell_matrix, bell_offdiagonal_frobenius,
    bell_populations, circuit_unitary, cnot_wire_pairs, crosses_party_cut,
    deltas_all_paulis_noisy, effective_operator, final_state_6q_noisy,
    format_sequence, ideal_local_fredkin, initial_state, isotropic_rho,
    local_block_unitary, phi_plus_fidelity,
)
from pqec_distill.noise import replacement_depolarizing  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "data" / "locc_vd_noisy"
FIG = ROOT / "results" / "figures" / "locc_vd_noisy"

SEED = 20261007
Q_GRID = (0.0, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
EPS_GRID = (0.0, 0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 2.0 / 3.0,
            0.75, 0.90, 1.0)
ROUNDOFF_SCALE = 1e-12          # what counts as "consistent with round-off"


# ===========================================================================
# Route B: an independent noisy-circuit implementation
# ===========================================================================
#
# Route A is the module's merged-program path plus
# pqec_distill.noise.replacement_depolarizing.  Route B below applies the gates
# one at a time from their own matrices and implements the channel with its own
# einsum partial trace, sharing no helper with route A.

_ROW, _COL = "abcdef", "ghijkl"
_I2 = np.eye(2, dtype=complex)


def _depol_route_b(sigma: np.ndarray, pair, q: float) -> np.ndarray:
    if q == 0.0:
        return sigma
    i, j = pair
    rest = [w for w in range(N_WIRES) if w not in (i, j)]
    t = sigma.reshape([2] * (2 * N_WIRES))
    row, col = list(_ROW), list(_COL)
    for w in (i, j):
        col[w] = row[w]
    reduced = np.einsum("".join(row) + "".join(col) + "->"
                        + "".join(row[w] for w in rest)
                        + "".join(col[w] for w in rest), t)
    out_idx = "".join(row[w] for w in rest) + "".join(col[w] for w in rest)
    target = "".join(_ROW) + "".join(_COL)
    rebuilt = np.einsum(f"{out_idx},{_ROW[i]}{_COL[i]},{_ROW[j]}{_COL[j]}"
                        f"->{target}", reduced, _I2, _I2)
    return (1 - q) * sigma + q * 0.25 * rebuilt.reshape(DIM, DIM)


def _gate_route_b(kind, arg) -> np.ndarray:
    h = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    t = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
    one = {"H": h, "T": t, "Tdg": t.conj().T}
    if kind == "CNOT":
        c, tgt = arg
        perm = np.arange(DIM)
        for idx in range(DIM):
            bits = [(idx >> (N_WIRES - 1 - w)) & 1 for w in range(N_WIRES)]
            if bits[c] == 1:
                bits[tgt] ^= 1
            perm[idx] = sum(b << (N_WIRES - 1 - w) for w, b in enumerate(bits))
        m = np.zeros((DIM, DIM), dtype=complex)
        for col, row in enumerate(perm):
            m[row, col] = 1.0
        return m
    factors = [_I2] * N_WIRES
    factors[arg] = one[kind]
    out = np.array([[1.0 + 0j]])
    for f in factors:
        out = np.kron(out, f)
    return out


def final_state_route_b(rho: np.ndarray, q: float) -> np.ndarray:
    sigma = initial_state(rho)
    for kind, arg in L16_OPS:
        g = _gate_route_b(kind, arg)
        sigma = g @ sigma @ g.conj().T
        if kind == "CNOT":
            sigma = _depol_route_b(sigma, tuple(arg), q)
    return sigma


# ===========================================================================
# structural and convention checks
# ===========================================================================

def structural_checks() -> dict:
    out = {
        "n_cnot_alice": N_CNOT_ALICE, "n_cnot_bob": N_CNOT_BOB,
        "n_cnot_total": N_CNOT,
        "cross_party_two_qubit_gates": crosses_party_cut(),
        "alice_block_vs_ideal_fredkin": float(np.max(np.abs(
            local_block_unitary("alice") - ideal_local_fredkin("alice")))),
        "bob_block_vs_ideal_fredkin": float(np.max(np.abs(
            local_block_unitary("bob") - ideal_local_fredkin("bob")))),
        "full_unitary_unitarity": float(np.max(np.abs(
            circuit_unitary().conj().T @ circuit_unitary() - np.eye(DIM)))),
    }
    # the two local blocks must commute (disjoint wires)
    ua, ub = local_block_unitary("alice"), local_block_unitary("bob")
    out["local_blocks_commutator"] = float(np.max(np.abs(ua @ ub - ub @ ua)))
    return out


def noise_convention_checks(rng, n_trials=10) -> dict:
    """The q used here must be the same parameter as the existing studies.

    (i) against the parent repository's eps2 Kraus channel
        ``noisy_bell_state.global_depol_kraus``, embedded on a wire pair;
    (ii) against the parent repository's own ``replacement_depol`` used by the
        iterated Type-3 analysis (5 wires, its native size).
    """
    res = {}
    # ---- (i) parent eps2 Kraus channel ---------------------------------
    paulis = [np.eye(2, dtype=complex),
              np.array([[0, 1], [1, 0]], dtype=complex),
              np.array([[0, -1j], [1j, 0]], dtype=complex),
              np.array([[1, 0], [0, -1]], dtype=complex)]
    two_q = [np.kron(p, s) for p in paulis for s in paulis]

    def embed(mat2q, pair):
        i, j = pair
        t = mat2q.reshape(2, 2, 2, 2)                 # (i,j | i',j')
        full = np.zeros([2] * (2 * N_WIRES), dtype=complex)
        eye = np.eye(2, dtype=complex)
        others = [w for w in range(N_WIRES) if w not in (i, j)]
        big = t
        for _ in others:
            big = np.tensordot(big, eye, axes=0)
        # axes: i, j, i', j', then (w, w') pairs for each other wire
        order = [0, 1] + [4 + 2 * k for k in range(len(others))]
        order += [2, 3] + [5 + 2 * k for k in range(len(others))]
        big = big.transpose(order)
        pos = [i, j] + others
        perm_row = [pos.index(w) for w in range(N_WIRES)]
        perm = perm_row + [N_WIRES + p for p in perm_row]
        full = big.transpose(perm)
        return full.reshape(DIM, DIM)

    worst_kraus = 0.0
    for _ in range(n_trials):
        m = rng.normal(size=(DIM, DIM)) + 1j * rng.normal(size=(DIM, DIM))
        sigma = m @ m.conj().T
        sigma /= np.trace(sigma)
        for pair in ((0, 2), (1, 3), (2, 4)):
            for q in (0.05, 0.2, 0.5):
                # parent eps2 convention: (1-eps) rho + (eps/16) sum_P P rho P
                acc = (1 - q) * sigma
                for p2 in two_q:
                    e = embed(p2, pair)
                    acc = acc + (q / 16.0) * (e @ sigma @ e.conj().T)
                ours = replacement_depolarizing(sigma, pair, q, n_qubits=N_WIRES)
                worst_kraus = max(worst_kraus, float(np.max(np.abs(acc - ours))))
    res["vs_parent_eps2_kraus_channel"] = worst_kraus

    # ---- (ii) parent replacement_depol on its native 5 wires ------------
    try:
        from pqec_distill.swap_test_source import load_parent_module
        parent = load_parent_module()
        worst_parent = 0.0
        for _ in range(5):
            m = rng.normal(size=(32, 32)) + 1j * rng.normal(size=(32, 32))
            s5 = m @ m.conj().T
            s5 /= np.trace(s5)
            for pair in ((0, 1), (1, 3), (2, 4)):
                for q in (0.05, 0.3):
                    a = parent.replacement_depol(s5, pair, q)
                    b = replacement_depolarizing(s5, pair, q, n_qubits=5)
                    worst_parent = max(worst_parent, float(np.max(np.abs(a - b))))
        res["vs_parent_iterated_replacement_depol"] = worst_parent
    except Exception as exc:                                  # pragma: no cover
        res["vs_parent_iterated_replacement_depol"] = f"unavailable: {exc}"

    # ---- channel properties on the 6-wire register ----------------------
    worst = {"trace": 0.0, "herm": 0.0, "min_eig": 0.0, "identity_at_q0": 0.0}
    for _ in range(5):
        m = rng.normal(size=(DIM, DIM)) + 1j * rng.normal(size=(DIM, DIM))
        sigma = m @ m.conj().T
        sigma /= np.trace(sigma)
        for pair in cnot_wire_pairs()[:6]:
            for q in (0.0, 0.1, 0.5, 1.0):
                o = replacement_depolarizing(sigma, pair, q, n_qubits=N_WIRES)
                worst["trace"] = max(worst["trace"], abs(np.trace(o) - 1.0))
                worst["herm"] = max(worst["herm"],
                                    np.linalg.norm(o - o.conj().T))
                worst["min_eig"] = min(worst["min_eig"], float(
                    np.linalg.eigvalsh(0.5 * (o + o.conj().T)).min()))
            worst["identity_at_q0"] = max(worst["identity_at_q0"], float(
                np.max(np.abs(replacement_depolarizing(sigma, pair, 0.0,
                                                       n_qubits=N_WIRES) - sigma))))
    res.update({f"channel_{k}": float(v) for k, v in worst.items()})
    return res


def state_validity(rho, q) -> dict:
    sigma = final_state_6q_noisy(rho, q)
    eig = np.linalg.eigvalsh(0.5 * (sigma + sigma.conj().T))
    return {"trace_minus_1": abs(float(np.real(np.trace(sigma))) - 1.0),
            "hermiticity": float(np.linalg.norm(sigma - sigma.conj().T)),
            "min_eig": float(eig.min())}


# ===========================================================================
# test states
# ===========================================================================

def bell_diagonal_test_set(rng, n_random=24) -> list:
    sets = []
    for k, name in enumerate(BELL_ORDER):
        p = [0.0] * 4
        p[k] = 1.0
        sets.append((f"vertex_{name}", "vertex", p))
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for i, j in edges:
        for w in (0.5, 0.8):
            p = [0.0] * 4
            p[i], p[j] = w, 1 - w
            sets.append((f"edge_{BELL_ORDER[i]}_{BELL_ORDER[j]}_{w:g}",
                         "edge", p))
    faces = [(0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)]
    for tri in faces:
        p = [0.0] * 4
        for t in tri:
            p[t] = 1.0 / 3.0
        sets.append((f"face_{''.join(BELL_ORDER[t][:3] for t in tri)}",
                     "face", p))
    for name, p in (("asym_strong", [0.97, 0.02, 0.005, 0.005]),
                    ("asym_mid", [0.7, 0.2, 0.08, 0.02]),
                    ("asym_flat", [0.28, 0.26, 0.24, 0.22]),
                    ("uniform", [0.25, 0.25, 0.25, 0.25])):
        sets.append((name, "asymmetric", p))
    for k in range(n_random):
        sets.append((f"dirichlet_{k}", "interior",
                     list(rng.dirichlet(np.ones(4) * (0.3 if k % 2 else 3.0)))))
    return sets


def general_test_states(rng, n=8) -> list:
    out = []
    for k in range(n):
        m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        r = m @ m.conj().T
        out.append((f"general_mixed_{k}", r / np.trace(r)))
    for k in range(4):
        v = rng.normal(size=4) + 1j * rng.normal(size=4)
        v /= np.linalg.norm(v)
        out.append((f"general_pure_{k}", np.outer(v, v.conj())))
    return out


# ===========================================================================
# the three studies
# ===========================================================================

def check_q0_against_frozen_ideal(rng) -> dict:
    """The decomposed circuit at q = 0 must reproduce the frozen ideal 6Q."""
    states = [(f"bell_{n}", bell_diagonal_rho([1.0 if i == k else 0.0
                                               for i in range(4)]))
              for k, n in enumerate(BELL_ORDER)]
    states += [(f"isotropic_{e:g}", isotropic_rho(e))
               for e in (0.0, 0.1, 0.3, 2.0 / 3.0, 1.0)]
    states += [(f"belldiag_{k}", bell_diagonal_rho(rng.dirichlet(np.ones(4))))
               for k in range(6)]
    states += general_test_states(rng, 6)
    worst_d = worst_t = 0.0
    worst_at = None
    for name, rho in states:
        noisy0 = deltas_all_paulis_noisy(rho, 0.0)
        ideal = delta_6q_all_paulis(rho)
        for lab in PAULI_LABELS:
            e = abs(noisy0[lab][0] - ideal[lab][0])
            if e > worst_d:
                worst_d, worst_at = e, (name, lab)
        t_new = reconstruct_operator({k: v[0] for k, v in noisy0.items()})
        t_old = reconstruct_operator({k: v[0] for k, v in ideal.items()})
        worst_t = max(worst_t, float(np.linalg.norm(t_new - t_old)))
    return {"n_states": len(states), "max_abs_delta_diff": worst_d,
            "worst_state_observable": worst_at,
            "max_frobenius_T0_diff": worst_t}


def study_a_bell_diagonality(bd_states, q_grid) -> tuple:
    rows, worst = [], {"c": -1.0}
    for name, family, p in bd_states:
        rho = bell_diagonal_rho(p)
        for q in q_grid:
            info = effective_operator(rho, q)
            r = info["R"]
            c_t = bell_offdiagonal_frobenius(info["T"])
            c_r = float("nan") if r is None else bell_offdiagonal_frobenius(r)
            row = {"state": name, "family": family, "q": q,
                   "p_in": ";".join(f"{x:.6f}" for x in p),
                   "C_Bell_T": c_t, "C_Bell_R": c_r, "D_q": info["D"],
                   "T_hermiticity": info["T_hermiticity"],
                   "R_min_eig": info.get("R_min_eig"),
                   "R_trace": info.get("R_trace"),
                   "R_is_psd": info.get("R_is_psd_1e-10")}
            rows.append(row)
            if not np.isnan(c_r) and c_r > worst["c"]:
                worst = {"c": c_r, "state": name, "q": q, "family": family}
    return rows, worst


def study_b_isotropy(eps_grid, q_grid) -> tuple:
    rows = []
    worst_iso = {"d": -1.0}
    for eps in eps_grid:
        rho = isotropic_rho(eps)
        for q in q_grid:
            info = effective_operator(rho, q)
            r = info["R"]
            if r is None:
                continue
            p = bell_populations(r)
            d_iso = max(abs(p[1] - p[2]), abs(p[1] - p[3]), abs(p[2] - p[3]))
            pairs = {"Phi-_vs_Psi+": abs(p[1] - p[2]),
                     "Phi-_vs_Psi-": abs(p[1] - p[3]),
                     "Psi+_vs_Psi-": abs(p[2] - p[3])}
            rows.append({"epsilon": eps, "q": q,
                         "p_Phi_plus": p[0], "p_Phi_minus": p[1],
                         "p_Psi_plus": p[2], "p_Psi_minus": p[3],
                         "pop_sum": float(p.sum()),
                         "delta_iso": d_iso, **pairs,
                         "C_Bell_R": bell_offdiagonal_frobenius(r),
                         "D_q": info["D"], "R_min_eig": info["R_min_eig"]})
            if d_iso > worst_iso["d"]:
                worst_iso = {"d": d_iso, "epsilon": eps, "q": q}
    return rows, worst_iso


def delta_f(eps: float, q: float) -> float:
    """F_out - F_in, both from the circuit-reconstructed operator."""
    info = effective_operator(isotropic_rho(eps), q)
    if info["R"] is None:
        return float("nan")
    return phi_plus_fidelity(info["R"]) - (1.0 - 3.0 * eps / 4.0)


def study_c_fidelity(eps_grid, q_grid_fine) -> tuple:
    rows = []
    for eps in eps_grid:
        f_in = 1.0 - 3.0 * eps / 4.0
        for q in q_grid_fine:
            info = effective_operator(isotropic_rho(eps), q)
            r = info["R"]
            f_out = float("nan") if r is None else phi_plus_fidelity(r)
            rows.append({"epsilon": eps, "q": q, "F_in": f_in,
                         "F_out": f_out, "Delta_F": f_out - f_in,
                         "D_q": info["D"],
                         "R_min_eig": info.get("R_min_eig")})
    return rows


def break_even(eps: float, q_scan, tol=1e-10,
               degenerate_tol=1e-10) -> dict:
    """Bisect Delta_F = 0.  Reports whether the sign change is unique."""
    vals = [(q, delta_f(eps, q)) for q in q_scan]
    finite = [v for _q, v in vals if not np.isnan(v)]
    scan_absmax = max(abs(v) for v in finite)
    base = {"epsilon": eps, "scan_min": min(finite), "scan_max": max(finite),
            "scan_abs_max": scan_absmax,
            "Delta_F_at_q0": vals[0][1], "tolerance": tol}
    # A degenerate input (for example the maximally mixed state) gives
    # Delta_F = 0 identically; any sign change there is round-off, so no root
    # is reported.
    if scan_absmax < degenerate_tol:
        return {**base, "q_gain_num": None, "n_sign_changes": 0,
                "unique": False,
                "status": "degenerate: Delta_F = 0 to round-off at every q"}
    changes = [k for k in range(len(vals) - 1)
               if not np.isnan(vals[k][1]) and not np.isnan(vals[k + 1][1])
               and np.sign(vals[k][1]) != np.sign(vals[k + 1][1])]
    if not changes:
        sgn = "no gain at any tested q" if max(finite) <= 0 else \
              "gain at every tested q"
        return {**base, "q_gain_num": None, "n_sign_changes": 0,
                "unique": False, "status": sgn}
    k = changes[0]
    lo, hi = vals[k][0], vals[k + 1][0]
    f_lo = vals[k][1]
    it = 0
    while hi - lo > tol and it < 200:
        mid = 0.5 * (lo + hi)
        f_mid = delta_f(eps, mid)
        if np.sign(f_mid) == np.sign(f_lo):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
        it += 1
    return {**base, "q_gain_num": 0.5 * (lo + hi),
            "bracket_lo": lo, "bracket_hi": hi, "bracket_width": hi - lo,
            "iterations": it, "n_sign_changes": len(changes),
            "unique": len(changes) == 1, "status": "root bracketed"}


def independent_route_check(rng, qs=(0.0, 0.02, 0.10), n_states=5) -> dict:
    states = [("bell_Phi+", bell_diagonal_rho([1, 0, 0, 0])),
              ("isotropic_0.3", isotropic_rho(0.3))]
    states += [(f"belldiag_{k}", bell_diagonal_rho(rng.dirichlet(np.ones(4))))
               for k in range(n_states)]
    worst_state = worst_obs = 0.0
    for _name, rho in states:
        for q in qs:
            a = final_state_6q_noisy(rho, q)
            b = final_state_route_b(rho, q)
            worst_state = max(worst_state, float(np.max(np.abs(a - b))))
            ia = effective_operator(rho, q)
            d_b = {}
            from pqec_distill.locc_vd import correlator_6q as corr
            for lab in PAULI_LABELS:
                o = pauli_product(lab)
                d_b[lab] = corr(b, "X", o) - corr(b, "Y", o)
            worst_obs = max(worst_obs, max(abs(ia["deltas"][l] - d_b[l])
                                           for l in PAULI_LABELS))
    return {"n_states": len(states), "q_values": list(qs),
            "max_abs_density_matrix_diff": worst_state,
            "max_abs_delta_P_diff": worst_obs}


# ===========================================================================
# figures
# ===========================================================================

BELL_COLOR = {"Phi+": "#2a78d6", "Phi-": "#eb6834",
              "Psi+": "#1baf7a", "Psi-": "#eda100"}
BELL_MARKER = {"Phi+": "o", "Phi-": "s", "Psi+": "^", "Psi-": "D"}
BELL_TEX = {"Phi+": r"$\Phi^{+}$", "Phi-": r"$\Phi^{-}$",
            "Psi+": r"$\Psi^{+}$", "Psi-": r"$\Psi^{-}$"}
EPS_RAMP = ["#86b6ef", "#5b95e0", "#3a74cf", "#2457ae", "#163c84", "#0d366b"]
INK, INK2, GRID = "#101010", "#4a4a4a", "#d9d9d9"


def use_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": 11,
        "axes.titlesize": 11.5, "axes.labelsize": 11.5,
        "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9.5,
        "axes.facecolor": "white", "axes.edgecolor": GRID,
        "axes.labelcolor": INK, "axes.linewidth": 0.9,
        "axes.grid": True, "axes.axisbelow": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "grid.alpha": 0.8,
        "xtick.color": INK2, "ytick.color": INK2,
        "lines.linewidth": 1.8, "lines.markersize": 4.5,
        "legend.frameon": False,
    })


def _despine(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def save(fig, stem):
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}", dpi=300 if ext == "png" else None)
    import matplotlib.pyplot as plt
    plt.close(fig)


def plot_bell_offdiagonal(rows):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 5.0))
    _despine(ax)
    picks = ["vertex_Phi+", "edge_Phi+_Psi-_0.5", "face_PhiPhiPsi",
             "asym_strong", "asym_flat", "uniform",
             "dirichlet_0", "dirichlet_1", "dirichlet_3"]
    shown = 0
    for name in picks:
        sub = sorted((r for r in rows if r["state"] == name),
                     key=lambda r: r["q"])
        if not sub:
            continue
        ax.semilogy([r["q"] for r in sub],
                    [max(r["C_Bell_R"], 1e-300) for r in sub],
                    marker="o", markeredgecolor="white", linewidth=1.5,
                    label=name)
        shown += 1
    ax.axhline(ROUNDOFF_SCALE, color=INK2, linestyle=(0, (4, 3)), linewidth=1.2)
    ax.annotate(r"round-off scale $10^{-12}$", xy=(0.03, 0.90),
                xycoords="axes fraction", fontsize=9.5, color=INK2)
    ax.set_xlabel(r"per-CNOT noise $q$")
    ax.set_ylabel(r"$C_{\rm Bell}$ of $R_q(\rho)$  (Frobenius, off-diagonal)")
    ax.legend(loc="lower right", labelcolor=INK, ncol=2, fontsize=8.5)
    ax.set_title("Bell off-diagonal content of the reconstructed operator\n"
                 f"for {shown} arbitrary Bell-diagonal inputs", color=INK, pad=8)
    save(fig, "locc_vd_noisy_bell_offdiagonal")


def plot_populations(rows, eps_list=(0.05, 0.20, 0.50, 0.90)):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(eps_list),
                             figsize=(3.5 * len(eps_list), 4.3),
                             gridspec_kw=dict(wspace=0.32))
    for col, eps in enumerate(eps_list):
        ax = axes[col]
        _despine(ax)
        sub = sorted((r for r in rows if abs(r["epsilon"] - eps) < 1e-12),
                     key=lambda r: r["q"])
        q = [r["q"] for r in sub]
        for key, name in (("p_Phi_plus", "Phi+"), ("p_Phi_minus", "Phi-"),
                          ("p_Psi_plus", "Psi+"), ("p_Psi_minus", "Psi-")):
            style = dict(linewidth=1.4, linestyle=(0, (5, 3))) \
                if name == "Psi-" else dict(linewidth=1.9)
            ax.plot(q, [r[key] for r in sub], color=BELL_COLOR[name],
                    marker=BELL_MARKER[name], markeredgecolor="white",
                    label=BELL_TEX[name], **style)
        ax.set_xlabel(r"per-CNOT noise $q$")
        if col == 0:
            ax.set_ylabel("Bell population of $R_q$")
        ax.set_title(rf"$\epsilon = {eps:g}$", color=INK, fontsize=11)
        ax.set_ylim(0, 1)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.07), labelcolor=INK)
    fig.suptitle("One noisy 6Q-L16 round: Bell populations of the "
                 "reconstructed operator", fontsize=13, color=INK, y=1.02)
    save(fig, "locc_vd_noisy_populations")


def plot_delta_f(rows, eps_grid, q_grid, breakeven):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8),
                             gridspec_kw=dict(wspace=0.26))
    for ax in axes:
        _despine(ax)
    eps_u = sorted({r["epsilon"] for r in rows})
    q_u = sorted({r["q"] for r in rows})
    grid = np.full((len(eps_u), len(q_u)), np.nan)
    for r in rows:
        grid[eps_u.index(r["epsilon"]), q_u.index(r["q"])] = r["Delta_F"]
    lim = float(np.nanmax(np.abs(grid)))
    im = axes[0].pcolormesh(np.array(q_u), np.array(eps_u), grid,
                            cmap="RdBu", vmin=-lim, vmax=lim, shading="nearest")
    cs = axes[0].contour(np.array(q_u), np.array(eps_u), grid, levels=[0.0],
                         colors=INK, linewidths=1.6)
    axes[0].clabel(cs, fmt={0.0: r"$\Delta F = 0$"}, fontsize=9)
    fig.colorbar(im, ax=axes[0], label=r"$\Delta F = F_{\rm out} - F_{\rm in}$")
    axes[0].set_xlabel(r"per-CNOT noise $q$")
    axes[0].set_ylabel(r"input mixing $\epsilon$")
    axes[0].grid(False)
    axes[0].set_title(r"one-round $\Delta F(\epsilon, q)$", color=INK,
                      fontsize=11)

    ok = [b for b in breakeven if b["q_gain_num"] is not None]
    axes[1].plot([b["epsilon"] for b in ok], [b["q_gain_num"] for b in ok],
                 color=EPS_RAMP[3], marker="o", markeredgecolor="white",
                 linewidth=1.9)
    axes[1].set_xlabel(r"input mixing $\epsilon$")
    axes[1].set_ylabel(r"$q_{\rm gain}^{\rm num}(\epsilon)$")
    axes[1].set_ylim(0, None)
    axes[1].set_title("numerical one-round break-even noise\n"
                      r"(circuit points, no fit)", color=INK, fontsize=11)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.text(-0.13, 1.06, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK)
    fig.suptitle("Noisy 6Q-L16, one round only", fontsize=13, color=INK,
                 y=1.02)
    save(fig, "locc_vd_noisy_fidelity_gain")


def plot_isotropy_breaking(rows):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6),
                             gridspec_kw=dict(wspace=0.28))
    for ax in axes:
        _despine(ax)
    eps_u = sorted({r["epsilon"] for r in rows})
    for k, eps in enumerate(eps_u):
        sub = sorted((r for r in rows if r["epsilon"] == eps),
                     key=lambda r: r["q"])
        colour = EPS_RAMP[k % len(EPS_RAMP)]
        axes[0].semilogy([r["q"] for r in sub],
                         [max(r["delta_iso"], 1e-300) for r in sub],
                         color=colour, marker="o", markersize=3.4,
                         markeredgecolor="white", linewidth=1.4,
                         label=rf"$\epsilon = {eps:g}$")
        axes[1].semilogy([r["q"] for r in sub],
                         [max(r["Psi+_vs_Psi-"], 1e-300) for r in sub],
                         color=colour, marker="s", markersize=3.4,
                         markeredgecolor="white", linewidth=1.4)
    for ax, lab in ((axes[0], r"$\delta_{\rm iso}$"),
                    (axes[1], r"$|p_{\Psi^+} - p_{\Psi^-}|$")):
        ax.axhline(ROUNDOFF_SCALE, color=INK2, linestyle=(0, (4, 3)),
                   linewidth=1.2)
        ax.set_xlabel(r"per-CNOT noise $q$")
        ax.set_ylabel(lab)
    axes[0].legend(loc="lower right", labelcolor=INK, ncol=2, fontsize=8)
    axes[0].set_title("isotropy breaking", color=INK, fontsize=11)
    axes[1].set_title(r"the $\Psi^+ \leftrightarrow \Psi^-$ relation",
                      color=INK, fontsize=11)
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.text(-0.14, 1.06, tag, transform=ax.transAxes, fontsize=12,
                fontweight="bold", color=INK)
    fig.suptitle("Bell-isotropic inputs after one noisy 6Q-L16 round",
                 fontsize=13, color=INK, y=1.02)
    save(fig, "locc_vd_noisy_isotropy")


# ===========================================================================
# driver
# ===========================================================================

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)

    print("=" * 78)
    print("NOISY 6Q-L16 -- CIRCUIT-LEVEL NUMERICAL STUDY (one round only)")
    print("=" * 78)
    print(f"  ordering : [a, b, A1, B1, A2, B2] = 0..5")
    print(f"  Alice    : Fredkin(a; A1, A2) on wires {ALICE_TRIPLE}")
    print(f"  Bob      : Fredkin(b; B1, B2) on wires {BOB_TRIPLE}")
    print(f"  source   : {CIRCUIT_SOURCE}")
    print(f"  noise    : {NOISE_CONVENTION}\n")
    print(format_sequence())

    print("\n" + "=" * 78)
    print("STRUCTURAL CHECKS")
    print("=" * 78)
    structural = structural_checks()
    for k, v in structural.items():
        print(f"    {k:<36} {v:.4e}" if isinstance(v, float)
              else f"    {k:<36} {v}")

    print("\n  NOISE-CONVENTION CHECKS")
    conv = noise_convention_checks(rng)
    for k, v in conv.items():
        print(f"    {k:<42} {v:.4e}" if isinstance(v, float)
              else f"    {k:<42} {v}")

    print("\n  final-state validity (eps = 0.3)")
    valid = {f"q={q:g}": state_validity(isotropic_rho(0.3), q)
             for q in (0.0, 0.05, 0.3)}
    for k, d in valid.items():
        print(f"    {k:<8} |tr-1|={d['trace_minus_1']:.2e}  "
              f"herm={d['hermiticity']:.2e}  min eig={d['min_eig']:.2e}")

    print("\n" + "=" * 78)
    print("q = 0 AGAINST THE FROZEN IDEAL 6Q RESULT")
    print("=" * 78)
    q0 = check_q0_against_frozen_ideal(rng)
    for k, v in q0.items():
        print(f"    {k:<34} {v:.4e}" if isinstance(v, float)
              else f"    {k:<34} {v}")
    assert q0["max_abs_delta_diff"] < 1e-12, "q=0 must reproduce the ideal result"

    print("\n" + "=" * 78)
    print("STUDY A -- DOES BELL-DIAGONALITY SURVIVE?")
    print("=" * 78)
    bd_states = bell_diagonal_test_set(rng)
    rows_a, worst_a = study_a_bell_diagonality(bd_states, Q_GRID)
    print(f"    {len(bd_states)} Bell-diagonal inputs x {len(Q_GRID)} q values "
          f"= {len(rows_a)} reconstructions")
    print(f"    max C_Bell of R_q over everything : {worst_a['c']:.3e}")
    print(f"      worst input    : {worst_a['state']} ({worst_a['family']})")
    print(f"      worst q        : {worst_a['q']}")
    print(f"      round-off scale: {ROUNDOFF_SCALE:.0e}  -> "
          f"{'consistent with round-off' if worst_a['c'] < ROUNDOFF_SCALE else 'GENUINELY NONZERO'}")
    by_q = {}
    for r in rows_a:
        by_q.setdefault(r["q"], []).append(r["C_Bell_R"])
    print("    max C_Bell by q:")
    for q in Q_GRID:
        print(f"      q={q:<8g} {max(by_q[q]):.3e}")
    neg = [r for r in rows_a if r["R_is_psd"] is False]
    print(f"    reconstructions with a negative eigenvalue below -1e-10: {len(neg)}")
    for r in neg[:5]:
        print(f"      {r['state']} q={r['q']}: min eig = {r['R_min_eig']:.3e}")
    print(f"    min |D_q| over everything: "
          f"{min(abs(r['D_q']) for r in rows_a):.3e}")

    print("\n" + "=" * 78)
    print("STUDY B -- DOES THE BELL-ISOTROPIC FAMILY SURVIVE?")
    print("=" * 78)
    rows_b, worst_b = study_b_isotropy(EPS_GRID, Q_GRID)
    print(f"    max delta_iso = {worst_b['d']:.3e} at eps={worst_b['epsilon']:g}, "
          f"q={worst_b['q']:g}")
    print(f"    max |p_Psi+ - p_Psi-| over everything = "
          f"{max(r['Psi+_vs_Psi-'] for r in rows_b):.3e}")
    print(f"    max |p_Phi- - p_Psi+| over everything = "
          f"{max(r['Phi-_vs_Psi+'] for r in rows_b):.3e}")
    print(f"    max |pop sum - 1| = "
          f"{max(abs(r['pop_sum'] - 1.0) for r in rows_b):.3e}")
    print("\n    representative populations (eps = 0.3)")
    print(f"    {'q':>8} {'p_Phi+':>12} {'p_Phi-':>12} {'p_Psi+':>12} "
          f"{'p_Psi-':>12} {'delta_iso':>11}")
    for r in sorted((r for r in rows_b if abs(r["epsilon"] - 0.3) < 1e-12),
                    key=lambda r: r["q"]):
        print(f"    {r['q']:>8g} {r['p_Phi_plus']:>12.8f} "
              f"{r['p_Phi_minus']:>12.8f} {r['p_Psi_plus']:>12.8f} "
              f"{r['p_Psi_minus']:>12.8f} {r['delta_iso']:>11.2e}")

    print("\n" + "=" * 78)
    print("STUDY C -- ONE-ROUND Phi+ FIDELITY")
    print("=" * 78)
    q_fine = sorted(set(list(Q_GRID) + list(np.linspace(0.0, 0.30, 31))))
    rows_c = study_c_fidelity(EPS_GRID, q_fine)
    print(f"    {'eps':>7} {'F_in':>10} " + " ".join(
        f"{'dF(q=' + format(q, 'g') + ')':>14}" for q in (0.0, 0.02, 0.05, 0.1)))
    for eps in EPS_GRID:
        f_in = 1.0 - 3.0 * eps / 4.0
        cells = []
        for q in (0.0, 0.02, 0.05, 0.1):
            v = next(r["Delta_F"] for r in rows_c
                     if r["epsilon"] == eps and abs(r["q"] - q) < 1e-15)
            cells.append(f"{v:>14.8f}")
        print(f"    {eps:>7.4f} {f_in:>10.6f} " + " ".join(cells))

    print("\n  numerical break-even q (bisection on the circuit, tol 1e-10)")
    q_scan = list(np.linspace(0.0, 0.5, 51))
    breakeven = [break_even(eps, q_scan) for eps in EPS_GRID]
    print(f"    {'eps':>7} {'q_gain_num':>16} {'unique?':>9} "
          f"{'sign changes':>13} {'dF(q=0)':>12}   status")
    for b in breakeven:
        qg = "none" if b["q_gain_num"] is None else f"{b['q_gain_num']:.12f}"
        print(f"    {b['epsilon']:>7.4f} {qg:>16} {str(b['unique']):>9} "
              f"{b['n_sign_changes']:>13} {b.get('Delta_F_at_q0', float('nan')):>12.6f}"
              f"   {b.get('status', '')}")

    print("\n" + "=" * 78)
    print("INDEPENDENT ROUTE CHECK")
    print("=" * 78)
    route = independent_route_check(rng)
    for k, v in route.items():
        print(f"    {k:<34} {v:.4e}" if isinstance(v, float)
              else f"    {k:<34} {v}")

    # ---- outputs --------------------------------------------------------
    for name, rowset in (("bell_diagonal_preservation.csv", rows_a),
                         ("isotropic_breaking.csv", rows_b),
                         ("fidelity_gain.csv", rows_c),
                         ("break_even.csv", breakeven)):
        with open(OUT / name, "w", newline="") as fh:
            cols = sorted({k for r in rowset for k in r})
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rowset)

    payload = {}
    for eps in (0.0, 0.1, 0.3, 0.5, 1.0):
        for q in (0.0, 0.02, 0.1, 0.3):
            info = effective_operator(isotropic_rho(eps), q)
            tag = f"eps{eps:g}_q{q:g}"
            payload[f"rho__{tag}"] = isotropic_rho(eps)
            payload[f"T__{tag}"] = info["T"]
            if info["R"] is not None:
                payload[f"R__{tag}"] = info["R"]
            payload[f"deltas__{tag}"] = np.array(
                [info["deltas"][l] for l in PAULI_LABELS])
    payload["pauli_labels"] = np.array(PAULI_LABELS)
    np.savez(OUT / "representative_operators.npz", **payload)

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "protocol": "6Q-L16: two party-local Type-3 Fredkin decompositions",
        "status": "circuit-level numerical study, ONE ROUND ONLY; "
                  "no analytic noisy recurrence used or derived",
        "wire_order": ["a", "b", "A1", "B1", "A2", "B2"],
        "alice_triple": list(ALICE_TRIPLE), "bob_triple": list(BOB_TRIPLE),
        "circuit_source": CIRCUIT_SOURCE,
        "noise_convention": NOISE_CONVENTION,
        "noise_only_after_cnots": True,
        "n_cnot_alice": N_CNOT_ALICE, "n_cnot_bob": N_CNOT_BOB,
        "n_cnot_total": N_CNOT, "n_ancillas": 2,
        "measurement_settings": ["XX", "YY"],
        "all_cnots_party_local": crosses_party_cut() == [],
        "cnot_wire_pairs": [list(p) for p in cnot_wire_pairs()],
        "comparison_metadata_note": (
            "the existing 5Q Type-3 baseline also uses 16 CNOTs under the same "
            "per-CNOT replacement-depolarizing convention; no performance "
            "comparison is drawn here"),
        "structural_checks": structural,
        "noise_convention_checks": conv,
        "final_state_validity": valid,
        "q0_vs_frozen_ideal": q0,
        "study_a_worst": worst_a,
        "study_a_max_C_Bell_by_q": {str(q): float(max(by_q[q])) for q in Q_GRID},
        "study_a_roundoff_scale": ROUNDOFF_SCALE,
        "study_b_worst_delta_iso": worst_b,
        "study_b_max_psi_plus_minus_psi_minus":
            float(max(r["Psi+_vs_Psi-"] for r in rows_b)),
        "study_b_max_phi_minus_vs_psi_plus":
            float(max(r["Phi-_vs_Psi+"] for r in rows_b)),
        "break_even": breakeven,
        "independent_route_check": route,
        "q_grid": list(Q_GRID), "eps_grid": list(EPS_GRID), "seed": SEED,
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    use_style()
    plot_bell_offdiagonal(rows_a)
    plot_populations(rows_b)
    plot_isotropy_breaking(rows_b)
    plot_delta_f(rows_c, EPS_GRID, q_fine, breakeven)

    print("\n" + "=" * 78)
    print("FILES")
    print("=" * 78)
    for p in sorted(OUT.glob("*")) + sorted(FIG.glob("*")):
        print(f"  {p.relative_to(ROOT)}")
    print(f"\n  elapsed {meta['elapsed_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
