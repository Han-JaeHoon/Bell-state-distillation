"""Blind circuit-level verification of the ideal six-qubit distributed (LOCC)
SWAP-test estimator against the ordinary five-qubit SWAP test.

Both estimators are computed from full density-matrix circuits built out of
gates.  No analytic matrix-square expression, no Bell-diagonal recurrence and
no previously frozen estimator formula is used anywhere in this script; the
reconstructed two-qubit operator is called T_6Q(rho) and is left uninterpreted.

Run:
    python scripts/verify_locc_vd.py
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401  (puts src/ on sys.path)

from pqec_distill.locc_vd import (  # noqa: E402
    ALICE_WIRES_6Q, BOB_WIRES_6Q, N_QUBITS_5Q, N_QUBITS_6Q, PAULI,
    PAULI_LABELS, WIRES_5Q, WIRES_6Q, bell_fidelity_numerator_from_deltas,
    delta_5q, delta_5q_all_paulis, delta_6q, delta_6q_all_paulis,
    embed_three_qubit_gate, final_state_5q, final_state_6q, fredkin_matrix,
    normalised_bell_fidelity, pauli_product, reconstruct_operator, s_a_6q,
    s_b_6q, unitary_5q, unitary_5q_permutation, unitary_6q_route_a,
    unitary_6q_route_b,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "data" / "locc_vd"

TOL = 1e-12
SEED = 20261007

# ---------------------------------------------------------------------------
# input state families
# ---------------------------------------------------------------------------

PHI_P = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PHI_M = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
PSI_P = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_M = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
BELL_VECS = (PHI_P, PHI_M, PSI_P, PSI_M)


def bell_diagonal(p) -> np.ndarray:
    rho = np.zeros((4, 4), dtype=complex)
    for w, v in zip(p, BELL_VECS):
        rho = rho + float(w) * np.outer(v, v.conj())
    return rho


def isotropic(eps: float) -> np.ndarray:
    e = float(eps)
    return (1 - e) * np.outer(PHI_P, PHI_P.conj()) + e * np.eye(4, dtype=complex) / 4


def random_pure(rng) -> np.ndarray:
    v = rng.normal(size=4) + 1j * rng.normal(size=4)
    v = v / np.linalg.norm(v)
    return np.outer(v, v.conj())


def random_mixed(rng, rank=4) -> np.ndarray:
    m = rng.normal(size=(4, rank)) + 1j * rng.normal(size=(4, rank))
    rho = m @ m.conj().T
    return rho / np.trace(rho)


def build_states(n_pure=40, n_full=40, n_rank2=20, n_rank3=20) -> list:
    """(name, family, rho).  Deterministic: one seeded generator, fixed order."""
    rng = np.random.default_rng(SEED)
    states = []

    # A. pure Bell states
    for name, v in (("Phi+", PHI_P), ("Phi-", PHI_M),
                    ("Psi+", PSI_P), ("Psi-", PSI_M)):
        states.append((f"bell_{name}", "A_pure_bell", np.outer(v, v.conj())))

    # B. Bell-isotropic
    for eps in (0.0, 0.01, 0.1, 0.3, 2.0 / 3.0, 0.8, 1.0):
        states.append((f"isotropic_eps{eps:.6f}", "B_isotropic", isotropic(eps)))

    # C. general asymmetric Bell-diagonal, including simplex boundary
    simplex = [(1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0),
               (0.5, 0.5, 0.0, 0.0), (0.5, 0.0, 0.5, 0.0),
               (0.0, 0.5, 0.25, 0.25), (0.7, 0.2, 0.1, 0.0),
               (0.6, 0.3, 0.05, 0.05), (0.25, 0.25, 0.25, 0.25),
               (0.9, 0.1, 0.0, 0.0), (0.4, 0.3, 0.2, 0.1)]
    for k, p in enumerate(simplex):
        states.append((f"belldiag_fixed_{k}", "C_bell_diagonal", bell_diagonal(p)))
    for k in range(10):                      # random interior simplex points
        w = rng.dirichlet(np.ones(4))
        states.append((f"belldiag_rand_{k}", "C_bell_diagonal", bell_diagonal(w)))

    # D. random pure states with complex amplitudes
    for k in range(n_pure):
        states.append((f"pure_{k}", "D_random_pure", random_pure(rng)))

    # E. random full-rank mixed states
    for k in range(n_full):
        states.append((f"mixed_full_{k}", "E_random_full_rank", random_mixed(rng)))

    # F. rank-deficient mixed states
    for k in range(n_rank2):
        states.append((f"mixed_rank2_{k}", "F_rank_deficient",
                       random_mixed(rng, rank=2)))
    for k in range(n_rank3):
        states.append((f"mixed_rank3_{k}", "F_rank_deficient",
                       random_mixed(rng, rank=3)))
    return states


def validate_state(rho: np.ndarray) -> dict:
    herm = float(np.linalg.norm(rho - rho.conj().T))
    tr = float(np.real(np.trace(rho)))
    eig = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    return {"herm_error": herm, "trace_error": abs(tr - 1.0),
            "min_eig": float(eig.min()), "rank": int(np.sum(eig > 1e-10))}


# ---------------------------------------------------------------------------
# structural checks
# ---------------------------------------------------------------------------

def _single_qubit_on(mat, wire, n):
    factors = [np.eye(2, dtype=complex)] * n
    factors[wire] = mat
    out = np.array([[1.0 + 0j]])
    for f in factors:
        out = np.kron(out, f)
    return out


def structural_checks() -> dict:
    g = fredkin_matrix()
    u_alice = embed_three_qubit_gate(g, (0, 2, 4), N_QUBITS_6Q)
    u_bob = embed_three_qubit_gate(g, (1, 3, 5), N_QUBITS_6Q)
    ua, ub = unitary_6q_route_a(), unitary_6q_route_b()
    u5, u5p = unitary_5q(), unitary_5q_permutation()
    sa, sb = s_a_6q(), s_b_6q()

    # locality: each party's gate commutes with every Pauli on the other party
    worst_alice = worst_bob = 0.0
    for wire in BOB_WIRES_6Q:
        for name in ("X", "Y", "Z"):
            p = _single_qubit_on(PAULI[name], wire, N_QUBITS_6Q)
            worst_alice = max(worst_alice, float(np.max(np.abs(u_alice @ p - p @ u_alice))))
    for wire in ALICE_WIRES_6Q:
        for name in ("X", "Y", "Z"):
            p = _single_qubit_on(PAULI[name], wire, N_QUBITS_6Q)
            worst_bob = max(worst_bob, float(np.max(np.abs(u_bob @ p - p @ u_bob))))

    # S_A S_B must exchange the complete bipartite copies on every basis state
    prod = sa @ sb
    worst_copy_swap = 0.0
    eye = np.eye(64, dtype=complex)
    for idx in range(64):
        bits = [(idx >> (5 - w)) & 1 for w in range(6)]
        want = [bits[0], bits[1], bits[4], bits[5], bits[2], bits[3]]
        tgt = sum(b << (5 - w) for w, b in enumerate(want))
        worst_copy_swap = max(worst_copy_swap,
                              float(np.max(np.abs(prod @ eye[:, idx] - eye[:, tgt]))))

    # the four control branches
    branches = {"00": np.eye(64, dtype=complex), "01": sb, "10": sa, "11": sa @ sb}
    worst_branch = 0.0
    for key, branch in branches.items():
        a_bit, b_bit = int(key[0]), int(key[1])
        for data in range(16):
            bits = [a_bit, b_bit] + [(data >> (3 - w)) & 1 for w in range(4)]
            idx = sum(b << (5 - w) for w, b in enumerate(bits))
            worst_branch = max(worst_branch, float(
                np.max(np.abs(ua @ eye[:, idx] - branch @ eye[:, idx]))))

    return {
        "fredkin_unitarity": float(np.max(np.abs(g.conj().T @ g - np.eye(8)))),
        "alice_embedding_unitarity":
            float(np.max(np.abs(u_alice.conj().T @ u_alice - np.eye(64)))),
        "bob_embedding_unitarity":
            float(np.max(np.abs(u_bob.conj().T @ u_bob - np.eye(64)))),
        "route_a_unitarity": float(np.max(np.abs(ua.conj().T @ ua - np.eye(64)))),
        "route_b_unitarity": float(np.max(np.abs(ub.conj().T @ ub - np.eye(64)))),
        "u5_unitarity": float(np.max(np.abs(u5.conj().T @ u5 - np.eye(32)))),
        "alice_gate_leaks_onto_bob_wires": worst_alice,
        "bob_gate_leaks_onto_alice_wires": worst_bob,
        "local_fredkins_commutator": float(np.max(np.abs(u_alice @ u_bob - u_bob @ u_alice))),
        "S_A_S_B_commutator": float(np.max(np.abs(sa @ sb - sb @ sa))),
        "S_A_S_B_is_complete_copy_swap_error": worst_copy_swap,
        "four_branch_structure_error": worst_branch,
        "route_a_minus_route_b_matrix": float(np.max(np.abs(ua - ub))),
        "u5_gate_minus_permutation_matrix": float(np.max(np.abs(u5 - u5p))),
    }


# ---------------------------------------------------------------------------
# the sweep
# ---------------------------------------------------------------------------

def run(states) -> tuple:
    ua, ub = unitary_6q_route_a(), unitary_6q_route_b()
    u5 = unitary_5q()
    ops = {lab: pauli_product(lab) for lab in PAULI_LABELS}

    rows, per_state, errors = [], [], []
    route_diffs, state_validity = [], []
    worst = {"err": -np.inf, "state": None, "obs": None}
    worst_xx = {"err": -np.inf, "state": None, "obs": None,
                "c_xx": None, "ref": None}
    worst_yy = {"err": -np.inf, "state": None, "obs": None,
                "c_yy": None, "ref": None}
    final_validity_6q = {"trace": 0.0, "herm": 0.0, "min_eig": np.inf}
    final_validity_5q = {"trace": 0.0, "herm": 0.0, "min_eig": np.inf}

    for name, family, rho in states:
        v = validate_state(rho)
        state_validity.append({"state": name, "family": family, **v})
        assert v["herm_error"] < TOL and v["trace_error"] < TOL, name
        assert v["min_eig"] > -TOL, name

        sigma6 = final_state_6q(rho, ua)
        sigma6_b = final_state_6q(rho, ub)
        sigma5 = final_state_5q(rho, u5)
        route_diffs.append({"state": name, "family": family,
                            "max_abs_diff": float(np.max(np.abs(sigma6 - sigma6_b)))})

        for sigma, acc in ((sigma6, final_validity_6q), (sigma5, final_validity_5q)):
            acc["trace"] = max(acc["trace"], abs(float(np.real(np.trace(sigma))) - 1.0))
            acc["herm"] = max(acc["herm"], float(np.linalg.norm(sigma - sigma.conj().T)))
            acc["min_eig"] = min(acc["min_eig"], float(
                np.linalg.eigvalsh(0.5 * (sigma + sigma.conj().T)).min()))

        d6, d5 = {}, {}
        for lab, o in ops.items():
            dd, c_xx, c_yy = delta_6q(sigma6, o)
            ref = delta_5q(sigma5, o)
            d6[lab], d5[lab] = dd, ref
            err = abs(dd - ref)
            errors.append(err)
            rows.append({"state": name, "family": family, "observable": lab,
                         "Delta_6Q": dd, "Delta_5Q": ref, "abs_error": err,
                         "C_XX": c_xx, "C_YY": c_yy,
                         "C_XX_minus_Delta_5Q": c_xx - ref,
                         "C_YY_minus_Delta_5Q": c_yy - ref})
            if err > worst["err"]:
                worst = {"err": err, "state": name, "obs": lab}
            if abs(c_xx - ref) > worst_xx["err"]:
                worst_xx = {"err": abs(c_xx - ref), "state": name, "obs": lab,
                            "c_xx": c_xx, "ref": ref}
            if abs(c_yy - ref) > worst_yy["err"]:
                worst_yy = {"err": abs(c_yy - ref), "state": name, "obs": lab,
                            "c_yy": c_yy, "ref": ref}

        t6 = reconstruct_operator(d6)
        t5 = reconstruct_operator(d5)
        eig6 = np.linalg.eigvalsh(0.5 * (t6 + t6.conj().T))
        per_state.append({
            "state": name, "family": family,
            "max_abs_delta_error": float(max(abs(d6[l] - d5[l]) for l in PAULI_LABELS)),
            "T_frobenius_6Q_minus_5Q": float(np.linalg.norm(t6 - t5)),
            "T_6Q_trace": float(np.real(np.trace(t6))),
            "T_6Q_hermiticity_error": float(np.linalg.norm(t6 - t6.conj().T)),
            "T_6Q_min_eig": float(eig6.min()),
            "T_6Q_max_eig": float(eig6.max()),
            "bell_fidelity_numerator_6Q": bell_fidelity_numerator_from_deltas(d6),
            "bell_fidelity_numerator_5Q": bell_fidelity_numerator_from_deltas(d5),
            "normalised_bell_fidelity_6Q": normalised_bell_fidelity(d6),
            "normalised_bell_fidelity_5Q": normalised_bell_fidelity(d5),
            "state_rank": v["rank"], "state_min_eig": v["min_eig"],
            "route_a_minus_route_b": float(np.max(np.abs(sigma6 - sigma6_b))),
        })

    err = np.array(errors)
    summary = {
        "n_states": len(states), "n_observables": len(PAULI_LABELS),
        "n_comparisons": int(err.size),
        "max_abs_error": float(err.max()), "mean_abs_error": float(err.mean()),
        "rms_error": float(np.sqrt(np.mean(err ** 2))),
        "worst_state": worst["state"], "worst_observable": worst["obs"],
        "max_route_a_minus_route_b": float(max(r["max_abs_diff"] for r in route_diffs)),
        "final_state_6q_validity": final_validity_6q,
        "final_state_5q_validity": final_validity_5q,
        "worst_C_XX_alone": worst_xx, "worst_C_YY_alone": worst_yy,
        "max_T_frobenius_6Q_minus_5Q":
            float(max(r["T_frobenius_6Q_minus_5Q"] for r in per_state)),
        "max_T_6Q_hermiticity_error":
            float(max(r["T_6Q_hermiticity_error"] for r in per_state)),
        "max_bell_fidelity_6Q_minus_5Q":
            float(max(abs(r["normalised_bell_fidelity_6Q"]
                          - r["normalised_bell_fidelity_5Q"]) for r in per_state)),
    }
    return rows, per_state, state_validity, route_diffs, summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pure", type=int, default=40)
    ap.add_argument("--n-full", type=int, default=40)
    ap.add_argument("--n-rank2", type=int, default=20)
    ap.add_argument("--n-rank3", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 78)
    print("IDEAL 6Q LOCC SWAP-TEST ESTIMATOR -- BLIND CIRCUIT-LEVEL VERIFICATION")
    print("=" * 78)
    print(f"  6Q ordering : {list(WIRES_6Q)}  ->  indices 0..5")
    print("  Alice gate  : Fredkin(a; A1, A2) = Fredkin(0; 2, 4)")
    print("  Bob gate    : Fredkin(b; B1, B2) = Fredkin(1; 3, 5)")
    print(f"  5Q ordering : {list(WIRES_5Q)}  ->  indices 0..4")
    print("  5Q gates    : Fredkin(c; A1, A2) and Fredkin(c; B1, B2)")
    print("  ideal gates, no noise, no decomposition, no postselection\n")

    print("STRUCTURAL CHECKS")
    checks = structural_checks()
    for k, v in checks.items():
        print(f"    {k:<44} {v:.3e}")

    states = build_states(args.n_pure, args.n_full, args.n_rank2, args.n_rank3)
    non_bd = sum(1 for _n, f, _r in states
                 if f in ("D_random_pure", "E_random_full_rank", "F_rank_deficient"))
    print(f"\n  {len(states)} input states "
          f"({non_bd} of them general non-Bell-diagonal), "
          f"{len(PAULI_LABELS)} observables each")

    rows, per_state, validity, route_diffs, summary = run(states)

    print("\n" + "=" * 78)
    print("BLIND COMPARISON  Delta_6Q(O)  versus  Delta_5Q(O)")
    print("=" * 78)
    print(f"    comparisons              {summary['n_comparisons']}")
    print(f"    max  |Delta_6Q - Delta_5Q|   {summary['max_abs_error']:.3e}"
          f"   (state {summary['worst_state']}, observable {summary['worst_observable']})")
    print(f"    mean |Delta_6Q - Delta_5Q|   {summary['mean_abs_error']:.3e}")
    print(f"    RMS  |Delta_6Q - Delta_5Q|   {summary['rms_error']:.3e}")

    print("\n  per-family maxima")
    fams = sorted({r["family"] for r in rows})
    for fam in fams:
        sub = [r["abs_error"] for r in rows if r["family"] == fam]
        print(f"    {fam:<22} n={len(sub):>5}  max={max(sub):.3e}  "
              f"mean={np.mean(sub):.3e}")

    print("\n  two independent 6Q routes")
    print(f"    max |rho_6Q(route A) - rho_6Q(route B)|  "
          f"{summary['max_route_a_minus_route_b']:.3e}")

    print("\n  final-state validity")
    for tag, d in (("6Q", summary["final_state_6q_validity"]),
                   ("5Q", summary["final_state_5q_validity"])):
        print(f"    {tag}: |trace-1| <= {d['trace']:.3e}   "
              f"hermiticity <= {d['herm']:.3e}   min eig >= {d['min_eig']:.3e}")

    print("\n  does the XX - YY subtraction do real work?")
    wx, wy = summary["worst_C_XX_alone"], summary["worst_C_YY_alone"]
    print(f"    C_XX alone worst: state={wx['state']}, O={wx['obs']},  "
          f"C_XX={wx['c_xx']:+.8f} vs Delta_5Q={wx['ref']:+.8f}  "
          f"|diff|={wx['err']:.3e}")
    print(f"    C_YY alone worst: state={wy['state']}, O={wy['obs']},  "
          f"C_YY={wy['c_yy']:+.8f} vs Delta_5Q={wy['ref']:+.8f}  "
          f"|diff|={wy['err']:.3e}")
    n_xx = sum(1 for r in rows if abs(r["C_XX_minus_Delta_5Q"]) > 1e-6)
    n_yy = sum(1 for r in rows if abs(r["C_YY_minus_Delta_5Q"]) > 1e-6)
    print(f"    comparisons where C_XX alone differs by > 1e-6: {n_xx}")
    print(f"    comparisons where C_YY alone differs by > 1e-6: {n_yy}")

    print("\n  reconstructed operator T_6Q (no interpretation at this stage)")
    print(f"    max ||T_6Q - T_5Q||_F            "
          f"{summary['max_T_frobenius_6Q_minus_5Q']:.3e}")
    print(f"    max hermiticity error of T_6Q    "
          f"{summary['max_T_6Q_hermiticity_error']:.3e}")
    print("    representative T_6Q spectra")
    for r in per_state:
        if r["family"] in ("A_pure_bell", "B_isotropic") or r["state"].endswith("_0"):
            print(f"      {r['state']:<26} Tr={r['T_6Q_trace']:+.10f}  "
                  f"eig in [{r['T_6Q_min_eig']:+.3e}, {r['T_6Q_max_eig']:+.3e}]")

    print("\n  Bell-fidelity estimator from local Pauli products "
          "(Phi+ = (II + XX - YY + ZZ)/4)")
    print(f"    {'state':<26} {'6Q':>14} {'5Q':>14} {'|diff|':>10}")
    for r in per_state:
        if r["family"] in ("A_pure_bell", "B_isotropic"):
            d = abs(r["normalised_bell_fidelity_6Q"] - r["normalised_bell_fidelity_5Q"])
            print(f"    {r['state']:<26} {r['normalised_bell_fidelity_6Q']:14.10f} "
                  f"{r['normalised_bell_fidelity_5Q']:14.10f} {d:10.2e}")
    print(f"    max |F_6Q - F_5Q| over all {len(per_state)} states: "
          f"{summary['max_bell_fidelity_6Q_minus_5Q']:.3e}")

    # ---- outputs -------------------------------------------------------
    with open(OUT / "blind_verification.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(OUT / "per_state_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_state[0]))
        w.writeheader()
        w.writerows(per_state)
    with open(OUT / "input_state_validity.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(validity[0]))
        w.writeheader()
        w.writerows(validity)

    # freeze representative operators and their inputs
    rep_names = ([f"bell_{n}" for n in ("Phi+", "Phi-", "Psi+", "Psi-")]
                 + [n for n, f, _ in states if f == "B_isotropic"]
                 + ["belldiag_fixed_5", "belldiag_rand_0", "pure_0", "pure_1",
                    "mixed_full_0", "mixed_full_1", "mixed_rank2_0",
                    "mixed_rank3_0"])
    by_name = {n: r for n, _f, r in states}
    payload = {}
    for name in rep_names:
        if name not in by_name:
            continue
        rho = by_name[name]
        d6 = {k: v[0] for k, v in delta_6q_all_paulis(rho).items()}
        d5 = delta_5q_all_paulis(rho)
        payload[f"rho__{name}"] = rho
        payload[f"T_6Q__{name}"] = reconstruct_operator(d6)
        payload[f"T_5Q__{name}"] = reconstruct_operator(d5)
        payload[f"deltas_6Q__{name}"] = np.array([d6[l] for l in PAULI_LABELS])
        payload[f"deltas_5Q__{name}"] = np.array([d5[l] for l in PAULI_LABELS])
    payload["pauli_labels"] = np.array(PAULI_LABELS)
    np.savez(OUT / "representative_operators.npz", **payload)

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "status": "BLIND circuit-level result, frozen before any analytic comparison",
        "analytic_input_used": False,
        "matrix_square_used_anywhere": False,
        "six_qubit_ordering": list(WIRES_6Q),
        "five_qubit_ordering": list(WIRES_5Q),
        "alice_gate": "Fredkin(a; A1, A2) = Fredkin(wire 0; wires 2, 4)",
        "bob_gate": "Fredkin(b; B1, B2) = Fredkin(wire 1; wires 3, 5)",
        "gate_order_in_simulator": "Alice then Bob (they commute; checked)",
        "five_qubit_gates": ["Fredkin(c; A1, A2)", "Fredkin(c; B1, B2)"],
        "ancilla_initial_state": "|+> on every ancilla",
        "ideal": True, "noise": None, "postselection": False,
        "twirling": False, "bell_diagonal_projection": False,
        "estimators": {
            "Delta_6Q": "<X_a X_b (x) O_(A1B1) (x) I_(A2B2)> - <Y_a Y_b (x) O (x) I>",
            "Delta_5Q": "<X_c (x) O_(A1B1) (x) I_(A2B2)>",
        },
        "observables": list(PAULI_LABELS),
        "seed": SEED,
        "structural_checks": checks,
        "summary": summary,
        "n_states_by_family": {f: sum(1 for _n, ff, _r in states if ff == f)
                               for f in sorted({f for _n, f, _r in states})},
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, default=str))

    print("\n" + "=" * 78)
    print("FILES")
    print("=" * 78)
    for p in sorted(OUT.glob("*")):
        print(f"  {p.relative_to(ROOT)}")
    print(f"\n  elapsed {meta['elapsed_seconds']} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
