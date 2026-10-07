"""Independent analytic cross-check of the FROZEN blind 6Q LOCC results.

The blind circuit run (scripts/verify_locc_vd.py, results/data/locc_vd/) is
treated as read-only evidence.  This script derives the analytic predictions
from scratch -- matrix square, SWAP identity, Pauli reconstruction, Bell
specialisations -- and compares them against that frozen data.  Nothing here
regenerates, rewrites or touches the frozen files; their SHA-256 digests are
recorded in the output metadata so the comparison is pinned to the exact bytes
that were compared.

The analytic side uses no circuit.  The circuit side is read from the frozen
CSV/NPZ.  The only thing shared between them is the deterministic input-state
generator, which is reused (read-only) so that each frozen row can be matched
to the density matrix it was produced from; that reuse is verified against the
frozen record before any comparison is made.

Run:
    python scripts/verify_locc_vd_analytic.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

import _bootstrap  # noqa: F401

# read-only: the frozen input-state generator and the frozen circuit helpers.
# verify_locc_vd.build_states gives the same 151 states, in the same order,
# from the same seed.  locc_vd supplies the circuit used ONLY in section 12
# (non-identical inputs), which the blind run did not cover.
from verify_locc_vd import build_states  # noqa: E402
from pqec_distill.locc_vd import (  # noqa: E402
    PAULI_LABELS, delta_5q, delta_6q, pauli_product, plus_state_dm,
    unitary_5q, unitary_6q_route_a,
)

ROOT = Path(__file__).resolve().parents[1]
FROZEN = ROOT / "results" / "data" / "locc_vd"
OUT = ROOT / "results" / "data" / "locc_vd_analytic"

PHI_P = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(2)
PHI_M = np.array([1, 0, 0, -1], dtype=complex) / np.sqrt(2)
PSI_P = np.array([0, 1, 1, 0], dtype=complex) / np.sqrt(2)
PSI_M = np.array([0, 1, -1, 0], dtype=complex) / np.sqrt(2)
BELL_VECS = (PHI_P, PHI_M, PSI_P, PSI_M)


# ===========================================================================
# 1. the analytic side, written from the derivation and nothing else
# ===========================================================================

def rho_squared(rho: np.ndarray) -> np.ndarray:
    """rho^2 -- the analytic claim for T_6Q(rho)."""
    return rho @ rho


def analytic_delta(o: np.ndarray, rho: np.ndarray) -> float:
    """Tr(O rho^2):  the claimed value of C_XX(O) - C_YY(O)."""
    return float(np.real(np.trace(o @ rho_squared(rho))))


def analytic_delta_nonidentical(o: np.ndarray, rho: np.ndarray,
                                sigma: np.ndarray) -> float:
    """(1/2) Tr[ O (rho sigma + sigma rho) ] for two different input copies."""
    return float(np.real(np.trace(o @ (rho @ sigma + sigma @ rho)))) * 0.5


def analytic_vd_expectation(o: np.ndarray, rho: np.ndarray) -> float:
    """Tr(O rho^2) / Tr(rho^2) -- an estimator of an expectation value."""
    return analytic_delta(o, rho) / analytic_delta(np.eye(4, dtype=complex), rho)


def bell_populations(rho: np.ndarray) -> np.ndarray:
    return np.array([float(np.real(v.conj() @ rho @ v)) for v in BELL_VECS])


def analytic_virtual_bell_populations(p) -> np.ndarray:
    """p_i' = p_i^2 / sum_j p_j^2 for a Bell-diagonal input."""
    p = np.asarray(p, dtype=float)
    return p ** 2 / np.sum(p ** 2)


def analytic_F_VD_isotropic(eps: float) -> float:
    """(1 - 3 eps/4)^2 / [ (1 - 3 eps/4)^2 + 3 (eps/4)^2 ]."""
    p0, p1 = 1.0 - 3.0 * eps / 4.0, eps / 4.0
    return p0 ** 2 / (p0 ** 2 + 3.0 * p1 ** 2)


# ---------------------------------------------------------------------------
# the SWAP identity, verified numerically as well as proved in the document
# ---------------------------------------------------------------------------

def copy_swap(dim=4) -> np.ndarray:
    """S = sum_{ij} |i><j| (x) |j><i| on two ``dim``-dimensional copies."""
    eye = np.eye(dim, dtype=complex)
    s = np.zeros((dim * dim, dim * dim), dtype=complex)
    for i in range(dim):
        for j in range(dim):
            s += np.kron(np.outer(eye[i], eye[j]), np.outer(eye[j], eye[i]))
    return s


def partial_trace_second(m: np.ndarray, dim=4) -> np.ndarray:
    return np.einsum("ikjk->ij", m.reshape(dim, dim, dim, dim))


def check_swap_identity(n_trials=50, seed=11) -> dict:
    """Tr_2[(A (x) B) S] = A B  and  Tr_2[S (A (x) B)] = B A."""
    rng = np.random.default_rng(seed)
    s = copy_swap()
    worst_ab = worst_ba = worst_full = 0.0
    for _ in range(n_trials):
        a = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        b = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        ab = np.kron(a, b)
        worst_ab = max(worst_ab, float(np.max(np.abs(
            partial_trace_second(ab @ s) - a @ b))))
        worst_ba = max(worst_ba, float(np.max(np.abs(
            partial_trace_second(s @ ab) - b @ a))))
        worst_full = max(worst_full, abs(complex(np.trace(ab @ s))
                                         - complex(np.trace(a @ b))))
    return {"max_err_Tr2_AB_S_equals_AB": worst_ab,
            "max_err_Tr2_S_AB_equals_BA": worst_ba,
            "max_err_Tr_AB_S_equals_Tr_AB": worst_full,
            "n_trials": n_trials}


def check_pauli_orthogonality() -> dict:
    """Tr(P Q) = 4 delta_PQ, the step that makes the reconstruction exact."""
    worst_diag = worst_off = 0.0
    for lp in PAULI_LABELS:
        for lq in PAULI_LABELS:
            val = complex(np.trace(pauli_product(lp) @ pauli_product(lq)))
            if lp == lq:
                worst_diag = max(worst_diag, abs(val - 4.0))
            else:
                worst_off = max(worst_off, abs(val))
    return {"max_err_Tr_PP_minus_4": worst_diag,
            "max_abs_Tr_PQ_offdiagonal": worst_off}


def check_identical_copy_commutation(states, n=20) -> dict:
    """S (rho (x) rho) S = rho (x) rho, hence S R = R S."""
    s = copy_swap()
    worst_inv = worst_comm = 0.0
    for _name, _fam, rho in states[:n]:
        r = np.kron(rho, rho)
        worst_inv = max(worst_inv, float(np.max(np.abs(s @ r @ s - r))))
        worst_comm = max(worst_comm, float(np.max(np.abs(s @ r - r @ s))))
    return {"max_err_S_R_S_minus_R": worst_inv,
            "max_err_SR_minus_RS": worst_comm, "n_states": min(n, len(states))}


# ===========================================================================
# 2. frozen-data loading (strictly read-only)
# ===========================================================================

def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_frozen():
    rows = list(csv.DictReader((FROZEN / "blind_verification.csv").open()))
    per_state = list(csv.DictReader((FROZEN / "per_state_summary.csv").open()))
    npz = np.load(FROZEN / "representative_operators.npz", allow_pickle=False)
    digests = {p.name: sha256(p) for p in sorted(FROZEN.glob("*"))}
    return rows, per_state, npz, digests


def check_states_match_frozen_record(states, npz, per_state) -> dict:
    """The regenerated inputs must be the ones the frozen run actually used."""
    by_name = {n: r for n, _f, r in states}
    worst_rho = 0.0
    n_checked = 0
    for key in npz.files:
        if not key.startswith("rho__"):
            continue
        name = key[len("rho__"):]
        if name not in by_name:
            raise KeyError(f"frozen npz holds a state the generator did not: {name}")
        worst_rho = max(worst_rho, float(np.max(np.abs(npz[key] - by_name[name]))))
        n_checked += 1
    frozen_names = [r["state"] for r in per_state]
    same_order = frozen_names == [n for n, _f, _r in states]
    return {"max_abs_rho_diff_vs_frozen_npz": worst_rho,
            "n_states_cross_checked_against_npz": n_checked,
            "state_list_and_order_identical": bool(same_order),
            "n_states_regenerated": len(states),
            "n_states_in_frozen_summary": len(frozen_names)}


# ===========================================================================
# 3. the comparisons
# ===========================================================================

def compare_deltas(rows, states):
    """Frozen Delta_6Q and Delta_5Q against the analytic Tr(O rho^2)."""
    by_name = {n: r for n, _f, r in states}
    ops = {lab: pauli_product(lab) for lab in PAULI_LABELS}
    out, e6, e5 = [], [], []
    worst6 = {"err": -np.inf}
    worst5 = {"err": -np.inf}
    for r in rows:
        rho = by_name[r["state"]]
        pred = analytic_delta(ops[r["observable"]], rho)
        d6, d5 = float(r["Delta_6Q"]), float(r["Delta_5Q"])
        err6, err5 = abs(d6 - pred), abs(d5 - pred)
        e6.append(err6)
        e5.append(err5)
        if err6 > worst6["err"]:
            worst6 = {"err": err6, "state": r["state"], "observable": r["observable"],
                      "frozen": d6, "analytic": pred}
        if err5 > worst5["err"]:
            worst5 = {"err": err5, "state": r["state"], "observable": r["observable"],
                      "frozen": d5, "analytic": pred}
        out.append({"state": r["state"], "family": r["family"],
                    "observable": r["observable"],
                    "frozen_Delta_6Q": d6, "frozen_Delta_5Q": d5,
                    "analytic_Tr_O_rho2": pred,
                    "abs_error_6Q_vs_analytic": err6,
                    "abs_error_5Q_vs_analytic": err5})
    e6, e5 = np.array(e6), np.array(e5)
    stats = {
        "n_comparisons": int(e6.size),
        "6Q_vs_analytic": {"max": float(e6.max()), "mean": float(e6.mean()),
                           "rms": float(np.sqrt(np.mean(e6 ** 2))),
                           "worst": worst6},
        "5Q_vs_analytic": {"max": float(e5.max()), "mean": float(e5.mean()),
                           "rms": float(np.sqrt(np.mean(e5 ** 2))),
                           "worst": worst5},
    }
    per_family = {}
    for fam in sorted({r["family"] for r in out}):
        sub = np.array([r["abs_error_6Q_vs_analytic"] for r in out
                        if r["family"] == fam])
        per_family[fam] = {"n": int(sub.size), "max": float(sub.max()),
                           "mean": float(sub.mean()),
                           "rms": float(np.sqrt(np.mean(sub ** 2)))}
    stats["per_family_6Q_vs_analytic"] = per_family
    return out, stats


def compare_operators(npz, states):
    """Frozen T_6Q against the analytic rho^2."""
    by_name = {n: r for n, _f, r in states}
    rows, worst = [], {"err": -np.inf}
    for key in sorted(npz.files):
        if not key.startswith("T_6Q__"):
            continue
        name = key[len("T_6Q__"):]
        rho = by_name[name]
        t_frozen = npz[key]
        t_analytic = rho_squared(rho)
        fro = float(np.linalg.norm(t_frozen - t_analytic))
        mx = float(np.max(np.abs(t_frozen - t_analytic)))
        t5 = npz[f"T_5Q__{name}"]
        rows.append({"state": name, "frobenius_T6Q_minus_rho2": fro,
                     "max_abs_T6Q_minus_rho2": mx,
                     "frobenius_T5Q_minus_rho2":
                         float(np.linalg.norm(t5 - t_analytic)),
                     "trace_T6Q": float(np.real(np.trace(t_frozen))),
                     "trace_rho2": float(np.real(np.trace(t_analytic)))})
        if fro > worst["err"]:
            worst = {"err": fro, "state": name}
    return rows, {"max_frobenius_T6Q_minus_rho2":
                  float(max(r["frobenius_T6Q_minus_rho2"] for r in rows)),
                  "max_frobenius_T5Q_minus_rho2":
                  float(max(r["frobenius_T5Q_minus_rho2"] for r in rows)),
                  "worst_representative": worst["state"],
                  "n_representatives": len(rows)}


def compare_bell_specialisations(per_state, states):
    """Bell-diagonal p_i' and the Bell-isotropic F_VD(epsilon) formula."""
    by_name = {n: r for n, _f, r in states}
    fam = {n: f for n, f, _r in states}
    iso_rows, bd_rows = [], []
    for r in per_state:
        name = r["state"]
        rho = by_name[name]
        f_frozen = float(r["normalised_bell_fidelity_6Q"])
        p = bell_populations(rho)
        p_prime = analytic_virtual_bell_populations(p)
        if fam[name] == "B_isotropic":
            # epsilon is recovered FROM THE STATE, p_0 = 1 - 3 eps/4, not from
            # the state's name: the name rounds epsilon to six decimals, so
            # eps = 2/3 is stored as "0.666667" and evaluating the formula at
            # the label instead of the state shifts F by ~4e-7.  The label
            # value is kept as a column so that effect stays visible.
            eps_label = float(name.split("eps")[1])
            eps = 4.0 * (1.0 - float(p[0])) / 3.0
            f_formula = analytic_F_VD_isotropic(eps)
            iso_rows.append({
                "state": name, "epsilon_from_state": eps,
                "epsilon_from_label": eps_label,
                "label_minus_state_epsilon": eps_label - eps,
                "frozen_F_6Q": f_frozen,
                "frozen_F_5Q": float(r["normalised_bell_fidelity_5Q"]),
                "analytic_F_VD_formula": f_formula,
                "analytic_F_VD_formula_at_label_epsilon":
                    analytic_F_VD_isotropic(eps_label),
                "analytic_p0_squared_over_sum": float(p_prime[0]),
                "abs_error_formula_vs_frozen": abs(f_formula - f_frozen),
                "abs_error_label_route_vs_frozen":
                    abs(analytic_F_VD_isotropic(eps_label) - f_frozen),
                "abs_error_formula_vs_population_route":
                    abs(f_formula - float(p_prime[0]))})
        if fam[name] in ("A_pure_bell", "B_isotropic", "C_bell_diagonal"):
            # the Bell-diagonal claim: rho^2 keeps the Bell basis, p_i' = p_i^2/sum
            r2 = rho_squared(rho)
            p_from_square = bell_populations(r2)
            p_from_square = p_from_square / p_from_square.sum()
            bd_rows.append({
                "state": name, "family": fam[name],
                "max_abs_p_prime_error":
                    float(np.max(np.abs(p_from_square - p_prime))),
                "frozen_F_6Q": f_frozen,
                "analytic_p0_prime": float(p_prime[0]),
                "abs_error_F_vs_p0_prime": abs(f_frozen - float(p_prime[0])),
                "offdiagonal_weight_of_rho_squared": float(np.sum(np.abs(
                    np.array([[v.conj() @ r2 @ w for w in BELL_VECS]
                              for v in BELL_VECS])
                    - np.diag(np.diag(np.array([[v.conj() @ r2 @ w
                                                 for w in BELL_VECS]
                                                for v in BELL_VECS])))) ** 2))})
    stats = {
        "isotropic": {
            "n": len(iso_rows),
            "max_abs_error_formula_vs_frozen":
                float(max(r["abs_error_formula_vs_frozen"] for r in iso_rows)),
            "mean_abs_error_formula_vs_frozen":
                float(np.mean([r["abs_error_formula_vs_frozen"] for r in iso_rows])),
            "max_abs_error_if_epsilon_taken_from_the_label_instead":
                float(max(r["abs_error_label_route_vs_frozen"] for r in iso_rows)),
            "note": ("epsilon is recovered from the state; taking it from the "
                     "rounded state name instead shifts F by ~4e-7 at eps = 2/3"),
        },
        "bell_diagonal": {
            "n": len(bd_rows),
            "max_abs_p_prime_error":
                float(max(r["max_abs_p_prime_error"] for r in bd_rows)),
            "max_abs_error_F_vs_p0_prime":
                float(max(r["abs_error_F_vs_p0_prime"] for r in bd_rows)),
            "max_offdiagonal_weight_of_rho_squared":
                float(max(r["offdiagonal_weight_of_rho_squared"] for r in bd_rows)),
        },
    }
    return iso_rows, bd_rows, stats


# ===========================================================================
# 4. section 12 -- non-identical inputs (a NEW test set, kept separate)
# ===========================================================================

def nonidentical_check(n_pairs=60, seed=909) -> tuple:
    """rho (x) sigma with rho != sigma, run through the frozen circuits.

    This is NOT part of the identical-copy VD claim.  It tests the
    pre-symmetry expression (1/2) Tr[O (rho sigma + sigma rho)].
    """
    rng = np.random.default_rng(seed)
    u6, u5 = unitary_6q_route_a(), unitary_5q()
    plus = plus_state_dm()
    ops = {lab: pauli_product(lab) for lab in PAULI_LABELS}

    def rand_mixed(rank=4):
        m = rng.normal(size=(4, rank)) + 1j * rng.normal(size=(4, rank))
        r = m @ m.conj().T
        return r / np.trace(r)

    def rand_pure():
        v = rng.normal(size=4) + 1j * rng.normal(size=4)
        v = v / np.linalg.norm(v)
        return np.outer(v, v.conj())

    rows, e6, e5, e_sym = [], [], [], []
    for k in range(n_pairs):
        rho = rand_pure() if k % 3 == 0 else rand_mixed(2 if k % 3 == 1 else 4)
        sigma = rand_mixed(3 if k % 2 == 0 else 4)
        data = np.kron(rho, sigma)
        sig6 = u6 @ np.kron(plus, np.kron(plus, data)) @ u6.conj().T
        sig5 = u5 @ np.kron(plus, data) @ u5.conj().T
        for lab, o in ops.items():
            d6, _xx, _yy = delta_6q(sig6, o)
            d5 = delta_5q(sig5, o)
            pred = analytic_delta_nonidentical(o, rho, sigma)
            naive = analytic_delta(o, rho)       # the identical-copy formula
            e6.append(abs(d6 - pred))
            e5.append(abs(d5 - pred))
            e_sym.append(abs(d6 - naive))
            rows.append({"pair": k, "observable": lab,
                         "circuit_Delta_6Q": d6, "circuit_Delta_5Q": d5,
                         "analytic_half_Tr_O_rs_plus_sr": pred,
                         "identical_copy_formula_Tr_O_rho2": naive,
                         "abs_error_6Q": abs(d6 - pred),
                         "abs_error_5Q": abs(d5 - pred),
                         "abs_gap_to_identical_copy_formula": abs(d6 - naive)})
    e6, e5, e_sym = np.array(e6), np.array(e5), np.array(e_sym)
    stats = {"n_pairs": n_pairs, "n_comparisons": int(e6.size),
             "max_abs_error_6Q": float(e6.max()),
             "mean_abs_error_6Q": float(e6.mean()),
             "rms_error_6Q": float(np.sqrt(np.mean(e6 ** 2))),
             "max_abs_error_5Q": float(e5.max()),
             "max_gap_to_identical_copy_formula": float(e_sym.max()),
             "mean_gap_to_identical_copy_formula": float(e_sym.mean())}
    return rows, stats


# ===========================================================================
# 5. driver
# ===========================================================================

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print("=" * 78)
    print("INDEPENDENT ANALYTIC CROSS-CHECK OF THE FROZEN BLIND 6Q RESULTS")
    print("=" * 78)

    rows, per_state, npz, digests = load_frozen()
    print("  frozen inputs (read-only, SHA-256):")
    for name, dig in digests.items():
        print(f"    {name:<34} {dig[:16]}...")

    states = build_states()
    match = check_states_match_frozen_record(states, npz, per_state)
    print("\n  regenerated inputs match the frozen record:")
    for k, v in match.items():
        print(f"    {k:<44} {v}")
    assert match["max_abs_rho_diff_vs_frozen_npz"] == 0.0
    assert match["state_list_and_order_identical"]

    print("\n" + "=" * 78)
    print("ANALYTIC INGREDIENTS (proved in the document, checked here)")
    print("=" * 78)
    swap = check_swap_identity()
    pauli = check_pauli_orthogonality()
    comm = check_identical_copy_commutation(states)
    for group, d in (("SWAP identity", swap), ("Pauli orthogonality", pauli),
                     ("identical-copy commutation", comm)):
        print(f"  {group}")
        for k, v in d.items():
            print(f"      {k:<42} {v:.3e}" if isinstance(v, float)
                  else f"      {k:<42} {v}")

    print("\n" + "=" * 78)
    print("FROZEN Delta  vs  ANALYTIC  Tr(O rho^2)")
    print("=" * 78)
    comp, stats = compare_deltas(rows, states)
    for tag in ("6Q_vs_analytic", "5Q_vs_analytic"):
        s = stats[tag]
        w = s["worst"]
        print(f"  {tag}")
        print(f"      max  {s['max']:.3e}   mean {s['mean']:.3e}   "
              f"rms {s['rms']:.3e}")
        print(f"      worst: state={w['state']}, observable={w['observable']}, "
              f"frozen={w['frozen']:+.12f}, analytic={w['analytic']:+.12f}")
    print("\n  per family (6Q vs analytic)")
    for fam, s in stats["per_family_6Q_vs_analytic"].items():
        print(f"      {fam:<22} n={s['n']:>5}  max={s['max']:.3e}  "
              f"mean={s['mean']:.3e}  rms={s['rms']:.3e}")

    print("\n" + "=" * 78)
    print("FROZEN T_6Q  vs  ANALYTIC  rho^2")
    print("=" * 78)
    op_rows, op_stats = compare_operators(npz, states)
    print(f"  representatives                      {op_stats['n_representatives']}")
    print(f"  max ||T_6Q - rho^2||_F               "
          f"{op_stats['max_frobenius_T6Q_minus_rho2']:.3e}")
    print(f"  max ||T_5Q - rho^2||_F               "
          f"{op_stats['max_frobenius_T5Q_minus_rho2']:.3e}")
    print(f"  worst representative                 {op_stats['worst_representative']}")
    for r in op_rows:
        print(f"      {r['state']:<26} ||diff||_F={r['frobenius_T6Q_minus_rho2']:.3e}"
              f"   Tr T_6Q={r['trace_T6Q']:+.12f}   Tr rho^2={r['trace_rho2']:+.12f}")

    print("\n" + "=" * 78)
    print("BELL SPECIALISATIONS")
    print("=" * 78)
    iso_rows, bd_rows, bell_stats = compare_bell_specialisations(per_state, states)
    print(f"  {'epsilon':>18} {'frozen F_6Q':>16} {'analytic formula':>18} "
          f"{'|diff|':>10}")
    for r in iso_rows:
        print(f"  {r['epsilon_from_state']:18.15f} {r['frozen_F_6Q']:16.12f} "
              f"{r['analytic_F_VD_formula']:18.12f} "
              f"{r['abs_error_formula_vs_frozen']:10.2e}")
    print("  (epsilon recovered from the state; using the rounded state name "
          "instead would give")
    print(f"   a spurious error of up to "
          f"{bell_stats['isotropic']['max_abs_error_if_epsilon_taken_from_the_label_instead']:.2e}"
          " at eps = 2/3)")
    print(f"  max |formula - frozen| over the isotropic points: "
          f"{bell_stats['isotropic']['max_abs_error_formula_vs_frozen']:.3e}")
    bd = bell_stats["bell_diagonal"]
    print(f"\n  Bell-diagonal inputs ({bd['n']} of them)")
    print(f"      max |p_i' from rho^2  -  p_i^2/sum_j p_j^2|   "
          f"{bd['max_abs_p_prime_error']:.3e}")
    print(f"      max |frozen F_6Q  -  p_0^2/sum_j p_j^2|       "
          f"{bd['max_abs_error_F_vs_p0_prime']:.3e}")
    print(f"      max Bell off-diagonal weight of rho^2         "
          f"{bd['max_offdiagonal_weight_of_rho_squared']:.3e}")

    print("\n" + "=" * 78)
    print("SECTION 12 (SEPARATE): NON-IDENTICAL INPUTS  rho (x) sigma")
    print("  not part of the identical-copy VD claim")
    print("=" * 78)
    ni_rows, ni_stats = nonidentical_check()
    print(f"  {ni_stats['n_pairs']} fresh (rho, sigma) pairs x 16 observables = "
          f"{ni_stats['n_comparisons']} comparisons")
    print(f"      max |Delta_6Q - (1/2)Tr[O(rho sigma + sigma rho)]|  "
          f"{ni_stats['max_abs_error_6Q']:.3e}")
    print(f"      mean                                                "
          f"{ni_stats['mean_abs_error_6Q']:.3e}")
    print(f"      rms                                                 "
          f"{ni_stats['rms_error_6Q']:.3e}")
    print(f"      max for the 5Q baseline                             "
          f"{ni_stats['max_abs_error_5Q']:.3e}")
    print(f"      max gap to the identical-copy formula Tr(O rho^2)   "
          f"{ni_stats['max_gap_to_identical_copy_formula']:.3e}"
          "   <- large, as it must be")

    # ---- outputs -------------------------------------------------------
    for name, rowset in (("analytic_comparison.csv", comp),
                         ("operator_comparison.csv", op_rows),
                         ("bell_isotropic_comparison.csv", iso_rows),
                         ("bell_diagonal_comparison.csv", bd_rows),
                         ("nonidentical_inputs.csv", ni_rows)):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rowset[0]))
            w.writeheader()
            w.writerows(rowset)

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "role": "independent analytic cross-check of frozen blind circuit data",
        "frozen_inputs_read_only": True,
        "frozen_sha256": digests,
        "frozen_state_record_match": match,
        "analytic_claims": {
            "Delta(O)": "Tr(O rho^2)",
            "Delta(I)": "Tr(rho^2)",
            "T_6Q": "rho^2",
            "normalised_estimator": "Tr(O rho^2) / Tr(rho^2)",
            "bell_diagonal": "p_i' = p_i^2 / sum_j p_j^2",
            "bell_isotropic": "(1-3e/4)^2 / [(1-3e/4)^2 + 3 (e/4)^2]",
            "nonidentical_inputs": "(1/2) Tr[O (rho sigma + sigma rho)]",
        },
        "ingredient_checks": {"swap_identity": swap,
                              "pauli_orthogonality": pauli,
                              "identical_copy_commutation": comm},
        "delta_comparison": stats,
        "operator_comparison": op_stats,
        "bell_comparison": bell_stats,
        "nonidentical_section": ni_stats,
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
