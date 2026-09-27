"""Bell-population dynamics of the four noisy purification protocols.

Default run: eps = 0.15, q = 0.02, n = 0..50, plus a 5000-round Type 5 run.
Everything is parameterised, so a later sweep is
``python scripts/bell_population_dynamics.py --eps 0.3 --q 0.05``.

Writes, under results/data/bell_population_dynamics/ and results/figures/:

  <protocol>_eps<...>_q<...>.csv      round-by-round populations + diagnostics
  summary.json                        the requested tables and all checks
  metadata.json                       code commit, parent-repo provenance, checks
  bellpop_stacked_<protocol>.png      Figure A
  bellpop_lines_<protocol>.png        Figure B
  bellpop_type5_offdiagonal.png       C_Bell(n), short and long
  bellpop_type5_longtime_linear.png   long-time F, linear n
  bellpop_type5_longtime_log.png      long-time F, log10(n+1)
  bellpop_type5_vs_calibrated.png     Type 5 vs Type5cal (escape diagnosis)
  bellpop_overview_F.png              F_n for all four protocols
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

import numpy as np

from _bootstrap import DATA_DIR, FIG_DIR  # noqa: E402
from pqec_distill import swap_test_source  # noqa: E402
from pqec_distill.bell_population_dynamics import (  # noqa: E402
    BELL_NAMES, PROTOCOLS, PROTOCOL_INFO, compilation_residuals,
    cross_check_exact_vs_dm, isotropic_populations, plateau_and_escape,
    run_dynamics, verify_cnot_counts, verify_noise_convention,
)
from pqec_distill.bell_population_plots import (  # noqa: E402
    plot_fidelity_overview, plot_long_time_fidelity, plot_offdiagonal,
    plot_pair_comparison, plot_population_lines, plot_stacked_area,
)

OUT = DATA_DIR / "bell_population_dynamics"
EXPECTED_CNOTS = {"Type3": 16, "Type4": 14, "Type5": 14, "4Q": 5}
REPORT_ROUNDS = (0, 1, 2, 5, 10, 50)


def _tag(eps: float, q: float) -> str:
    return f"eps{eps:g}_q{q:g}".replace(".", "p")


def write_csv(dyn, path) -> None:
    rows = dyn.rows()
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# verification battery (section 8 of the task)
# ---------------------------------------------------------------------------

def run_checks(eps: float, q: float, n_rounds: int, runs: dict) -> dict:
    checks: dict = {}

    p0 = isotropic_populations(eps)
    checks["initial_state"] = {
        "eps": eps,
        "populations": p0.tolist(),
        "expected": [1 - 0.75 * eps, 0.25 * eps, 0.25 * eps, 0.25 * eps],
        "max_abs_error": float(np.max(np.abs(
            p0 - np.array([1 - 0.75 * eps, 0.25 * eps, 0.25 * eps, 0.25 * eps])))),
        "sum_minus_1": float(p0.sum() - 1.0),
        "F_0": float(p0[0]),
        "F_0_equals_1_minus_3eps_over_4": bool(abs(p0[0] - (1 - 0.75 * eps)) < 1e-15),
    }

    counts = verify_cnot_counts()
    checks["cnot_counts"] = {
        "measured": counts, "expected": EXPECTED_CNOTS,
        "all_match": all(counts[k] == v for k, v in EXPECTED_CNOTS.items()),
    }

    checks["noise_convention"] = verify_noise_convention()
    checks["noise_convention"]["statement"] = (
        "q is the two-qubit REPLACEMENT depolarizing strength, applied after every "
        "CNOT on that CNOT's two qubits. The parent repository writes it as eps2 "
        "and implements it via noisy_bell_state.global_depol_kraus; the identity of "
        "the two implementations at the same numerical q is checked above, on random "
        "2-qubit and 5-qubit states.")

    checks["compilation_residual_vs_ideal_gadget"] = {
        k: {"max_abs": v[0], "frobenius": v[1]}
        for k, v in compilation_residuals().items()}

    checks["exact_map_vs_full_density_matrix"] = {}
    for proto in ("Type3", "Type4", "4Q"):
        r = cross_check_exact_vs_dm(proto, eps, q, n_rounds)
        checks["exact_map_vs_full_density_matrix"][proto] = {
            "max_abs_population_diff": r["max_abs_pop_diff"],
            "max_abs_F_diff": r["max_abs_F_diff"],
            "argmax_round": r["argmax_round"],
        }
    checks["exact_map_vs_full_density_matrix"]["Type5"] = (
        "no Bell-diagonal exact map exists for Type 5 and none was used; "
        "its noisy round map leaves the Bell-diagonal manifold")

    per_state = {}
    for name, dyn in runs.items():
        d = dyn.diagnostics
        per_state[name] = {
            "max_trace_err": max(x["trace_err"] for x in d),
            "max_herm_err": max(x["herm_err"] for x in d),
            "min_eigenvalue_over_run": min(x["min_eig"] for x in d),
            "max_population_sum_err": max(x["pop_sum_err"] for x in d),
            "max_C_Bell": float(dyn.c_bell.max()),
            "bell_diagonal_preserved_numerically": bool(dyn.c_bell.max() < 1e-20),
            "expected_bell_diagonal": PROTOCOL_INFO[name]["bell_diagonal_preserved"],
        }
    checks["per_round_state_validity"] = per_state
    return checks


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps", type=float, default=0.15)
    ap.add_argument("--q", type=float, default=0.02)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--long-rounds", type=int, default=5000)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    eps, q, N, NL = args.eps, args.q, args.rounds, args.long_rounds
    OUT.mkdir(parents=True, exist_ok=True)
    tag = _tag(eps, q)

    # ---- short runs, all four protocols -------------------------------
    runs = {p: run_dynamics(p, eps, q, n_rounds=N, backend="dm") for p in PROTOCOLS}
    for p, dyn in runs.items():
        write_csv(dyn, OUT / f"{p}_{tag}_n{N}.csv")

    # ---- long runs ----------------------------------------------------
    long_runs = {"Type5": run_dynamics("Type5", eps, q, n_rounds=NL, backend="dm"),
                 "Type5cal": run_dynamics("Type5cal", eps, q, n_rounds=NL, backend="dm")}
    for p, dyn in long_runs.items():
        write_csv(dyn, OUT / f"{p}_{tag}_n{NL}.csv")
    # long runs for the other three come from the verified exact map (cheap, and
    # the exact/dm agreement over N rounds is checked separately)
    long_exact = {p: run_dynamics(p, eps, q, n_rounds=NL, backend="exact",
                                  record_every=max(1, NL // 500))
                  for p in ("Type3", "Type4", "4Q")}
    for p, dyn in long_exact.items():
        write_csv(dyn, OUT / f"{p}_{tag}_n{NL}_exact.csv")

    # ---- checks -------------------------------------------------------
    checks = run_checks(eps, q, N, runs)

    # ---- requested tables ---------------------------------------------
    tables = {}
    for p, dyn in runs.items():
        long_ref = long_runs.get(p) or long_exact.get(p)
        tables[p] = {
            "populations": {f"n={n}": dyn.pops[n].tolist() for n in REPORT_ROUNDS
                            if n <= N},
            "F": {f"F_{n}": float(dyn.fidelity[n]) for n in REPORT_ROUNDS if n <= N},
            "C_Bell": {f"n={n}": float(dyn.c_bell[n]) for n in REPORT_ROUNDS if n <= N},
            "long_time": {
                "n": int(long_ref.n[-1]),
                "populations": long_ref.pops[-1].tolist(),
                "F": float(long_ref.fidelity[-1]),
                "C_Bell": float(long_ref.c_bell[-1]),
                "backend": long_ref.backend,
            },
            "per_round_weight_n1": float(dyn.weight[1]),
            "weight_meaning": ("postselection success probability"
                               if PROTOCOL_INFO[p]["family"] == "postselect"
                               else "parity weight Tr(tau_A), NOT a probability"),
        }
    escape = {p: plateau_and_escape(d) for p, d in long_runs.items()}
    for p, d in long_exact.items():
        escape[p] = plateau_and_escape(d)

    summary = {"eps": eps, "q": q, "rounds": N, "long_rounds": NL,
               "bell_order": BELL_NAMES, "tables": tables,
               "plateau_and_escape": escape, "checks": checks}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    meta = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                 text=True, check=True).stdout.strip(),
        "eps": eps, "q": q, "rounds": N, "long_rounds": NL,
        "noise_model": ("two-qubit replacement depolarizing D_q after every CNOT, "
                        "on that CNOT's two qubits"),
        "protocols": {p: PROTOCOL_INFO[p] for p in PROTOCOLS},
        "parent_repo_provenance": swap_test_source.provenance(),
    }
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2))

    # ---- figures ------------------------------------------------------
    if not args.no_figures:
        import matplotlib.pyplot as plt
        for p, dyn in runs.items():
            plot_stacked_area(dyn).savefig(
                FIG_DIR / f"bellpop_stacked_{p}.png", dpi=150); plt.close("all")
            plot_population_lines(dyn).savefig(
                FIG_DIR / f"bellpop_lines_{p}.png", dpi=150); plt.close("all")
        plot_fidelity_overview([runs[p] for p in PROTOCOLS]).savefig(
            FIG_DIR / "bellpop_overview_F.png", dpi=150); plt.close("all")
        plot_offdiagonal(runs["Type5"], long_runs["Type5"], escape=escape["Type5"]).savefig(
            FIG_DIR / "bellpop_type5_offdiagonal.png", dpi=150); plt.close("all")
        ann = escape["Type5"]
        plot_long_time_fidelity(long_runs["Type5"], log_x=False, annotate=ann).savefig(
            FIG_DIR / "bellpop_type5_longtime_linear.png", dpi=150); plt.close("all")
        plot_long_time_fidelity(long_runs["Type5"], log_x=True, annotate=ann).savefig(
            FIG_DIR / "bellpop_type5_longtime_log.png", dpi=150); plt.close("all")
        plot_pair_comparison(long_runs["Type5"], long_runs["Type5cal"]).savefig(
            FIG_DIR / "bellpop_type5_vs_calibrated.png", dpi=150); plt.close("all")

    # ---- console report ------------------------------------------------
    print(f"eps = {eps}, q = {q}, rounds = {N}, long rounds = {NL}\n")
    print("CHECKS")
    c = checks
    print(f"  initial state max error         {c['initial_state']['max_abs_error']:.2e}"
          f"   F_0 = {c['initial_state']['F_0']}")
    print(f"  CNOT counts                     {c['cnot_counts']['measured']}  "
          f"all match: {c['cnot_counts']['all_match']}")
    print(f"  q == parent eps2 (2q Kraus)     "
          f"{c['noise_convention']['max_abs_diff_2q_vs_parent_kraus']:.2e}")
    print(f"  q == parent eps2 (5q channel)   "
          f"{c['noise_convention']['max_abs_diff_5q_vs_parent_replacement']:.2e}")
    for k, v in c["compilation_residual_vs_ideal_gadget"].items():
        print(f"  compilation residual {k:<7}    Frobenius {v['frobenius']:.3e}")
    for k, v in c["exact_map_vs_full_density_matrix"].items():
        if isinstance(v, dict):
            print(f"  exact vs full DM {k:<9}      max |dp| "
                  f"{v['max_abs_population_diff']:.2e}")
    for k, v in c["per_round_state_validity"].items():
        print(f"  {k:<6} trace {v['max_trace_err']:.1e}  herm {v['max_herm_err']:.1e}  "
              f"min eig {v['min_eigenvalue_over_run']:.3e}  "
              f"pop sum {v['max_population_sum_err']:.1e}  "
              f"max C_Bell {v['max_C_Bell']:.2e}")

    print("\nBELL POPULATIONS  (Phi+, Phi-, Psi+, Psi-)")
    for p in PROTOCOLS:
        print(f"  {p}")
        for n in REPORT_ROUNDS:
            if n > N:
                continue
            v = runs[p].pops[n]
            print(f"     n={n:<3} {v[0]:.12f}  {v[1]:.12f}  {v[2]:.12f}  {v[3]:.12f}"
                  f"   sum={v.sum():.15f}")
        lt = tables[p]["long_time"]
        print(f"     n={lt['n']:<3} " + "  ".join(f"{x:.12f}" for x in lt['populations'])
              + f"   [{lt['backend']}]")

    print("\nF TABLE")
    hdr = "  protocol  " + "".join(f"F_{n:<14}" for n in REPORT_ROUNDS) + "F_long"
    print(hdr)
    for p in PROTOCOLS:
        row = "".join(f"{runs[p].fidelity[n]:<16.10f}" for n in REPORT_ROUNDS if n <= N)
        print(f"  {p:<10}{row}{tables[p]['long_time']['F']:.10f}")

    print("\nPLATEAU / ESCAPE")
    for p, e in escape.items():
        print(f"  {p:<9} plateau F = {e['plateau_F']}  n_escape = {e['n_escape']}  "
              f"F_end = {e.get('F_end')}  C_Bell_end = {e.get('c_bell_end'):.4e}"
              + (f"  C_Bell growth/round = {e['c_bell_growth_per_round']:.6f}"
                 if e.get("c_bell_growth_per_round") else ""))
    print(f"\nwrote {OUT}/ and figures in {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
