"""Type 4 repeated dynamics along the Bell-isotropic initial line.

One protocol (Type 4, the resynthesised 14-CNOT circuit), one noise strength,
several inputs: how do the Bell populations move, and does the starting eps
change where they end up?

Default run: q = 0.02, eps in {0.02, 0.05, 0.10, 0.15, 0.20, 0.30}, n = 0..50.
Everything is parameterised, so sweeping q later is

    python scripts/type4_epsilon_sweep.py --q 0.05
    python scripts/type4_epsilon_sweep.py --q 0.12 --eps 0.05 0.2 0.5

The dynamics come from the VERIFIED exact Bell-diagonal map
(``pqec_distill.analytic_exact_map.analytic_map_type4``), which was validated
row-by-row against the frozen circuit-only reference.  No map is re-derived
here.  A full density-matrix simulation of the actual 14-CNOT circuit is run
alongside it as a cross-check at every eps.

Writes under results/data/type4_epsilon_sweep/ and results/figures/:

  type4_eps_<eps>_q<q>.csv          round-by-round populations, per eps
  summary.csv, summary.json         the tables, the checks, the fixed point
  type4_eps_<eps>_stacked.png       Figure A, per eps
  type4_eps_<eps>_lines.png         Figure B, per eps
  type4_F_overlay_q<q>.png          Figure C
  type4_summary_vs_eps_q<q>.png     Figure D
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
from pqec_distill.bell_population_dynamics import (  # noqa: E402
    BELL_NAMES, PROTOCOL_INFO, cross_check_exact_vs_dm, isotropic_populations,
    run_dynamics, verify_cnot_counts, verify_noise_convention,
)
from pqec_distill.bell_population_plots import (  # noqa: E402
    plot_fidelity_epsilon_overlay, plot_population_lines,
    plot_stacked_area, plot_summary_vs_epsilon,
)

PROTOCOL = "Type4"
OUT = DATA_DIR / "type4_epsilon_sweep"
DEFAULT_EPS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
REPORT_ROUNDS = (0, 1, 2, 5, 10, 50)


def fmt(x: float) -> str:
    """0.15 -> '0p15', for file names."""
    return f"{x:g}".replace(".", "p")


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

def sweep(eps_values, q: float, n_rounds: int, long_rounds: int) -> dict:
    """Run Type 4 from every eps, on the exact map, plus a long run each."""
    short = {e: run_dynamics(PROTOCOL, e, q, n_rounds=n_rounds, backend="exact")
             for e in eps_values}
    long = {e: run_dynamics(PROTOCOL, e, q, n_rounds=long_rounds, backend="exact",
                            record_every=max(1, long_rounds // 400))
            for e in eps_values}
    return {"short": short, "long": long}


def convergence_round(dyn, target, tol: float) -> int | None:
    """First recorded round whose populations are within ``tol`` of ``target``
    and stay there for the rest of the run."""
    far = np.max(np.abs(dyn.pops - target), axis=1) >= tol
    idx = np.where(far)[0]
    if idx.size == 0:
        return int(dyn.n[0])
    last = int(idx[-1])
    return int(dyn.n[last + 1]) if last + 1 < len(dyn.n) else None


def symmetry_residual(dyn) -> float:
    """max |p_Psi+ - p_Psi-| over the run: the Type 4 invariant relation."""
    i_p, i_m = BELL_NAMES.index("Psi+"), BELL_NAMES.index("Psi-")
    return float(np.max(np.abs(dyn.pops[:, i_p] - dyn.pops[:, i_m])))


def anisotropy_round(dyn, tol: float = 1e-12) -> int | None:
    """First round at which the three error components are no longer all equal."""
    err = dyn.pops[:, 1:]
    spread = err.max(axis=1) - err.min(axis=1)
    hit = np.where(spread > tol)[0]
    return int(dyn.n[hit[0]]) if hit.size else None


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

def run_checks(eps_values, q: float, n_rounds: int, runs: dict) -> dict:
    checks: dict = {"map_used": (
        "pqec_distill.analytic_exact_map.analytic_map_type4 -- the verified "
        "exact Bell-diagonal map; no map was re-derived for this analysis")}

    counts = verify_cnot_counts()
    checks["cnot_count"] = {"Type4_measured": counts["Type4"], "expected": 14,
                            "match": counts["Type4"] == 14, "all_protocols": counts}

    nc = verify_noise_convention(n_trials=3)
    nc["statement"] = (
        "q is the two-qubit REPLACEMENT depolarizing strength applied after every "
        "CNOT on that CNOT's two qubits; identical to the parent repository's eps2 "
        "(noisy_bell_state.global_depol_kraus) at the same numerical value")
    checks["noise_convention"] = nc

    per_eps = {}
    for e, dyn in runs["short"].items():
        cc = cross_check_exact_vs_dm(PROTOCOL, e, q, n_rounds=n_rounds)
        per_eps[f"{e:g}"] = {
            "population_sum_max_err": float(np.max(np.abs(dyn.pops.sum(axis=1) - 1))),
            "min_population": float(dyn.pops.min()),
            "no_negative_population": bool(dyn.pops.min() >= -1e-15),
            "max_abs_Psi_plus_minus_Psi_minus": symmetry_residual(dyn),
            "exact_vs_full_dm_max_population_diff": cc["max_abs_pop_diff"],
            "exact_vs_full_dm_max_F_diff": cc["max_abs_F_diff"],
            "full_dm_max_C_Bell": cc["max_C_Bell_dm"],
        }
    checks["per_epsilon"] = per_eps
    return checks


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=float, default=0.02)
    ap.add_argument("--eps", type=float, nargs="+", default=list(DEFAULT_EPS))
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--long-rounds", type=int, default=2000)
    ap.add_argument("--conv-tol", type=float, default=1e-12)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    q, eps_values, N, NL = args.q, sorted(args.eps), args.rounds, args.long_rounds
    OUT.mkdir(parents=True, exist_ok=True)

    runs = sweep(eps_values, q, N, NL)
    for e, dyn in runs["short"].items():
        with open(OUT / f"type4_eps_{fmt(e)}_q{fmt(q)}.csv", "w", newline="") as fh:
            rows = dyn.rows()
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    # the fixed point, taken from the long runs
    finals = {e: runs["long"][e].pops[-1] for e in eps_values}
    ref = finals[eps_values[-1]]
    spread = float(max(np.max(np.abs(v - ref)) for v in finals.values()))
    fixed_point = {
        "populations": ref.tolist(),
        "F": float(ref[0]),
        "max_spread_across_epsilon": spread,
        "same_for_every_epsilon": bool(spread < 1e-12),
        "n_long": NL,
    }

    checks = run_checks(eps_values, q, N, runs)

    # per-eps tables
    tables = {}
    for e in eps_values:
        dyn = runs["short"][e]
        lng = runs["long"][e]
        tables[f"{e:g}"] = {
            "initial_populations": isotropic_populations(e).tolist(),
            "populations": {f"n={n}": dyn.pops[n].tolist()
                            for n in REPORT_ROUNDS if n <= N},
            "F": {f"F_{n}": float(dyn.fidelity[n]) for n in REPORT_ROUNDS if n <= N},
            "converged_populations": lng.pops[-1].tolist(),
            "F_infty": float(lng.pops[-1][0]),
            "n_converged": convergence_round(dyn, lng.pops[-1], args.conv_tol),
            "n_anisotropic": anisotropy_round(dyn),
            "max_abs_Psi_plus_minus_Psi_minus": symmetry_residual(dyn),
            "F_infty_minus_F_0": float(lng.pops[-1][0] - dyn.fidelity[0]),
            "round_improves_on_input": bool(lng.pops[-1][0] > dyn.fidelity[0]),
        }

    summary = {"protocol": PROTOCOL, "q": q, "eps_values": eps_values,
               "rounds": N, "long_rounds": NL, "bell_order": BELL_NAMES,
               "fixed_point": fixed_point, "tables": tables, "checks": checks,
               "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True, check=True).stdout.strip(),
               "n_cnot": PROTOCOL_INFO[PROTOCOL]["n_cnot"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))

    with open(OUT / "summary.csv", "w", newline="") as fh:
        cols = (["eps"] + [f"F_{n}" for n in REPORT_ROUNDS if n <= N]
                + ["F_infty", "F_infty_minus_F_0", "n_converged", "n_anisotropic",
                   "max_abs_PsiP_minus_PsiM"])
        w = csv.writer(fh)
        w.writerow(cols)
        for e in eps_values:
            t = tables[f"{e:g}"]
            w.writerow([e] + [t["F"][f"F_{n}"] for n in REPORT_ROUNDS if n <= N]
                       + [t["F_infty"], t["F_infty_minus_F_0"], t["n_converged"],
                          t["n_anisotropic"], t["max_abs_Psi_plus_minus_Psi_minus"]])

    # ---- figures ------------------------------------------------------
    if not args.no_figures:
        import matplotlib.pyplot as plt
        for e in eps_values:
            dyn = runs["short"][e]
            plot_stacked_area(dyn).savefig(
                FIG_DIR / f"type4_eps_{fmt(e)}_stacked.png", dpi=150)
            plt.close("all")
            # Psi- is expected to lie exactly on Psi+ for Type 4, so draw it
            # dashed: two coincident curves, not one missing curve
            plot_population_lines(dyn, dashed=("Psi-",)).savefig(
                FIG_DIR / f"type4_eps_{fmt(e)}_lines.png", dpi=150)
            plt.close("all")
        plot_fidelity_epsilon_overlay([runs["short"][e] for e in eps_values]).savefig(
            FIG_DIR / f"type4_F_overlay_q{fmt(q)}.png", dpi=150)
        plt.close("all")
        # F_0 (the input), F_1 (what one round does) and F at the last round;
        # F_infty is the dashed reference line, since it coincides with F_last
        picks = [0, 1, max(r for r in REPORT_ROUNDS if r <= N)]
        series = {rf"$F_{{{n}}}$": [tables[f"{e:g}"]["F"][f"F_{n}"] for e in eps_values]
                  for n in dict.fromkeys(picks)}
        plot_summary_vs_epsilon(eps_values, series, q,
                                fixed_point=fixed_point["F"]).savefig(
            FIG_DIR / f"type4_summary_vs_eps_q{fmt(q)}.png", dpi=150)
        plt.close("all")

    # ---- console -------------------------------------------------------
    print(f"Type 4 ({checks['cnot_count']['Type4_measured']} CNOT), q = {q}, "
          f"rounds = {N}, long rounds = {NL}\n")
    print("CHECKS")
    print(f"  exact Bell-diagonal map used   {checks['map_used'].splitlines()[0][:60]}...")
    print(f"  CNOT count = 14                {checks['cnot_count']['match']}")
    print(f"  q == parent eps2 (2q / 5q)     "
          f"{checks['noise_convention']['max_abs_diff_2q_vs_parent_kraus']:.1e} / "
          f"{checks['noise_convention']['max_abs_diff_5q_vs_parent_replacement']:.1e}")
    for e, c in checks["per_epsilon"].items():
        print(f"  eps={e:<5} sum-1 {c['population_sum_max_err']:.1e}  "
              f"min p {c['min_population']:.4f}  "
              f"|PsiP-PsiM| {c['max_abs_Psi_plus_minus_Psi_minus']:.1e}  "
              f"exact vs DM {c['exact_vs_full_dm_max_population_diff']:.1e}")

    print("\nBELL POPULATIONS  (Phi+, Phi-, Psi+, Psi-)")
    for e in eps_values:
        t = tables[f"{e:g}"]
        print(f"  eps = {e:g}")
        for n in REPORT_ROUNDS:
            if n > N:
                continue
            v = t["populations"][f"n={n}"]
            print(f"     n={n:<3} " + "  ".join(f"{x:.12f}" for x in v)
                  + f"   sum={sum(v):.15f}")
        print(f"     n->inf " + "  ".join(f"{x:.12f}" for x in t["converged_populations"]))

    print("\nSUMMARY TABLE")
    head = ("  eps    " + "".join(f"F_{n:<14}" for n in REPORT_ROUNDS if n <= N)
            + "F_infty         dF=F_inf-F_0   n_conv  n_aniso")
    print(head)
    for e in eps_values:
        t = tables[f"{e:g}"]
        row = "".join(f"{t['F'][f'F_{n}']:<16.10f}" for n in REPORT_ROUNDS if n <= N)
        print(f"  {e:<7g}{row}{t['F_infty']:<16.12f}{t['F_infty_minus_F_0']:+.10f}   "
              f"{str(t['n_converged']):<7} {t['n_anisotropic']}")
    print(f"\nfixed point identical across eps: {fixed_point['same_for_every_epsilon']} "
          f"(max spread {fixed_point['max_spread_across_epsilon']:.2e})")
    print(f"F_infty = {fixed_point['F']:.15f}")
    print(f"\nwrote {OUT}/ and figures in {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
