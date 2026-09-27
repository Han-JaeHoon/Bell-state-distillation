"""Repeated dynamics along the Bell-isotropic initial line, per protocol.

One noise strength, several inputs: how do the Bell populations move, and does
the starting eps change where they end up?

Default run: Type 3, Type 4 and 4Q, q = 0.02,
eps in {0.02, 0.05, 0.10, 0.15, 0.20, 0.30}, n = 0..50.

    python scripts/epsilon_sweep.py                          # all three
    python scripts/epsilon_sweep.py --protocol 4Q --q 0.05
    python scripts/epsilon_sweep.py --q 0.12 --eps 0.05 0.2 0.5

The dynamics come from the VERIFIED exact Bell-diagonal maps in
``pqec_distill.analytic_exact_map``, which were validated row-by-row against
the frozen circuit-only reference.  No map is re-derived here.  A full
density-matrix simulation of the actual circuit is run alongside as a
cross-check at every eps.  For 4Q the round is a physical postselection, so its
per-round success probability is recorded as well.

Type 5 is deliberately excluded: its noisy dynamics does not stay Bell-diagonal,
so there is no exact Bell-diagonal map for it.

Writes under results/data/epsilon_sweep/ and results/figures/:

  <tag>_eps_<eps>_q<q>.csv          round-by-round populations, per eps
  <tag>_summary.csv, <tag>_summary.json
  comparison.csv, comparison.json   the three protocols side by side
  <tag>_eps_<eps>_stacked.png       Figure A, per eps
  <tag>_eps_<eps>_lines.png         Figure B, per eps
  <tag>_F_overlay_q<q>.png          Figure C
  <tag>_summary_vs_eps_q<q>.png     Figure D
  4q_Psucc_overlay_q<q>.png         4Q only: success probability per round
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
    BELL_NAMES, INVARIANT_PAIR, PROTOCOL_INFO, cross_check_exact_vs_dm,
    isotropic_populations, run_dynamics, verify_cnot_counts,
    verify_noise_convention,
)
from pqec_distill.bell_population_plots import (  # noqa: E402
    plot_fidelity_epsilon_overlay, plot_population_lines, plot_stacked_area,
    plot_success_epsilon_overlay, plot_summary_vs_epsilon,
)

SWEEP_PROTOCOLS = ("Type3", "Type4", "4Q")
TAG = {"Type3": "type3", "Type4": "type4", "4Q": "4q"}
OUT = DATA_DIR / "epsilon_sweep"
DEFAULT_EPS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
REPORT_ROUNDS = (0, 1, 2, 5, 10, 50)


def fmt(x: float) -> str:
    """0.15 -> '0p15', for file names."""
    return f"{x:g}".replace(".", "p")


# ---------------------------------------------------------------------------
# dynamics
# ---------------------------------------------------------------------------

def sweep(protocol: str, eps_values, q: float, n_rounds: int,
          long_rounds: int) -> dict:
    """Run ``protocol`` from every eps, on the exact map, plus a long run each."""
    short = {e: run_dynamics(protocol, e, q, n_rounds=n_rounds, backend="exact")
             for e in eps_values}
    long = {e: run_dynamics(protocol, e, q, n_rounds=long_rounds, backend="exact",
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
    """max |p_a - p_b| over the run for the protocol's invariant pair:
    Psi+ = Psi- for Type 3/4, Phi- = Psi- for 4Q."""
    a, b = INVARIANT_PAIR[dyn.protocol]
    i_a, i_b = BELL_NAMES.index(a), BELL_NAMES.index(b)
    return float(np.max(np.abs(dyn.pops[:, i_a] - dyn.pops[:, i_b])))


def contraction_rate(dyn, f_inf: float, skip: int = 3,
                     floor: float = 1e-13) -> dict:
    """How fast |F_n - F_inf| shrinks, two ways.

    ``fit`` is a log-linear fit over the stretch past the initial transient and
    still above round-off: an AVERAGE over that window, so it is still pulled up
    by whatever transient is left in it.  ``last_ratio`` is the single ratio
    r_n / r_{n-1} at the last round above the floor, which is the better
    estimate of the asymptotic rate -- the Jacobian spectral radius at the fixed
    point, which the ratio sequence approaches.
    """
    resid = np.abs(dyn.fidelity - f_inf)
    m = (resid > floor) & (dyn.n >= skip)
    out = {"fit": None, "last_ratio": None, "n_last": None}
    if m.sum() >= 3:
        out["fit"] = float(np.exp(np.polyfit(dyn.n[m], np.log(resid[m]), 1)[0]))
    idx = np.where(m)[0]
    if idx.size:
        j = int(idx[-1])
        if j > 0 and resid[j - 1] > 0:
            out["last_ratio"] = float(resid[j] / resid[j - 1])
            out["n_last"] = int(dyn.n[j])
    return out


def anisotropy_round(dyn, tol: float = 1e-12) -> int | None:
    """First round at which the three error components are no longer all equal."""
    err = dyn.pops[:, 1:]
    spread = err.max(axis=1) - err.min(axis=1)
    hit = np.where(spread > tol)[0]
    return int(dyn.n[hit[0]]) if hit.size else None


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------

_MAP_NAME = {"Type3": "analytic_map_type3", "Type4": "analytic_map_type4",
             "4Q": "analytic_map_4q"}


def run_checks(protocol: str, eps_values, q: float, n_rounds: int,
               runs: dict) -> dict:
    expected_cnot = PROTOCOL_INFO[protocol]["n_cnot"]
    checks: dict = {"map_used": (
        f"pqec_distill.analytic_exact_map.{_MAP_NAME[protocol]} -- the verified "
        "exact Bell-diagonal map; no map was re-derived for this analysis"),
        "invariant_relation_checked": " = ".join(
            f"p_{x}" for x in INVARIANT_PAIR[protocol])}

    counts = verify_cnot_counts()
    checks["cnot_count"] = {"measured": counts[protocol], "expected": expected_cnot,
                            "match": counts[protocol] == expected_cnot,
                            "all_protocols": counts}

    nc = verify_noise_convention(n_trials=3)
    nc["statement"] = (
        "q is the two-qubit REPLACEMENT depolarizing strength applied after every "
        "CNOT on that CNOT's two qubits; identical to the parent repository's eps2 "
        "(noisy_bell_state.global_depol_kraus) at the same numerical value")
    checks["noise_convention"] = nc

    per_eps = {}
    for e, dyn in runs["short"].items():
        cc = cross_check_exact_vs_dm(protocol, e, q, n_rounds=n_rounds)
        per_eps[f"{e:g}"] = {
            "population_sum_max_err": float(np.max(np.abs(dyn.pops.sum(axis=1) - 1))),
            "min_population": float(dyn.pops.min()),
            "no_negative_population": bool(dyn.pops.min() >= -1e-15),
            "invariant_pair_max_abs_diff": symmetry_residual(dyn),
            "exact_vs_full_dm_max_population_diff": cc["max_abs_pop_diff"],
            "exact_vs_full_dm_max_F_diff": cc["max_abs_F_diff"],
            "full_dm_max_C_Bell": cc["max_C_Bell_dm"],
        }
    checks["per_epsilon"] = per_eps
    return checks


# ---------------------------------------------------------------------------
# one protocol
# ---------------------------------------------------------------------------

def analyse(protocol: str, eps_values, q: float, N: int, NL: int,
            conv_tol: float, figures: bool) -> dict:
    tag = TAG[protocol]
    runs = sweep(protocol, eps_values, q, N, NL)

    for e, dyn in runs["short"].items():
        with open(OUT / f"{tag}_eps_{fmt(e)}_q{fmt(q)}.csv", "w", newline="") as fh:
            rows = dyn.rows()
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)

    finals = {e: runs["long"][e].pops[-1] for e in eps_values}
    ref = finals[eps_values[-1]]
    spread = float(max(np.max(np.abs(v - ref)) for v in finals.values()))
    f_inf = float(ref[0])
    fixed_point = {
        "populations": ref.tolist(), "F": f_inf,
        "max_spread_across_epsilon": spread,
        "same_for_every_epsilon": bool(spread < 1e-12),
        "n_long": NL,
        "break_even_eps": 4.0 * (1.0 - f_inf) / 3.0,
        "break_even_note": ("the eps at which F_0 = F_infty; NOT a threshold, "
                            "just where the input crosses the attractor"),
    }
    if PROTOCOL_INFO[protocol]["family"] == "postselect":
        fixed_point["P_succ_at_fixed_point"] = float(runs["long"][eps_values[-1]].weight[-1])

    checks = run_checks(protocol, eps_values, q, N, runs)

    tables = {}
    for e in eps_values:
        dyn, lng = runs["short"][e], runs["long"][e]
        t = {
            "initial_populations": isotropic_populations(e).tolist(),
            "populations": {f"n={n}": dyn.pops[n].tolist()
                            for n in REPORT_ROUNDS if n <= N},
            "F": {f"F_{n}": float(dyn.fidelity[n]) for n in REPORT_ROUNDS if n <= N},
            "converged_populations": lng.pops[-1].tolist(),
            "F_infty": float(lng.pops[-1][0]),
            "n_converged": convergence_round(dyn, lng.pops[-1], conv_tol),
            "n_anisotropic": anisotropy_round(dyn),
            "invariant_pair": list(INVARIANT_PAIR[protocol]),
            "invariant_pair_max_abs_diff": symmetry_residual(dyn),
            "contraction_rate": contraction_rate(dyn, f_inf),
            "F_infty_minus_F_0": float(lng.pops[-1][0] - dyn.fidelity[0]),
            "round_improves_on_input": bool(lng.pops[-1][0] > dyn.fidelity[0]),
        }
        if PROTOCOL_INFO[protocol]["family"] == "postselect":
            t["P_succ"] = {f"n={n}": (None if not np.isfinite(dyn.weight[n])
                                      else float(dyn.weight[n]))
                           for n in REPORT_ROUNDS if n <= N}
            t["P_succ_infty"] = float(lng.weight[-1])
        tables[f"{e:g}"] = t

    summary = {"protocol": protocol, "n_cnot": PROTOCOL_INFO[protocol]["n_cnot"],
               "q": q, "eps_values": list(eps_values), "rounds": N,
               "long_rounds": NL, "bell_order": BELL_NAMES,
               "fixed_point": fixed_point, "tables": tables, "checks": checks,
               "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
               "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                        text=True, check=True).stdout.strip()}
    (OUT / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2))

    with open(OUT / f"{tag}_summary.csv", "w", newline="") as fh:
        cols = (["protocol", "eps"] + [f"F_{n}" for n in REPORT_ROUNDS if n <= N]
                + ["F_infty", "F_infty_minus_F_0", "n_converged", "n_anisotropic",
                   "contraction_rate_fit", "contraction_rate_last_ratio",
                   "invariant_pair", "invariant_pair_max_abs_diff"])
        if PROTOCOL_INFO[protocol]["family"] == "postselect":
            cols += ["P_succ_n1", "P_succ_infty"]
        w = csv.writer(fh)
        w.writerow(cols)
        for e in eps_values:
            t = tables[f"{e:g}"]
            row = ([protocol, e] + [t["F"][f"F_{n}"] for n in REPORT_ROUNDS if n <= N]
                   + [t["F_infty"], t["F_infty_minus_F_0"], t["n_converged"],
                      t["n_anisotropic"], t["contraction_rate"]["fit"],
                      t["contraction_rate"]["last_ratio"],
                      "=".join(t["invariant_pair"]), t["invariant_pair_max_abs_diff"]])
            if PROTOCOL_INFO[protocol]["family"] == "postselect":
                row += [t["P_succ"]["n=1"], t["P_succ_infty"]]
            w.writerow(row)

    if figures:
        import matplotlib.pyplot as plt
        dashed = (INVARIANT_PAIR[protocol][1],)
        for e in eps_values:
            dyn = runs["short"][e]
            plot_stacked_area(dyn).savefig(
                FIG_DIR / f"{tag}_eps_{fmt(e)}_stacked.png", dpi=150)
            plt.close("all")
            # the second member of the invariant pair is drawn dashed on top of
            # the first, so their exact coincidence reads as two curves
            plot_population_lines(dyn, dashed=dashed).savefig(
                FIG_DIR / f"{tag}_eps_{fmt(e)}_lines.png", dpi=150)
            plt.close("all")
        ordered = [runs["short"][e] for e in eps_values]
        plot_fidelity_epsilon_overlay(ordered).savefig(
            FIG_DIR / f"{tag}_F_overlay_q{fmt(q)}.png", dpi=150)
        plt.close("all")
        picks = list(dict.fromkeys([0, 1, max(r for r in REPORT_ROUNDS if r <= N)]))
        series = {rf"$F_{{{n}}}$": [tables[f"{e:g}"]["F"][f"F_{n}"] for e in eps_values]
                  for n in picks}
        plot_summary_vs_epsilon(eps_values, series, q, fixed_point=f_inf,
                                label=PROTOCOL_INFO[protocol]["label"] + " -").savefig(
            FIG_DIR / f"{tag}_summary_vs_eps_q{fmt(q)}.png", dpi=150)
        plt.close("all")
        if PROTOCOL_INFO[protocol]["family"] == "postselect":
            plot_success_epsilon_overlay(
                ordered, PROTOCOL_INFO[protocol]["label"]).savefig(
                FIG_DIR / f"{tag}_Psucc_overlay_q{fmt(q)}.png", dpi=150)
            plt.close("all")

    return {"summary": summary, "runs": runs, "tables": tables,
            "fixed_point": fixed_point, "checks": checks}


def report(protocol: str, result: dict, eps_values, q: float, N: int) -> None:
    tables, checks, fp = result["tables"], result["checks"], result["fixed_point"]
    info = PROTOCOL_INFO[protocol]
    postselect = info["family"] == "postselect"
    print(f"\n{'=' * 78}\n{info['label']}   |   q = {q}, rounds = {N}\n{'=' * 78}")
    print("CHECKS")
    print(f"  exact map            {checks['map_used'].split(' -- ')[0]}")
    print(f"  CNOT count = {checks['cnot_count']['expected']:<3}     "
          f"measured {checks['cnot_count']['measured']}  "
          f"match {checks['cnot_count']['match']}")
    print(f"  invariant relation   {checks['invariant_relation_checked']}")
    print(f"  q == parent eps2     "
          f"{checks['noise_convention']['max_abs_diff_2q_vs_parent_kraus']:.1e} (2q) / "
          f"{checks['noise_convention']['max_abs_diff_5q_vs_parent_replacement']:.1e} (5q)")
    for e, c in checks["per_epsilon"].items():
        print(f"    eps={e:<5} sum-1 {c['population_sum_max_err']:.1e}  "
              f"min p {c['min_population']:.4f}  "
              f"invariant {c['invariant_pair_max_abs_diff']:.1e}  "
              f"exact vs DM {c['exact_vs_full_dm_max_population_diff']:.1e}  "
              f"C_Bell {c['full_dm_max_C_Bell']:.1e}")

    print("\nBELL POPULATIONS  (Phi+, Phi-, Psi+, Psi-)")
    for e in eps_values:
        t = tables[f"{e:g}"]
        print(f"  eps = {e:g}")
        for n in REPORT_ROUNDS:
            if n > N:
                continue
            v = t["populations"][f"n={n}"]
            line = (f"     n={n:<3} " + "  ".join(f"{x:.12f}" for x in v)
                    + f"   sum={sum(v):.15f}")
            if postselect and t["P_succ"][f"n={n}"] is not None:
                line += f"   P_succ={t['P_succ'][f'n={n}']:.12f}"
            print(line)
        print("     n->inf " + "  ".join(f"{x:.12f}" for x in t["converged_populations"])
              + (f"   P_succ={t['P_succ_infty']:.12f}" if postselect else ""))

    print("\nSUMMARY TABLE")
    print("  eps    " + "".join(f"F_{n:<14}" for n in REPORT_ROUNDS if n <= N)
          + "F_infty         dF=F_inf-F_0   n_conv  rate")
    for e in eps_values:
        t = tables[f"{e:g}"]
        row = "".join(f"{t['F'][f'F_{n}']:<16.10f}" for n in REPORT_ROUNDS if n <= N)
        rate = t["contraction_rate"]["last_ratio"]
        print(f"  {e:<7g}{row}{t['F_infty']:<16.12f}{t['F_infty_minus_F_0']:+.10f}   "
              f"{str(t['n_converged']):<7} " + (f"{rate:.6f}" if rate else "n/a"))
    print(f"  fixed point identical across eps: {fp['same_for_every_epsilon']} "
          f"(max spread {fp['max_spread_across_epsilon']:.2e})")
    print(f"  F_infty = {fp['F']:.15f}   break-even eps (F_0 = F_infty) = "
          f"{fp['break_even_eps']:.12f}")


# ---------------------------------------------------------------------------
# cross-protocol comparison
# ---------------------------------------------------------------------------

def write_comparison(results: dict, eps_values, q: float, N: int) -> None:
    rows = []
    for protocol, res in results.items():
        fp = res["fixed_point"]
        for e in eps_values:
            t = res["tables"][f"{e:g}"]
            row = {
                "protocol": protocol,
                "n_cnot": PROTOCOL_INFO[protocol]["n_cnot"],
                "q": q, "eps": e,
                **{f"F_{n}": t["F"][f"F_{n}"] for n in REPORT_ROUNDS if n <= N},
                "F_infty": t["F_infty"],
                "F_infty_minus_F_0": t["F_infty_minus_F_0"],
                "improves": t["round_improves_on_input"],
                "n_converged": t["n_converged"],
                "contraction_rate_fit": t["contraction_rate"]["fit"],
                "contraction_rate_last_ratio": t["contraction_rate"]["last_ratio"],
                "invariant_pair": "=".join(t["invariant_pair"]),
                "invariant_pair_max_abs_diff": t["invariant_pair_max_abs_diff"],
                "break_even_eps": fp["break_even_eps"],
                "fixed_point_same_for_every_eps": fp["same_for_every_epsilon"],
                "P_succ_infty": t.get("P_succ_infty", ""),
            }
            rows.append(row)
    with open(OUT / "comparison.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    per_protocol = {}
    for protocol, res in results.items():
        fp = res["fixed_point"]
        rates = [res["tables"][f"{e:g}"]["contraction_rate"]["last_ratio"]
                 for e in eps_values]
        rates = [r for r in rates if r is not None]
        per_protocol[protocol] = {
            "n_cnot": PROTOCOL_INFO[protocol]["n_cnot"],
            "fixed_point_populations": fp["populations"],
            "F_infty": fp["F"],
            "fixed_point_same_for_every_eps": fp["same_for_every_epsilon"],
            "max_spread_across_epsilon": fp["max_spread_across_epsilon"],
            "break_even_eps": fp["break_even_eps"],
            "invariant_pair": list(INVARIANT_PAIR[protocol]),
            "invariant_pair_worst": max(
                res["tables"][f"{e:g}"]["invariant_pair_max_abs_diff"]
                for e in eps_values),
            "contraction_rate_last_ratio_median": (float(np.median(rates))
                                                  if rates else None),
            "n_converged_max": max(res["tables"][f"{e:g}"]["n_converged"]
                                   for e in eps_values),
            "eps_that_lose_fidelity": [e for e in eps_values
                                       if not res["tables"][f"{e:g}"]["round_improves_on_input"]],
        }
        if "P_succ_at_fixed_point" in fp:
            per_protocol[protocol]["P_succ_at_fixed_point"] = fp["P_succ_at_fixed_point"]
    (OUT / "comparison.json").write_text(json.dumps(
        {"q": q, "eps_values": list(eps_values), "rounds": N,
         "per_protocol": per_protocol}, indent=2))

    print(f"\n{'=' * 78}\nCOMPARISON  (q = {q})\n{'=' * 78}")
    print(f"  {'protocol':<9}{'CNOT':<6}{'F_infty':<18}{'break-even eps':<17}"
          f"{'rate/round':<13}{'invariant':<16}{'P_succ_inf'}")
    for protocol, d in per_protocol.items():
        print(f"  {protocol:<9}{d['n_cnot']:<6}{d['F_infty']:<18.12f}"
              f"{d['break_even_eps']:<17.12f}"
              f"{d['contraction_rate_last_ratio_median']:<13.9f}"
              f"{'='.join(d['invariant_pair']):<16}"
              + (f"{d['P_succ_at_fixed_point']:.9f}"
                 if "P_succ_at_fixed_point" in d else "-"))
    print("\n  converged populations (Phi+, Phi-, Psi+, Psi-)")
    for protocol, d in per_protocol.items():
        print(f"    {protocol:<9}" + "  ".join(f"{x:.12f}"
                                               for x in d["fixed_point_populations"]))
    print("\n  eps whose fidelity FALLS under repetition (F_0 > F_infty)")
    for protocol, d in per_protocol.items():
        print(f"    {protocol:<9}{d['eps_that_lose_fidelity'] or 'none'}")


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--protocol", nargs="+", default=list(SWEEP_PROTOCOLS),
                    choices=list(SWEEP_PROTOCOLS))
    ap.add_argument("--q", type=float, default=0.02)
    ap.add_argument("--eps", type=float, nargs="+", default=list(DEFAULT_EPS))
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--long-rounds", type=int, default=2000)
    ap.add_argument("--conv-tol", type=float, default=1e-12)
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    q, eps_values = args.q, sorted(args.eps)
    N, NL = args.rounds, args.long_rounds
    OUT.mkdir(parents=True, exist_ok=True)

    results = {}
    for protocol in [p for p in SWEEP_PROTOCOLS if p in args.protocol]:
        results[protocol] = analyse(protocol, eps_values, q, N, NL,
                                    args.conv_tol, not args.no_figures)
        report(protocol, results[protocol], eps_values, q, N)

    if len(results) > 1:
        write_comparison(results, eps_values, q, N)
    print(f"\nwrote {OUT}/ and figures in {FIG_DIR}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
