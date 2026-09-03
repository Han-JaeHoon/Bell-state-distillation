"""Phase 2: noisy-CNOT sweep over (epsilon, p) for BOTH noise conventions.

Noise is applied after each of the five CNOTs on the two qubits it acted on;
the final H on q3 is ideal.  The FULL 4x4 retained density matrix is tracked
(never projected back onto the Bell-diagonal family).

Outputs
-------
results/data/noisy_sweep_replace.csv
results/data/noisy_sweep_pauli.csv
results/figures/noisy_{F_out_vs_p,P_success_vs_p,gain_region,
                       concurrence_gain,bell_leakage}_{convention}.png
"""

from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import DATA_DIR, FIG_DIR  # noqa: E402
from pqec_distill.analytics import isotropic_populations, matrix_square_normalized  # noqa: E402
from pqec_distill.bell_states import (  # noqa: E402
    bell_diagonal_state, bell_offdiagonal_norm, bell_populations,
)
from pqec_distill.entanglement import concurrence, negativity  # noqa: E402
from pqec_distill.measurement import postselect_branch  # noqa: E402
from pqec_distill.noise import NOISE_CONVENTIONS, convention_name, noisy_full_channel_dm  # noqa: E402

EPS_VALUES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]
P_VALUES = np.linspace(0.0, 0.2, 41)

FIELDS = [
    "convention", "eps", "p", "F_in", "C_in",
    "P_success", "F_out", "gain", "purity_out",
    "C_out", "N_out", "concurrence_gain",
    "bell_leakage", "trace_distance_to_ideal", "min_eigenvalue",
]


def trace_distance(rho, sigma):
    diff = rho - sigma
    return float(0.5 * np.sum(np.abs(np.linalg.eigvalsh((diff + diff.conj().T) / 2))))


def evaluate(eps: float, p: float, convention: str) -> dict:
    rho_in = bell_diagonal_state(isotropic_populations(eps))
    rho4 = np.kron(rho_in, rho_in)
    out4 = noisy_full_channel_dm(rho4, p, convention)
    rho_tilde, prob = postselect_branch(out4, 0, 0)
    rho_out = rho_tilde / prob if prob > 1e-14 else np.full((4, 4), np.nan)

    f_in = float(bell_populations(rho_in)[0])
    f_out = float(bell_populations(rho_out)[0])
    c_in = concurrence(rho_in)
    c_out = concurrence(rho_out)
    ideal = matrix_square_normalized(rho_in)

    return {
        "convention": convention,
        "eps": eps,
        "p": float(p),
        "F_in": f_in,
        "C_in": c_in,
        "P_success": prob,
        "F_out": f_out,
        "gain": f_out - f_in,
        "purity_out": float(np.real(np.trace(rho_out @ rho_out))),
        "C_out": c_out,
        "N_out": negativity(rho_out),
        "concurrence_gain": c_out - c_in,
        "bell_leakage": bell_offdiagonal_norm(rho_out),
        "trace_distance_to_ideal": trace_distance(rho_out, ideal),
        "min_eigenvalue": float(np.linalg.eigvalsh((rho_out + rho_out.conj().T) / 2).min()),
    }


def make_figures(rows, convention):
    label = "replacement" if convention == "replace" else "Pauli"
    tag = f"({label} depolarizing)"

    def series(eps, key):
        sel = sorted([r for r in rows if r["eps"] == eps], key=lambda r: r["p"])
        return [r["p"] for r in sel], [r[key] for r in sel]

    for key, ylabel, title, fname in [
        ("F_out", r"$F_{out}$", "Output Bell fidelity", "F_out_vs_p"),
        ("P_success", r"$P_{success}$", "Postselection success probability", "P_success_vs_p"),
        ("gain", r"$F_{out}-F_{in}$", "Purification gain", "gain_region"),
        ("concurrence_gain", r"$C_{out}-C_{in}$", "Concurrence gain", "concurrence_gain"),
        ("bell_leakage", "Bell off-diagonal norm", "Bell-diagonal leakage", "bell_leakage"),
    ]:
        fig, ax = plt.subplots(figsize=(6.4, 4.6))
        for eps in EPS_VALUES:
            xs, ys = series(eps, key)
            ax.plot(xs, ys, lw=1.8, marker="o", ms=2.5, label=rf"$\epsilon={eps}$")
        if key in ("gain", "concurrence_gain"):
            ax.axhline(0.0, color="k", ls="--", lw=1)
        ax.set_xlabel(r"per-CNOT noise $p$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\n{tag}", fontsize=10)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"noisy_{fname}_{convention}.png", dpi=150)
        plt.close(fig)


def break_even_p(rows, eps):
    """Largest p for which the gain is still positive (linear interpolation)."""
    sel = sorted([r for r in rows if r["eps"] == eps], key=lambda r: r["p"])
    prev = None
    for r in sel:
        if prev is not None and prev["gain"] > 0 >= r["gain"]:
            x0, y0 = prev["p"], prev["gain"]
            x1, y1 = r["p"], r["gain"]
            return x0 + (x1 - x0) * y0 / (y0 - y1)
        prev = r
    return float("nan") if sel[-1]["gain"] <= 0 else float("inf")


def main() -> int:
    summary = {}
    for convention in NOISE_CONVENTIONS:
        rows = [evaluate(e, p, convention) for e in EPS_VALUES for p in P_VALUES]
        path = DATA_DIR / f"noisy_sweep_{convention}.csv"
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        make_figures(rows, convention)
        summary[convention] = rows

        print(f"=== convention: {convention} ===")
        print(f"    {convention_name(convention)}")
        ideal = [r for r in rows if r["p"] == 0.0]
        print(f"    max Bell leakage at p=0 : {max(r['bell_leakage'] for r in ideal):.3e}")
        print(f"    max leakage over sweep  : {max(r['bell_leakage'] for r in rows):.3e}")
        print(f"    min eigenvalue over sweep: {min(r['min_eigenvalue'] for r in rows):.3e}")
        print(f"    {'eps':>6} {'p*(break-even)':>16}")
        for eps in EPS_VALUES:
            print(f"    {eps:>6.2f} {break_even_p(rows, eps):>16.6f}")
        print()

    # Cross-check the exact reparameterization between the two conventions.
    worst = 0.0
    for r_p in summary["pauli"]:
        target = 16.0 * r_p["p"] / 15.0
        if target > 0.2 + 1e-12:
            continue
        match = evaluate(r_p["eps"], target, "replace")
        worst = max(worst, abs(match["F_out"] - r_p["F_out"]),
                    abs(match["P_success"] - r_p["P_success"]))
    print(f"cross-check  D_Pauli,p == D_replace,16p/15 : max |diff| = {worst:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
