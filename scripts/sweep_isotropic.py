"""Bell-isotropic epsilon sweep: circuit vs analytic, data + figures.

Outputs
-------
results/data/isotropic_sweep.csv
results/data/isotropic_reference_points.csv
results/figures/{fidelity_in_out,eps_vs_epsprime,success_probability,
                 purification_gain,concurrence}.png
"""

from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import DATA_DIR, FIG_DIR  # noqa: E402
from pqec_distill.analytics import (  # noqa: E402
    F_of_eps, F_out_isotropic, eps_prime, isotropic_populations,
    p_success_isotropic,
)
from pqec_distill.bell_states import bell_diagonal_state, bell_populations  # noqa: E402
from pqec_distill.entanglement import concurrence, negativity  # noqa: E402
from pqec_distill.measurement import run_circuit_success  # noqa: E402

N_POINTS = 501
REFERENCE_EPS = [0.0, 0.01, 0.1, 0.4, 2.0 / 3.0, 0.7, 0.8, 0.9, 1.0]

FIELDS = [
    "eps", "F_in_circuit", "F_in_analytic",
    "P_success_circuit", "P_success_analytic",
    "F_out_circuit", "F_out_analytic",
    "eps_prime_circuit", "eps_prime_analytic",
    "gain_circuit", "purity_out", "C_in", "C_out", "N_in", "N_out",
    "abs_err_P_success", "abs_err_F_out", "abs_err_eps_prime",
]


def evaluate(eps: float) -> dict:
    """Everything measured from the CIRCUIT, plus the independent analytics."""
    rho_in = bell_diagonal_state(isotropic_populations(eps))
    rho_out, prob = run_circuit_success(rho_in)

    f_in = float(bell_populations(rho_in)[0])
    f_out = float(bell_populations(rho_out)[0])
    ep_circuit = (1.0 - f_out) * 4.0 / 3.0

    row = {
        "eps": eps,
        "F_in_circuit": f_in,
        "F_in_analytic": float(F_of_eps(eps)),
        "P_success_circuit": prob,
        "P_success_analytic": float(p_success_isotropic(eps)),
        "F_out_circuit": f_out,
        "F_out_analytic": float(F_out_isotropic(eps)),
        "eps_prime_circuit": ep_circuit,
        "eps_prime_analytic": float(eps_prime(eps)),
        "gain_circuit": f_out - f_in,
        "purity_out": float(np.real(np.trace(rho_out @ rho_out))),
        "C_in": concurrence(rho_in),
        "C_out": concurrence(rho_out),
        "N_in": negativity(rho_in),
        "N_out": negativity(rho_out),
    }
    row["abs_err_P_success"] = abs(row["P_success_circuit"] - row["P_success_analytic"])
    row["abs_err_F_out"] = abs(row["F_out_circuit"] - row["F_out_analytic"])
    row["abs_err_eps_prime"] = abs(row["eps_prime_circuit"] - row["eps_prime_analytic"])
    return row


def write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_figures(rows):
    eps = np.array([r["eps"] for r in rows])
    f_in = np.array([r["F_in_circuit"] for r in rows])
    f_out = np.array([r["F_out_circuit"] for r in rows])
    ep = np.array([r["eps_prime_circuit"] for r in rows])
    ps = np.array([r["P_success_circuit"] for r in rows])
    c_in = np.array([r["C_in"] for r in rows])
    c_out = np.array([r["C_out"] for r in rows])

    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.plot(f_in, f_out, lw=2, label="circuit")
    ax.plot(f_in, f_in, "k--", lw=1, label="no purification ($F_{out}=F_{in}$)")
    ax.axvline(0.5, color="grey", ls=":", lw=1)
    ax.set_xlabel(r"input Bell fidelity $F_{in}$")
    ax.set_ylabel(r"output Bell fidelity $F_{out}$")
    ax.set_title("Bell-isotropic input: input vs output fidelity")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fidelity_in_out.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.plot(eps, ep, lw=2, label=r"$\epsilon'$ (circuit)")
    ax.plot(eps, eps ** 2 / 4.0, ls="--", lw=1.4, label=r"$\epsilon^2/4$ (leading order)")
    ax.plot(eps, eps, "k:", lw=1, label=r"$\epsilon'=\epsilon$")
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$\epsilon'$")
    ax.set_title(r"$\epsilon' = \epsilon^2/(4-6\epsilon+3\epsilon^2)$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eps_vs_epsprime.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.plot(eps, ps, lw=2)
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$P_{success}$")
    ax.set_title(r"Success probability $P = \mathrm{Tr}(\rho^2) = (4-6\epsilon+3\epsilon^2)/4$")
    ax.set_ylim(0, 1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "success_probability.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.plot(eps, f_out - f_in, lw=2)
    ax.axhline(0.0, color="k", ls="--", lw=1)
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel(r"$F_{out} - F_{in}$")
    ax.set_title("Purification gain (one ideal round)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "purification_gain.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.6))
    ax.plot(eps, c_in, lw=2, label="input concurrence")
    ax.plot(eps, c_out, lw=2, label="output concurrence (postselected)")
    ax.axvline(2.0 / 3.0, color="grey", ls=":", lw=1.2)
    ax.annotate(r"input separable for $\epsilon \geq 2/3$", xy=(2.0 / 3.0, 0.55),
                xytext=(0.30, 0.72), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.set_xlabel(r"$\epsilon$")
    ax.set_ylabel("concurrence")
    ax.set_title("Concurrence before and after one round")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "concurrence.png", dpi=150)
    plt.close(fig)


def main() -> int:
    rows = [evaluate(float(e)) for e in np.linspace(0.0, 1.0, N_POINTS)]
    write_csv(DATA_DIR / "isotropic_sweep.csv", rows)

    ref = [evaluate(float(e)) for e in REFERENCE_EPS]
    write_csv(DATA_DIR / "isotropic_reference_points.csv", ref)

    make_figures(rows)

    max_err = max(
        max(r["abs_err_P_success"], r["abs_err_F_out"], r["abs_err_eps_prime"])
        for r in rows
    )
    print(f"swept {len(rows)} points on eps in [0,1]")
    print(f"max |circuit - analytic| over all sweep points: {max_err:.3e}\n")
    header = f"{'eps':>8} {'F_in':>10} {'P_succ':>12} {'F_out':>14} {'eps_prime':>14} {'C_out':>9}"
    print(header)
    print("-" * len(header))
    for r in ref:
        print(
            f"{r['eps']:>8.5f} {r['F_in_circuit']:>10.6f} {r['P_success_circuit']:>12.8f} "
            f"{r['F_out_circuit']:>14.10f} {r['eps_prime_circuit']:>14.12f} {r['C_out']:>9.6f}"
        )
    if max_err > 1e-12:
        print("\nWARNING: circuit and analytics disagree beyond tolerance")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
