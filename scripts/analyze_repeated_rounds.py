"""Repeated ideal purification: circuit iteration vs the analytic recurrence.

Outputs
-------
results/data/repeated_rounds.csv
results/figures/repeated_rounds.png
"""

from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import DATA_DIR, FIG_DIR  # noqa: E402
from pqec_distill.analytics import (  # noqa: E402
    isotropic_populations, p_total_repeated, repeated_populations,
    repeated_populations_closed_form,
)
from pqec_distill.bell_states import bell_diagonal_state, bell_populations  # noqa: E402
from pqec_distill.measurement import run_circuit_success  # noqa: E402

EPS_VALUES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]
MAX_LEVELS = 6

FIELDS = [
    "eps", "level", "F_circuit", "F_recurrence", "F_closed_form",
    "P_level", "P_total_tree", "Tr_rho_pow", "err_F", "err_P_total",
    "state_vs_matrix_power_fro",
]


def main() -> int:
    rows = []
    worst_f = 0.0
    worst_p = 0.0
    worst_state = 0.0

    for eps in EPS_VALUES:
        p0 = isotropic_populations(eps)
        rho0 = bell_diagonal_state(p0)
        rho = rho0.copy()
        per_level = []

        for level in range(1, MAX_LEVELS + 1):
            rho, prob = run_circuit_success(rho)
            per_level.append(prob)

            f_circuit = float(bell_populations(rho)[0])
            f_recurrence = float(repeated_populations(p0, level)[0])
            f_closed = float(repeated_populations_closed_form(p0, level)[0])

            tree_total = 1.0
            for k, pk in enumerate(per_level, start=1):
                tree_total *= pk ** (2 ** (level - k))

            power = np.linalg.matrix_power(rho0, 2 ** level)
            tr_power = float(np.real(np.trace(power)))
            state_err = float(np.linalg.norm(rho - power / tr_power, "fro"))

            row = {
                "eps": eps, "level": level,
                "F_circuit": f_circuit,
                "F_recurrence": f_recurrence,
                "F_closed_form": f_closed,
                "P_level": prob,
                "P_total_tree": tree_total,
                "Tr_rho_pow": tr_power,
                "err_F": abs(f_circuit - f_recurrence),
                "err_P_total": abs(tree_total - tr_power),
                "state_vs_matrix_power_fro": state_err,
            }
            rows.append(row)
            worst_f = max(worst_f, row["err_F"])
            worst_p = max(worst_p, row["err_P_total"])
            worst_state = max(worst_state, state_err)

    with open(DATA_DIR / "repeated_rounds.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for eps in EPS_VALUES:
        sel = [r for r in rows if r["eps"] == eps]
        levels = [0] + [r["level"] for r in sel]
        fid = [float(isotropic_populations(eps)[0])] + [r["F_circuit"] for r in sel]
        axes[0].plot(levels, fid, marker="o", ms=3.5, lw=1.5, label=rf"$\epsilon={eps}$")
        axes[1].semilogy(
            [r["level"] for r in sel], [max(r["P_total_tree"], 1e-300) for r in sel],
            marker="o", ms=3.5, lw=1.5, label=rf"$\epsilon={eps}$",
        )
    axes[0].axhline(1.0, color="k", ls=":", lw=1)
    axes[0].set_xlabel("purification level $\\ell$")
    axes[0].set_ylabel(r"$F_\ell$")
    axes[0].set_title("Bell fidelity under repeated IDEAL rounds")
    axes[0].legend(fontsize=8)
    axes[1].set_xlabel("purification level $\\ell$")
    axes[1].set_ylabel(r"$P_{total}(\ell)=\mathrm{Tr}[\rho^{2^\ell}]$")
    axes[1].set_title("Full-tree success probability ($2^\\ell$ input copies)")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "repeated_rounds.png", dpi=150)
    plt.close(fig)

    print(f"levels 1..{MAX_LEVELS}, eps in {EPS_VALUES}")
    print(f"max |F_circuit - F_recurrence|            : {worst_f:.3e}")
    print(f"max |P_total(tree) - Tr[rho^(2^l)]|       : {worst_p:.3e}")
    print(f"max ||rho_l - rho^(2^l)/Tr||_F            : {worst_state:.3e}\n")
    print(f"{'eps':>6} {'l':>3} {'F_circuit':>14} {'P_level':>12} {'P_total_tree':>14}")
    print("-" * 54)
    for r in rows:
        if r["level"] <= 4:
            print(f"{r['eps']:>6.2f} {r['level']:>3d} {r['F_circuit']:>14.10f} "
                  f"{r['P_level']:>12.8f} {r['P_total_tree']:>14.6e}")
    ok = max(worst_f, worst_p, worst_state) < 1e-10
    print(f"\nall consistent: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
