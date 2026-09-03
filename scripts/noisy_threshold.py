"""One-round operational CNOT-noise threshold p*(eps) from the closed form.

Outputs
-------
results/data/noisy_threshold.csv
results/figures/noisy_threshold_vs_eps.png
results/figures/noisy_one_round_fidelity.png
"""

from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import DATA_DIR, FIG_DIR  # noqa: E402
from pqec_distill.analytics import F_of_eps  # noqa: E402
from pqec_distill.noisy_analytics import (  # noqa: E402
    one_round_fidelity, qbar_replace, threshold_p,
)

BLUE, ORANGE, GREEN, GREY = "#0072B2", "#D55E00", "#009E73", "#5A5A5A"
GRID_BREAK_EVEN = {0.05: 0.033697, 0.10: 0.061156, 0.20: 0.102816,
                   0.30: 0.132114, 0.50: 0.166546}


def slope_K(eb):
    return eb * (12 * eb ** 3 - 3 * eb ** 2 + 30 * eb + 25) / (4 * (3 * eb ** 2 + 1) ** 2)


def main() -> int:
    eps_grid = np.linspace(0.002, 0.998, 499)
    rows = []
    for e in eps_grid:
        rows.append({"eps": float(e), "F_in": float(F_of_eps(e)),
                     "p_star_replace": threshold_p(e), "p_star_pauli": threshold_p(e, "pauli"),
                     "weak_noise_slope_K": slope_K(1 - e)})
    with open(DATA_DIR / "noisy_threshold.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    ps = np.array([r["p_star_replace"] for r in rows])
    k = int(np.argmax(ps))
    print("one-round threshold p*(eps): F_out(eps, p*) = F_in(eps)")
    print(f"  {'eps':>7} {'p* replace':>11} {'p* pauli':>10} {'K(eb)':>8}")
    for e in (0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 2 / 3, 0.8, 0.9):
        print(f"  {e:>7.4f} {threshold_p(e):>11.6f} {threshold_p(e, 'pauli'):>10.6f} {slope_K(1 - e):>8.4f}")
    print(f"  maximum p* = {ps[k]:.6f} at eps = {eps_grid[k]:.3f}")

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(eps_grid, ps, color=BLUE, lw=2, label="replacement convention")
    ax.plot(eps_grid, [r["p_star_pauli"] for r in rows], color=ORANGE, lw=1.6, ls="--",
            label=r"Pauli convention ($=\frac{15}{16}\,p^*_{\rm replace}$)")
    ax.scatter(list(GRID_BREAK_EVEN), list(GRID_BREAK_EVEN.values()), s=28, color="black",
               zorder=5, label="earlier 41-point grid break-even")
    ax.axvline(2 / 3, color=GREY, ls=":", lw=1)
    ax.text(2 / 3 - 0.01, 0.193, "input separable for $\\epsilon \\geq 2/3$", fontsize=8,
            color=GREY, ha="right", va="top")
    ax.set_xlabel(r"input noise $\epsilon$")
    ax.set_ylabel(r"one-round threshold $p^*$")
    ax.set_title(r"Per-CNOT noise below which one round improves $F$")
    ax.set_xlim(0, 1); ax.set_ylim(0, 0.2)
    ax.legend(fontsize=8.5, loc="lower right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "noisy_threshold_vs_eps.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    pp = np.linspace(0, 0.25, 251)
    for e, c in zip((0.05, 0.1, 0.2, 0.3, 0.5), plt.cm.viridis(np.linspace(0.1, 0.85, 5))):
        ax.plot(pp, [one_round_fidelity(e, qbar_replace(p)) for p in pp], color=c, lw=1.8,
                label=rf"$\epsilon={e}$")
        ax.axhline(float(F_of_eps(e)), color=c, lw=0.8, ls=":")
        ax.plot([threshold_p(e)], [float(F_of_eps(e))], "o", color=c, ms=4)
    ax.set_xlabel(r"per-CNOT noise $p$ (replacement)")
    ax.set_ylabel(r"$F_{\rm out}$ after one round  (dotted: $F_{\rm in}$)")
    ax.set_title("One-round output fidelity; dots mark $p^*$")
    ax.legend(fontsize=8.5)
    fig.tight_layout(); fig.savefig(FIG_DIR / "noisy_one_round_fidelity.png", dpi=150); plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
