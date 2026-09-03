"""Repeated noisy purification: fixed points vs p, trajectories, stability.

Outputs
-------
results/data/repeated_fixed_points.csv
results/data/repeated_trajectories.csv
results/figures/repeated_Fstar_vs_p.png
results/figures/repeated_trajectories.png
results/figures/repeated_stability.png
results/figures/repeated_offbell_decay.png
"""

from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import DATA_DIR, FIG_DIR  # noqa: E402
from pqec_distill.analytics import isotropic_populations  # noqa: E402
from pqec_distill.bell_states import bell_diagonal_state  # noqa: E402
from pqec_distill.entanglement import concurrence  # noqa: E402
from pqec_distill.gates import I2, PAULI_Z, kron_list  # noqa: E402
from pqec_distill.noisy_analytics import (  # noqa: E402
    entanglement_limit_p, fidelity_uv, fixed_point_uv, jacobian_uv,
    saddle_node_p,
)
from pqec_distill.repeated_noisy import (  # noqa: E402
    PAULI_LABELS_2Q, effective_map, fixed_point_dense, full_jacobian, iterate,
    off_bell_projection_norm, to_pauli_coords,
)

BLUE, ORANGE, GREEN, GREY, PURPLE = "#0072B2", "#D55E00", "#009E73", "#5A5A5A", "#CC79A7"
IXX, IYY, IZZ = (PAULI_LABELS_2Q.index(k) for k in ("XX", "YY", "ZZ"))
CONV = "replace"


def fid(rho):
    r = to_pauli_coords(rho)
    return (1 + r[IXX] - r[IYY] + r[IZZ]) / 4


def main() -> int:
    p_sn, p_ent = saddle_node_p(), entanglement_limit_p()
    print(f"saddle node  p_SN  = {p_sn:.12f}")
    print(f"entanglement p_ent = {p_ent:.12f}\n")

    # ---- fixed points vs p (closed form + dense) --------------------------
    p_dense = [0.001, 0.003, 0.01, 0.02, 0.03, 0.05, 0.07, 0.1, 0.12, 0.15, 0.17]
    rows = []
    print(f"{'p':>6} {'u*':>9} {'v*':>9} {'F*':>9} {'C*':>8} {'P_succ*':>8} {'rho(J_2D)':>9} {'rho(J_15D)':>10} {'max offBell |lam|':>17} {'dense-closed':>12}")
    for p in p_dense:
        qb = 1 - p
        u, v = fixed_point_uv(qb)[0]
        rho, n, res = fixed_point_dense(bell_diagonal_state(isotropic_populations(0.1)), p, CONV)
        lam2 = np.abs(np.linalg.eigvals(jacobian_uv(u, v, qb))).max()
        lam15 = np.sort(np.abs(np.linalg.eigvals(full_jacobian(rho, p, CONV))))[::-1]
        r = to_pauli_coords(rho)
        row = {"p": p, "u_star": u, "v_star": v, "F_star": fidelity_uv(u, v),
               "C_star": max(0.0, 2 * fidelity_uv(u, v) - 1),
               "P_success_star": (1 + qb ** 5 * u * u + (qb ** 5 + qb ** 3) * v * v) / 4,
               "spectral_radius_2D": lam2, "spectral_radius_15D": lam15[0],
               "max_offbell_eig": lam15[3], "dense_minus_closed_F": abs(fid(rho) - fidelity_uv(u, v)),
               "dense_n_iter": n}
        rows.append(row)
        print(f"{p:>6.3f} {u:>9.6f} {v:>9.6f} {row['F_star']:>9.6f} {row['C_star']:>8.5f} {row['P_success_star']:>8.5f} "
              f"{lam2:>9.6f} {lam15[0]:>10.6f} {lam15[3]:>17.1e} {row['dense_minus_closed_F']:>12.1e}")
    with open(DATA_DIR / "repeated_fixed_points.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- F* vs p figure (dense closed-form curve + unstable branch) -------
    pp = np.linspace(0.0, p_sn, 400)
    hi, lo = [], []
    for p in pp:
        fps = fixed_point_uv(1 - p)
        hi.append(fidelity_uv(*fps[0]) if fps else np.nan)
        lo.append(fidelity_uv(*fps[1]) if len(fps) > 1 else np.nan)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.plot(pp, hi, color=BLUE, lw=2, label=r"$F^*(p)$, attracting branch")
    ax.plot(pp, lo, color=BLUE, lw=1.2, ls="--", label="unstable branch")
    ax.plot(pp, 1 - pp - 23 / 8 * pp ** 2, color=GREY, lw=1, ls=":", label=r"$1 - p - \frac{23}{8}p^2$")
    ax.scatter([r["p"] for r in rows], [r["F_star"] for r in rows], s=22, color="black", zorder=5,
               label="dense 32x32 iteration")
    ax.axhline(0.5, color=ORANGE, lw=0.8, ls=":")
    ax.axvline(p_ent, color=ORANGE, lw=0.9, ls="--"); ax.text(p_ent - 0.004, 0.62, r"$p_{\rm ent}$", color=ORANGE, ha="right", fontsize=9)
    ax.axvline(p_sn, color=GREEN, lw=0.9, ls="--"); ax.text(p_sn + 0.003, 0.62, r"$p_{\rm SN}$", color=GREEN, fontsize=9)
    ax.set_xlabel(r"per-CNOT noise $p$ (replacement)"); ax.set_ylabel(r"fixed-point Bell fidelity $F^*$")
    ax.set_title("Fixed point of repeated noisy purification"); ax.set_xlim(0, 0.2); ax.set_ylim(0.2, 1.02)
    ax.legend(fontsize=8.5, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG_DIR / "repeated_Fstar_vs_p.png", dpi=150); plt.close(fig)

    # ---- trajectories ----------------------------------------------------
    traj_rows = []
    p_list = [0.01, 0.05, 0.1, 0.15, 0.19]
    eps0_list = [0.1, 0.5, 0.8]
    fig, axes = plt.subplots(1, len(p_list), figsize=(13.5, 3.4), sharey=True)
    for ax, p in zip(axes, p_list):
        for e0, c in zip(eps0_list, (BLUE, GREEN, PURPLE)):
            states = iterate(bell_diagonal_state(isotropic_populations(e0)), p, CONV, 40)
            fs = [fid(s) for s in states]
            for n, f in enumerate(fs):
                traj_rows.append({"p": p, "eps0": e0, "n": n, "F": f, "C": concurrence(states[n])})
            ax.plot(range(len(fs)), fs, marker="o", ms=2.5, lw=1.3, color=c, label=rf"$\epsilon_0={e0}$")
        fps = fixed_point_uv(1 - p)
        if fps:
            ax.axhline(fidelity_uv(*fps[0]), color="black", lw=0.8, ls=":")
        ax.axhline(0.5, color=ORANGE, lw=0.7, ls=":")
        ax.set_title(rf"$p={p}$" + ("" if p < p_sn else r"  ($>p_{\rm SN}$)"), fontsize=10)
        ax.set_xlabel("round $n$")
    axes[0].set_ylabel(r"$F_n$"); axes[0].legend(fontsize=8)
    fig.suptitle("Bell fidelity under repeated noisy purification (dotted: closed-form $F^*$)", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG_DIR / "repeated_trajectories.png", dpi=150); plt.close(fig)
    with open(DATA_DIR / "repeated_trajectories.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(traj_rows[0])); w.writeheader(); w.writerows(traj_rows)

    # ---- stability figure ------------------------------------------------
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.semilogy([r["p"] for r in rows], [r["spectral_radius_2D"] for r in rows], "o-", color=BLUE, lw=1.6, ms=4, label="Bell-sector (2D) spectral radius")
    ax.semilogy([r["p"] for r in rows], [r["spectral_radius_15D"] for r in rows], "s--", color=ORANGE, lw=1.2, ms=5, mfc="none", label="full-state (15D) spectral radius")
    ax.semilogy([r["p"] for r in rows], [max(r["max_offbell_eig"], 1e-20) for r in rows], "^-", color=GREEN, lw=1.2, ms=4, label="largest off-Bell eigenvalue (12 of them)")
    ax.axhline(1.0, color="black", lw=0.8, ls=":")
    ax.set_xlabel(r"per-CNOT noise $p$"); ax.set_ylabel(r"$|\lambda|$ at the fixed point")
    ax.set_title("Fixed point is a full-state attractor: 15D radius = 2D radius < 1")
    ax.set_ylim(1e-19, 2); ax.legend(fontsize=8.5, loc="center right")
    fig.tight_layout(); fig.savefig(FIG_DIR / "repeated_stability.png", dpi=150); plt.close(fig)

    # ---- off-Bell seed decay ---------------------------------------------
    seed = (kron_list([PAULI_Z, I2]) + kron_list([I2, PAULI_Z])) / np.sqrt(2)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    print("\nseeded off-Bell perturbation (ZI+IZ)/sqrt2, norm per round:")
    for p, c in zip((0.01, 0.05, 0.1), (BLUE, GREEN, ORANGE)):
        rho_star = fixed_point_dense(bell_diagonal_state(isotropic_populations(0.1)), p, CONV)[0]
        for eta, ls in ((1e-2, "-"), (1e-4, "--")):
            rho = rho_star + eta / 4 * seed
            ds = []
            for _ in range(5):
                ds.append(max(off_bell_projection_norm(rho), 1e-18))
                rho, _ = effective_map(rho, p, CONV)
            ax.semilogy(range(5), ds, marker="o", ms=3.5, ls=ls, color=c, label=rf"$p={p}$, $\eta=10^{{{int(np.log10(eta))}}}$")
            print(f"  p={p} eta={eta:.0e}: " + " -> ".join(f"{d:.1e}" for d in ds))
    ax.set_xlabel("round"); ax.set_ylabel("off-Bell Pauli norm")
    ax.set_title(r"Off-Bell seed at $\rho^*$ decays quadratically ($\delta \to \delta^2$)")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG_DIR / "repeated_offbell_decay.png", dpi=150); plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
