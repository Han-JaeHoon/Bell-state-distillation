"""Compare the repeated-round fixed point with the parent project's 5-qubit
SWAP-test gadgets (Steps 3/4/5) at the SAME per-CNOT replacement noise.

Steps 3 and 4 are evaluated from the exact (u, v) recursions of the parent
project's notes 01/02 (re-implemented here and checked against the frozen
CSV); Step 5, which has no fixed point on the Phi+ branch, is taken from the
frozen trajectory CSV in results/data/external/ (see PROVENANCE.md there).
The 4-qubit curve comes from noisy_analytics.

Outputs
-------
results/data/compare_fixed_points.csv
results/figures/compare_fidelity_vs_round.png
results/figures/compare_Fstar_vs_q.png
"""

from __future__ import annotations

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _bootstrap import DATA_DIR, FIG_DIR  # noqa: E402
from pqec_distill.noisy_analytics import (  # noqa: E402
    entanglement_limit_p, fidelity_uv, fixed_point_branch, noisy_map_uv,
    saddle_node_p,
)
from pqec_distill.swap_test_reference import (  # noqa: E402
    Q_ENT_STEP3, Q_ENT_STEP4, Q_SN_STEP3, Q_SN_STEP4, STEP5_PLATEAU,
    fidelity_34, fixed_point_34, step3_map, step4_map,
)

EXT = DATA_DIR / "external" / "pqec_operational_threshold__fidelity_vs_round.csv"
C3, C4, C5, CN = "#0072B2", "#D55E00", "#009E73", "#CC79A7"
EB, Q, N = 0.9, 0.01, 5000


def read_parent_series():
    out = {}
    with open(EXT) as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["series"], {})[int(r["n"])] = float(r["F"])
    return out


def trajectory_34(step, qb):
    u = v = EB
    fs = [fidelity_34(u, v)]
    for _ in range(N):
        u, v = step(u, v, qb)
        fs.append(fidelity_34(u, v))
    return np.array(fs)


def trajectory_4q(qb):
    u = v = EB
    fs = [fidelity_uv(u, v)]
    for _ in range(N):
        u, v = noisy_map_uv(u, v, qb)
        fs.append(fidelity_uv(u, v))
    return np.array(fs)


def main() -> int:
    parent = read_parent_series()
    qb = 1 - Q
    f3, f4, fn = trajectory_34(step3_map, qb), trajectory_34(step4_map, qb), trajectory_4q(qb)
    n5 = np.array(sorted(parent["step5"]))
    f5 = np.array([parent["step5"][k] for k in n5])

    d3 = max(abs(parent["step3"][k] - f3[k]) for k in parent["step3"])
    d4 = max(abs(parent["step4"][k] - f4[k]) for k in parent["step4"])
    print(f"re-implemented Step 3/4 recursions vs frozen parent CSV: max|dF| = {d3:.1e} / {d4:.1e}")

    # ---- figure 1: same condition as the parent's Figure 1 ------------------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, early in ((ax[0], True), (ax[1], False)):
        xs = range(11) if early else range(N + 1)
        sl = slice(0, 11) if early else slice(None)
        a.plot(xs, f3[sl], color=C3, lw=1.6, label="Step 3 — textbook 16-CNOT")
        a.plot(xs, f4[sl], color=C4, lw=1.6, label="Step 4 — resynthesized 14-CNOT")
        m = n5 <= 10 if early else np.ones_like(n5, bool)
        a.plot(n5[m], f5[m], color=C5, lw=1.6, label="Step 5 — learned 14-CNOT (calibrated)")
        a.plot(xs, fn[sl], color=CN, lw=2.2, ls="--", label="4-qubit 5-CNOT (postselected)")
    ax[0].set(xlabel="PQEC round $n$", ylabel=r"Bell fidelity $F_n$", title="(a) early rounds", ylim=(0.92, 1.0))
    for f, c in ((f3[10], C3), (f4[10], C4), (f5[n5 == 10][0], C5), (fn[10], CN)):
        ax[0].text(10.2, f, f"{f:.4f}", color=c, fontsize=8, va="center")
    ax[1].set(xscale="log", xlim=(1, N), ylim=(0.35, 1.02), xlabel="PQEC round $n$", title="(b) long-time behaviour")
    ax[1].axhline(0.5, color="grey", lw=0.7, ls=":")
    ax[1].text(1.2, 0.51, "separable below", fontsize=7.5, color="grey")
    ax[1].annotate("Step 5 leaves its\nmetastable plateau", xy=(900, 0.6), xytext=(60, 0.62), fontsize=8,
                   color=C5, arrowprops=dict(arrowstyle="->", color=C5, lw=0.8))
    ax[0].legend(fontsize=7.5, loc="lower right")
    fig.suptitle(r"Same input $\rho(\bar\epsilon,\bar\epsilon)$, $\bar\epsilon=0.9$;  same per-CNOT replacement depolarizing $q=0.01$", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "compare_fidelity_vs_round.png", dpi=150)
    plt.close(fig)

    # ---- figure 2 + CSV: F*(q) ----------------------------------------------
    p_sn, p_ent = saddle_node_p(), entanglement_limit_p()
    qq = np.linspace(0.0, 0.2, 801)
    rows = []
    for x in qq:
        fp = fixed_point_branch(1 - x)
        u3, v3 = fixed_point_34(x, 3)
        u4, v4 = fixed_point_34(x, 4)
        row = {"q": float(x),
               "F_star_step3": fidelity_34(u3, v3) if not np.isnan(u3) else np.nan,
               "F_star_step4": fidelity_34(u4, v4) if not np.isnan(u4) else np.nan,
               "F_star_4qubit": fidelity_uv(*fp) if fp else np.nan,
               "P_success_star_4qubit": ((1 + (1 - x) ** 5 * fp[0] ** 2 + ((1 - x) ** 5 + (1 - x) ** 3) * fp[1] ** 2) / 4) if fp else np.nan}
        rows.append(row)
    with open(DATA_DIR / "compare_fixed_points.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.plot(qq, [r["F_star_step3"] for r in rows], color=C3, lw=1.8, label=r"Step 3 (16 CNOT): $1-F^*=\frac{17}{8}q+\dots$, saddle in full space")
    ax.plot(qq, [r["F_star_step4"] for r in rows], color=C4, lw=1.8, label=r"Step 4 (14 CNOT): $1-F^*=\frac{7}{4}q+\dots$, saddle in full space")
    ax.plot(qq, [r["F_star_4qubit"] for r in rows], color=CN, lw=2.4, ls="--", label=r"4-qubit (5 CNOT): $1-F^*=q+\dots$, full-state attractor")
    ax.scatter(list(STEP5_PLATEAU), list(STEP5_PLATEAU.values()), marker="x", s=45, color=C5, zorder=5,
               label="Step 5 (14 CNOT): metastable plateau only (escapes)")
    for x, c in ((Q_SN_STEP3, C3), (Q_SN_STEP4, C4), (p_sn, CN)):
        ax.axvline(x, color=c, lw=0.8, ls=":"); ax.text(x, 0.27, "SN", color=c, fontsize=7.5, ha="center")
    ax.axhline(0.5, color="grey", lw=0.7, ls=":")
    ax.set(xlabel="per-CNOT noise $q$ (replacement depolarizing)", ylabel=r"fixed-point Bell fidelity $F^*$",
           title="Repeated-round fixed point: 5-qubit SWAP-test gadgets vs 4-qubit circuit", ylim=(0.25, 1.01), xlim=(0, 0.2))
    ax.title.set_fontsize(10); ax.legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "compare_Fstar_vs_q.png", dpi=150)
    plt.close(fig)

    # ---- table ----------------------------------------------------------------
    print(f"\n{'q':>6} | {'S3 F*':>8} {'S4 F*':>8} {'S5 plateau':>10} | {'4q F*':>8} {'4q P_succ*':>10}")
    for x in (0.001, 0.01, 0.03, 0.05, 0.1, 0.12, 0.15, 0.17):
        fp = fixed_point_branch(1 - x); u, v = fp; qb_ = 1 - x
        ps = (1 + qb_ ** 5 * u * u + (qb_ ** 5 + qb_ ** 3) * v * v) / 4
        s5 = f"{STEP5_PLATEAU[x]:.6f}" if x in STEP5_PLATEAU else "—"
        print(f"{x:>6.3f} | {fidelity_34(*fixed_point_34(x, 3)):>8.6f} {fidelity_34(*fixed_point_34(x, 4)):>8.6f} {s5:>10} | {fidelity_uv(u, v):>8.6f} {ps:>10.4f}")
    print(f"\nq_SN : S3 {Q_SN_STEP3:.6f}  S4 {Q_SN_STEP4:.6f}  4q {p_sn:.6f}")
    print(f"q_ent: S3 {Q_ENT_STEP3:.6f}  S4 {Q_ENT_STEP4:.6f}  4q {p_ent:.6f}")
    print(f"F_5000 (eb=0.9, q=0.01): S3 {f3[-1]:.6f}  S4 {f4[-1]:.6f}  S5 {f5[-1]:.6f}  4q {fn[-1]:.6f}")
    return 0 if max(d3, d4) < 1e-12 else 1


if __name__ == "__main__":
    raise SystemExit(main())
