"""Figures for the repeated-dynamics section of the report.

Every number is recomputed here from the repository's implementations --
the verified exact Bell-diagonal recurrences for Type 3 / Type 4 / 4Q and the
full two-qubit density-matrix map for Type 5.  Nothing is read off a stored
plot, and scripts/report_verify.py runs first so the three implementations
(report note, repository map, full circuit simulation) are checked against one
another before anything is drawn.

Each round's exact output state is the next round's input.  The Bell-isotropic
form is used for the initial state only; no round twirls or re-projects.

    python scripts/make_report_figures.py
    python scripts/make_report_figures.py --q 0.05 --eps-main 0.2
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

from _bootstrap import DATA_DIR, ROOT  # noqa: E402
from pqec_distill.analytic_exact_map import (  # noqa: E402
    analytic_map_4q, analytic_map_type3, analytic_map_type4, analytic_success_4q,
)
from pqec_distill.bell_population_dynamics import (  # noqa: E402
    bell_offdiagonal_c, bell_populations, isotropic_dm, one_round,
)
from pqec_distill.report_style import (  # noqa: E402
    BELL_COLOR, BELL_DASH, BELL_MARKER, BELL_ORDER, BELL_TEX, EPS_MARKER,
    EPS_RAMP, GRID, INK, INK_2, QUANTITY_COLOR, panel_label, save, style_axes,
    use_report_style,
)
from report_recurrence import isotropic_pops  # noqa: E402
import report_verify  # noqa: E402

FIG_DIR = ROOT / "results" / "report_figures"
CSV_DIR = DATA_DIR / "report_figures"

EXACT_MAP = {"Type3": analytic_map_type3, "Type4": analytic_map_type4,
             "4Q": analytic_map_4q}
LABEL = {"Type3": "Type 3  (16 CNOT)", "Type4": "Type 4  (14 CNOT)",
         "Type5": "Type 5  (14 CNOT, learned)", "4Q": "4Q  (5 CNOT, postselected)"}
SHORT = {"Type3": "Type 3", "Type4": "Type 4", "Type5": "Type 5", "4Q": "4Q"}
TAG = {"Type3": "type3", "Type4": "type4", "Type5": "type5", "4Q": "4q"}
EPS_LIST = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)


# ===========================================================================
# trajectories
# ===========================================================================

def run_bell_diagonal(protocol: str, eps: float, q: float, n_rounds: int) -> dict:
    """Iterate the verified exact Bell-diagonal recurrence."""
    step = EXACT_MAP[protocol]
    p = isotropic_pops(eps)
    pops, succ = [p.copy()], [np.nan]
    for _ in range(n_rounds):
        if protocol == "4Q":
            succ.append(float(analytic_success_4q(p, q)))
        else:
            succ.append(np.nan)
        p = step(p, q)
        pops.append(p.copy())
    pops = np.array(pops)
    return {"protocol": protocol, "eps": eps, "q": q,
            "n": np.arange(n_rounds + 1), "pops": pops,
            "F": pops[:, 0].copy(), "C_Bell": np.zeros(n_rounds + 1),
            "P_succ": np.array(succ)}


def run_full_state(protocol: str, eps: float, q: float, n_rounds: int) -> dict:
    """Iterate the full two-qubit density matrix (required for Type 5)."""
    rho = isotropic_dm(eps)
    pops, cb = [], []
    for n in range(n_rounds + 1):
        pops.append(bell_populations(rho))
        cb.append(bell_offdiagonal_c(rho))
        if n < n_rounds:
            rho, _ = one_round(rho, q, protocol)
    pops = np.array(pops)
    return {"protocol": protocol, "eps": eps, "q": q,
            "n": np.arange(n_rounds + 1), "pops": pops,
            "F": pops[:, 0].copy(), "C_Bell": np.array(cb),
            "P_succ": np.full(n_rounds + 1, np.nan), "final_state": rho}


def attractor(protocol: str, eps: float, q: float, n_long: int = 2000) -> dict:
    """Populations the recurrence settles on, recomputed (never hard-coded)."""
    step = EXACT_MAP[protocol]
    p = isotropic_pops(eps)
    for _ in range(n_long):
        p = step(p, q)
    out = {"populations": p.copy(), "F": float(p[0]), "n_long": n_long}
    if protocol == "4Q":
        out["P_succ"] = float(analytic_success_4q(p, q))
    return out


def write_csv(path, rows, fieldnames) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def traj_rows(tr, with_cbell=False, with_psucc=False) -> list:
    rows = []
    for i, n in enumerate(tr["n"]):
        row = {"round": int(n),
               "p_phi_plus": tr["pops"][i, 0], "p_phi_minus": tr["pops"][i, 1],
               "p_psi_plus": tr["pops"][i, 2], "p_psi_minus": tr["pops"][i, 3],
               "fidelity": tr["F"][i]}
        if with_cbell:
            row["C_Bell"] = tr["C_Bell"][i]
        if with_psucc:
            row["P_succ"] = "" if not np.isfinite(tr["P_succ"][i]) else tr["P_succ"][i]
        rows.append(row)
    return rows


# ===========================================================================
# shared drawing helpers
# ===========================================================================

def draw_bell_curves(ax, tr, keys, every, markers=True):
    for k in keys:
        i = BELL_ORDER.index(k)
        ax.plot(tr["n"], tr["pops"][:, i], color=BELL_COLOR[k],
                linestyle=BELL_DASH[k],
                linewidth=1.5 if k == "Psi-" else 1.8,
                marker=BELL_MARKER[k] if markers else None,
                markevery=(BELL_ORDER.index(k), every),
                markeredgecolor="white", label=BELL_TEX[k])


def bell_legend(fig, ncol=4, y=0.0):
    handles = [Line2D([0], [0], color=BELL_COLOR[k], linestyle=BELL_DASH[k],
                      marker=BELL_MARKER[k], markeredgecolor="white",
                      linewidth=1.8, label=BELL_TEX[k]) for k in BELL_ORDER]
    fig.legend(handles=handles, loc="lower center", ncol=ncol,
               bbox_to_anchor=(0.5, y), columnspacing=2.4,
               labelcolor=INK)


def condition_note(fig, eps, q, x=0.5, y=0.955, ha="center"):
    fig.text(x, y, rf"$\epsilon = {eps:g}$,  $q = {q:g}$", ha=ha,
             va="bottom", fontsize=10.5, color=INK_2)


# ===========================================================================
# Figure 1
# ===========================================================================

def figure1(eps, q, n_rounds):
    traj = {p: run_bell_diagonal(p, eps, q, n_rounds) for p in ("Type3", "Type4", "4Q")}
    traj["Type5"] = run_full_state("Type5", eps, q, n_rounds)
    order = ["Type3", "Type4", "Type5", "4Q"]

    for p in order:
        write_csv(CSV_DIR / f"data_fig01_{TAG[p]}.csv",
                  traj_rows(traj[p], with_cbell=(p == "Type5")),
                  ["round", "p_phi_plus", "p_phi_minus", "p_psi_plus",
                   "p_psi_minus", "fidelity"] + (["C_Bell"] if p == "Type5" else []))

    fig = plt.figure(figsize=(11.0, 9.0))
    outer = GridSpec(2, 2, figure=fig, hspace=0.44, wspace=0.26,
                     left=0.075, right=0.985, top=0.865, bottom=0.105)
    every = max(1, n_rounds // 12)
    tags = "(a)", "(b)", "(c)", "(d)"

    for k, protocol in enumerate(order):
        inner = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[k // 2, k % 2],
                                        hspace=0.16, height_ratios=[1.25, 1.0])
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)
        tr = traj[protocol]

        draw_bell_curves(ax_top, tr, BELL_ORDER, every)
        ax_top.set_ylim(0, 1)
        ax_top.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_top.set_ylabel("Bell population")
        ax_top.set_title(LABEL[protocol], color=INK, pad=6)
        plt.setp(ax_top.get_xticklabels(), visible=False)

        draw_bell_curves(ax_bot, tr, BELL_ORDER[1:], every)
        err = tr["pops"][:, 1:]
        lo, hi = float(err.min()), float(err.max())
        ax_bot.set_ylim(max(0.0, lo - 0.10 * (hi - lo)), hi + 0.34 * (hi - lo))
        ax_bot.set_ylabel("error populations")
        ax_bot.set_xlabel(r"purification round $n$")
        ax_bot.set_xlim(-0.8, n_rounds)
        # the three error components all start at the same value
        ax_bot.plot([0], [tr["pops"][0, 1]], marker="o", markersize=8,
                    markerfacecolor="none", markeredgecolor=INK_2,
                    markeredgewidth=1.1, linestyle="none", zorder=6,
                    clip_on=False)
        ax_bot.annotate(rf"$\Phi^-=\Psi^+=\Psi^-={tr['pops'][0, 1]:.4f}$"
                        "\nat $n=0$",
                        xy=(0.13, 0.80), xycoords="axes fraction",
                        fontsize=9.5, color=INK_2)
        for ax in (ax_top, ax_bot):
            style_axes(ax)
        panel_label(ax_top, tags[k], dx=-0.16, dy=1.13)

    bell_legend(fig, ncol=4, y=0.004)
    fig.suptitle("Bell-basis population dynamics under repeated noisy purification",
                 fontsize=13.5, color=INK, y=0.972)
    condition_note(fig, eps, q, y=0.928)
    return save(fig, FIG_DIR, "fig01_bell_population_dynamics"), traj


# ===========================================================================
# Figure 2
# ===========================================================================

def _attractor_panels(axes, protocol, runs, fstar, n_rounds, show_ylabel=True):
    ax_f, ax_d = axes
    for j, (eps, tr) in enumerate(runs):
        colour, marker = EPS_RAMP[j], EPS_MARKER[j]
        ax_f.plot(tr["n"], tr["F"], color=colour, marker=marker,
                  markevery=(j, max(1, n_rounds // 12)), markeredgecolor="white",
                  label=rf"$\epsilon = {eps:g}$")
        ax_d.plot(tr["n"], np.maximum(np.abs(tr["F"] - fstar), 1e-17),
                  color=colour, marker=marker,
                  markevery=(j, max(1, n_rounds // 12)), markeredgecolor="white")
    ax_f.set_xlim(0, n_rounds)
    ax_f.set_ylim(0.74, 1.0)
    ax_d.set_xlim(0, n_rounds)
    ax_d.set_yscale("log")
    ax_d.set_ylim(3e-18, 1.0)
    if show_ylabel:
        ax_f.set_ylabel(r"$F_n$")
        ax_d.set_ylabel(r"$|F_n - F_\star|$")
    ax_f.text(0.97, 0.10, rf"$F_\star = {fstar:.9f}$", transform=ax_f.transAxes,
              ha="right", fontsize=9.5, color=INK)
    ax_f.set_title(SHORT[protocol], color=INK, pad=5)
    for ax in (ax_f, ax_d):
        style_axes(ax)


def figure2(q, n_rounds, eps_list):
    protocols = ("Type3", "Type4", "4Q")
    data = {}
    for protocol in protocols:
        runs = [(e, run_bell_diagonal(protocol, e, q, n_rounds)) for e in eps_list]
        fstar = attractor(protocol, eps_list[-1], q)["F"]
        data[protocol] = {"runs": runs, "fstar": fstar}
        rows = []
        for eps, tr in runs:
            for i, n in enumerate(tr["n"]):
                rows.append({"epsilon": eps, "round": int(n),
                             "p_phi_plus": tr["pops"][i, 0],
                             "p_phi_minus": tr["pops"][i, 1],
                             "p_psi_plus": tr["pops"][i, 2],
                             "p_psi_minus": tr["pops"][i, 3],
                             "fidelity": tr["F"][i],
                             "abs_F_minus_F_star": abs(tr["F"][i] - fstar),
                             "F_star": fstar})
        write_csv(CSV_DIR / f"data_fig02_{TAG[protocol]}.csv", rows,
                  ["epsilon", "round", "p_phi_plus", "p_phi_minus", "p_psi_plus",
                   "p_psi_minus", "fidelity", "abs_F_minus_F_star", "F_star"])

    # ---- A: all three protocols --------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(10.4, 10.4), sharex=True,
                             gridspec_kw=dict(hspace=0.34, wspace=0.26))
    for r, protocol in enumerate(protocols):
        _attractor_panels(axes[r], protocol, data[protocol]["runs"],
                          data[protocol]["fstar"], n_rounds)
        axes[r][1].set_title("")
        panel_label(axes[r][0], f"({'abc'[r]})", dx=-0.155, dy=1.06)
    for ax in axes[-1]:
        ax.set_xlabel(r"purification round $n$")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.004), columnspacing=1.8, labelcolor=INK)
    fig.suptitle("Bell-isotropic inputs approaching a protocol-dependent attractor",
                 fontsize=13.5, color=INK, y=0.975)
    fig.text(0.985, 0.938, rf"$q = {q:g}$", ha="right", fontsize=10.5, color=INK_2)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.905, bottom=0.085)
    paths_all = save(fig, FIG_DIR, "fig02_common_attractor_all")

    # ---- B: Type 4 only ------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.6),
                             gridspec_kw=dict(wspace=0.26))
    _attractor_panels(axes, "Type4", data["Type4"]["runs"],
                      data["Type4"]["fstar"], n_rounds)
    axes[0].set_title("")
    axes[1].set_title("")
    for ax, tag in zip(axes, ("(a)", "(b)")):
        ax.set_xlabel(r"purification round $n$")
        panel_label(ax, tag, dx=-0.15, dy=1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.004), columnspacing=1.8, labelcolor=INK)
    fig.suptitle("Type 4: Bell-isotropic inputs approaching a common attractor",
                 fontsize=13, color=INK, y=0.985)
    fig.text(0.985, 0.915, rf"$q = {q:g}$", ha="right", fontsize=10.5, color=INK_2)
    fig.subplots_adjust(left=0.095, right=0.985, top=0.855, bottom=0.185)
    paths_t4 = save(fig, FIG_DIR, "fig02_common_attractor_type4")
    return paths_all + paths_t4, data


# ===========================================================================
# Figure 3
# ===========================================================================

def figure3(q, eps_list, n_grid=401, eps_max=0.36):
    """Input fidelity, one round, and the long-depth limit, against eps.

    The two break-even points are different questions and are computed
    separately: F_1(eps) = F_0(eps) asks whether ONE round helps, and
    F_0(eps) = F_star asks whether REPEATING helps.  At q = 0.02 they differ in
    the fifth decimal of eps, which the inset resolves.
    """
    from scipy.optimize import brentq

    protocol = "Type4"
    grid = np.linspace(0.0, eps_max, n_grid)

    def f0(e):
        return 1 - 3 * np.asarray(e) / 4

    def f1(e):
        return float(EXACT_MAP[protocol](isotropic_pops(float(e)), q)[0])

    fstar = attractor(protocol, 0.30, q)["F"]
    f50 = {e: run_bell_diagonal(protocol, e, q, 50)["F"][-1] for e in eps_list}
    max_f50_gap = max(abs(v - fstar) for v in f50.values())

    eps_be_asym = 4.0 * (1.0 - fstar) / 3.0                 # F_0 = F_star
    eps_be_one = brentq(lambda e: f1(e) - float(f0(e)),     # F_1 = F_0
                        1e-9, 0.5, xtol=1e-15, rtol=8.9e-16)

    curve_f1 = np.array([f1(e) for e in grid])
    write_csv(CSV_DIR / "data_fig03_type4.csv",
              [{"epsilon": e, "F_0": float(f0(e)), "F_1": f1(e), "F_star": fstar}
               for e in grid],
              ["epsilon", "F_0", "F_1", "F_star"])
    write_csv(CSV_DIR / "data_fig03_type4_tested_eps.csv",
              [{"epsilon": e, "F_0": float(f0(e)), "F_1": f1(e), "F_50": f50[e],
                "F_star": fstar, "F_50_minus_F_star": f50[e] - fstar}
               for e in eps_list],
              ["epsilon", "F_0", "F_1", "F_50", "F_star", "F_50_minus_F_star"])

    fig, (ax, axi) = plt.subplots(1, 2, figsize=(11.4, 5.0),
                                  gridspec_kw=dict(width_ratios=[1.6, 1],
                                                   wspace=0.24))
    style_axes(ax)
    style_axes(axi)

    ax.plot(grid, f0(grid), color=QUANTITY_COLOR[0], label=r"$F_0$  (input)")
    ax.plot(grid, curve_f1, color=QUANTITY_COLOR[1], label=r"$F_1$  (one round)")
    ax.axhline(fstar, color=QUANTITY_COLOR[2], linestyle=(0, (6, 3)),
               label=r"$F_\star$  (long-depth limit)")
    for e in eps_list:
        ax.plot([e], [f0(e)], marker="o", color=QUANTITY_COLOR[0],
                markeredgecolor="white", markersize=6, linestyle="none")
        ax.plot([e], [f1(e)], marker="s", color=QUANTITY_COLOR[1],
                markeredgecolor="white", markersize=6, linestyle="none")
    ax.axvspan(0.0, eps_be_asym, color="#f1f1f1", zorder=0)
    ax.text(eps_be_asym / 2, 0.893, "repeating\nlowers $F$", ha="center",
            fontsize=10, color=INK_2)
    ax.set_xlim(0, eps_max)
    ax.set_ylim(0.70, 1.005)
    ax.set_xlabel(r"input mixing $\epsilon$")
    ax.set_ylabel("target Bell fidelity")
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.04), labelcolor=INK)
    panel_label(ax, "(a)", dx=-0.115, dy=1.03)

    # ---- (b) the two break-even points are not the same point ----------
    lo, hi = eps_be_one - 2.4e-5, eps_be_asym + 2.4e-5
    gi = np.linspace(lo, hi, 241)
    axi.plot(gi, (f0(gi) - fstar) * 1e5, color=QUANTITY_COLOR[0])
    axi.plot(gi, (np.array([f1(e) for e in gi]) - fstar) * 1e5,
             color=QUANTITY_COLOR[1])
    axi.axhline(0.0, color=QUANTITY_COLOR[2], linestyle=(0, (6, 3)))
    axi.plot([eps_be_one], [(f0(eps_be_one) - fstar) * 1e5], marker="X",
             markersize=10, color=INK, markeredgecolor="white", linestyle="none",
             zorder=9)
    axi.plot([eps_be_asym], [0.0], marker="P", markersize=10, color=INK,
             markeredgecolor="white", linestyle="none", zorder=9)
    axi.annotate(r"$F_1 = F_0$" "\n" rf"$\epsilon = {eps_be_one:.8f}$",
                 xy=(eps_be_one, (f0(eps_be_one) - fstar) * 1e5),
                 xytext=(-8, 16), textcoords="offset points", ha="right",
                 fontsize=9, color=INK)
    axi.annotate(r"$F_0 = F_\star$" "\n" rf"$\epsilon = {eps_be_asym:.8f}$",
                 xy=(eps_be_asym, 0.0), xytext=(6, -30),
                 textcoords="offset points", ha="right", fontsize=9, color=INK)
    axi.set_xlim(lo, hi)
    axi.set_xlabel(r"input mixing $\epsilon$")
    axi.set_ylabel(r"$(F - F_\star)\times 10^{5}$")
    axi.ticklabel_format(axis="x", useOffset=False, style="plain")
    axi.set_xticks([eps_be_one, eps_be_asym])
    axi.set_xticklabels([f"{eps_be_one:.6f}", f"{eps_be_asym:.6f}"], fontsize=9)
    axi.set_title(r"one round and long depth break even at different $\epsilon$"
                  "\n" rf"(difference ${eps_be_asym - eps_be_one:.2e}$)",
                  fontsize=10, color=INK, pad=6)
    panel_label(axi, "(b)", dx=-0.19, dy=1.03)

    fig.suptitle("Type 4: input, one round, and the long-depth limit",
                 fontsize=13, color=INK, y=1.0)
    fig.text(0.995, 0.945, rf"$q = {q:g}$", ha="right", fontsize=10.5, color=INK_2)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.83, bottom=0.125)
    return (save(fig, FIG_DIR, "fig03_type4_input_one_round_asymptotic"),
            {"F_star": fstar,
             "eps_break_even_one_round_F1_eq_F0": eps_be_one,
             "eps_break_even_asymptotic_F0_eq_Fstar": eps_be_asym,
             "difference": eps_be_asym - eps_be_one,
             "max_abs_F50_minus_Fstar_over_tested_eps": max_f50_gap,
             "F_50": f50})


# ===========================================================================
# Figure 4
# ===========================================================================

def figure4(eps, q, n_long, zoom=50):
    tr = run_full_state("Type5", eps, q, n_long)
    write_csv(CSV_DIR / "data_fig04_type5.csv", traj_rows(tr, with_cbell=True),
              ["round", "p_phi_plus", "p_phi_minus", "p_psi_plus", "p_psi_minus",
               "fidelity", "C_Bell"])

    F, C, n = tr["F"], tr["C_Bell"], tr["n"]
    plateau_idx = 5
    plateau = float(F[plateau_idx])
    below = np.where((n > plateau_idx) & (F < plateau - 0.01))[0]
    n_escape = int(n[below[0]]) if below.size else None
    stats = {"F_0": float(F[0]), "plateau_F": plateau,
             "plateau_measured_at_n": plateau_idx,
             "n_escape": n_escape,
             "C_Bell_at_escape": float(C[below[0]]) if below.size else None,
             "F_final": float(F[-1]), "n_final": int(n[-1]),
             "C_Bell_final": float(C[-1]), "C_Bell_n1": float(C[1]),
             "F_max": float(F.max()), "n_F_max": int(n[int(F.argmax())])}

    fig = plt.figure(figsize=(10.6, 7.6))
    gs = GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.26,
                  left=0.085, right=0.985, top=0.885, bottom=0.085)
    ax_a = fig.add_subplot(gs[0, :])
    x = np.log10(n + 1)
    ax_a.plot(x, F, color=BELL_COLOR["Phi+"], linewidth=2.0)
    ax_a.axhline(F[0], color=INK_2, linestyle=":", linewidth=1.0)
    ax_a.annotate(rf"$F_0 = {F[0]:.4f}$", xy=(0, F[0]), xytext=(8, -16),
                  textcoords="offset points", fontsize=10, color=INK_2)
    ax_a.annotate(rf"metastable plateau,  $F \approx {plateau:.6f}$",
                  xy=(np.log10(41), plateau), xytext=(0, -26),
                  textcoords="offset points", ha="center", fontsize=10,
                  color=INK)
    if n_escape:
        ax_a.axvline(np.log10(n_escape + 1), color=BELL_COLOR["Phi-"],
                     linestyle=(0, (5, 3)), linewidth=1.3)
        ax_a.annotate(rf"escape, $n \approx {n_escape}$",
                      xy=(np.log10(n_escape + 1), 0.62), xytext=(7, 0),
                      textcoords="offset points", fontsize=10,
                      color=BELL_COLOR["Phi-"])
    ax_a.annotate(rf"$F_{{{int(n[-1])}}} = {F[-1]:.6f}$", xy=(x[-1], F[-1]),
                  xytext=(-8, 14), textcoords="offset points", ha="right",
                  fontsize=10, color=INK)
    ax_a.set_xlim(0, x[-1])
    ax_a.set_ylim(0.3, 1.0)
    ax_a.set_xlabel(r"$\log_{10}(n+1)$")
    ax_a.set_ylabel(r"target Bell fidelity $F_n$")
    style_axes(ax_a)
    panel_label(ax_a, "(a)", dx=-0.075, dy=1.04)

    ax_b = fig.add_subplot(gs[1, 0])
    m = (n >= 1) & (n <= zoom)
    ax_b.plot(n[m], C[m], color=BELL_COLOR["Phi-"], linewidth=2.0)
    ax_b.set_yscale("log")
    ax_b.set_xlim(1, zoom)
    ax_b.set_xlabel(r"purification round $n$")
    ax_b.set_ylabel(r"$C_{\mathrm{Bell}}(n)$")
    ax_b.set_title(rf"$1 \leq n \leq {zoom}$", color=INK, pad=5)
    ax_b.annotate(rf"$C_{{\mathrm{{Bell}}}}(1) = {C[1]:.2e}$",
                  xy=(0.06, 0.12), xycoords="axes fraction", fontsize=9.5,
                  color=INK)
    style_axes(ax_b)
    panel_label(ax_b, "(b)", dx=-0.19, dy=1.06)

    ax_c = fig.add_subplot(gs[1, 1])
    m = n >= 1
    ax_c.plot(n[m], C[m], color=BELL_COLOR["Phi-"], linewidth=2.0)
    ax_c.set_yscale("log")
    ax_c.set_xlim(1, n[-1])
    if n_escape:
        ax_c.axvline(n_escape, color=INK_2, linestyle=(0, (5, 3)), linewidth=1.2)
    ax_c.set_xlabel(r"purification round $n$")
    ax_c.set_ylabel(r"$C_{\mathrm{Bell}}(n)$")
    ax_c.set_title(rf"$1 \leq n \leq {int(n[-1])}$", color=INK, pad=5)
    style_axes(ax_c)
    panel_label(ax_c, "(c)", dx=-0.19, dy=1.06)

    fig.suptitle("Type 5: a metastable high-fidelity plateau and its escape",
                 fontsize=13.5, color=INK, y=0.975)
    condition_note(fig, eps, q, y=0.915)
    return save(fig, FIG_DIR, "fig04_type5_metastability"), stats


# ===========================================================================
# Figure 5
# ===========================================================================

def figure5(eps, q, n_long):
    raw = run_full_state("Type5", eps, q, n_long)
    cal = run_full_state("Type5cal", eps, q, n_long)
    d_f = float(np.max(np.abs(raw["F"] - cal["F"])))
    d_c = float(np.max(np.abs(raw["C_Bell"] - cal["C_Bell"])))

    rows = []
    for i, n in enumerate(raw["n"]):
        rows.append({"round": int(n), "F_raw": raw["F"][i], "F_calibrated": cal["F"][i],
                     "abs_diff_F": abs(raw["F"][i] - cal["F"][i]),
                     "C_Bell_raw": raw["C_Bell"][i],
                     "C_Bell_calibrated": cal["C_Bell"][i]})
    write_csv(CSV_DIR / "data_fig05_type5_raw_vs_calibrated.csv", rows,
              ["round", "F_raw", "F_calibrated", "abs_diff_F",
               "C_Bell_raw", "C_Bell_calibrated"])

    def escape(tr):
        plateau = float(tr["F"][5])
        below = np.where((tr["n"] > 5) & (tr["F"] < plateau - 0.01))[0]
        return int(tr["n"][below[0]]) if below.size else None

    stats = {"max_abs_F_raw_minus_calibrated": d_f,
             "max_abs_C_Bell_raw_minus_calibrated": d_c,
             "n_escape_raw": escape(raw), "n_escape_calibrated": escape(cal),
             "F_final_raw": float(raw["F"][-1]),
             "F_final_calibrated": float(cal["F"][-1])}

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4),
                             gridspec_kw=dict(wspace=0.26))
    x = np.log10(raw["n"] + 1)
    styles = (dict(color=QUANTITY_COLOR[0], linewidth=3.0, alpha=0.85),
              dict(color=QUANTITY_COLOR[1], linewidth=1.4, linestyle=(0, (5, 3))))
    for style, tr, name in zip(styles, (raw, cal),
                               ("raw learned circuit", "compilation-calibrated")):
        axes[0].plot(x, tr["F"], label=name, **style)
        axes[1].plot(x, np.maximum(tr["C_Bell"], 1e-34), label=name, **style)
    axes[0].set_ylim(0.3, 1.0)
    axes[0].set_ylabel(r"target Bell fidelity $F_n$")
    axes[0].annotate(rf"$\max_n |F_n^{{\mathrm{{raw}}}} - "
                     rf"F_n^{{\mathrm{{cal}}}}| = {d_f:.1e}$",
                     xy=(0.05, 0.12), xycoords="axes fraction", fontsize=10,
                     color=INK)
    axes[1].set_yscale("log")
    axes[1].set_ylabel(r"$C_{\mathrm{Bell}}(n)$")
    for ax, tag, loc in zip(axes, ("(a)", "(b)"),
                            ("center left", "lower right")):
        ax.set_xlim(0, x[-1])
        ax.set_xlabel(r"$\log_{10}(n+1)$")
        ax.legend(loc=loc, labelcolor=INK)
        style_axes(ax)
        panel_label(ax, tag, dx=-0.15, dy=1.05)
    if stats["n_escape_raw"] == stats["n_escape_calibrated"]:
        axes[0].annotate(rf"both escape at $n = {stats['n_escape_raw']}$",
                         xy=(0.05, 0.20), xycoords="axes fraction", fontsize=10,
                         color=INK)
    else:
        axes[0].annotate(rf"escape at $n = {stats['n_escape_raw']}$ (raw), "
                         rf"$n = {stats['n_escape_calibrated']}$ (calibrated)",
                         xy=(0.05, 0.20), xycoords="axes fraction", fontsize=10,
                         color=INK)
    fig.suptitle("Type 5: the escape is not an artefact of the compilation residual",
                 fontsize=13, color=INK, y=1.02)
    fig.text(0.995, 0.965, rf"$\epsilon = {eps:g}$,  $q = {q:g}$", ha="right",
             fontsize=10.5, color=INK_2)
    return save(fig, FIG_DIR, "fig05_type5_raw_vs_calibrated"), stats


# ===========================================================================
# Figure S1
# ===========================================================================

def figureS1(q, n_rounds, eps_list):
    runs = [(e, run_bell_diagonal("4Q", e, q, n_rounds)) for e in eps_list]
    pstar = attractor("4Q", eps_list[-1], q)["P_succ"]
    rows = []
    for eps, tr in runs:
        for i, n in enumerate(tr["n"]):
            if n == 0:
                continue
            rows.append({"epsilon": eps, "round": int(n),
                         "P_succ": tr["P_succ"][i], "fidelity": tr["F"][i]})
    write_csv(CSV_DIR / "data_figS1_4q_success.csv", rows,
              ["epsilon", "round", "P_succ", "fidelity"])

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6),
                             gridspec_kw=dict(width_ratios=[1.5, 1], wspace=0.24))
    zoom = 8
    for ax, xmax in zip(axes, (n_rounds, zoom)):
        style_axes(ax)
        for j, (eps, tr) in enumerate(runs):
            m = np.isfinite(tr["P_succ"])
            ax.plot(tr["n"][m], tr["P_succ"][m], color=EPS_RAMP[j],
                    marker=EPS_MARKER[j],
                    markevery=(j, max(1, xmax // 10)), markeredgecolor="white",
                    label=rf"$\epsilon = {eps:g}$")
        ax.axhline(pstar, color=INK_2, linestyle=(0, (6, 3)), linewidth=1.1)
        ax.set_xlim(1, xmax)
        ax.set_xlabel(r"purification round $n$")
    axes[0].set_ylabel(r"$P_{\mathrm{succ}}^{(n)}$")
    axes[1].set_ylabel(r"$P_{\mathrm{succ}}^{(n)}$")
    axes[0].annotate(rf"$P_{{\mathrm{{succ}}}}^\star = {pstar:.9f}$",
                     xy=(0.35, 0.10), xycoords="axes fraction", fontsize=10,
                     color=INK)
    axes[0].set_title(rf"$1 \leq n \leq {n_rounds}$", color=INK, pad=5)
    axes[1].set_title(rf"$1 \leq n \leq {zoom}$  (the transient)", color=INK, pad=5)
    axes[1].set_xticks(range(1, zoom + 1))
    for ax, tag in zip(axes, ("(a)", "(b)")):
        panel_label(ax, tag, dx=-0.14, dy=1.05)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.004), columnspacing=1.8, labelcolor=INK)
    fig.suptitle("4Q: per-round postselection success", fontsize=13, color=INK,
                 y=0.99)
    fig.text(0.985, 0.925, rf"$q = {q:g}$", ha="right", fontsize=10.5, color=INK_2)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.855, bottom=0.215)
    return save(fig, FIG_DIR, "figS1_4q_success_probability"), {"P_succ_star": pstar}


# ===========================================================================

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eps-main", type=float, default=0.15)
    ap.add_argument("--q", type=float, default=0.02)
    ap.add_argument("--rounds", type=int, default=50)
    ap.add_argument("--long-rounds", type=int, default=5000)
    ap.add_argument("--eps", type=float, nargs="+", default=list(EPS_LIST))
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()
    eps, q, N, NL = args.eps_main, args.q, args.rounds, args.long_rounds
    eps_list = tuple(sorted(args.eps))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    setup = report_verify.print_setup(eps, q, N, NL)
    checks = None if args.skip_verify else report_verify.verify(eps, q, N)

    use_report_style()
    print("\n" + "=" * 78 + "\nFIGURES\n" + "=" * 78)
    written = []
    p1, traj1 = figure1(eps, q, N)
    written += p1
    p2, data2 = figure2(q, N, eps_list)
    written += p2
    p3, stats3 = figure3(q, eps_list)
    written += p3
    p4, stats4 = figure4(eps, q, NL)
    written += p4
    p5, stats5 = figure5(eps, q, NL)
    written += p5
    pS1, statsS1 = figureS1(q, N, eps_list)
    written += pS1
    for path in written:
        print(f"  {path.relative_to(ROOT)}")

    # ---- recomputed numbers ------------------------------------------
    fixed = {}
    for protocol in ("Type3", "Type4", "4Q"):
        a = attractor(protocol, eps, q)
        spread = float(max(np.max(np.abs(attractor(protocol, e, q)["populations"]
                                         - a["populations"])) for e in eps_list))
        fixed[protocol] = {
            "populations": a["populations"].tolist(), "F": a["F"],
            "max_spread_over_tested_eps": spread,
            "break_even_eps": 4.0 * (1.0 - a["F"]) / 3.0,
        }
        if "P_succ" in a:
            fixed[protocol]["P_succ"] = a["P_succ"]

    invariants, sums = {}, {}
    for protocol in ("Type3", "Type4", "Type5", "4Q"):
        tr = traj1[protocol]
        sums[protocol] = float(np.max(np.abs(tr["pops"].sum(axis=1) - 1.0)))
        if protocol == "Type5":
            invariants[protocol] = None
            continue
        a, b = ("Phi-", "Psi-") if protocol == "4Q" else ("Psi+", "Psi-")
        i_a, i_b = BELL_ORDER.index(a), BELL_ORDER.index(b)
        invariants[protocol] = {
            "pair": f"p_{a} = p_{b}",
            "max_abs_diff": float(np.max(np.abs(tr["pops"][:, i_a]
                                                - tr["pops"][:, i_b])))}

    summary = {
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "setup": setup, "verification": checks,
        "attractors_recomputed": fixed,
        "invariant_pairs_on_the_reported_trajectory": invariants,
        "max_abs_population_sum_minus_1": sums,
        "type5_metastability": stats4,
        "type5_raw_vs_calibrated": stats5,
        "type4_input_vs_asymptotic": stats3,
        "4q_success": statsS1,
        "figures": [str(p.relative_to(ROOT)) for p in written],
    }
    (CSV_DIR / "report_figures_summary.json").write_text(json.dumps(summary, indent=2))

    print("\n" + "=" * 78 + "\nRECOMPUTED VALUES\n" + "=" * 78)
    print(f"  attractors at q = {q} (from the recurrence, not hard-coded)")
    for protocol, d in fixed.items():
        print(f"    {protocol:<7} F = {d['F']:.12f}   "
              + "(" + ", ".join(f"{x:.6f}" for x in d["populations"]) + ")")
        print(f"            spread over the tested eps {d['max_spread_over_tested_eps']:.2e}"
              f"   break-even eps {d['break_even_eps']:.12f}"
              + (f"   P_succ {d['P_succ']:.12f}" if "P_succ" in d else ""))
    print("  invariant pair on the reported trajectory")
    for protocol, d in invariants.items():
        print(f"    {protocol:<7} " + ("not Bell-diagonal, no invariant pair"
                                       if d is None
                                       else f"{d['pair']:<19} {d['max_abs_diff']:.2e}"))
    print("  max |sum_i p_i - 1| over the trajectory")
    for protocol, v in sums.items():
        print(f"    {protocol:<7} {v:.2e}")
    print(f"  Type 5: plateau F = {stats4['plateau_F']:.12f}, escape at n = "
          f"{stats4['n_escape']}, F_{stats4['n_final']} = {stats4['F_final']:.12f}, "
          f"C_Bell final {stats4['C_Bell_final']:.6f}")
    print(f"  Type 5 raw vs calibrated: escape {stats5['n_escape_raw']} vs "
          f"{stats5['n_escape_calibrated']}, max|dF| = "
          f"{stats5['max_abs_F_raw_minus_calibrated']:.3e}")
    print(f"  Type 4 break-even eps: one round (F_1 = F_0) "
          f"{stats3['eps_break_even_one_round_F1_eq_F0']:.12f}, "
          f"asymptotic (F_0 = F_star) "
          f"{stats3['eps_break_even_asymptotic_F0_eq_Fstar']:.12f}, "
          f"difference {stats3['difference']:.3e}")
    print(f"  max |F_50 - F_star| over the tested eps = "
          f"{stats3['max_abs_F50_minus_Fstar_over_tested_eps']:.2e}")
    print(f"\nwrote figures to {FIG_DIR.relative_to(ROOT)}/ and data to "
          f"{CSV_DIR.relative_to(ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
