"""Figures for the Bell-population dynamics.

Reusable over (eps, q): every function takes a :class:`Dynamics` (or a list of
them) and returns the matplotlib figure, so a later sweep just calls them again.

Colour assignment is fixed by Bell state, never by position in a filter or by
rank, and is the validated 4-slot categorical order (blue, orange, aqua,
yellow).  Three of those slots sit below 3:1 contrast on a light surface, so the
relief rule applies and every figure carries direct labels in addition to its
legend; the CSV written next to each figure is the table view.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from .bell_population_dynamics import BELL_NAMES, Dynamics, PROTOCOL_INFO

__all__ = [
    "BELL_COLORS", "BELL_MARKERS", "STACK_ORDER", "SURFACE",
    "plot_stacked_area", "plot_population_lines", "plot_offdiagonal",
    "plot_long_time_fidelity", "plot_fidelity_overview", "plot_pair_comparison",
]

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#dedcd6"

#: categorical slots 1-4, assigned to Bell states and never reassigned
BELL_COLORS = {"Phi+": "#2a78d6", "Phi-": "#eb6834",
               "Psi+": "#1baf7a", "Psi-": "#eda100"}
#: secondary encoding, so identity never rests on colour alone
BELL_MARKERS = {"Phi+": "o", "Phi-": "s", "Psi+": "^", "Psi-": "D"}
#: bottom to top; Phi+ first so the bottom band's height IS the target fidelity
STACK_ORDER = ["Phi+", "Phi-", "Psi+", "Psi-"]

_PRETTY = {"Phi+": r"$\Phi^+$", "Phi-": r"$\Phi^-$",
           "Psi+": r"$\Psi^+$", "Psi-": r"$\Psi^-$"}


def _spread_end_labels(ax, items, min_gap_frac: float = 0.052, dx: int = 7,
                       inside: bool = False) -> None:
    """Direct labels at the right edge, pushed apart so they never overlap.

    ``items`` is a sequence of (y_data, text, colour).  Positions are resolved in
    axes fractions, so the spreading works whatever the y range is.
    """
    lo, hi = ax.get_ylim()
    span = hi - lo
    if span <= 0:
        return
    rows = sorted(((float(y) - lo) / span, txt, col) for y, txt, col in items)
    placed = []
    for frac, txt, col in rows:
        if placed and frac - placed[-1][0] < min_gap_frac:
            frac = placed[-1][0] + min_gap_frac
        placed.append((frac, txt, col))
    # if the stack overflowed the top, slide the whole group down
    over = placed[-1][0] - 1.0
    if over > 0:
        placed = [(f - over, t, c) for f, t, c in placed]
    for frac, txt, col in placed:
        if inside:
            ax.annotate(txt, xy=(1.0, min(max(frac, 0.0), 1.0)),
                        xycoords=ax.transAxes, textcoords="offset points",
                        xytext=(-dx, 0), va="center", ha="right",
                        fontsize=10, color=col, annotation_clip=False,
                        bbox=dict(facecolor=SURFACE, edgecolor="none",
                                  alpha=0.8, pad=1.4))
        else:
            ax.annotate(txt, xy=(1.0, min(max(frac, 0.0), 1.0)),
                        xycoords=ax.transAxes, textcoords="offset points",
                        xytext=(dx, 0), va="center", ha="left",
                        fontsize=10, color=col, annotation_clip=False)


def _subtitle(dyn: Dynamics) -> str:
    return (rf"$\epsilon = {dyn.eps:g}$,  $q = {dyn.q:g}$   "
            f"({PROTOCOL_INFO[dyn.protocol]['n_cnot']} CNOT, backend: {dyn.backend})")


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.7, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9)


def _fig(nrows=1, ncols=1, figsize=(7.2, 4.6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    fig.patch.set_facecolor(SURFACE)
    for ax in np.atleast_1d(axes).ravel():
        _style(ax)
    return fig, axes


# ---------------------------------------------------------------------------
# Figure A -- stacked area
# ---------------------------------------------------------------------------

def plot_stacked_area(dyn: Dynamics, order: Sequence[str] = None,
                      zoom: bool = True, label_min_height: float = 0.05):
    """Stacked Bell populations, Phi+ at the bottom, so the bottom band's height
    IS the target fidelity F_n.

    With ``zoom``, a second panel repeats the same stack over the top of the
    range only, where the three error bands live -- same measure, same stacking
    order, just a different slice of the y axis.
    """
    order = list(order or STACK_ORDER)
    idx = [BELL_NAMES.index(k) for k in order]
    series = [dyn.pops[:, i] for i in idx]
    cum = np.cumsum(np.vstack(series), axis=0)
    err_floor = float(np.min(cum[0]))                     # lowest Phi+ level

    ncols = 2 if zoom else 1
    fig, axes = _fig(1, ncols, figsize=(11.6, 4.6) if zoom else (7.2, 4.6))
    axes = np.atleast_1d(axes)

    for k, ax in enumerate(axes):
        ax.stackplot(dyn.n, *series,
                     colors=[BELL_COLORS[c] for c in order],
                     edgecolor=SURFACE, linewidth=1.4)     # 2px surface gap
        ax.set_xlim(dyn.n[0], dyn.n[-1])
        ax.set_xlabel("purification round $n$", color=INK_2)
        ax.set_ylabel("Bell-basis population", color=INK_2)
        if k == 0:
            ax.set_ylim(0, 1)
            ax.set_title("full range", color=INK, fontsize=10, loc="left")
        else:
            pad = max(0.02 * (1 - err_floor), 0.002)
            ax.set_ylim(max(0.0, err_floor - pad), 1.0)
            ax.set_title("zoom on the error bands", color=INK, fontsize=10, loc="left")
        lower = 0.0
        labels = []
        for c, top in zip(order, cum[:, -1]):
            height = top - lower
            visible = (min(top, ax.get_ylim()[1]) - max(lower, ax.get_ylim()[0]))
            frac = visible / (ax.get_ylim()[1] - ax.get_ylim()[0])
            if frac >= label_min_height:
                labels.append(((max(lower, ax.get_ylim()[0]) + min(top, ax.get_ylim()[1])) / 2,
                               _PRETTY[c], INK))
            lower = top
        _spread_end_labels(ax, labels, inside=True)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=BELL_COLORS[c], edgecolor="none")
               for c in order]
    fig.legend(handles[::-1], [_PRETTY[c] for c in order][::-1],
               loc="lower center", ncol=4, frameon=False, fontsize=10,
               labelcolor=INK_2, bbox_to_anchor=(0.5, -0.01),
               title="stack order, top to bottom", title_fontsize=8)
    fig.suptitle(f"{PROTOCOL_INFO[dyn.protocol]['label']}   |   {_subtitle(dyn)}",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0.09, 0.96, 0.93))
    return fig


# ---------------------------------------------------------------------------
# Figure B -- individual populations
# ---------------------------------------------------------------------------

def plot_population_lines(dyn: Dynamics, zoom: bool = True):
    """The four populations as lines; right panel zooms on the error components,
    where the initial threefold equality breaks."""
    ncols = 2 if zoom else 1
    fig, axes = _fig(1, ncols, figsize=(11.4, 4.4) if zoom else (7.2, 4.6))
    axes = np.atleast_1d(axes)
    every = max(1, len(dyn.n) // 18)

    for ax, keys in zip(axes, ([BELL_NAMES, BELL_NAMES[1:]] if zoom else [BELL_NAMES])):
        for k in keys:
            i = BELL_NAMES.index(k)
            ax.plot(dyn.n, dyn.pops[:, i], lw=2, color=BELL_COLORS[k],
                    marker=BELL_MARKERS[k], markevery=every, markersize=5,
                    markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label=_PRETTY[k])
        ax.set_xlim(dyn.n[0], dyn.n[-1])
        ax.set_xlabel("purification round $n$", color=INK_2)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="best")

    axes[0].set_ylim(0, 1)
    axes[0].set_ylabel("Bell-basis population", color=INK_2)
    for ax, keys in zip(axes, ([BELL_NAMES, BELL_NAMES[1:]] if zoom else [BELL_NAMES])):
        _spread_end_labels(ax, [(dyn.pops[-1, BELL_NAMES.index(k)], _PRETTY[k], INK)
                                for k in keys])
    axes[0].set_title("all four populations", color=INK, fontsize=10, loc="left")
    if zoom:
        axes[1].set_ylabel("Bell-basis population", color=INK_2)
        axes[1].set_title(r"error components only  (start: $\Phi^-=\Psi^+=\Psi^-$)",
                          color=INK, fontsize=10, loc="left")
    fig.suptitle(f"{PROTOCOL_INFO[dyn.protocol]['label']}   |   {_subtitle(dyn)}",
                 color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.97, 0.94))
    return fig


# ---------------------------------------------------------------------------
# Bell off-diagonal weight
# ---------------------------------------------------------------------------

def plot_offdiagonal(dyn_short: Dynamics, dyn_long: Dynamics = None,
                     escape: dict = None):
    """C_Bell(n) = sum_{i != j} |<B_i|rho_n|B_j>|^2 on a log axis.

    The initial state is Bell-diagonal, so C_Bell(0) is zero up to round-off
    (~1e-34) and would stretch the axis over thirty empty decades.  The curve
    therefore starts at n = 1 and the exact starting value is stated instead.
    One series, so no legend box: the title names it.
    """
    runs = [d for d in (dyn_short, dyn_long) if d is not None]
    fig, axes = _fig(1, len(runs), figsize=(11.6, 4.4) if len(runs) == 2 else (7.2, 4.4))
    axes = np.atleast_1d(axes)
    for ax, d in zip(axes, runs):
        m = d.n >= 1
        y = d.c_bell[m]
        ax.plot(d.n[m], y, lw=2, color=BELL_COLORS["Phi-"])
        ax.set_yscale("log")
        ax.set_xlim(d.n[m][0], d.n[-1])
        lo, hi = float(np.min(y)), float(np.max(y))
        ax.set_ylim(lo / 3, hi * 3)
        ax.set_xlabel("purification round $n$", color=INK_2)
        ax.set_ylabel(r"$C_{\mathrm{Bell}}(n)$", color=INK_2)
        ax.set_title(f"$1 \\leq n \\leq {d.n[-1]}$", color=INK, fontsize=10, loc="left")
        if escape and escape.get("n_escape") and escape["n_escape"] <= d.n[-1]:
            ax.axvline(escape["n_escape"], color=INK_2, ls="--", lw=1.2)
            ax.annotate(f"fidelity escapes here, $n \\approx {escape['n_escape']}$",
                        xy=(escape["n_escape"], lo * 2), textcoords="offset points",
                        xytext=(6, 0), fontsize=9, color=INK_2)
    axes[0].annotate(f"$C_{{\\mathrm{{Bell}}}}(0) = {dyn_short.c_bell[0]:.1e}$"
                     "  (Bell-diagonal input)",
                     xy=(0.02, 0.06), xycoords="axes fraction",
                     fontsize=9, color=INK_2)
    fig.suptitle(f"{PROTOCOL_INFO[dyn_short.protocol]['label']}   |   "
                 r"Bell off-diagonal weight  $C_{\mathrm{Bell}}=\sum_{i\neq j}"
                 r"|\langle B_i|\rho_n|B_j\rangle|^2$" + f"   |   {_subtitle(dyn_short)}",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


# ---------------------------------------------------------------------------
# long-time fidelity
# ---------------------------------------------------------------------------

def plot_long_time_fidelity(dyn: Dynamics, log_x: bool = False,
                            annotate: dict = None):
    """F_n over a long run; ``log_x`` plots against log10(n+1)."""
    fig, ax = _fig(figsize=(7.6, 4.6))
    x = np.log10(dyn.n + 1) if log_x else dyn.n
    ax.plot(x, dyn.fidelity, lw=2, color=BELL_COLORS["Phi+"], label=r"$F_n=p_{\Phi^+}$")
    ax.axhline(dyn.fidelity[0], color=INK_2, ls=":", lw=1)
    ax.annotate(f"$F_0 = {dyn.fidelity[0]:.4f}$", (x[0], dyn.fidelity[0]),
                textcoords="offset points", xytext=(8, -12), fontsize=9, color=INK_2)
    if annotate and annotate.get("n_escape"):
        xe = np.log10(annotate["n_escape"] + 1) if log_x else annotate["n_escape"]
        ax.axvline(xe, color=BELL_COLORS["Phi-"], ls="--", lw=1.2)
        ax.annotate(f"escape, $n \\approx {annotate['n_escape']}$",
                    (xe, 0.55), textcoords="offset points", xytext=(6, 0),
                    fontsize=9, color=BELL_COLORS["Phi-"])
    ax.annotate(f"$F_{{{dyn.n[-1]}}} = {dyn.fidelity[-1]:.6f}$",
                (x[-1], dyn.fidelity[-1]), textcoords="offset points",
                xytext=(-6, 10), ha="right", fontsize=9, color=INK)
    ax.set_xlim(x[0], x[-1])
    ax.set_ylim(0, 1)
    ax.set_xlabel(r"$\log_{10}(n+1)$" if log_x else "purification round $n$",
                  color=INK_2)
    ax.set_ylabel(r"target Bell fidelity $F_n$", color=INK_2)
    ax.set_title(f"{PROTOCOL_INFO[dyn.protocol]['label']}\n{_subtitle(dyn)}",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# cross-protocol comparisons
# ---------------------------------------------------------------------------

def plot_fidelity_overview(runs: Sequence[Dynamics]):
    """F_n for several protocols on one axis (same measure, same scale)."""
    slots = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
    fig, ax = _fig(figsize=(7.6, 4.8))
    every = max(1, len(runs[0].n) // 18)
    for colour, d in zip(slots, runs):
        marker = BELL_MARKERS[BELL_NAMES[slots.index(colour) % 4]]
        ax.plot(d.n, d.fidelity, lw=2, color=colour, marker=marker,
                markevery=every, markersize=5, markeredgecolor=SURFACE,
                markeredgewidth=0.8, label=d.protocol)
        ax.annotate(d.protocol, (d.n[-1], d.fidelity[-1]),
                    textcoords="offset points", xytext=(6, 0), fontsize=10,
                    color=INK, va="center")
    ax.axhline(runs[0].fidelity[0], color=INK_2, ls=":", lw=1)
    ax.set_xlim(runs[0].n[0], runs[0].n[-1] * 1.08)
    ax.set_xlabel("purification round $n$", color=INK_2)
    ax.set_ylabel(r"target Bell fidelity $F_n$", color=INK_2)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="lower right")
    ax.set_title("Target fidelity per round, four protocols\n"
                 rf"$\epsilon = {runs[0].eps:g}$,  $q = {runs[0].q:g}$",
                 color=INK, fontsize=11, loc="left")
    fig.tight_layout()
    return fig


def plot_pair_comparison(a: Dynamics, b: Dynamics, log_x: bool = True):
    """Two long runs of the same protocol family, F and C_Bell side by side."""
    fig, axes = _fig(1, 2, figsize=(11.6, 4.4))
    # the second run is drawn thin and dashed ON TOP of the first, so "the two
    # curves coincide" is something the reader can actually see
    styles = (dict(lw=3.4, color="#2a78d6", alpha=0.85),
              dict(lw=1.5, color="#eb6834", ls=(0, (5, 4))))
    for style, d, name in zip(styles, (a, b), (a.protocol, b.protocol)):
        x = np.log10(d.n + 1) if log_x else d.n
        axes[0].plot(x, d.fidelity, label=name, **style)
        axes[1].plot(x, np.maximum(d.c_bell, 1e-34), label=name, **style)
    m = min(len(a.fidelity), len(b.fidelity))
    dF = float(np.max(np.abs(a.fidelity[:m] - b.fidelity[:m])))
    axes[0].annotate(rf"$\max_n |F_n^{{\mathrm{{{a.protocol}}}}} - "
                     rf"F_n^{{\mathrm{{{b.protocol}}}}}| = {dF:.1e}$",
                     xy=(0.03, 0.10), xycoords="axes fraction",
                     fontsize=9.5, color=INK)
    for ax, ylab in zip(axes, (r"target Bell fidelity $F_n$",
                               r"$C_{\mathrm{Bell}}(n)$")):
        ax.set_xlabel(r"$\log_{10}(n+1)$" if log_x else "purification round $n$",
                      color=INK_2)
        ax.set_ylabel(ylab, color=INK_2)
        ax.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="best")
    axes[0].set_ylim(0, 1)
    axes[1].set_yscale("log")
    fig.suptitle(f"{a.protocol} vs {b.protocol}   |   "
                 rf"$\epsilon = {a.eps:g}$, $q = {a.q:g}$   |   "
                 "does the escape come from the compilation residual or from the noise?",
                 color=INK, fontsize=10, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig
