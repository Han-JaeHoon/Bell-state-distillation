"""Publication figure style shared by every report figure.

One place for the things that must not drift between figures: the serif face,
the type sizes, the recessive grid, and -- most importantly -- the Bell-basis
colour and marker convention, which is fixed by Bell state and identical in
every panel of every figure.
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

__all__ = [
    "BELL_ORDER", "BELL_TEX", "BELL_COLOR", "BELL_MARKER", "BELL_DASH",
    "EPS_RAMP", "EPS_MARKER", "QUANTITY_COLOR", "SURFACE", "INK", "INK_2", "GRID",
    "use_report_style", "style_axes", "save",
]

#: canonical ordering, used for every array, CSV column set and legend
BELL_ORDER = ("Phi+", "Phi-", "Psi+", "Psi-")
BELL_TEX = {"Phi+": r"$\Phi^{+}$", "Phi-": r"$\Phi^{-}$",
            "Psi+": r"$\Psi^{+}$", "Psi-": r"$\Psi^{-}$"}

#: categorical slots 1-4 of the validated palette, bound to Bell states and
#: never reassigned.  Two of the four sit below 3:1 against white, so every
#: figure using them carries a legend AND the CSV export is the table view.
BELL_COLOR = {"Phi+": "#2a78d6", "Phi-": "#eb6834",
              "Psi+": "#1baf7a", "Psi-": "#eda100"}
#: secondary encoding, so identity never rests on colour alone
BELL_MARKER = {"Phi+": "o", "Phi-": "s", "Psi+": "^", "Psi-": "D"}
#: Psi- is dashed: in Type 3/4 it lies exactly on Psi+, in 4Q exactly on Phi-,
#: and a dashed overlay shows two coincident curves instead of a missing one
BELL_DASH = {"Phi+": "solid", "Phi-": "solid", "Psi+": "solid",
             "Psi-": (0, (5, 3))}

#: eps is a MAGNITUDE: one hue, light to dark.  Six steps of the documented
#: blue ramp; the ramp holds five steps that also clear the ordinal lightness
#: gap, so every curve additionally carries a marker and a direct label.
EPS_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b")
EPS_MARKER = ("o", "s", "^", "D", "v", "P")

#: distinct QUANTITIES (F_0 vs F_1 vs F_inf) take categorical slots
QUANTITY_COLOR = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")

SURFACE = "#ffffff"
INK = "#101010"
INK_2 = "#4a4a4a"
GRID = "#d9d9d9"


def use_report_style() -> None:
    """Install the shared rcParams.  Idempotent."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "savefig.bbox": "tight",
        "savefig.dpi": 300,
        "pdf.fonttype": 42,          # embed TrueType, not Type 3
        "ps.fonttype": 42,
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 11,
        "axes.titlesize": 11.5,
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 0.8,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 1.8,
        "lines.markersize": 4.5,
        "lines.markeredgewidth": 0.7,
        "legend.frameon": False,
        "legend.handlelength": 2.2,
    })


def style_axes(ax) -> None:
    """Drop the top/right spines and keep the remaining ones recessive."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)


def panel_label(ax, text: str, dx: float = -0.16, dy: float = 1.04) -> None:
    ax.text(dx, dy, text, transform=ax.transAxes, fontsize=12,
            fontweight="bold", color=INK, va="bottom", ha="left")


def save(fig, out_dir, stem: str) -> list:
    """Write <stem>.png (300 dpi) and <stem>.pdf (vector).  Returns the paths."""
    paths = []
    for ext in ("png", "pdf"):
        path = out_dir / f"{stem}.{ext}"
        fig.savefig(path, dpi=300 if ext == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths
