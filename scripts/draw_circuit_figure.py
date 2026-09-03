"""Publication-quality figure of the 4-qubit / 5-CNOT Bell-purification circuit.

The gate list is read from ``pqec_distill.circuit.CNOT_SEQUENCE``, so the figure
can never drift out of sync with the circuit that is actually simulated.

Outputs
-------
results/figures/circuit_5cnot.pdf   (vector, primary)
results/figures/circuit_5cnot.png   (preview)
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Arc, Circle, FancyArrowPatch, Rectangle

from _bootstrap import FIG_DIR  # noqa: E402
from pqec_distill.circuit import CNOT_SEQUENCE, N_QUBITS, Q3, Q4  # noqa: E402

# Embed TrueType (not Type 3) so the PDF is publication-safe.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
matplotlib.rcParams["font.family"] = "serif"
matplotlib.rcParams["font.serif"] = ["DejaVu Serif"]

# Okabe-Ito
BLUE = "#0072B2"
GREEN = "#009E73"
ORANGE = "#D55E00"
GREY = "#5A5A5A"

WIRE_LW = 1.3
GATE_LW = 1.5
DOT_R = 0.075
TGT_R = 0.145

X0 = 0.0                      # wire start
DX = 0.95                     # CNOT column spacing
X_FIRST = 0.85                # first CNOT column
X_H = X_FIRST + 5 * DX + 0.35  # Hadamard column
X_M = X_H + 0.95              # measurement column
X_END = X_M + 0.72            # retained-wire end
X_BAND_L = -0.62              # band left edge
X_LABEL = -0.90               # pair-label right edge (outside the band)

#: row y-coordinates, q1 on top
Y = {0: 3.0, 1: 2.0, 2: 1.0, 3: 0.0}


def cnot_x(index: int) -> float:
    return X_FIRST + index * DX


def draw_wires(ax):
    for q in range(N_QUBITS):
        y = Y[q]
        # measured wires stop at the meter; retained wires run to the end
        x_end = X_M - 0.30 if q in (Q3, Q4) else X_END
        ax.plot([X0, x_end], [y, y], color="black", lw=WIRE_LW, zorder=1,
                solid_capstyle="butt")


def draw_cnot(ax, control: int, target: int, x: float):
    yc, yt = Y[control], Y[target]
    ax.plot([x, x], [yc, yt], color=BLUE, lw=GATE_LW, zorder=3,
            solid_capstyle="round")
    ax.add_patch(Circle((x, yc), DOT_R, facecolor=BLUE, edgecolor=BLUE, zorder=4))
    ax.add_patch(Circle((x, yt), TGT_R, facecolor="white", edgecolor=BLUE,
                        lw=GATE_LW, zorder=4))
    ax.plot([x - TGT_R, x + TGT_R], [yt, yt], color=BLUE, lw=GATE_LW, zorder=5)
    ax.plot([x, x], [yt - TGT_R, yt + TGT_R], color=BLUE, lw=GATE_LW, zorder=5)


def draw_hadamard(ax, qubit: int, x: float):
    y = Y[qubit]
    w = h = 0.46
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor="white",
                           edgecolor=GREEN, lw=GATE_LW, zorder=4))
    ax.text(x, y, "H", ha="center", va="center", fontsize=11, color=GREEN,
            zorder=5)


def draw_meter(ax, qubit: int, x: float, label: str):
    y = Y[qubit]
    w, h = 0.56, 0.46
    ax.add_patch(Rectangle((x - w / 2, y - h / 2), w, h, facecolor="white",
                           edgecolor="black", lw=GATE_LW, zorder=4))
    ax.add_patch(Arc((x, y - 0.10), 0.34, 0.30, theta1=0, theta2=180,
                     edgecolor="black", lw=1.0, zorder=5))
    ax.add_patch(FancyArrowPatch((x, y - 0.10), (x + 0.13, y + 0.10),
                                 arrowstyle="-|>", mutation_scale=7,
                                 color="black", lw=1.0, zorder=5))
    for dy in (-0.028, 0.028):
        ax.plot([x + w / 2, x + 0.80], [y + dy, y + dy], color="black", lw=0.8,
                zorder=2)
    ax.text(x + 0.90, y, label, ha="left", va="center", fontsize=10.5)


def draw_pair(ax, qubits, band_color, title, ket, title_color):
    """Shaded band behind a Bell pair, with its label OUTSIDE the band."""
    ys = [Y[q] for q in qubits]
    top, bot = max(ys) + 0.36, min(ys) - 0.36
    mid = (top + bot) / 2
    ax.add_patch(Rectangle((X_BAND_L, bot), (X_END + 0.10) - X_BAND_L,
                           top - bot, facecolor=band_color, edgecolor="none",
                           zorder=0))
    ax.text(X_LABEL, mid + 0.20, title, ha="right", va="center",
            fontsize=9.5, color=title_color)
    ax.text(X_LABEL, mid - 0.22, ket, ha="right", va="center",
            fontsize=11, color="black")


def main() -> int:
    fig, ax = plt.subplots(figsize=(8.6, 3.25))

    draw_pair(ax, [0, 1], "#DCE9F5", "pair A  (retained)",
              r"$|B_{ab}\rangle$", BLUE)
    draw_pair(ax, [2, 3], "#ECECEC", "pair B  (measured)",
              r"$|B_{cd}\rangle$", GREY)

    draw_wires(ax)

    for i, (control, target) in enumerate(CNOT_SEQUENCE):
        x = cnot_x(i)
        draw_cnot(ax, control, target, x)
        ax.text(x, 3.66, f"{i + 1}", ha="center", va="center", fontsize=8.5,
                color=BLUE)

    draw_hadamard(ax, Q3, X_H)
    draw_meter(ax, Q3, X_M, r"$m_3 = b \oplus d$")
    draw_meter(ax, Q4, X_M, r"$m_4 = a \oplus c$")

    for q, name in enumerate(["q_1", "q_2", "q_3", "q_4"]):
        ax.text(X0 - 0.20, Y[q], rf"${name}$", ha="right", va="center",
                fontsize=11)

    # retained output, at the right end of the q1/q2 wires
    ax.text(X_END + 0.14, 2.5, r"$|B_{ab}\rangle$ kept", ha="left",
            va="center", fontsize=10, color=BLUE)

    # CNOT-span bracket
    y_br = -0.72
    ax.annotate("", xy=(cnot_x(0) - 0.26, y_br), xytext=(cnot_x(4) + 0.26, y_br),
                arrowprops=dict(arrowstyle="<->", color=BLUE, lw=0.9))
    ax.text(cnot_x(2), y_br - 0.30, "5 CNOTs", ha="center", va="center",
            fontsize=9.5, color=BLUE)

    ax.text(X_M + 0.10, y_br - 0.02, r"postselect $(m_3,m_4)=(0,0)$",
            ha="center", va="center", fontsize=9.5, color=ORANGE)
    ax.text(X_M + 0.10, y_br - 0.34,
            r"$\Pi_{00}=\frac{1}{4}(I+XXXX)(I+ZZZZ)$",
            ha="center", va="center", fontsize=8.8, color=ORANGE)

    ax.set_title(
        "4-qubit / 5-CNOT Bell-label comparator\n"
        r"$V\,|B_{ab}\rangle_{12}|B_{cd}\rangle_{34}"
        r" = |B_{ab}\rangle_{12}\,|b\oplus d\rangle_3\,|a\oplus c\rangle_4$",
        fontsize=11, pad=10)

    ax.set_xlim(X_LABEL - 1.90, X_END + 1.55)
    ax.set_ylim(y_br - 0.62, 3.92)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"circuit_5cnot.{ext}", format=ext,
                    bbox_inches="tight", pad_inches=0.03,
                    dpi=300 if ext == "png" else None)
    plt.close(fig)

    print("gate list drawn from CNOT_SEQUENCE:")
    for i, (c, t) in enumerate(CNOT_SEQUENCE, start=1):
        print(f"  {i}. CNOT q{c + 1} -> q{t + 1}")
    print(f"  then H on q{Q3 + 1}; measure q{Q3 + 1}, q{Q4 + 1}")
    print(f"\nwrote {FIG_DIR / 'circuit_5cnot.pdf'}")
    print(f"wrote {FIG_DIR / 'circuit_5cnot.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
