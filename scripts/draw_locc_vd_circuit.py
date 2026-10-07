"""Draw the ideal six-qubit distributed (LOCC) SWAP-test circuit with
``qml.draw_mpl``, and the five-qubit single-ancilla baseline next to it.

The drawing is not decorative: before any figure is saved, the unitary of the
PennyLane circuit is compared against the unitary that
``scripts/verify_locc_vd.py`` actually verified, so the picture is guaranteed to
show the circuit the numbers came from.

Run:
    python scripts/draw_locc_vd_circuit.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pennylane as qml  # noqa: E402

import _bootstrap  # noqa: F401,E402

from pqec_distill.locc_vd import (  # noqa: E402
    unitary_5q, unitary_6q_route_a, unitary_6q_route_b,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "locc_vd"

# the implementation's fixed ordering
WIRES_6Q = ["a", "b", "A1", "B1", "A2", "B2"]
# the same six wires regrouped by party, for a readability-only second view
WIRES_6Q_BY_PARTY = ["a", "A1", "A2", "b", "B1", "B2"]
WIRES_5Q = ["c", "A1", "B1", "A2", "B2"]

ALICE = {"a", "A1", "A2"}
BOB = {"b", "B1", "B2"}
ALICE_COLOR, BOB_COLOR = "#2a78d6", "#eb6834"
INK = "#101010"


# ---------------------------------------------------------------------------
# the circuits, written once and reused for both the matrix check and the figure
# ---------------------------------------------------------------------------

def gates_6q():
    """Alice's local Fredkin, then Bob's.  No state preparation, no readout."""
    qml.CSWAP(wires=["a", "A1", "A2"])     # Alice: Fredkin(a ; A1, A2)
    qml.CSWAP(wires=["b", "B1", "B2"])     # Bob:   Fredkin(b ; B1, B2)


def gates_5q():
    """The single ancilla c controls both local swaps."""
    qml.CSWAP(wires=["c", "A1", "A2"])
    qml.CSWAP(wires=["c", "B1", "B2"])


def circuit_6q(ancilla_pauli="X"):
    """|+> preparation, the two local Fredkins, and the ancilla read-out.

    The verification script initialises both ancillas in the state |+>; the
    Hadamards drawn here are the equivalent state preparation from |0>.
    """
    qml.Hadamard(wires="a")
    qml.Hadamard(wires="b")
    gates_6q()
    obs = qml.X if ancilla_pauli == "X" else qml.Y
    return qml.expval(obs("a") @ obs("b") @ qml.Z("A1") @ qml.Z("B1"))


def circuit_5q():
    qml.Hadamard(wires="c")
    gates_5q()
    return qml.expval(qml.X("c") @ qml.Z("A1") @ qml.Z("B1"))


# ---------------------------------------------------------------------------
# the figure must show the circuit that was verified -- check it first
# ---------------------------------------------------------------------------

def check_drawing_matches_verified_circuit() -> dict:
    u_draw_6q = qml.matrix(gates_6q, wire_order=WIRES_6Q)()
    u_draw_5q = qml.matrix(gates_5q, wire_order=WIRES_5Q)()
    return {
        "6Q drawn vs verified route A": float(np.max(np.abs(u_draw_6q - unitary_6q_route_a()))),
        "6Q drawn vs verified route B": float(np.max(np.abs(u_draw_6q - unitary_6q_route_b()))),
        "5Q drawn vs verified baseline": float(np.max(np.abs(u_draw_5q - unitary_5q()))),
    }


# ---------------------------------------------------------------------------
# drawing
# ---------------------------------------------------------------------------

def _style():
    plt.rcParams.update({
        "figure.facecolor": "white", "savefig.facecolor": "white",
        "savefig.bbox": "tight", "savefig.dpi": 300,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })


def _colour_wire_labels(ax, wire_order):
    """Tint each wire label by its owner: Alice blue, Bob orange."""
    wanted = {w: (ALICE_COLOR if w in ALICE else
                  BOB_COLOR if w in BOB else INK) for w in wire_order}
    for txt in ax.texts:
        key = txt.get_text().strip()
        if key in wanted:
            txt.set_color(wanted[key])
            txt.set_fontweight("bold")


def _annotate(ax, wire_order, ancilla_pauli, data_obs="O"):
    """Left-hand input labels and right-hand read-out labels.

    ``qml.draw_mpl`` lays wires out top-to-bottom in ``wire_order`` at integer y,
    one gate column per integer x, with the read-out column one unit inside the
    right edge.  Both are read off the axes rather than hard-coded.
    """
    x_lo, x_hi = ax.get_xlim()
    x_meas = x_hi - 1.0
    x_in = x_lo - 1.0
    ax.set_xlim(x_in - 0.50, x_hi + 1.15)
    row = {w: k for k, w in enumerate(wire_order)}

    for w in ("a", "b", "c"):
        if w not in row:
            continue
        colour = ALICE_COLOR if w in ALICE else BOB_COLOR if w in BOB else INK
        ax.text(x_in, row[w], r"$|0\rangle$", va="center", ha="left",
                fontsize=12, color=colour)

    # the two input copies, when the ordering puts each copy on adjacent wires
    for label, pair in ((r"$\rho_{A_1B_1}$", ("A1", "B1")),
                        (r"$\rho_{A_2B_2}$", ("A2", "B2"))):
        r0, r1 = row[pair[0]], row[pair[1]]
        if abs(r0 - r1) != 1:
            continue
        top, bot = min(r0, r1), max(r0, r1)
        xb = x_in + 0.30
        ax.plot([xb, xb], [top - 0.18, bot + 0.18], color=INK, linewidth=1.3)
        for yy in (top - 0.18, bot + 0.18):
            ax.plot([xb, xb + 0.16], [yy, yy], color=INK, linewidth=1.3)
        ax.text(x_in + 0.12, 0.5 * (top + bot), label, va="center", ha="right",
                fontsize=11.5, color=INK)

    sym = {"X": r"$X$", "Y": r"$Y$"}[ancilla_pauli]
    for w in ("a", "b", "c"):
        if w not in row:
            continue
        colour = ALICE_COLOR if w in ALICE else BOB_COLOR if w in BOB else INK
        ax.text(x_meas + 0.78, row[w], sym, va="center", ha="center",
                fontsize=13, fontweight="bold", color=colour)
    for w in ("A1", "B1"):
        colour = ALICE_COLOR if w in ALICE else BOB_COLOR
        ax.text(x_meas + 0.78, row[w], rf"${data_obs}$", va="center",
                ha="center", fontsize=12, color=colour)
    ax.text(x_meas + 0.78, -0.72, "read out", va="center", ha="center",
            fontsize=10, color="#4a4a4a", style="italic")


def _save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        paths.append(p)
    plt.close(fig)
    return paths


def draw_6q(ancilla_pauli, wire_order, stem, title, party_cut=False):
    _style()
    drawer = qml.draw_mpl(circuit_6q, wire_order=wire_order, show_all_wires=True,
                          style="pennylane", decimals=None)
    fig, ax = drawer(ancilla_pauli)
    _colour_wire_labels(ax, wire_order)
    _annotate(ax, wire_order, ancilla_pauli)
    if party_cut:
        # wires run top to bottom in wire_order, so Alice occupies rows 0-2
        x_lo, x_hi = ax.get_xlim()
        ax.axhline(2.5, color="#4a4a4a", linestyle=(0, (5, 4)), linewidth=1.2)
        for name, y, colour in (("Alice", 1.0, ALICE_COLOR),
                                ("Bob", 4.0, BOB_COLOR)):
            ax.text(x_lo + 0.12, y, name, rotation=90, va="center", ha="center",
                    color=colour, fontsize=12, fontweight="bold")
        ax.set_xlim(x_lo, x_hi)
    fig.set_size_inches(10.0, 5.6)
    ax.set_title(title, color=INK, fontsize=12.5, pad=12)
    return _save(fig, stem)


def draw_5q(stem, title):
    _style()
    drawer = qml.draw_mpl(circuit_5q, wire_order=WIRES_5Q, show_all_wires=True,
                          style="pennylane", decimals=None)
    fig, ax = drawer()
    _colour_wire_labels(ax, WIRES_5Q)
    _annotate(ax, WIRES_5Q, "X")
    fig.set_size_inches(10.0, 5.0)
    ax.set_title(title, color=INK, fontsize=12.5, pad=12)
    return _save(fig, stem)


def main() -> int:
    print("=" * 74)
    print("DOES THE DRAWING SHOW THE CIRCUIT THAT WAS VERIFIED?")
    print("=" * 74)
    checks = check_drawing_matches_verified_circuit()
    for k, v in checks.items():
        print(f"  {k:<34} max |diff| = {v:.3e}")
    assert max(checks.values()) < 1e-12, "the drawn circuit is not the verified one"
    print("\n  text drawing (implementation ordering [a, b, A1, B1, A2, B2])\n")
    print(qml.draw(circuit_6q, wire_order=WIRES_6Q, show_all_wires=True)("X"))

    written = []
    written += draw_6q("X", WIRES_6Q, "locc_vd_6q_circuit_XX",
                       r"6Q LOCC SWAP-test circuit  —  $\langle X_a X_b \otimes "
                       r"O_{A_1B_1}\rangle$   (ordering $[a,b,A_1,B_1,A_2,B_2]$)")
    written += draw_6q("Y", WIRES_6Q, "locc_vd_6q_circuit_YY",
                       r"6Q LOCC SWAP-test circuit  —  $\langle Y_a Y_b \otimes "
                       r"O_{A_1B_1}\rangle$   (same gates, rotated read-out)")
    written += draw_6q("X", WIRES_6Q_BY_PARTY, "locc_vd_6q_circuit_by_party",
                       "Same circuit, wires regrouped by party "
                       "(readability only — the simulator uses "
                       r"$[a,b,A_1,B_1,A_2,B_2]$)", party_cut=True)
    written += draw_5q("locc_vd_5q_baseline",
                       r"5Q standard SWAP test (baseline)  —  $\langle X_c "
                       r"\otimes O_{A_1B_1}\rangle$")

    print("\n" + "=" * 74)
    print("FILES")
    print("=" * 74)
    for p in written:
        print(f"  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
