"""Draw the ideal six-qubit distributed (LOCC) SWAP-test circuit with
``qml.draw_mpl``, and the five-qubit single-ancilla baseline next to it.

Every measurement is drawn as a computational-basis (Z) measurement, with the
basis-change gates that make it one shown explicitly:

    measure X   ->   H            then Z
    measure Y   ->   S^dagger, H  then Z
    measure Z   ->                     Z

The drawing is not decorative.  Before any figure is saved the script checks
(a) that the CSWAP network of the PennyLane circuit is the unitary that
scripts/verify_locc_vd.py actually verified, and (b) that reading Z after the
drawn rotations reproduces the frozen correlators <X_a X_b (x) O> and
<Y_a Y_b (x) O> exactly.  A figure can therefore not drift from the code.

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
    correlator_6q, delta_5q, final_state_5q, final_state_6q, pauli_product,
    plus_state_dm, unitary_5q, unitary_6q_route_a, unitary_6q_route_b,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures" / "locc_vd"

#: the ordering the frozen simulator uses; the matrix checks use this one
WIRES_6Q_IMPL = ["a", "b", "A1", "B1", "A2", "B2"]
#: display order requested for the figure (same circuit, rows rearranged)
WIRES_6Q_DISPLAY = ["a", "A1", "B1", "A2", "B2", "b"]
#: display order grouped by party, so the Alice/Bob cut can be drawn
WIRES_6Q_BY_PARTY = ["a", "A1", "A2", "b", "B1", "B2"]
WIRES_5Q = ["c", "A1", "B1", "A2", "B2"]

ALICE = {"a", "A1", "A2"}
BOB = {"b", "B1", "B2"}
ALICE_COLOR, BOB_COLOR = "#2a78d6", "#eb6834"
INK, INK2 = "#101010", "#4a4a4a"


# ---------------------------------------------------------------------------
# circuits
# ---------------------------------------------------------------------------

def gates_6q():
    """Alice's local Fredkin, then Bob's.  No preparation, no read-out."""
    qml.CSWAP(wires=["a", "A1", "A2"])     # Alice: Fredkin(a ; A1, A2)
    qml.CSWAP(wires=["b", "B1", "B2"])     # Bob:   Fredkin(b ; B1, B2)


def gates_5q():
    qml.CSWAP(wires=["c", "A1", "A2"])
    qml.CSWAP(wires=["c", "B1", "B2"])


def rotate_to_z(pauli: str, wire) -> None:
    """Gates that turn a measurement of ``pauli`` into a Z measurement.

    X:  H                 because  H^dag Z H = X
    Y:  S^dag then H      because  (H S^dag)^dag Z (H S^dag) = S X S^dag = Y
    Z:  nothing
    """
    if pauli == "X":
        qml.Hadamard(wires=wire)
    elif pauli == "Y":
        qml.adjoint(qml.S(wires=wire))
        qml.Hadamard(wires=wire)
    elif pauli not in ("Z", "I"):
        raise ValueError(f"not a Pauli: {pauli!r}")


def _z_product(wires):
    obs = qml.Z(wires[0])
    for w in wires[1:]:
        obs = obs @ qml.Z(w)
    return obs


def _barrier(wires):
    """A visual-only separator.  ``only_visual=True`` means it carries no
    operation, so it cannot affect any matrix or expectation value; the matrix
    checks below run on ``gates_6q`` / ``gates_5q``, which have no barriers."""
    qml.Barrier(wires=wires, only_visual=True)


def circuit_6q(ancilla_pauli="X", data_label="ZZ"):
    """|+> preparation, the two local Fredkins, basis changes, Z read-out.

    The Hadamards at the left are the preparation of the |+> ancillas from
    |0>; the verification script initialises them as |+><+| directly.
    """
    qml.Hadamard(wires="a")
    qml.Hadamard(wires="b")
    _barrier(WIRES_6Q_IMPL)
    gates_6q()
    _barrier(WIRES_6Q_IMPL)
    rotate_to_z(ancilla_pauli, "a")
    rotate_to_z(ancilla_pauli, "b")
    rotate_to_z(data_label[0], "A1")
    rotate_to_z(data_label[1], "B1")
    measured = ["a", "b"] + [w for w, p in (("A1", data_label[0]),
                                            ("B1", data_label[1])) if p != "I"]
    return qml.expval(_z_product(measured))


def circuit_5q(data_label="ZZ"):
    qml.Hadamard(wires="c")
    _barrier(WIRES_5Q)
    gates_5q()
    _barrier(WIRES_5Q)
    rotate_to_z("X", "c")
    rotate_to_z(data_label[0], "A1")
    rotate_to_z(data_label[1], "B1")
    measured = ["c"] + [w for w, p in (("A1", data_label[0]),
                                       ("B1", data_label[1])) if p != "I"]
    return qml.expval(_z_product(measured))


# ---------------------------------------------------------------------------
# the figure must show the circuit that was verified
# ---------------------------------------------------------------------------

def _single(mat, wire, order):
    out = np.array([[1.0 + 0j]])
    for w in order:
        out = np.kron(out, mat if w == wire else np.eye(2, dtype=complex))
    return out


_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_SDG = np.array([[1, 0], [0, -1j]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)


def _rotation_unitary(pauli: str, wire, order) -> np.ndarray:
    dim = 2 ** len(order)
    if pauli == "X":
        return _single(_H, wire, order)
    if pauli == "Y":
        return _single(_H, wire, order) @ _single(_SDG, wire, order)
    return np.eye(dim, dtype=complex)


def _z_observable(wires, order) -> np.ndarray:
    out = np.eye(2 ** len(order), dtype=complex)
    for w in wires:
        out = out @ _single(_Z, w, order)
    return out


def check_drawing_matches_verified_circuit(n_states=12, seed=5150) -> dict:
    """(a) the CSWAP network is the verified unitary;
    (b) Z after the drawn rotations reproduces the frozen correlators."""
    checks = {
        "6Q CSWAP network vs verified route A":
            float(np.max(np.abs(qml.matrix(gates_6q, wire_order=WIRES_6Q_IMPL)()
                                - unitary_6q_route_a()))),
        "6Q CSWAP network vs verified route B":
            float(np.max(np.abs(qml.matrix(gates_6q, wire_order=WIRES_6Q_IMPL)()
                                - unitary_6q_route_b()))),
        "5Q CSWAP network vs verified baseline":
            float(np.max(np.abs(qml.matrix(gates_5q, wire_order=WIRES_5Q)()
                                - unitary_5q()))),
    }

    rng = np.random.default_rng(seed)
    order6, order5 = WIRES_6Q_IMPL, WIRES_5Q
    worst6 = worst5 = 0.0
    for _ in range(n_states):
        m = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        rho = m @ m.conj().T
        rho = rho / np.trace(rho)
        sigma6, sigma5 = final_state_6q(rho), final_state_5q(rho)
        for label in ("II", "IX", "XY", "YZ", "ZZ", "ZX", "YY", "XX"):
            data_wires = [w for w, p in (("A1", label[0]), ("B1", label[1]))
                          if p != "I"]
            for anc in ("X", "Y"):
                # rotate, then read Z on the measured wires
                u = np.eye(2 ** len(order6), dtype=complex)
                for wire, p in (("a", anc), ("b", anc),
                                ("A1", label[0]), ("B1", label[1])):
                    u = _rotation_unitary(p, wire, order6) @ u
                z = _z_observable(["a", "b"] + data_wires, order6)
                drawn = float(np.real(np.trace(z @ u @ sigma6 @ u.conj().T)))
                direct = correlator_6q(sigma6, anc, pauli_product(label))
                worst6 = max(worst6, abs(drawn - direct))
            u5 = np.eye(2 ** len(order5), dtype=complex)
            for wire, p in (("c", "X"), ("A1", label[0]), ("B1", label[1])):
                u5 = _rotation_unitary(p, wire, order5) @ u5
            z5 = _z_observable(["c"] + data_wires, order5)
            drawn5 = float(np.real(np.trace(z5 @ u5 @ sigma5 @ u5.conj().T)))
            worst5 = max(worst5, abs(drawn5 - delta_5q(sigma5,
                                                       pauli_product(label))))
    checks["6Q drawn Z read-out vs frozen <P_a P_b (x) O>"] = worst6
    checks["5Q drawn Z read-out vs frozen <X_c (x) O>"] = worst5
    return checks


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
    wanted = {w: (ALICE_COLOR if w in ALICE else
                  BOB_COLOR if w in BOB else INK) for w in wire_order}
    for txt in ax.texts:
        key = txt.get_text().strip()
        if key in wanted:
            txt.set_color(wanted[key])
            txt.set_fontweight("bold")


def _save(fig, stem):
    OUT.mkdir(parents=True, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        p = OUT / f"{stem}.{ext}"
        fig.savefig(p, dpi=300 if ext == "png" else None)
        paths.append(p)
    plt.close(fig)
    return paths


def draw_6q(ancilla_pauli, data_label, wire_order, stem, title,
            party_cut=False):
    _style()
    drawer = qml.draw_mpl(circuit_6q, wire_order=wire_order,
                          show_all_wires=True, style="pennylane", decimals=None)
    fig, ax = drawer(ancilla_pauli, data_label)
    _colour_wire_labels(ax, wire_order)
    if party_cut:
        # axhline and text inside the existing limits: no axis is resized, so
        # the aspect ratio stays exactly the one qml.draw_mpl produced
        x_lo, _x_hi = ax.get_xlim()
        ax.axhline(2.5, color=INK2, linestyle=(0, (5, 4)), linewidth=1.2)
        for name, y, colour in (("Alice", 1.0, ALICE_COLOR),
                                ("Bob", 4.0, BOB_COLOR)):
            ax.text(x_lo + 0.10, y, name, rotation=90, va="center", ha="center",
                    color=colour, fontsize=11, fontweight="bold")
    ax.set_title(title, color=INK, fontsize=12, pad=12)
    return _save(fig, stem)


def draw_5q(data_label, stem, title):
    _style()
    drawer = qml.draw_mpl(circuit_5q, wire_order=WIRES_5Q, show_all_wires=True,
                          style="pennylane", decimals=None)
    fig, ax = drawer(data_label)
    _colour_wire_labels(ax, WIRES_5Q)
    ax.set_title(title, color=INK, fontsize=12, pad=12)
    return _save(fig, stem)


def main() -> int:
    print("=" * 74)
    print("DOES THE DRAWING SHOW THE CIRCUIT THAT WAS VERIFIED?")
    print("=" * 74)
    checks = check_drawing_matches_verified_circuit()
    for k, v in checks.items():
        print(f"  {k:<46} {v:.3e}")
    assert max(checks.values()) < 1e-12, "the drawn circuit is not the verified one"

    print("\n  text drawing, display order "
          f"{WIRES_6Q_DISPLAY}, ancilla X read-out, O = Z(x)Z\n")
    print(qml.draw(circuit_6q, wire_order=WIRES_6Q_DISPLAY,
                   show_all_wires=True)("X", "ZZ"))
    print("\n  the same with the ancilla Y read-out\n")
    print(qml.draw(circuit_6q, wire_order=WIRES_6Q_DISPLAY,
                   show_all_wires=True)("Y", "ZZ"))

    written = []
    written += draw_6q(
        "X", "ZZ", WIRES_6Q_DISPLAY, "locc_vd_6q_circuit_XX",
        r"6Q LOCC SWAP-test circuit  —  $C_{XX}(O)$ for $O = Z\otimes Z$"
        "\n" r"ancilla $X$ read-out drawn as $H$ then $Z$")
    written += draw_6q(
        "Y", "ZZ", WIRES_6Q_DISPLAY, "locc_vd_6q_circuit_YY",
        r"6Q LOCC SWAP-test circuit  —  $C_{YY}(O)$ for $O = Z\otimes Z$"
        "\n" r"ancilla $Y$ read-out drawn as $S^{\dagger}$, $H$ then $Z$")
    written += draw_6q(
        "X", "XY", WIRES_6Q_DISPLAY, "locc_vd_6q_circuit_XX_data_XY",
        r"6Q LOCC SWAP-test circuit  —  $C_{XX}(O)$ for $O = X\otimes Y$"
        "\n" r"the data basis changes are drawn too: $H$ on $A_1$, "
        r"$S^{\dagger}H$ on $B_1$")
    written += draw_6q(
        "X", "ZZ", WIRES_6Q_BY_PARTY, "locc_vd_6q_circuit_by_party",
        "Same circuit, wires regrouped by party (readability only)",
        party_cut=True)
    written += draw_5q(
        "ZZ", "locc_vd_5q_baseline",
        r"5Q standard SWAP test (baseline)  —  $\Delta_{5Q}(O)$ for "
        r"$O = Z\otimes Z$" "\n" r"ancilla $X$ read-out drawn as $H$ then $Z$")

    print("\n" + "=" * 74)
    print("FILES")
    print("=" * 74)
    for p in written:
        print(f"  {p.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
