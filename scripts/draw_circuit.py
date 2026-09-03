"""Render the exact 5-CNOT circuit as ASCII, straight from CNOT_SEQUENCE.

The drawing is generated from the same gate list the simulator uses, so it can
never drift out of sync with the circuit.

Outputs
-------
results/data/circuit_diagram.txt   (also printed)
"""

from __future__ import annotations

from _bootstrap import DATA_DIR  # noqa: E402
from pqec_distill.circuit import CNOT_SEQUENCE, N_QUBITS, Q3  # noqa: E402

COL_W = 6
DOT, PLUS, WIRE, VERT = "●", "⊕", "─", "│"


def render() -> str:
    grid = [[WIRE * COL_W for _ in CNOT_SEQUENCE] for _ in range(N_QUBITS)]
    for col, (control, target) in enumerate(CNOT_SEQUENCE):
        lo, hi = min(control, target), max(control, target)
        for row in range(lo, hi + 1):
            if row == control:
                mark = DOT
            elif row == target:
                mark = PLUS
            else:
                mark = VERT
            cell = list(WIRE * COL_W)
            cell[2] = mark
            grid[row][col] = "".join(cell)

    tails = {
        0: "  keep (retained pair A)",
        1: "  keep (retained pair A)",
        2: f"{WIRE}[H]{WIRE} measure  m3",
        3: f"{WIRE * 5} measure  m4",
    }
    lines = []
    for row in range(N_QUBITS):
        body = "".join(grid[row])
        lines.append(f"q{row + 1} {WIRE}{body}{WIRE}{tails[row]}")

    header = (
        "gate order:  "
        + "  ".join(
            f"{i + 1}.q{c + 1}->q{t + 1}" for i, (c, t) in enumerate(CNOT_SEQUENCE)
        )
        + f"   then H on q{Q3 + 1}"
    )
    return header + "\n\n" + "\n".join(lines)


def main() -> int:
    text = render()
    (DATA_DIR / "circuit_diagram.txt").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
