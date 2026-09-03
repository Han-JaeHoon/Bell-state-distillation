"""Run all 16 Bell-product inputs through the exact circuit and record the result.

Outputs
-------
results/data/bell_product_cases.csv
results/data/bell_product_cases.json
"""

from __future__ import annotations

import csv
import json

import numpy as np

from _bootstrap import DATA_DIR  # noqa: E402
from pqec_distill.analytics import predicted_measurement_bits  # noqa: E402
from pqec_distill.bell_states import (  # noqa: E402
    BELL_LABELS, BELL_NAMES, bell_index, bell_state,
)
from pqec_distill.circuit import CNOT_SEQUENCE_LABELS, full_unitary  # noqa: E402


def basis_vector(bit3: int, bit4: int) -> np.ndarray:
    v = np.zeros(4, dtype=complex)
    v[2 * bit3 + bit4] = 1.0
    return v


def phase_aligned_error(got: np.ndarray, expected: np.ndarray) -> float:
    overlap = np.vdot(expected, got)
    phase = overlap / abs(overlap) if abs(overlap) > 1e-14 else 1.0
    return float(np.linalg.norm(got - phase * expected))


def main() -> int:
    v = full_unitary()
    rows = []
    worst = 0.0
    for (a, b) in BELL_LABELS:
        for (c, d) in BELL_LABELS:
            psi = np.kron(bell_state(a, b), bell_state(c, d))
            out = v @ psi
            m3, m4 = predicted_measurement_bits(a, b, c, d)
            expected = np.kron(bell_state(a, b), basis_vector(m3, m4))
            err = phase_aligned_error(out, expected)
            worst = max(worst, err)

            # measured bits read directly off the simulated statevector
            probs = np.zeros((2, 2))
            for i, amp in enumerate(out):
                probs[(i >> 1) & 1, i & 1] += abs(amp) ** 2
            obs3, obs4 = np.unravel_index(int(np.argmax(probs)), (2, 2))

            rows.append(
                {
                    "first_pair": BELL_NAMES[bell_index(a, b)],
                    "second_pair": BELL_NAMES[bell_index(c, d)],
                    "a": a, "b": b, "c": c, "d": d,
                    "predicted_m3": m3, "predicted_m4": m4,
                    "observed_m3": int(obs3), "observed_m4": int(obs4),
                    "branch_probability": float(probs[obs3, obs4]),
                    "retained_pair": BELL_NAMES[bell_index(a, b)],
                    "statevector_error": err,
                    "success_branch": bool((m3, m4) == (0, 0)),
                }
            )

    with open(DATA_DIR / "bell_product_cases.csv", "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "circuit_cnot_sequence": CNOT_SEQUENCE_LABELS,
        "final_gate": "H on q3",
        "measured_qubits": ["q3", "q4"],
        "qubit_order": "|q1 q2 q3 q4>, q1 most significant",
        "n_cases": len(rows),
        "max_statevector_error": worst,
        "all_matched": bool(
            all(r["predicted_m3"] == r["observed_m3"] for r in rows)
            and all(r["predicted_m4"] == r["observed_m4"] for r in rows)
            and worst < 1e-12
        ),
        "cases": rows,
    }
    with open(DATA_DIR / "bell_product_cases.json", "w") as fh:
        json.dump(payload, fh, indent=2)

    print(f"{'first':>5} x {'second':<6} -> (m3,m4)  pred   err")
    for r in rows:
        print(
            f"{r['first_pair']:>5} x {r['second_pair']:<6} -> "
            f"({r['observed_m3']},{r['observed_m4']})     "
            f"({r['predicted_m3']},{r['predicted_m4']})   {r['statevector_error']:.2e}"
        )
    print(f"\nmax statevector error over all 16 cases: {worst:.3e}")
    print(f"all matched: {payload['all_matched']}")
    return 0 if payload["all_matched"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
