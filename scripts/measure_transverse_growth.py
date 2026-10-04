"""Transverse stability of the Bell-diagonal manifold for the Type-3 round.

The threshold search in verify_type3_qabs_circuit.py works on exactly
Bell-diagonal inputs, so it never leaves the manifold except through
round-off.  This script asks the separate question the round-off study
raised: if a small Bell off-diagonal component is present, does the Type-3
round contract or amplify it?

Method
------
rho_0 = isotropic_rho(eps), then a traceless Hermitian off-diagonal seed of
amplitude `seed` is added in the Bell basis (the Phi+ / Phi- coherence).  The
exact circuit-level round is iterated and

    a_n = sqrt(C_off(rho_n)),   C_off = sum_{i != j} |<B_i|rho|B_j>|^2

is recorded.  The growth rate is the geometric mean of a_{n+1}/a_n over a
window taken after the first few rounds, so the number reported is the
asymptotic amplitude multiplier per round, not a transient.

Nothing here feeds the threshold: the populations stay Bell-diagonal to
round-off in the actual threshold runs, and the binding violation there sits
at round 0.  This measures only how long a trajectory started slightly off
the manifold stays near it.
"""

from __future__ import annotations

import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verify_type3_qabs_circuit import (  # noqa: E402
    BELL, BELL_ORDER, bell_offdiagonal_weight, isotropic_rho, one_round,
    q_abs_full,
)

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(HERE, "results", "type3_qabs")

SEED = 1e-12          # seed amplitude: far above round-off, far below the state
N_ROUNDS = 60         # iterations per trajectory
FIT_FROM, FIT_TO = 10, 40   # window for the geometric-mean rate


def seeded_rho(eps: float, seed: float, pair=("Phi+", "Phi-")) -> np.ndarray:
    """rho_0(eps) plus a traceless Hermitian Bell off-diagonal perturbation."""
    u, v = (BELL[pair[0]], BELL[pair[1]])
    delta = np.outer(u, v.conj()) + np.outer(v, u.conj())
    return isotropic_rho(eps) + seed * delta


def growth_rate(eps: float, q: float, seed=SEED, n_rounds=N_ROUNDS,
                fit_from=FIT_FROM, fit_to=FIT_TO) -> dict:
    """Geometric-mean amplitude multiplier per round for the off-diagonal part."""
    rho = seeded_rho(eps, seed)
    amps = [float(np.sqrt(bell_offdiagonal_weight(rho)))]
    for _ in range(n_rounds):
        rho, _info = one_round(rho, q)
        amps.append(float(np.sqrt(bell_offdiagonal_weight(rho))))
    a = np.array(amps)
    lo, hi = fit_from, min(fit_to, len(a) - 1)
    if a[lo] <= 0.0 or a[hi] <= 0.0 or hi <= lo:
        rate = float("nan")
    else:
        rate = float((a[hi] / a[lo]) ** (1.0 / (hi - lo)))
    return {"rate": rate, "a_initial": a[0], "a_final": a[-1],
            "amps": a, "locked": bool(a[hi] == a[lo])}


def sweep_at_threshold(eps_values, margin=1e-6) -> list:
    """Rate at q = q_abs(eps) - margin, i.e. just inside the suppressing side."""
    rows = []
    for eps in eps_values:
        qa = q_abs_full(eps)["q_abs"]
        q = max(qa - margin, 0.0)
        g = growth_rate(eps, q)
        rows.append({"epsilon": eps, "q_abs": qa, "q": q,
                     "growth_rate": g["rate"], "a_initial": g["a_initial"],
                     "a_final": g["a_final"], "locked": g["locked"]})
        print(f"    eps={eps:<5g} q={q:.12f}  rate={g['rate']:.6f}"
              f"  a: {g['a_initial']:.3e} -> {g['a_final']:.3e}"
              + ("  [fp-locked]" if g["locked"] else ""))
    return rows


def sweep_in_q(eps_values, q_values) -> list:
    """Rate as a function of q at fixed eps -- is it a function of q alone?"""
    rows = []
    for eps in eps_values:
        for q in q_values:
            g = growth_rate(eps, q)
            rows.append({"epsilon": eps, "q": q, "growth_rate": g["rate"],
                         "locked": g["locked"]})
            print(f"    eps={eps:<5g} q={q:<6g}  rate={g['rate']:.6f}"
                  + ("  [fp-locked]" if g["locked"] else ""))
    return rows


def write_csv(path: str, rows: list, cols: list) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


def main() -> int:
    print(__doc__.strip().splitlines()[0])
    print(f"\n  seed amplitude {SEED:.0e}, {N_ROUNDS} rounds, "
          f"rate from rounds {FIT_FROM}..{FIT_TO}\n")

    eps_values = (0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20,
                  0.25, 0.30, 0.40, 0.50, 0.60)
    print("  rate just below the threshold  (q = q_abs(eps) - 1e-6)")
    at_thr = sweep_at_threshold(eps_values)

    print("\n  rate versus q at fixed eps")
    in_q = sweep_in_q((0.15, 0.50), (0.0, 0.005, 0.01, 0.02, 0.03, 0.05))

    write_csv(os.path.join(OUTDIR, "type3_transverse_growth.csv"), at_thr,
              ["epsilon", "q_abs", "q", "growth_rate", "a_initial", "a_final",
               "locked"])
    write_csv(os.path.join(OUTDIR, "type3_transverse_growth_vs_q.csv"), in_q,
              ["epsilon", "q", "growth_rate", "locked"])

    print("\n  results/type3_qabs/type3_transverse_growth.csv")
    print("  results/type3_qabs/type3_transverse_growth_vs_q.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
