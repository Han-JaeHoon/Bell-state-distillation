# Type 4 repeated dynamics along the Bell-isotropic initial line

How the Bell populations move under repeated Type 4 rounds, and how the answer
depends on the strength `eps` of the input mixing.  Reference run: `q = 0.02`,
`eps in {0.02, 0.05, 0.10, 0.15, 0.20, 0.30}`, `n = 0..50`.

```
python scripts/type4_epsilon_sweep.py --q 0.02
python scripts/type4_epsilon_sweep.py --q 0.12 --eps 0.05 0.2 0.5   # sweep q
```

## 1. What was used

The dynamics come from the **verified exact Bell-diagonal map**
`pqec_distill.analytic_exact_map.analytic_map_type4`, which was validated
row-by-row against the frozen circuit-only reference.  No map was re-derived or
guessed.  A full density-matrix simulation of the actual 14-CNOT circuit runs
alongside it as a cross-check at every `eps`.

`q` is the two-qubit **replacement** depolarizing strength applied after every
CNOT on that CNOT's two qubits — the same convention, at the same numerical
value, as the parent repository's `eps2`.

## 2. Checks

| check | result |
|---|---|
| Type 4 CNOT count | **14** |
| `q` = parent `eps2`, 2-qubit Kraus / 5-qubit channel | `4.4e-16` / `6.9e-18` |
| `\|sum_i p_i - 1\|`, all `eps` | `<= 2.2e-16` |
| smallest population, all `eps` | `0.0050` (`eps=0.02`), `0.0102` otherwise — no negatives |
| `max \|p_Psi+ - p_Psi-\|`, all `eps` | `<= 2.8e-17` |
| exact map vs full density matrix, 50 rounds | `<= 6.7e-16` in populations |
| Bell off-diagonal weight in the full simulation | `<= 9.5e-28` — the state stays Bell-diagonal |

## 3. Summary table (`q = 0.02`)

| `eps` | `F_0` | `F_1` | `F_2` | `F_5` | `F_10` | `F_50` | `F_inf` | `F_inf - F_0` | rounds to converge |
|---|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.9850000000 | 0.9650787098 | 0.9644105150 | 0.9643811172 | 0.9643811149 | 0.9643811149 | 0.964381114877 | **−0.0206188851** | 8 |
| 0.05 | 0.9625000000 | 0.9643183787 | 0.9643793783 | 0.9643811148 | 0.9643811149 | 0.9643811149 | 0.964381114877 | +0.0018811149 | 7 |
| 0.10 | 0.9250000000 | 0.9620827465 | 0.9642850504 | 0.9643811078 | 0.9643811149 | 0.9643811149 | 0.964381114877 | +0.0393811149 | 8 |
| 0.15 | 0.8875000000 | 0.9583945214 | 0.9641202649 | 0.9643810955 | 0.9643811149 | 0.9643811149 | 0.964381114877 | +0.0768811149 | 9 |
| 0.20 | 0.8500000000 | 0.9529234534 | 0.9638543399 | 0.9643810756 | 0.9643811149 | 0.9643811149 | 0.964381114877 | +0.1143811149 | 9 |
| 0.30 | 0.7750000000 | 0.9349802959 | 0.9627926487 | 0.9643809953 | 0.9643811149 | 0.9643811149 | 0.964381114877 | +0.1893811149 | 9 |

"rounds to converge" is the first round after which the populations stay within
`1e-12` of the fixed point.

Converged populations, **identical for every `eps`** (max spread across the six
runs: `0.00e+00`):

```
(p_Phi+, p_Phi-, p_Psi+, p_Psi-) = (0.964381114877, 0.015188365715, 0.010215259704, 0.010215259704)
```

## 4. Observations

1. **`p_Psi+ = p_Psi-` holds.**  Maximum deviation `2.8e-17` over every round and
   every `eps` — round-off only.  This is the invariant plane `y = -x` of the
   Pauli coordinates, which Type 4 preserves exactly.
2. **The isotropic error splits in one round.**  At `n = 1` already,
   `p_Phi- != p_Psi+`; the relative spread of the three error components is
   0.21 (`eps = 0.30`) to 0.42 (`eps = 0.02`).  `Phi-` is the component that
   grows relative to the other two, at every `eps`.
3. **Convergence takes 7–9 rounds** to `1e-12`, and about 3 rounds to `1e-4`.
   The residual `|F_n - F_inf|` falls geometrically by a factor `0.0425` per
   round, essentially the same rate for every `eps`.
4. **Every `eps` reaches the same fixed point**, bit-identically.
5. **`F_inf` does not depend on `eps`.**  It is a property of the map at
   `q = 0.02` alone.  This matches the independent repeated-dynamics analysis,
   which found the Type 4 target branch at `q = 0.02` at
   `F = 0.96438111487671` with Jacobian spectral radius `0.04276005297530256` —
   the same fixed point and the same contraction rate measured in point 3.
6. **No — `F_inf > F_0` is not true for every `eps`.**  At `eps = 0.02` the
   input starts at `F_0 = 0.985`, *above* the fixed point, and every round
   pushes it **down** to `0.964381`; the trajectory is monotonically
   decreasing.  `F_0 = F_inf` at `eps = 0.0474918468`, so at `q = 0.02` inputs
   cleaner than that are degraded by the protocol and inputs noisier than that
   are improved.  All five other `eps` values improve.
7. **Intuition for a later threshold discussion** (nothing is defined here).
   Two separate questions are visible in this data and should not be conflated:
   whether the fixed point is *useful* (where `F_inf(q)` sits — e.g. above 1/2,
   or above some target), and whether *this particular input* is improved
   (whether `F_0` is below or above `F_inf(q)`).  At `q = 0.02` the second
   question has an answer that depends on `eps` even though the first does not.
   The `eps`-independence of `F_inf` also means a single number per `q`
   characterises the attractor, while the break-even `eps` is a derived
   quantity, not an independent one.

## 5. Files

Figures in `results/figures/`:

| file | content |
|---|---|
| `type4_eps_<eps>_stacked.png` | Figure A, per `eps`: stacked populations, `Phi+` at the bottom so the bottom band's height is `F_n`; second panel zooms on the error bands |
| `type4_eps_<eps>_lines.png` | Figure B, per `eps`: the four populations as lines; second panel isolates the error components.  `Psi-` is drawn dashed on top of `Psi+`, so their exact coincidence is visible rather than looking like a missing curve |
| `type4_F_overlay_q0p02.png` | Figure C: `F_n` for all six `eps`; second panel shows `\|F_n - F_inf\|` on a log axis |
| `type4_summary_vs_eps_q0p02.png` | Figure D: `F_0`, `F_1`, `F_50` against `eps`, with `F_inf` as a reference line |

Data in `results/data/type4_epsilon_sweep/`: one CSV per `eps`, plus
`summary.csv` and `summary.json` (tables, fixed point, all checks, commit).

Colour note: `eps` is a magnitude, so the overlay uses one hue light-to-dark
rather than categorical hues.  The documented blue ramp holds five steps that
clear the ordinal lightness-gap gate on a light surface; six `eps` values need
a sixth step, so the darkest pair is closer than that gate wants.  Every curve
therefore carries its own marker and a direct label, and no two are separated
by lightness alone.
