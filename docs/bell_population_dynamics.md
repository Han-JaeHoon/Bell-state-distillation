# Bell-population dynamics under repeated noisy purification

What happens to the four Bell populations of the retained pair when a noisy
purification round is applied over and over.  Reference run: `eps = 0.15`,
`q = 0.02`, `n = 0..50`, plus a 5000-round Type 5 run.

Reproduce (or sweep) with

```
python scripts/bell_population_dynamics.py --eps 0.15 --q 0.02 --rounds 50 --long-rounds 5000
```

## 1. Setup

Initial state, fixed for every protocol:

```
rho_0(eps) = (1-eps) |Phi+><Phi+| + eps I/4
p_Phi+ = 1 - 3 eps / 4 = 0.8875      p_Phi- = p_Psi+ = p_Psi- = eps / 4 = 0.0375
```

Noise: after **every** CNOT, on that CNOT's two qubits, the two-qubit
**replacement** depolarizing channel

```
D_q(rho) = (1-q) rho + q [ I_ij/4 (x) Tr_ij(rho) ]
```

The parent repository calls this strength `eps2` and implements it through
`noisy_bell_state.global_depol_kraus`; this package calls it `p` with
`convention="replace"`.  They are the same channel at the same numerical value,
checked on random 2-qubit and 5-qubit states (§3).  It is **not** the Pauli
depolarizing convention, which is the same family only after
`p_replace = 16 p_pauli / 15`.

One round consumes two identical copies of the current state:

| family | round |
|---|---|
| Type 3 / 4 / 5 (5 qubits) | `tau_A = Tr_{a,B}[(Z_a (x) I) E_q(\|0><0\|_a (x) rho (x) rho)]`, then `rho -> tau_A / Tr tau_A`.  `Tr tau_A` is a **parity weight, not a probability**. |
| 4Q (4 qubits) | noisy 5-CNOT + H on `rho (x) rho`, project `(q3,q4)` onto `(0,0)`, trace out, normalise.  Here the trace **is** the per-round success probability. |

## 2. Which map each protocol uses

| protocol | CNOTs | round map used | Bell-diagonal manifold |
|---|---|---|---|
| Type 3 | 16 | full density matrix (parent `iterated_noisy_pqec`), cross-checked against the verified exact map | preserved |
| Type 4 | 14 | same | preserved |
| Type 5 | 14 | **full density matrix only** | **not preserved** |
| 4Q | 5 | full density matrix (`noise` + `measurement`), cross-checked against the verified exact map | preserved |

Type 5 has no Bell-diagonal reduced map and none was used or invented.  Its
learned gate sequence also reproduces the ideal gadget unitary only to
`2.04e-7` in Frobenius norm, so a calibrated variant (`Type5cal`, same gates,
same noise, residual removed) is run beside it purely as a diagnostic.

## 3. Verification

Everything below was executed, not asserted; the values are in
`results/data/bell_population_dynamics/summary.json`.

| check | result |
|---|---|
| initial populations vs `1-3eps/4, eps/4, eps/4, eps/4` | max error `0` |
| `F_0` | `0.8875` |
| CNOT counts | Type 3 **16**, Type 4 **14**, Type 5 **14**, 4Q **5** — all as expected |
| `q` = parent `eps2`, 2-qubit Kraus vs replacement channel | `4.4e-16` |
| `q` = parent `eps2`, 5-qubit channel after a CNOT | `6.9e-18` |
| compilation residual vs the ideal gadget unitary | Type 3 `1.6e-15`, Type 4 `4.4e-15`, **Type 5 `2.0e-7`** |
| exact Bell-diagonal map vs full density matrix, 50 rounds | Type 3 `5.6e-16`, Type 4 `4.4e-16`, 4Q `3.3e-16` |
| `\|Tr rho_n - 1\|` | `<= 3.3e-16` for all protocols |
| anti-Hermitian part | `0` at every round |
| smallest eigenvalue over the run | Type 3 `1.31e-2`, Type 4 `1.02e-2`, Type 5 `6.75e-3`, 4Q `5.35e-3` — all positive |
| `\|sum_i p_i - 1\|` | `<= 4.4e-16` |
| `C_Bell` over 50 rounds | Type 3 `3.9e-30`, Type 4 `9.5e-28`, 4Q `1.5e-33`, **Type 5 `1.8e-6`** |

The quadratic round map amplifies any floating-point anti-Hermitian residue by
exactly x2 per round, so each iterate is projected onto the Hermitian manifold
(an exact-arithmetic identity) and the discarded norm is recorded; it stayed at
`0` throughout.

## 4. What the numbers show

### F per round

| protocol | `F_0` | `F_1` | `F_2` | `F_5` | `F_10` | `F_50` | long time |
|---|---|---|---|---|---|---|---|
| Type 3 | 0.8875 | 0.9490413223 | 0.9553473282 | 0.9558024372 | 0.9558025618 | 0.9558025618 | 0.9558025618 |
| Type 4 | 0.8875 | 0.9583945214 | 0.9641202649 | 0.9643810955 | 0.9643811149 | 0.9643811149 | 0.9643811149 |
| Type 5 | 0.8875 | 0.9702463653 | 0.9749440676 | 0.9750267249 | 0.9750267217 | 0.9750265514 | **0.4056964814** |
| 4Q | 0.8875 | 0.9705881195 | 0.9783819971 | 0.9787757280 | 0.9787757599 | 0.9787757599 | 0.9787757599 |

### Which equalities survive

The input has `p_Phi- = p_Psi+ = p_Psi-`.  After one noisy round:

* **Type 3 / Type 4** keep `p_Psi+ = p_Psi-` exactly (to `1e-14`) and push
  `Phi-` up: this is the invariant plane `y = -x` of the Pauli coordinates.
  The dominant error component is `Phi-`.
* **4Q** keeps `p_Phi- = p_Psi-` exactly and pushes `Psi+` up: the invariant
  plane `y = -z`.  The dominant error component is `Psi+`.
* **Type 5** keeps **none** of the three equalities.  At `n = 1` already
  `(Phi-, Psi+, Psi-) = (0.008378, 0.010835, 0.010541)` — three different
  numbers — and off-diagonal Bell coherence appears at `C_Bell = 1.6e-6`.

### Type 5: plateau, escape, asymptote

| | value |
|---|---|
| plateau `F` (`n ~ 2..300`) | `0.974944` |
| `C_Bell` growth on the plateau | `x1.03234` per round (e-folding ~31 rounds) |
| escape (first round `F` falls 0.01 below the plateau) | `n = 321` |
| `C_Bell` at escape | `0.0208` |
| `F` at `n = 5000` | `0.405696` |
| populations at `n = 5000` | `(0.40570, 0.08557, 0.13612, 0.37262)` |
| `C_Bell` at `n = 5000` | `0.6508` |

The escape is **not** caused by the `2e-7` compilation residual.  `Type5cal`,
with that residual removed, escapes at the *same* round `321` with the same
growth rate `1.032348`; and raw Type 5 at `q = 0` does not escape at all
(`F -> 1` and `C_Bell` only reaches `5e-10` by `n = 3000`).  The seed is the
CNOT noise: at `q = 0.02` the learned circuit's noise placement produces
`C_Bell = 1.6e-6` after a single round, and repeated application grows it
exponentially until the high-fidelity fixed point is lost.

### Per-round weight at `n = 1`

Type 3 `0.6126`, Type 4 `0.6295`, Type 5 `0.6214` — these are parity weights
`Tr(tau_A)`, **not** success probabilities.  4Q `0.7465`, which **is** a success
probability, rising to `0.8989` once the state settles.

## 5. Figures

All under `results/figures/`:

| file | content |
|---|---|
| `bellpop_stacked_<protocol>.png` | Figure A: stacked populations, `Phi+` at the bottom so the bottom band's height is `F_n`; second panel zooms on the error bands |
| `bellpop_lines_<protocol>.png` | Figure B: the four populations as lines; second panel isolates the error components and their splitting |
| `bellpop_overview_F.png` | `F_n` for all four protocols on one axis |
| `bellpop_type5_offdiagonal.png` | `C_Bell(n)`, short and long run |
| `bellpop_type5_longtime_linear.png`, `bellpop_type5_longtime_log.png` | long-time `F_n`, linear `n` and `log10(n+1)` |
| `bellpop_type5_vs_calibrated.png` | Type 5 vs Type5cal — the escape diagnosis |

## 6. Data and provenance

`results/data/bell_population_dynamics/` holds one CSV per protocol (round,
four populations, their sum, `F`, `C_Bell`, the per-round weight and the state
diagnostics), `summary.json` with the tables and every check above, and
`metadata.json` with the commit and the parent-repository provenance: the
parent commit, the revision the round map was taken from, and a SHA-256 for
every circuit file whose contents entered the calculation.  The parent
repository is only **read** — the round-map file is extracted with `git show`
into a temporary directory; no checkout, branch switch or write is performed.

## 7. Scope

This note is about the observed behaviour of the repeated dynamics at one
`(eps, q)` point.  It does not define or compute any threshold, and it is not a
statement about arbitrary Bell-diagonal inputs — only about the Bell-isotropic
initial line.
