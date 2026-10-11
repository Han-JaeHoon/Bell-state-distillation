# `q_th(epsilon)`: the all-round fidelity-improvement threshold, from the full circuits

**Status: circuit-level numerical observation. This is NOT an analytic
derivation.**

The primary path uses no analytic recurrence, no closed-form map, no symbolic
equations, no previously derived one-round `q_th` equation, and no stored
`q_abs` / `q_rel` value as a hint or bracket. The three protocol
implementations import nothing from `pqec_distill`. The hierarchy and
protocol comparisons happen once, in clearly separated SECONDARY steps, after
the primary tables are frozen.

Reproduce with

```
pip install python-flint            # optional, for the high-precision route
python scripts/verify_qth_full_circuit.py --time-budget 1500
pytest -q tests/test_type3_qth_circuit.py tests/test_type4_qth_circuit.py \
          tests/test_4q_qth_circuit.py
```

Recorded run: 3738 s, Python 3.11.15, NumPy 2.4.6, python-flint 0.9.0.

---

## 1. The quantity measured

With `F_n = <Phi+| rho_n |Phi+>` on the effective two-qubit state of round `n`,

```
q_th_full(eps)  = sup { q : F_(n+1) > F_n  for EVERY n >= 0 }     <-- the target
q_th_first(eps) = sup { q : F_1 > F_0 }                           <-- diagnostic
```

Whether the two coincide is the central question and is an **output**. `q_th`
is kept strictly distinct from `q_abs`, `q_rel`, `q_fixed`, `q_target`,
saddle-node thresholds and `F* = 1/2` crossings: only `F_(n+1) > F_n` enters
the predicate anywhere.

`F_n` is computed from an independently constructed `Phi+` projector
`(I + XX)(I + ZZ)/4`, never from a coordinate formula; it agrees with
`<Phi+|rho|Phi+>` to `1.1e-16` and with the `Phi+` Bell population to `1e-14`
on random states.

## 2. The three frozen circuits

All three transcriptions are verified against the repository source at run
time, textually (the frozen scripts are parsed, never imported).

| | wires | structure | CNOTs | effective state |
|---|---|---|---|---|
| **Type 3** | `[a, A1, A2, B1, B2]` | `H_a`, `Fredkin(a;1,3)`, `Fredkin(a;2,4)`, `H_a`, each Fredkin the Clifford+T `_c2;_tof;_c2` block | **16** | parity-weighted Pauli tomography |
| **Type 4** | same | the Qiskit-resynthesised `{u, cx}` sequence, 36 operations | **14** | parity-weighted Pauli tomography |
| **4Q** | `[q1, q2, q3, q4]` | `CNOT q3->q4, q2->q4, q1->q4, q3->q2, q3->q1`, then `H(q3)` | **5** | physical postselection |

Provenance checks, all `True`:

| transcription | checked against |
|---|---|
| Type 3 | `scripts/verify_type3_qabs_circuit.py` `fredkin_ops` (16 CNOTs) |
| Type 4 | `scripts/verify_type4_qabs_circuit.py`, `scripts/verify_type4_qrel_circuit.py`, and `PQEC-Operational-Threshold/pqec_resynth_noise.py` `GATES` |
| 4Q | `src/pqec_distill/circuit.py` `CNOT_SEQUENCE` and `scripts/verify_4q_qabs_circuit.py` |

**Type 3 / Type 4 read-out.** Ancilla `a = 0`, retained data `(1, 2)`,
discarded `(3, 4)`. All sixteen coefficients of the unnormalised effective
operator are **physical circuit expectation values**

```
Delta_P = < Z_a (x) P_(1,2) (x) I_(3,4) >      on the final five-wire state
T       = 1/4 sum_P Delta_P P ,      rho_(n+1) = T / Tr T
```

A test rebuilds `T` from the sixteen numbers independently and checks each
`Delta_P` against `Tr[(Z_a (x) P (x) I) sigma]` to `1e-13`.

**4Q read-out.** Measure `(q3, q4)`, keep the `(0,0)` branch, trace it out,
normalise by `P_succ`; two copies of that exact 4x4 state feed the next round.

No Bell projection, twirling, isotropic replacement, Bell-diagonal truncation
or invariant-plane projection anywhere, in any protocol.

## 3. Noise

After **every** CNOT on wires `(i, j)`:
`D_q(sigma) = (1-q) sigma + q [I_ij/4 (x) Tr_ij(sigma)]`, the repository's
two-qubit **replacement** depolarizing channel — never converted to a 15-Pauli
probability. Single-qubit gates, preparation, measurement and classical
processing are ideal. `q = 0` is the exact identity channel, and at `q = 1`
the pair's reduced state is exactly `I/4` with the other wires untouched
(both verified per protocol, to `1e-12`).

## 4. The three-valued predicate and its tolerances

| name | value | role |
|---|---|---|
| `VIOL_TOL` | `1e-12` | FAIL iff `Delta F_n < -VIOL_TOL` **while the orbit is trusted** |
| `STEP_TOL` | `1e-14` | full-state convergence |
| `FID_TOL` / `CONV_RUN` | `1e-14` / 10 | fidelity convergence: `\|Delta F\|` below `FID_TOL` for 10 consecutive rounds |
| `OFFDIAG_TRUST` | `1e-18` | Bell off-diagonal weight above which the orbit is no longer trusted |
| `DENOM_FLOOR` | `1e-9` | Type 3/4 parity-weight safety floor |
| `N_MAX` / `N_MAX_HP` | 400 / 600 | maximum rounds (double / high precision); **600 actually used** |
| `BISECT_TOL` | `1e-10` | bisection target |
| q scan | `[0, 0.9]`, 91 points | coarse scan |

The second convergence criterion is necessary, not cosmetic: for Type 3/4 the
growing transverse mode keeps the full state moving long after the fidelity
has settled, so `STEP_TOL` alone would never fire.

Statuses: **PASS** (converged while trusted), **FAIL** (`Delta F` below
`-VIOL_TOL` while trusted), **AMBIGUOUS** (transverse contamination first),
**UNSAFE** (parity denominator below the floor), **MAXROUNDS**.

## 5. The transverse instability, and how it was removed

Type 3 and Type 4 amplify the Bell-off-diagonal direction. The Bell-isotropic
input is exactly Bell diagonal, so in double precision the transverse sector
starts at round-off and grows geometrically. For Type 3 at `eps = 2/3`,
`q = 0.1192`:

| round | 20 | 40 | 60 | 80 | 100 | 150 |
|---|---|---|---|---|---|---|
| `C_off` | 8.3e-27 | 5.6e-21 | 3.7e-15 | 2.5e-09 | 1.6e-03 | 2.7e-01 |
| `Delta F` | +1.6e-04 | +4.1e-07 | +1.0e-09 | -4.4e-10 | **-2.9e-04** | **-6.6e-08** |

The large negative `Delta F` from round ~80 onwards occurs when the state is no
longer Bell diagonal at all: it is **numerical contamination, not physics**.
Such a trajectory is reported AMBIGUOUS and never FAIL.

The resolution is an optional **ball-arithmetic route** (`python-flint`, 400
bits) that repeats ROUTE A exactly. There the transverse sector is certified
to be zero: `C_off < 1e-100` and the per-round ball radius is `1.4e-116`.
Two implementation notes, both stated because they are assumptions:

* ball radii are re-centred after each round (`tighten`). Ball arithmetic
  propagates a pessimistic bound that widens by ~`1e4` per round, so at 200
  bits the enclosure is useless after ~15 rounds. Re-centring keeps the
  per-round enclosure as a certified bound on that round — which is what the
  reported radius is — while the accumulated true error behaves like round-off
  at 400 bits, some sixty orders of magnitude below the double-precision seed
  this route exists to remove.
* the frozen Type-4 U3 angles are float64 renderings of multiples of `pi/4`.
  The high-precision route snaps them to the exact multiple; the snap residual
  is `7.77e-16` over 66 angles. At double precision the verbatim floats are
  used.

`Delta F` is formed **in high precision** (differencing two doubles would floor
it at `1e-16` and make the convergence test meaningless near the fixed point).

The predicate is hybrid: double precision first, ball arithmetic only where
double precision is undecided.

## 6. Independent implementation routes

| | ROUTE A | ROUTE B |
|---|---|---|
| CNOT | `\|0><0\| (x) I + \|1><1\| (x) X` by Kronecker product | the computational-basis permutation |
| 1-qubit gates | Kronecker product | explicit index arithmetic |
| propagation | merged unitary program | gate by gate, no merging |
| channel | einsum partial trace + identity re-insertion | independently coded einsum, own index letters |
| Type 3/4 effective state | **Pauli tomography** from 16 circuit expectation values | **parity-weighted partial trace** `Tr_B[<0\|sigma\|0> - <1\|sigma\|1>]` |
| 4Q effective state | diagonal sub-block | explicit 16x16 projector + einsum partial trace |

Worst disagreement over 27 `(protocol, eps, q)` points x 5 rounds:

| quantity | max difference |
|---|---|
| retained `rho_(n+1)` | 1.728e-15 |
| `F` | 6.661e-16 |
| `Delta F` | 4.441e-16 |
| Bell populations | 7.772e-16 |
| normalisation denominator / `P_succ` | 6.661e-16 |
| ROUTE A float64 vs 400-bit ball arithmetic, 7 rounds | 2.475e-15 |

## 7. Main results

`F_in = 1 - 3 eps/4`. The interval is `[largest certified PASS, smallest
certified FAIL]`; `estimate` is its midpoint.

### 7.1 4Q postselected (5 CNOTs)

| eps | `q_th_full` | `q_th_first` | full − first | binding round | rounds to convergence | `P_succ` min |
|---|---|---|---|---|---|---|
| 0.01 | 0.0073349383 | 0.0073349383 | +3.7e-11 | **0** | 7 | |
| 0.02 | 0.0143538144 | 0.0143538143 | +3.7e-11 | **0** | 9 | |
| 0.05 | 0.0337019472 | 0.0337019472 | +3.7e-11 | **0** | 12 | |
| 0.10 | 0.0611603567 | 0.0611603567 | +3.7e-11 | **0** | 16 | |
| 0.15 | 0.0838639422 | 0.0838639422 | +3.7e-11 | **0** | 20 | |
| 0.20 | 0.1028193437 | 0.1028193437 | +3.7e-11 | **0** | 25 | |
| 0.30 | 0.1321146134 | 0.1321146134 | +3.7e-11 | **0** | 38 | |
| 0.40 | 0.1526881335 | 0.1526881335 | +3.7e-11 | **0** | 59 | |
| 0.50 | 0.1665413712 | 0.1665413711 | +3.7e-11 | **0** | 99 | |
| 0.60 | 0.1747745780 | 0.1747745780 | +3.7e-11 | **0** | 184 | |
| 2/3 | 0.1774161043 | 0.1774161042 | +3.7e-11 | **0** | 279 | |
| 0.75 | **0.1775910934** | 0.1775910934 | +3.7e-11 | **0** | 310 | |
| 0.90 | 0.1685871538 | 0.1685871538 | +3.7e-11 | **0** | 153 | |

Every row trusted, bracket width `7.45e-11`, exactly **one** predicate sign
change. `+3.7e-11` is the bisection half-width: the certified PASS endpoint
equals `q_th_first` bit for bit at every epsilon. Minimum `P_succ` over all
rows: `0.2534`.

### 7.2 5Q Type 3 (16 CNOTs)

| eps | `q_th_full` | `q_th_first` | full − first | binding round | `Delta F` at upper fail | rounds | trusted |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.0034500423 | 0.0035069580 | **−5.69e-05** | **3** | −1.00e-12 | 6 | yes |
| 0.02 | 0.0068494503 | 0.0069687189 | **−1.19e-04** | **4** | −1.00e-12 | 7 | yes |
| 0.05 | 0.0168001983 | 0.0170787832 | **−2.79e-04** | **5** | −1.00e-12 | 9 | yes |
| 0.10 | 0.0325328549 | 0.0329909427 | **−4.58e-04** | **7** | −1.00e-12 | 11 | yes |
| 0.15 | 0.0471557452 | 0.0477026618 | **−5.47e-04** | **8** | −1.00e-12 | 13 | yes |
| 0.20 | 0.0606205200 | 0.0611943332 | **−5.74e-04** | **10** | −1.00e-12 | 15 | yes |
| 0.30 | 0.0839793389 | 0.0844986723 | **−5.19e-04** | **13** | −1.00e-12 | 20 | yes |
| 0.40 | 0.1025474464 | 0.1029486671 | **−4.01e-04** | **16** | −1.01e-12 | 26 | yes |
| 0.50 | 0.1163873782 | 0.1166566045 | **−2.69e-04** | **21** | −1.03e-12 | 46 | yes |
| 0.60 | 0.1255728516 | 0.1257147018 | **−1.42e-04** | **26** | −1.49e-12 | 66 | yes |
| 2/3 | 0.1291134401 | 0.1291737116 | **−6.03e-05** | **28** | −1.42e-12 | 49 | yes |
| 0.75 | **[0.1299625217, 0.1305267722]** | 0.1305267721 | −2.82e-04 | 0 | −9.53e-11 | 600 | **no** |
| 0.90 | 0.1239891087 | 0.1239891087 | +3.7e-11 | **0** | −1.18e-12 | 219 | yes |

### 7.3 5Q Type 4 (14 CNOTs)

| eps | `q_th_full` | `q_th_first` | full − first | binding round | `Delta F` at upper fail | rounds | trusted |
|---|---|---|---|---|---|---|---|
| 0.01 | 0.0041160944 | 0.0042709354 | **−1.55e-04** | **3** | −1.00e-12 | 6 | yes |
| 0.02 | 0.0081669774 | 0.0085117971 | **−3.45e-04** | **4** | −1.00e-12 | 7 | yes |
| 0.05 | 0.0201841348 | 0.0210461683 | **−8.62e-04** | **5** | −1.00e-12 | 9 | yes |
| 0.10 | 0.0397172856 | 0.0412625469 | **−1.55e-03** | **7** | −1.00e-12 | 11 | yes |
| 0.15 | 0.0586016929 | 0.0605516153 | **−1.95e-03** | **8** | −1.00e-12 | 13 | yes |
| 0.20 | 0.0766713572 | 0.0788172615 | **−2.15e-03** | **10** | −1.00e-12 | 15 | yes |
| 0.30 | 0.1098226314 | 0.1119110493 | **−2.09e-03** | **13** | −1.00e-12 | 19 | yes |
| 0.40 | 0.1381482143 | 0.1398609783 | **−1.71e-03** | **17** | −1.00e-12 | 35 | yes |
| 0.50 | 0.1608530627 | 0.1620746125 | **−1.22e-03** | **22** | −1.00e-12 | 44 | yes |
| 0.60 | 0.1773125336 | 0.1780271054 | **−7.15e-04** | **29** | −1.04e-12 | 61 | yes |
| 2/3 | 0.1845431818 | 0.1849266187 | **−3.83e-04** | **36** | −1.02e-12 | 78 | yes |
| 0.75 | **[0.1885644917, 0.1890354177]** | 0.1890354177 | −2.36e-04 | 0 | −5.77e-11 | 600 | **no** |
| 0.90 | 0.1823089591 | 0.1823089591 | +9.3e-12 | **0** | −1.10e-11 | 241 | yes |

### 7.4 Predicate structure: how many transitions?

Every coarse scan has exactly **one PASS -> FAIL boundary**. The extra label
transitions counted in `predicate_sign_changes` (2 or 3 for Type 3/4) are not
additional thresholds:

```
type3, eps = 0.40:   PASS [0, 0.11)   FAIL [0.11, 0.86)   UNSAFE [0.86, 0.9]
type4, eps = 0.40:   PASS [0, 0.14)   FAIL [0.14, 0.86)   UNSAFE [0.86, 0.9]
4q:                  PASS             FAIL                 (1 transition)
```

The `UNSAFE` region at `q >~ 0.86` is the Type 3/4 **parity-weight collapse**:
`Tr T` falls below the `1e-9` floor (e.g. `6.5e-11` at `q = 0.89`), so the
normalised effective state is not numerically meaningful there. It is far
above every threshold (all `<= 0.19`). Where an AMBIGUOUS band also appears it
straddles the single PASS/FAIL boundary and is resolved by the
high-precision route.

### 7.5 The two unresolved rows

| | interval | width | non-decisive q values | min `Delta F` in the band | rounds |
|---|---|---|---|---|---|
| Type 3, `eps = 0.75` | [0.12996252, 0.13052677] | 5.64e-04 | 22 | **+6.62e-15** | 600 |
| Type 4, `eps = 0.75` | [0.18856449, 0.18903542] | 4.71e-04 | 23 | **+6.83e-15** | 600 |

In the band, 600 high-precision rounds show `Delta F` **strictly positive** —
there is no hidden violation, the orbit is simply not yet converged (critical
slowing as `q -> q_th_first`). The bisection is two-sided: the PASS edge and
the FAIL edge are each tightened separately, so the reported interval is the
tightest decisive one. 37 of 39 rows are fully resolved to `7.45e-11`.

## 8. `q_th_full` versus `q_th_first`

This is the central output, and the answer is **protocol dependent**.

* **4Q**: `q_th_full = q_th_first` at every epsilon, to the bisection width
  `7.45e-11`, and the certified PASS endpoint equals `q_th_first` exactly.
  The binding round is **0**.
* **Type 3 / Type 4**: `q_th_full < q_th_first` **strictly**, by `6e-05` to
  `5.7e-04` (Type 3) and `1.5e-04` to `2.2e-03` (Type 4). The binding round is
  **not 0**; it grows with epsilon, reaching **28** (Type 3) and **36**
  (Type 4) at `eps = 2/3`.
* At `eps = 0.90` both Type 3 and Type 4 revert to binding round **0** with
  `q_th_full = q_th_first` within the bisection width. The `eps = 0.75` rows
  sit at that crossover and are the two unresolved ones.

### Tolerance dependence — important caveat

`q_th_first` is located where a steep first-round `Delta F` crosses
`-VIOL_TOL`, so it barely moves with the tolerance. `q_th_full` for Type 3/4
is located where a **late-round dip** crosses `-VIOL_TOL`, and that dip
deepens very slowly with `q`, so it inherits a much larger tolerance
dependence. Measured by re-bisecting the predicate at each tolerance
(`eps = 0.40`):

| protocol | `VIOL_TOL = 1e-10` | `1e-12` | `1e-14` | spread |
|---|---|---|---|---|
| 4Q | 0.1526881336 | 0.1526881335 | 0.1526881335 | **7.45e-11** |
| Type 3 | 0.1025476150 | 0.1025474463 | 0.1025474325 | **1.83e-07** |
| Type 4 | 0.1381505089 | 0.1381482143 | 0.1381479171 | **2.59e-06** |

So the Type 3/4 `q_th_full` values, although bisected to `7.45e-11`, carry a
definitional uncertainty of order `1e-7` to `1e-6` from the choice of
`VIOL_TOL`. That is still two to four orders of magnitude smaller than the gap
`q_th_first - q_th_full`, so the strict inequality is unaffected — but the
digits beyond the sixth should not be read as physical. `q_th_first` itself is
stable to `4.6e-11 ... 9.5e-11` over the same tolerance range.

## 9. Non-monotone fidelity trajectories

**107 cases** of `F_1 > F_0` followed by `F_(m+1) < F_m` at a later round `m`,
all in Type 3 and Type 4. **0 cases** of the opposite pattern (an initial drop
followed by recovery) in any protocol.

Deepest example, Type 3, `eps = 2/3`, `q = 0.1291134401`
(`results/qth/qth_nonmonotone_cases.csv`):

```
F_0 .. F_28 :  0.500000000000  0.500089464542  0.500236484142  0.500341890490
               0.500405709065  0.500441937492  ...  0.500485369231
               0.500485369239   <- F_28, the maximum
F_29        :  0.500485369237   <- Delta F = -1.418643e-12
Bell populations at the violation : 0.500485369239  0.183401393250
                                    0.158056618756  0.158056618756
Bell off-diagonal weight there    : 3.65e-23      (trust threshold 1e-18)
```

The fidelity rises monotonically for 28 rounds and then turns over. `C_off` is
five orders of magnitude below the trust threshold, so this is a **genuine
late-round violation**, not contamination. This is exactly the mechanism that
makes `q_th_full < q_th_first`.

## 10. Near-threshold trajectories

12 `(protocol, epsilon)` blocks at `eps in {0.05, 0.15, 0.40, 2/3}`, three `q`
values each (`q_th +- delta`, `delta >= 5e-4`), 60 rounds, 1126 recorded
rounds in `results/qth/qth_trajectories.csv`, plotted in
`qth_trajectories_F.*` and `qth_trajectories_dF.*` with `F_n`, `Delta F_n`,
the three non-target Bell populations and the off-diagonal weight. They show
the threshold being controlled by a late-round turnover for Type 3/4 and by
the first round for 4Q.

## 11. Full-state diagnostics

| | max trace error | max hermiticity residual | min eigenvalue | max `C_off` | min denominator / `P_succ` |
|---|---|---|---|---|---|
| Type 3 | 2.22e-16 | 0.0 | 2.17e-03 | 8.99e-21 | 6.77e-02 (parity weight) |
| Type 4 | 2.22e-16 | 0.0 | 2.07e-03 | 1.67e-18 | 3.41e-02 (parity weight) |
| 4Q | 2.22e-16 | 3.93e-17 | 1.88e-03 | 1.18e-32 | 2.53e-01 (`P_succ`) |

The effective states stay physical (positive, unit trace) along every threshold
trajectory. 4Q shows no transverse contamination at all; Type 3/4 reach the
trust threshold only where the high-precision route takes over.

## 12. Degenerate endpoints

Handled separately for all three protocols, never forced into an interior root.

* **`eps = 0`**: `F_0 = 1`. Fidelity cannot exceed one, so `F_1 > F_0` is
  impossible; the largest `Delta F_1` seen over the probes is `+2.22e-16`,
  i.e. round-off. Degenerate by definition, no root reported.
* **`eps = 1`**: `F_0 = 1/4`. Measured, not assumed: `Delta F_1 = 0.0` exactly
  at every probed `q` for all three protocols — the maximally mixed state is a
  fixed point, so the strict inequality fails there too. No root reported.

## 13. Negative controls

All 15 detected (`results/qth/qth_negative_controls.csv`). Each is placed at a
position a perturbation scan measured to be effective, because the scan turned
up a genuine structural fact:

| protocol | control | position | `max\|Delta rho\|` | `q_th_first` shift |
|---|---|---|---|---|
| Type 3 | A noise `+1e-3` | CNOT #8 | 3.93e-04 | −2.40e-04 |
| Type 3 | B channel removed | CNOT #16 | 2.02e-02 | +2.96e-02 |
| Type 3 | C CNOT reversed | op 9 | parity weight collapses | no root |
| Type 3 | D angle `+1e-3` | op 30 | 8.98e-05 | −7.23e-08 |
| Type 3 | F wrong parity sign | `(+1,+1)` | 2.38e-01 | −1.03e-01 |
| Type 4 | A / B / C / D / F | 7 / 7 / 7 / 8 / — | 3.99e-04 … collapse | up to +6.38e-02 |
| 4Q | A / B / C / D / E | 4 / 4 / 3 / 5 / `(0,1)` | 3.02e-04 … 2.90e-01 | up to +4.99e-02 |

**Perturbations the effective state annihilates** (a measured property of the
normalised estimator, reported rather than hidden):

```
type3   noise offset / channel removal on CNOTs  [3, 5, 6, 7, 11, 13, 14, 15]   (8 of 16)
type4   noise offset / channel removal on CNOTs  [2, 4, 5, 6, 9, 11, 12, 13]    (8 of 14)
type3, type4   the leading ancilla Hadamard (op 0)
4q      none -- every perturbation bites
```

For Type 3/4 the effective state is `T / Tr T`, so a perturbation that merely
rescales `T` cancels; exactly half the CNOT noise positions turn out to leave
the normalised read-out untouched. 4Q has no such position.

## 14. SECONDARY: `q_abs <= q_rel <= q_th`

Read only after the primary tables were frozen, from the already-frozen
results (`results/type{3,4}_q{abs,rel}/`, `results/4q_qabs/`, and the 4Q
`q_rel` study via `git show feature/4q-qrel-circuit-study:...`, read-only —
that branch is never merged or modified; a test checks every frozen file's
hash is unchanged).

**The hierarchy holds at all 39 compared `(protocol, epsilon)` points**, with
no violations. Representative values:

| protocol | eps | `q_abs` | `q_rel` | `q_th` | `q_th − q_rel` | `q_rel − q_abs` |
|---|---|---|---|---|---|---|
| Type 3 | 0.15 | 0.0395626315 | 0.0403462582 | 0.0471557452 | +6.81e-03 | +7.84e-04 |
| Type 3 | 0.40 | 0.0885675115 | 0.0925305820 | 0.1025474464 | +1.00e-02 | +3.96e-03 |
| Type 4 | 0.15 | 0.0478101298 | 0.0489816861 | 0.0586016929 | +9.62e-03 | +1.17e-03 |
| Type 4 | 0.40 | 0.1136410411 | 0.1205788680 | 0.1381482143 | +1.76e-02 | +6.94e-03 |
| 4Q | 0.15 | 0.0606654173 | 0.0626662258 | 0.0838639422 | +2.12e-02 | +2.00e-03 |
| 4Q | 0.40 | 0.1173995783 | 0.1265132944 | 0.1526881335 | +2.62e-02 | +9.11e-03 |

## 15. SECONDARY: protocol comparison

| eps range | ordering (high to low) |
|---|---|
| 0.01 … 0.50 | **4Q > Type 4 > Type 3** |
| 0.60 … 0.90 | **Type 4 > 4Q > Type 3** |

One crossing, between `eps = 0.50` and `eps = 0.60` (the grid does not
localise it further). Type 3 is lowest at every epsilon tested.

Maxima over the tested grid, all at `eps = 0.75`:

| protocol | max `q_th` | at |
|---|---|---|
| Type 4 | 0.188800 | `eps = 0.75` (interval row, see 7.5) |
| 4Q | 0.177591 | `eps = 0.75` |
| Type 3 | 0.130245 | `eps = 0.75` (interval row, see 7.5) |

The grid is coarse near the maximum and two of the three maxima fall on the
unresolved rows, so these locate the maxima only to the grid.

## 16. Conclusions

### VERIFIED IMPLEMENTATION FACT

* The three gate lists used here are the repository's own frozen
  specifications, checked textually against every available source: Type 3
  **16** CNOTs, Type 4 **14** CNOTs, 4Q **5** CNOTs.
* The noise is the repository's two-qubit replacement depolarizing channel
  after every CNOT, on exactly that CNOT's wires; `q = 0` is the exact
  identity and `q = 1` replaces the pair by `I/4` exactly.
* For Type 3/4 all sixteen coefficients of the effective operator are physical
  circuit expectation values `<Z_a (x) P (x) I>`, verified against an
  independent rebuild to `1e-13`; for 4Q the state is the physically
  postselected `(0,0)` branch.
* Effective states stay physical along every threshold trajectory (trace error
  `<= 2.22e-16`, min eigenvalue `>= 1.88e-03`).
* Two numerically independent implementation routes — including two different
  effective-state rules for Type 3/4 — agree to `1.73e-15` on the state and
  `6.66e-16` on the fidelity; double precision and 400-bit ball arithmetic
  agree to `2.48e-15`.
* Type 3 and Type 4 amplify the Bell-off-diagonal direction, so double-precision
  orbits stop being faithful after ~40–80 rounds; in 400-bit ball arithmetic
  the transverse sector is certified below `1e-100`.
* All 15 negative controls change the output. Exactly 8 of 16 (Type 3) and 8
  of 14 (Type 4) CNOT noise positions, and the leading ancilla Hadamard, are
  annihilated by the normalisation; 4Q has no such position.

### NUMERICAL OBSERVATION

* The `q_th` tables of section 7; 37 of 39 rows bisected to `7.45e-11`, two
  reported as intervals of width `~5e-04`.
* **4Q**: `q_th_full = q_th_first` to the bisection width at every epsilon,
  binding round 0, one predicate transition, no transverse contamination.
* **Type 3 / Type 4**: `q_th_full < q_th_first` strictly over `0.01 <= eps <=
  2/3`, with a binding round that is not 0 and grows with epsilon (up to 28 and
  36). At `eps = 0.90` both revert to binding round 0 and
  `q_th_full = q_th_first` within the bisection width.
* 107 trajectories with `F_1 > F_0` and a later `F_(m+1) < F_m`, verified
  uncontaminated (`C_off` five orders below the trust threshold at the
  violation); 0 trajectories with an initial drop followed by recovery.
* `q_th_full` for Type 3/4 carries a `VIOL_TOL` dependence of `1.8e-07`
  (Type 3) to `2.6e-06` (Type 4) at `eps = 0.40`; 4Q's is `7.5e-11`.
* `q_abs <= q_rel <= q_th` at all 39 compared points.
* Protocol ordering `4Q > Type 4 > Type 3` up to `eps = 0.50`, crossing to
  `Type 4 > 4Q > Type 3` by `eps = 0.60`.
* `eps = 0` and `eps = 1` are degenerate for all three protocols.

### NOT YET PROVED ANALYTICALLY

* **The exact analytic formula for `q_th(epsilon)`**, for any of the three
  protocols. Every number above is a bisection result from the full noisy
  circuit, with the tolerance caveat of section 8.
* **Whether `q_th_full = q_th_first` exactly for 4Q**, as opposed to agreeing
  to `7.45e-11`; and whether the strict inequality for Type 3/4 holds for all
  epsilon rather than on the tested grid.
* **Whether the observed binding round is universally valid** — round 0 for
  4Q; a growing late round for Type 3/4 below `eps ~ 0.7`; round 0 again at
  `eps = 0.90`. The crossover location is not resolved: the `eps = 0.75` rows
  are the two unresolved ones.
* **The exact protocol-crossing location** between `eps = 0.50` and
  `eps = 0.60`.
* **The exact locations and values of the `q_th` maxima**; the grid places all
  three near `eps = 0.75`, where two of the three rows are intervals.
* Whether `q_abs(eps) <= q_rel(eps) <= q_th(eps)` holds as an inequality for
  all epsilon.
* Whether the Type 3/4 effective map preserves Bell diagonality exactly, and
  whether exactly half the CNOT noise positions are annihilated by the
  normalisation as an exact statement.

The implementation study says WHAT happens; the analytic work will say WHY.

## 17. Files

```
scripts/_qth_core.py                     three frozen circuits, routes A and B
scripts/_qth_highprec.py                 optional python-flint ball-arithmetic route
scripts/verify_qth_full_circuit.py       the study
tests/_qth_testlib.py                    shared test helpers
tests/test_type3_qth_circuit.py          22 tests
tests/test_type4_qth_circuit.py          22 tests
tests/test_4q_qth_circuit.py             20 tests
docs/qth_full_circuit_numerical.md       this document
results/qth/
    qth_summary.csv                      the main table (all protocols)
    qth_type3.csv / qth_type4.csv / qth_4q.csv
    qth_first_vs_full.csv                section 8
    qth_trajectories.csv                 near-threshold records
    qth_nonmonotone_cases.csv            the 107 late violations
    (no qth_initial_drop_recovery.csv -- no such trajectory was found)
    qth_hierarchy_comparison.csv         section 14
    qth_protocol_crossings.csv           section 15
    qth_tolerance_sensitivity.csv        q_th_first vs VIOL_TOL
    qth_full_tolerance_sensitivity.csv   q_th_full vs VIOL_TOL
    qth_dip_depth_scan.csv               worst dip depth vs q
    qth_route_comparison.csv             routes A / B / high precision
    qth_negative_controls.csv            the 15 controls
    qth_degenerate_endpoints.csv         eps = 0 and eps = 1
    qth_verification.json                every number quoted above
    qth_curves.*, qth_binding_round.*, qth_trajectories_F.*,
    qth_trajectories_dF.*, qth_hierarchy.*
```

No frozen `q_abs` or `q_rel` output is modified; the 4Q `q_rel` study on
`feature/4q-qrel-circuit-study` is read with `git show` and that branch is
left untouched.
