# Raw noisy estimator comparison: 5Q Type-3 versus 6Q-L16

**Status: numerical experiment, independent of any analytic argument.**

This document records a full density-matrix, circuit-level numerical test of
whether the raw (unnormalised) noisy estimator of the five-qubit Type-3 SWAP
test equals that of the six-qubit distributed LOCC circuit `6Q-L16`, under the
same per-CNOT replacement depolarizing noise.

Nothing below uses an analytic equivalence argument, the Type-3 Bell-diagonal
recurrence, a closed-form noisy map, or any previously derived fidelity or
threshold formula. `pqec_distill.analytic_exact_map` is never imported, and
Bell diagonality, isotropy and equality of the two estimators are never
assumed. The only inputs are the two physical gate sequences, the physical
noise channel, and the two estimator definitions.

---

## 1. The quantity compared

Five qubits, ordering `[c, A1, B1, A2, B2]` = wires `0..4`:

```
Delta_5(q; O, R) = < X_c (x) O_(A1B1) (x) I_(A2B2) >     (route A)
```

evaluated on the full 32x32 state immediately **before** the final ancilla
Hadamard, or equivalently

```
Delta_5(q; O, R) = < Z_c (x) O_(A1B1) (x) I_(A2B2) >     (route B)
```

**after** that Hadamard. Route A is primary; route A = route B is measured,
not assumed.

Six qubits, ordering `[a, b, A1, B1, A2, B2]` = wires `0..5`:

```
Delta_6(q; O, R) = < X_a X_b (x) O_(A1B1) (x) I_(A2B2) >
                 - < Y_a Y_b (x) O_(A1B1) (x) I_(A2B2) >
```

Both are **raw and unnormalised**. No partial trace precedes the observable, no
denominator is divided out, and `Delta(I)` is compared separately so that a
hidden q-dependent prefactor could not escape detection.

## 2. The two circuits

| | 5Q Type-3 | 6Q-L16 |
|---|---|---|
| ordering | `[c, A1, B1, A2, B2]` | `[a, b, A1, B1, A2, B2]` |
| structure | `H_c`, `Fredkin(c; A1, A2)`, `Fredkin(c; B1, B2)`, `H_c` | `Fredkin(a; A1, A2)` (Alice), `Fredkin(b; B1, B2)` (Bob) |
| Fredkin wires | `(0; 1, 3)` and `(0; 2, 4)` | `(0; 2, 4)` and `(1; 3, 5)` |
| CNOTs | 16 | 16 = 8 Alice + 8 Bob |
| cross-party 2-qubit gates | n/a (single ancilla) | 0 |

Each Fredkin is the canonical Type-3 decomposition
`_c2(b,a); _tof(q,a,b); _c2(b,a)` with the Clifford+T Toffoli, 8 CNOTs,
single-qubit gates left in their original positions. Neither circuit is
collapsed to an ideal Fredkin for the noisy run: the decomposition itself is
part of the channel. The 5Q wire-index gate list is identical to the frozen
`scripts/verify_type3_qabs_circuit.py` `TYPE3_OPS`.

Measured CNOT lists:

```
5Q : (3,1) (1,3) (0,3) (1,3) (0,3) (0,1) (0,1) (3,1)
     (4,2) (2,4) (0,4) (2,4) (0,4) (0,2) (0,2) (4,2)
6Q : (4,2) (2,4) (0,4) (2,4) (0,4) (0,2) (0,2) (4,2)      [Alice]
     (5,3) (3,5) (1,5) (3,5) (1,5) (1,3) (1,3) (5,3)      [Bob]
```

## 3. Noise

After **every** CNOT on wires `(i, j)`:

```
D_q^(ij)(sigma) = (1-q) sigma + q [ I_ij/4 (x) Tr_ij(sigma) ]
```

applied through this repository's own
`pqec_distill.noise.replacement_depolarizing`. No noise on single-qubit gates,
the `|+>` preparation, the final H, read-out rotations, state preparation or
measurement.

Verified: both modules bind the *same function object*; both `NOISE_CONVENTION`
strings are identical; each program contains exactly 16 channel insertions whose
wire pairs equal the circuit's CNOT list; the package channel agrees with an
independently coded einsum channel to `6.9e-18` (5 wires) and `3.5e-18`
(6 wires); and `q = 0` is the exact identity channel (`0.0`).

## 4. Code paths

| path | state construction | read-out |
|---|---|---|
| 5Q main | `pqec_distill.type3_5q_raw_noisy` (own dense gates, own register) | own `X_c (x) O (x) I` / `Z_c (x) O (x) I` |
| 5Q reference | script-local: CNOT as basis permutation, gate by gate, own einsum channel | script-local observable |
| 6Q main | frozen `pqec_distill.locc_vd_noisy` program | frozen `pqec_distill.locc_vd.correlator_6q` |
| 6Q reference | script-local: CNOT as basis permutation, gate by gate, own einsum channel | script-local `XX`/`YY` observables |

The two sides share only the physical noise channel, deliberately, because the
noise model must be identical. No high-level "equivalent estimator" helper is
shared.

## 5. Test sets

* **94 two-qubit states**, 11 families: the four Bell states; the four
  computational-basis states; non-Bell-diagonal coherent states; states with
  explicitly imaginary coherences; near-pure; near-maximally-mixed;
  asymmetric Bell-diagonal; Bell-isotropic; 25 random complex pure; 25 random
  full-rank mixed; 16 rank-deficient (rank 2 and rank 3). Seeded.
* **22 arbitrary four-qubit inputs `R`** (16x16), 7 families: random pure 4Q,
  random full-rank, rank-deficient, non-identical products `rho (x) sigma`,
  classically correlated across the copy cut, entangled across `A1B1 | A2B2`,
  and strong complex coherences. These are *not* of the form `rho (x) rho`.
* **56 observables**: all 16 two-qubit Pauli products; 30 random Hermitian
  `O = (M + M^dagger)/2` normalised in Frobenius norm, with substantial
  imaginary off-diagonal weight; the four Bell projectors and 6 random rank-1
  projectors.
* **q grid** (15 values): `0, 1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.10,
  0.15, 0.20, 0.30, 0.50, 0.75, 1.0`. `q = 1` is kept: the raw comparison is
  well defined there even where a normalised estimator is not.

Seeds: `pure 20240517`, `mixed 7761103`, `rank_def 31415926`,
`observables 1618033`, `projectors 1414213`, `general_r 2718281`.

## 6. Results

Tolerances recorded, never used to clip a residual: sanity `1e-12`,
equivalence `1e-12`, negative-control minimum signal `1e-6`.

### 6.1 Primary comparison, `R = rho (x) rho`

| quantity | value |
|---|---|
| comparisons | **78 960** |
| max `\|Delta_5 - Delta_6\|` | **7.771561e-16** |
| mean | 2.103829e-17 |
| RMS | 4.669585e-17 |
| max relative (`/max(1,\|Delta_5\|,\|Delta_6\|)`) | 7.771561e-16 |
| worst `(rho, O, q)` | `coh_00_plus_01`, `II`, `q = 1e-6` |
| Pauli family alone (22 560) | max 7.771561e-16 |
| random Hermitian + projectors (56 400) | max 4.440892e-16 |
| `Delta_5(I)` vs `Delta_6(I)` (1 410) | max 7.771561e-16 |
| route A vs route B | max 4.440892e-16 |

### 6.2 Pauli-tomographic consequence

`T_5 = 1/4 sum_P Delta_5(P) P` built in this script; `T_6` through the frozen
`reconstruct_operator`.

| quantity | value |
|---|---|
| max `\|\|T_5 - T_6\|\|_F` | 6.684428e-16 |
| max entry difference | 6.661338e-16 |
| max `\|Tr T_5 - Tr T_6\|` | 7.771561e-16 |
| max `\|\|R_5 - R_6\|\|_F` (denominator `> 1e-9`) | 8.092673e-16 |

### 6.3 Arbitrary four-qubit `R`

| quantity | value |
|---|---|
| comparisons | **18 480** |
| max `\|Delta_5 - Delta_6\|` | 2.220446e-16 |
| worst `(R, O, q)` | `R_pure4q_00`, `II`, `q = 0` |
| max `\|\|T_5 - T_6\|\|_F` | 2.397812e-16 |
| max `\|\|R_5 - R_6\|\|_F` | 1.662361e-15 |
| `Delta_5(I)` vs `Delta_6(I)` | max 2.220446e-16 |

The equivalence therefore does **not** require two identical copies in the
tested range.

### 6.4 Independent routes

| comparison | max difference |
|---|---|
| 5Q main vs 5Q reference | 2.220446e-16 |
| 6Q main vs 6Q reference | 4.440892e-16 |
| 5Q at `q = 0` vs frozen ideal `delta_5q` | 6.661338e-16 |
| 6Q at `q = 0` vs frozen ideal `delta_6q` | 5.551115e-16 |
| general-R 6Q runner vs frozen `final_state_6q_noisy` at `R = rho (x) rho` | 2.775558e-16 |

### 6.5 Negative controls

| control | perturbation | max `\|Delta_5 - Delta_6^mod\|` | detected |
|---|---|---|---|
| A | first 6Q CNOT noise set to `q + 1e-3` | 4.754831e-04 | yes |
| B | depolarizing channel after 6Q CNOT #3 removed | 2.712845e-02 | yes |
| C | one Alice-block CNOT control/target exchanged (still unitary) | 5.154405e-01 | yes |
| D | `C_XX` alone, without `- C_YY` | 2.258545e-01 | yes |

All four perturbations move the metric by 9 to 15 orders of magnitude above the
unperturbed residual, so the comparison is sensitive, not degenerate.

### 6.6 Mechanism diagnostic (not evidence by itself)

With `Omega_5 = <0|_c sigma_5 |1>_c` taken after both noisy Fredkins and
`Omega_6 = <00|_(ab) sigma_6 |11>_(ab)` after both local noisy Fredkins, both
16x16 operators on `(A1, B1, A2, B2)`:

```
max | Omega_5 - 2 Omega_6 |  =  2.789401e-16
```

over 8 states x 15 q values. This is consistent with the ancilla-block relation
`Omega_5 = 2 Omega_6`, which is strictly stronger than the observable-level
equality, but it is reported only as a mechanism check.

## 7. Conclusion

The full circuit-level numerical evidence is consistent with exact equality of
the noisy 5Q Type-3 and 6Q-L16 estimators: over 97 440 raw comparisons spanning
94 two-qubit states, 22 arbitrary four-qubit inputs, 56 Hermitian observables
and 15 values of q from 0 to 1, the largest observed
`|Delta_5 - Delta_6|` is `7.77e-16`, with mean `2.1e-17`, while deliberate
perturbations of the 6Q circuit raise the same metric to `4.8e-04` and above.

Numerical simulation does not by itself prove exact equality; the exact
statement is a separate analytic result, to be compared against these frozen
numbers.

## 8. Files

```
src/pqec_distill/type3_5q_raw_noisy.py              raw noisy 5Q estimator (additive)
scripts/verify_5q_6q_type3_noisy_equivalence.py     the experiment
tests/test_5q_6q_type3_noisy_equivalence.py         19 automated tests
results/data/5q_6q_type3_noisy_equivalence/
    metadata.json                                   every number quoted above
    arbitrary_rho_pauli.csv                         22 560 per-comparison rows
    arbitrary_rho_random_observable.csv             per (state, q), aggregated
                                                    over the 40 non-Pauli
                                                    observables, worst one named
    operator_reconstruction.csv                     T_5 / T_6 / R_5 / R_6
    general_R.csv                                   arbitrary four-qubit inputs
    negative_controls.csv                           512 control rows
    independent_routes.csv                          route cross-check
    local_coherence_diagnostic.csv                  mechanism diagnostic
```

Reproduce with

```
python scripts/verify_5q_6q_type3_noisy_equivalence.py
pytest -q tests/test_5q_6q_type3_noisy_equivalence.py
```

Environment of the recorded run: Python 3.11.15, NumPy 2.4.6, 76.4 s total.
