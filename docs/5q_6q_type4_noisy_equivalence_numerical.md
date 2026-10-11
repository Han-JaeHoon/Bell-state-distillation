# Raw noisy estimator comparison: 5Q Type-4 versus 6Q-L14

**Status: numerical experiment, independent of any analytic argument.**

Full density-matrix, circuit-level test of whether the raw (unnormalised) noisy
estimator of the five-qubit Type-4 SWAP test equals that of the distributed
two-ancilla six-qubit circuit `6Q-L14`, under the same per-CNOT replacement
depolarizing noise.

The primary path uses no analytic Type-4 Bell-diagonal recurrence, no
closed-form noisy map, and no previously derived Type-4 threshold;
`pqec_distill.analytic_exact_map` is never imported by it. Bell diagonality,
isotropy and equality of the two estimators are never assumed. A secondary
cross-check against the known Type-4 Bell-diagonal map is reported at the end
and enters no conclusion.

Prerequisite: `docs/type4_7cnot_factorization.md`, which establishes that the
frozen 14-CNOT Type-4 circuit really does factorise into two identical
standalone 7-CNOT Fredkin blocks `G7`.

---

## 1. The quantities compared

Five qubits, ordering `[c, A1, B1, A2, B2]` = wires `0..4`, the regrouped
Type-4 circuit `H_c -> G7(c;A1,A2) -> G7(c;B1,B2) -> H_c`:

```
Delta_5_T4(q; O, R) = < X_c (x) O_(A1B1) (x) I_(A2B2) >   (route A, before final H)
                    = < Z_c (x) O_(A1B1) (x) I_(A2B2) >   (route B, after  final H)
```

Six qubits, ordering `[a, b, A1, B1, A2, B2]` = wires `0..5`, `6Q-L14`:

```
C_XX = < X_a X_b (x) O (x) I >
C_YY = < Y_a Y_b (x) O (x) I >
Delta_6_T4(q; O, R) = C_XX - C_YY
```

Both raw and unnormalised; evaluated on the full 32x32 and 64x64 noisy density
matrices. No partial trace precedes the observable, no Bell projection, no
twirling, no re-isotropisation, and nothing is divided by `Delta(I)`.

## 2. The two circuits

| | 5Q Type-4 | 6Q-L14 |
|---|---|---|
| ordering | `[c, A1, B1, A2, B2]` | `[a, b, A1, B1, A2, B2]` |
| structure | `H_c`, `G7(c;A1,A2)`, `G7(c;B1,B2)`, `H_c` | `G7(a;A1,A2)` (Alice), `G7(b;B1,B2)` (Bob) |
| block wires | `(0;1,3)` and `(0;2,4)` | `(0;2,4)` and `(1;3,5)` |
| CNOTs | 14 | 14 = 7 Alice + 7 Bob |
| cross-party 2-qubit gates | n/a (single ancilla) | **0** |
| ancillas | `|+>_c` | `|+>_a |+>_b` |

Measured CNOT lists:

```
5Q (both orderings, identical):
   (1,3) (0,3) (1,3) (0,3) (0,1) (0,1) (3,1)
   (2,4) (0,4) (2,4) (0,4) (0,2) (0,2) (4,2)
6Q-L14:
   (2,4) (0,4) (2,4) (0,4) (0,2) (0,2) (4,2)   [Alice]
   (3,5) (1,5) (3,5) (1,5) (1,3) (1,3) (5,3)   [Bob]
```

Ideal structure of 6Q-L14 (measured, with the phase as an output):

| | global phase | max residual |
|---|---|---|
| Alice block vs ideal `Fredkin(a;A1,A2)` | `-0.125 pi` | 5.978734e-16 |
| Bob block vs ideal `Fredkin(b;B1,B2)` | `-0.125 pi` | 6.206336e-16 |
| full 6Q unitary vs product of the two ideal Fredkins | `-0.250 pi` | 1.110223e-15 |
| local blocks commutator | — | 0.0 |
| unitarity of the full 6Q unitary | — | 8.881784e-16 |

## 3. Noise

After **every** CNOT on wires `(i, j)`:

```
D_q^(ij)(sigma) = (1-q) sigma + q [ I_ij/4 (x) Tr_ij(sigma) ]
```

through this repository's own `pqec_distill.noise.replacement_depolarizing`.
No noise on U3 gates, the `|+>` preparation, the final H, read-out rotations,
state preparation or measurement.

Verified: both modules bind the **same function object**; both
`NOISE_CONVENTION` strings are identical; each of the three programs (5Q
regrouped, 5Q original, 6Q-L14) contains exactly 14 channel insertions whose
wire pairs equal that circuit's own CNOT list; the package channel agrees with
an independently coded einsum channel to `1.4e-17` (5 wires) and `6.9e-18`
(6 wires); `q = 0` is the exact identity channel (`0.0`).

## 4. Guards against a self-comparison

| guard | max difference |
|---|---|
| 5Q route A (`X_c` before final H) vs route B (`Z_c` after final H) | 3.330669e-16 |
| 5Q regrouped ordering vs the **ORIGINAL frozen 14-CNOT gate ordering** (route B) | 4.440892e-16 |
| 5Q main path vs self-contained reference route | 4.440892e-16 |
| 6Q main path vs self-contained reference route (estimator) | 5.551115e-16 |
| 6Q main path vs reference route (full 64x64 state) | 1.183031e-16 |
| 6Q `q = 0` vs frozen ideal `delta_6q` | 8.689246e-16 |

The reference routes build each CNOT as a computational-basis permutation, each
U3 from its matrix definition, apply gates one at a time with no merged
unitaries, implement the depolarizing channel as two einsum contractions, and
construct the 5Q and 6Q observables separately. The only shared ingredient
between the 5Q and 6Q main paths is the physical noise channel, shared on
purpose. No high-level "equivalent estimator" helper is shared.

## 5. Ideal (`q = 0`) validation of 6Q-L14

Over all 94 states and all 56 observables:

| comparison | max difference |
|---|---|
| `Delta_6_L14(0)` vs frozen ideal 6Q `delta_6q` | 9.992007e-16 |
| `Delta_6_L14(0)` vs `Tr(O rho^2)` | 9.129404e-16 |
| `Delta_6_L14(0)` vs `Delta_5_T4(0)` | 4.440892e-16 |

Checked on generic complex states, not only Bell-diagonal ones.

## 6. Test sets

* **94 two-qubit states**, 11 families: four Bell states; four
  computational-basis states; non-Bell-diagonal coherent states; explicitly
  imaginary coherences; near-pure; near-maximally-mixed; asymmetric
  Bell-diagonal; Bell-isotropic; 25 random complex pure; 25 random full-rank
  mixed; 16 rank-deficient (rank 2 and 3).
* **22 arbitrary four-qubit inputs `R`** (16x16), 7 families: random pure 4Q;
  random full-rank; rank-deficient; non-identical products `rho (x) sigma`;
  classically correlated across the copy cut; entangled across `A1B1 | A2B2`;
  strong complex coherences. None of these is of the form `rho (x) rho`.
* **56 observables**: all 16 two-qubit Pauli products; 30 random Hermitian
  `O = (M + M^dagger)/2` normalised in Frobenius norm, with substantial
  imaginary off-diagonal weight; four Bell projectors; 6 random rank-1
  projectors.
* **q grid** (15 values): `0, 1e-6, 1e-4, 1e-3, 0.005, 0.01, 0.02, 0.05, 0.10,
  0.15, 0.20, 0.30, 0.50, 0.75, 1.0`. `q = 1` is kept.

Seeds: `pure 20261011`, `mixed 5772156`, `rank_def 27182818`,
`observables 31415926`, `projectors 14142135`, `general_r 16180339`.

Tolerances recorded, never used to clip a residual: sanity `1e-12`,
equivalence `1e-12`, negative-control minimum signal `1e-6`.

## 7. Results

### 7.1 Primary comparison, `R = rho (x) rho`

| quantity | value |
|---|---|
| comparisons | **78 960** |
| max `\|Delta_5 - Delta_6\|` | **5.551115e-16** |
| mean | 1.537532e-17 |
| RMS | 3.294783e-17 |
| max relative (`/max(1,\|Delta_5\|,\|Delta_6\|)`) | 5.551115e-16 |
| worst `(rho, O, q)` | `comp_10` (= `\|10><10\|`), `ZI`, `q = 1e-3` |
| Pauli family alone (22 560) | max 5.551115e-16 |
| random Hermitian + projectors (56 400) | max 3.330669e-16, worst `rand_pure_01` / `proj_rand1_03` / `q = 0.005` |

### 7.2 Identity signal / scale check

| quantity | value |
|---|---|
| `Delta_5(I)` vs `Delta_6(I)`, 1 410 comparisons | max **4.440892e-16** (mean 4.26e-17) |
| worst | `comp_11`, `q = 0.01` |

So `alpha(q) = 1` within floating point: there is no common q-dependent
prefactor hiding behind agreeing normalised quantities.

### 7.3 Pauli-tomographic effective operators

`T_5 = 1/4 sum_P Delta_5(P) P` built in this script; `T_6` through the frozen
`reconstruct_operator`.

| quantity | value |
|---|---|
| max `\|\|T_5 - T_6\|\|_F` | 3.823324e-16 |
| max entry difference | 3.330669e-16 |
| max `\|Tr T_5 - Tr T_6\|` | 5.551115e-16 |
| max `\|\|R_5 - R_6\|\|_F` (denominator `> 1e-9`) | 4.965372e-16 |

### 7.4 Arbitrary four-qubit `R`

| quantity | value |
|---|---|
| comparisons | **18 480** |
| max `\|Delta_5 - Delta_6\|` | 2.220446e-16 |
| worst `(R, O, q)` | `R_pure4q_00`, `II`, `q = 1e-6` |
| `Delta_5(I)` vs `Delta_6(I)` | max 2.220446e-16 |
| max `\|\|T_5 - T_6\|\|_F` | 1.842567e-16 |
| max `\|\|R_5 - R_6\|\|_F` | 1.677415e-14 |
| regrouped vs original 5Q ordering | max 2.775558e-16 |

The one visibly larger number, `1.68e-14` for the *normalised* `R`, comes from
dividing by a small `Tr T` for some general-`R` inputs; the raw `T` difference
there is `1.8e-16`. Raw equality is the fundamental statement.

The equivalence therefore does **not** require two identical copies in the
tested range — the same structural strengthening as in the Type-3 case.

### 7.5 Negative controls

| control | perturbation | max `\|Delta_5 - Delta_6^mod\|` | detected |
|---|---|---|---|
| A | first 6Q CNOT noise set to `q + 1e-3` | 5.268511e-04 | yes |
| B | depolarizing channel after 6Q CNOT #3 removed | 2.920812e-02 | yes |
| C | one Alice-block CNOT control/target exchanged (still unitary) | 5.549543e-01 | yes |
| D | `C_XX` alone, without `- C_YY` | 2.502543e-01 | yes |

All four move the metric by 12 to 15 orders of magnitude above the unperturbed
residual, so the comparison is sensitive, not degenerate.

### 7.6 Local-coherence mechanism diagnostic

With `Omega_5 = <0|_c sigma_5 |1>_c` after both noisy 7-CNOT blocks and before
the final H, and `Omega_6 = <00|_(ab) sigma_6 |11>_(ab)`, both 16x16 operators
on `(A1, B1, A2, B2)`:

```
max | Omega_5 - 2 Omega_6 |  =  1.716581e-16
```

over 8 states x 15 q values (worst `comp_11`, `q = 1e-4`). Consistent with
`Omega_5 = 2 Omega_6`, strictly stronger than the observable-level equality,
and exactly the relation already observed in the Type-3 study
(`2.79e-16` there). Reported as a mechanism check only.

### 7.7 Secondary, non-primary cross-check

Clearly separated, and not part of any conclusion above: the normalised
effective Bell populations reconstructed from the 6Q-L14 circuit agree with the
known Type-4 Bell-diagonal map `analytic_map_type4` to `5.551115e-16` over
Bell-isotropic inputs `eps in {0.05, 0.2, 0.5, 0.8}` and
`q in {0, 0.01, 0.05, 0.2}`. The import sits inside a single function that the
primary path never calls.

## 8. Where Type-4 departs from Type-3

Nowhere, structurally, in the tested range. The first structural point at which
the two cases could have differed is the factorisation itself: Type-3's
`_fred(q,a,b)` is a hand-written 8-CNOT block and is *already* local by
construction, whereas Type-4 is a single Qiskit-resynthesised 14-CNOT block in
which the two local pieces are interleaved at the boundaries. Making 6Q-L14
well defined therefore required the extra verification step of
`docs/type4_7cnot_factorization.md` (two interleaved operations, moved across
disjoint support). Past that point every measured quantity behaves the same way
as in Type-3: raw equality at floating point, `Delta(I)` equality, the
general-`R` strengthening, and the `Omega_5 = 2 Omega_6` ancilla-block relation.

| quantity | Type-3 / 6Q-L16 | Type-4 / 6Q-L14 |
|---|---|---|
| CNOTs per party | 8 | 7 |
| local block vs ideal Fredkin | exact (phase 0) | `exp(-i pi/8)` × Fredkin |
| regrouping needed? | no | yes, 18 disjoint-support moves |
| max raw `\|Delta_5 - Delta_6\|` | 7.77e-16 | 5.55e-16 |
| `Delta_5(I)` vs `Delta_6(I)` | 7.77e-16 | 4.44e-16 |
| general-`R` max | 2.22e-16 | 2.22e-16 |
| `\|Omega_5 - 2 Omega_6\|` | 2.79e-16 | 1.72e-16 |

## 9. Conclusion (CASE 1 of the task's interpretation rule)

All raw residuals sit at floating-point scale, including the unnormalised
identity signal. Therefore:

> **Full circuit-level numerical evidence is consistent with exact equality of
> the noisy 5Q Type-4 and 6Q-L14 estimators under the present noise model.**

Stated with the separations the task requires:

* **VERIFIED FACT** — the 14-CNOT 5Q Type-4 circuit factorises into two local
  7-CNOT Fredkin decompositions, each equal to `exp(-i pi/8)` times the ideal
  Fredkin, and the regrouping leaves both the ideal unitary and the noisy
  channel unchanged (`8.42e-17`).
* **NUMERICAL OBSERVATION** — over 97 440 raw comparisons the two noisy
  estimators coincide to at most `5.55e-16`, while deliberate perturbations
  raise the same metric to `5.27e-04` and above.
* **NOT YET PROVED ANALYTICALLY** — exact noisy estimator equivalence. Numerical
  simulation does not by itself prove exact equality; an analytic proof is a
  separate result.

## 10. Files

```
src/pqec_distill/type4_7cnot_fredkin.py             gate list, G7, regrouping certificate
src/pqec_distill/type4_5q_raw_noisy.py              raw noisy 5Q Type-4 estimator
src/pqec_distill/locc_vd_type4_noisy.py             6Q-L14
scripts/verify_type4_7cnot_factorization.py         sections 1-4
scripts/verify_5q_6q_type4_noisy_equivalence.py     this experiment
tests/test_type4_7cnot_factorization.py             14 tests
tests/test_5q_6q_type4_noisy_equivalence.py         20 tests
results/data/type4_7cnot_factorization/             factorization evidence
results/data/5q_6q_type4_noisy_equivalence/
    metadata.json                                   every number quoted above
    arbitrary_rho_pauli.csv                         22 560 per-comparison rows
    arbitrary_rho_random_observable.csv             per (state, q), aggregated over
                                                    the 40 non-Pauli observables
    operator_reconstruction.csv                     T_5 / T_6 / R_5 / R_6
    general_R.csv                                   arbitrary four-qubit inputs
    negative_controls.csv                           512 control rows
    independent_routes.csv                          route cross-check
    local_coherence_diagnostic.csv                  mechanism diagnostic
    ideal_q0_validation.csv                         q = 0 validation
    secondary_analytic_sanity.csv                   SECONDARY only
```

Reproduce with

```
python scripts/verify_type4_7cnot_factorization.py
python scripts/verify_5q_6q_type4_noisy_equivalence.py
pytest -q tests/test_type4_7cnot_factorization.py \
          tests/test_5q_6q_type4_noisy_equivalence.py
```

Environment of the recorded run: Python 3.11.15, NumPy 2.4.6, 85.3 s total.
