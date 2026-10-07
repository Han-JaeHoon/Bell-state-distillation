# Analytic derivation and cross-check of the 6Q distributed SWAP-test estimator

This document derives, from the circuit structure alone, what the six-qubit
distributed estimator computes, and compares the derivation against the
**frozen** blind circuit data in `results/data/locc_vd/`. Those files are read
only; their SHA-256 digests are recorded in
`results/data/locc_vd_analytic/metadata.json` so the comparison is pinned to
the exact bytes.

Companion: `docs/locc_vd_circuit_verification.md` (the blind run),
`scripts/verify_locc_vd_analytic.py` (this cross-check),
`tests/test_locc_vd_analytic.py`.

---

## 1. System

Data input, two copies on `(A1,B1)` and `(A2,B2)`:

```
R = rho_(A1B1) (x) rho_(A2B2),      with  rho_(A1B1) = rho_(A2B2) = rho .
```

`S_A = SWAP(A1,A2)`, `S_B = SWAP(B1,B2)`, `S = S_A S_B`. Both ancillas start in
`|+>_a |+>_b`. The simulator's ordering is `[a, b, A1, B1, A2, B2]`, so the data
register factorises as `(A1B1) (x) (A2B2)` — copy 1 tensor copy 2 — and `S` is
the SWAP of those two four-dimensional factors.

Branches: `W_00 = I`, `W_01 = S_B`, `W_10 = S_A`, `W_11 = S_A S_B`, with the
first bit of `x = (a,b)` controlling `S_A`.

Nothing below assumes the final identity; it is obtained.

---

## 2. The full post-control state

`|+> = (|0> + |1>)/sqrt(2)`, so

```
|++><++|_(ab) = (1/2 sum_x |x>) (1/2 sum_y <y|) = (1/4) sum_{x,y} |x><y| ,
```

the four terms of each `|+><+|` contributing `1/2 * 1/2 = 1/4` in total — the
normalisation is `1/4`, once, for the pair of ancillas.

The controlled operation is `U = sum_x |x><x| (x) W_x`. Conjugating:

```
Omega = U ( |++><++| (x) R ) U^dagger
      = (1/4) sum_{x,y} |x><y| (x) W_x R W_y^dagger .                    (1)
```

Check of the factors: `Tr Omega = (1/4) sum_{x,y} <y|x> Tr[W_x R W_y^dagger]
= (1/4) sum_x Tr[W_x R W_x^dagger] = (1/4) * 4 * Tr R = 1`. ✓

---

## 3. The XX correlator

Write `Obar = O_(A1B1) (x) I_(A2B2)`. Since `X (x) X` maps `|ab>` to the
bit-complement `|a-bar b-bar>`,

```
<y| X (x) X |x> = delta_{y, x-bar} ,
```

so tracing (1) against `X_a X_b (x) Obar` keeps only the four terms whose two
branch labels are complementary:

```
C_XX(O) = Tr[(X_a X_b (x) Obar) Omega]
        = (1/4) sum_x <x-bar| X (x) X |x> ... 
        = (1/4) { Tr[Obar R S]            (x = 00, y = 11)
                + Tr[Obar S R]            (x = 11, y = 00)
                + Tr[Obar S_B R S_A]      (x = 01, y = 10)
                + Tr[Obar S_A R S_B] }    (x = 10, y = 01)        (2)
```

using `W_00 R W_11^dagger = R S`, `W_11 R W_00^dagger = S R`,
`W_01 R W_10^dagger = S_B R S_A`, `W_10 R W_01^dagger = S_A R S_B`, and
`S_A^dagger = S_A`, `S_B^dagger = S_B`, `S^dagger = S`.

Exactly four interference terms, in two pairs: an **`I <-> S_A S_B`** pair and
an **`S_A <-> S_B`** pair. Not simplified yet.

---

## 4. The YY correlator

`Y|0> = i|1>`, `Y|1> = -i|0>`, so `<1|Y|0> = i` and `<0|Y|1> = -i`. Again only
complementary labels survive, but now with signs:

| branch pair | `<y|XX|x>` | `<y|YY|x>` |
|---|---|---|
| `x=00, y=11` | `+1` | `(i)(i) = -1` |
| `x=11, y=00` | `+1` | `(-i)(-i) = -1` |
| `x=01, y=10` | `+1` | `(i)(-i) = +1` |
| `x=10, y=01` | `+1` | `(-i)(i) = +1` |

Hence

```
C_YY(O) = (1/4) { - Tr[Obar R S] - Tr[Obar S R]
                  + Tr[Obar S_B R S_A] + Tr[Obar S_A R S_B] } .      (3)
```

**The `I <-> S_A S_B` coherences change sign; the `S_A <-> S_B` coherences do
not.** That asymmetry is the whole mechanism.

---

## 5. The subtraction

Both `X (x) X` and `Y (x) Y` are supported only on complementary pairs, so the
difference has entries `1 - (-1) = 2` on the `00 <-> 11` pair and `1 - 1 = 0`
on the `01 <-> 10` pair:

```
X (x) X - Y (x) Y = 2 ( |00><11| + |11><00| ) .                        (4)
```

(Derived from the table above, not assumed; checked as a matrix identity in
`tests/test_locc_vd_analytic.py::test_xx_minus_yy_is_twice_the_outer_coherence`.)

Subtracting (3) from (2), the `S_A <-> S_B` terms cancel and the other pair
doubles:

```
C_XX(O) - C_YY(O) = (1/4) * 2 * { Tr[Obar R S] + Tr[Obar S R] }
                  = (1/2) Tr[ Obar ( R S + S R ) ] .                   (5)
```

This holds for **any** `R`, identical copies or not. The ordering matters and is
kept: `R S` and `S R` are different operators in general.

### Why the cross term dies, in words

The ancillas control two *different* operations, `S_A` on Alice's side and
`S_B` on Bob's. The ancilla pair observable `X_a X_b` is blind to which of the
four branch coherences it reads: it weights all four equally. Rotating both
ancillas to `Y` flips the sign of a coherence exactly when the two branch labels
differ in *both* bits — because each `<.|Y|.>` contributes a factor `±i` and
two such factors multiply to `-1`, whereas a coherence differing in both bits in
the *opposite* senses (`01 <-> 10`) picks up `(i)(-i) = +1`. The `I <-> S_A S_B`
coherence is of the first kind, the `S_A <-> S_B` coherence of the second.
Subtracting therefore keeps only the interference between "neither party
swapped" and "both parties swapped", which is precisely the full-copy SWAP
interference the ordinary single-ancilla test measures.

### The same calculation for the 5Q baseline

One ancilla in `|+>` controlling the full `S`: `Omega_5 = (1/2) sum_{x,y in
{0,1}} |x><y| (x) W_x R W_y^dagger` with `W_0 = I`, `W_1 = S`, and
`<y|X|x> = delta_{y, 1-x}`, so

```
Delta_5Q(O) = <X_c (x) Obar> = (1/2) { Tr[Obar R S] + Tr[Obar S R] } ,
```

**identical to (5) for any `R`**. This is why the blind run found the two
estimators equal to machine precision on every input, including the ones that
are not Bell-diagonal, and it predicts the same for non-identical copies
(section 10 below).

---

## 6. Identical copies

For `R = rho (x) rho`, `S` exchanges the two copies, so

```
S R S = (rho (x) rho) with the factors exchanged = rho (x) rho = R ,
```

and since `S^2 = I`, right-multiplying by `S` gives `S R = R S`. Then (5)
collapses:

```
C_XX(O) - C_YY(O) = (1/2) Tr[ Obar * 2 R S ] = Tr[ (O (x) I) (rho (x) rho) S ] .
                                                                        (6)
```

---

## 7. The SWAP identity, proved

Let `{|i>}` be any orthonormal basis of the single-copy space (here `C^4`), and

```
S = sum_{ij} |i><j| (x) |j><i| .
```

Then, with `Tr_2` the partial trace over the second copy,

```
Tr_2[ (A (x) B) S ] = sum_{ij} A|i><j| * Tr[ B|j><i| ]
                    = sum_{ij} A|i><j| <i|B|j>
                    = A ( sum_i |i><i| ) B ( sum_j |j><j| )
                    = A B .                                             (7)
```

The mirrored version, needed in section 10, is

```
Tr_2[ S (A (x) B) ] = sum_{ij} |i><j|A * <i|B|j> = B A .               (7')
```

Setting `A = B = rho` in (7): `Tr_2[(rho (x) rho) S] = rho^2`. Because `O` acts
only on copy 1 and the identity on copy 2, (6) becomes

```
C_XX(O) - C_YY(O) = Tr[ O * Tr_2[(rho (x) rho) S] ] = Tr( O rho^2 ) .   (8)
```

**This is the main analytic claim.**

---

## 8. Denominator and the normalised estimator

`O = I` in (8):

```
C_XX(I) - C_YY(I) = Tr(rho^2) ,                                         (9)
```

the purity. Hence

```
<O>_VD = [C_XX(O) - C_YY(O)] / [C_XX(I) - C_YY(I)] = Tr(O rho^2) / Tr(rho^2) .
```

**This is an estimator of an expectation value.** It is the number one would
obtain by measuring `O` on the state `rho^2 / Tr(rho^2)` — but that state is
never prepared. The six-qubit circuit uses no postselection, every shot is kept,
the final six-qubit state has unit trace, and the reduced state of `(A1,B1)`
after the circuit is **not** `rho^2 / Tr(rho^2)`. The ratio is formed in
classical post-processing.

---

## 9. The Pauli-tomographic operator

Define `d_P = C_XX(P) - C_YY(P)` for the sixteen two-qubit Pauli products. By
(8), `d_P = Tr(P rho^2)`. Pauli products satisfy `Tr(P Q) = 4 delta_PQ` and span
the 4x4 matrices, so any `M` obeys `M = (1/4) sum_P Tr(P M) P`. With
`M = rho^2`:

```
T_6Q := (1/4) sum_P d_P P = (1/4) sum_P Tr(P rho^2) P = rho^2 .        (10)
```

So the operator the blind run reconstructed and deliberately left unnamed **is
the matrix square**.

---

## 10. Non-identical copies (separate claim)

Keeping `R = rho (x) sigma` in (5) and using (7) and (7'):

```
Tr_2[(rho (x) sigma) S] = rho sigma ,     Tr_2[S (rho (x) sigma)] = sigma rho ,
```

so

```
Delta_6Q(O) = Delta_5Q(O) = (1/2) Tr[ O ( rho sigma + sigma rho ) ] .  (11)
```

This is **not** part of the identical-copy VD claim; it is the pre-symmetry
expression, and it reduces to (8) when `sigma = rho`. It is reported in its own
section below and in `results/data/locc_vd_analytic/nonidentical_inputs.csv`.

---

## 11. Bell specialisations

### Bell-diagonal

For `rho = sum_i p_i |B_i><B_i|` the Bell states are orthonormal eigenvectors,
so `rho^2 = sum_i p_i^2 |B_i><B_i|` — the Bell basis is preserved exactly and no
off-diagonal weight appears. Normalising (10) by (9):

```
p_i' = p_i^2 / sum_j p_j^2 ,        F_VD = p_0^2 / sum_j p_j^2  (target Phi+).
```

### Bell-isotropic

`I/4 = (1/4) sum_i |B_i><B_i|`, so

```
rho_eps = (1-eps)|Phi+><Phi+| + eps I/4
       => p_0 = (1-eps) + eps/4 = 1 - 3 eps/4 ,   p_1 = p_2 = p_3 = eps/4 ,
```

and therefore

```
F_VD(eps) = (1 - 3 eps/4)^2 / [ (1 - 3 eps/4)^2 + 3 (eps/4)^2 ] .      (12)
```

Spot values: `eps = 0 -> 1`; `eps = 2/3 -> 0.25/(0.25 + 3/36) = 3/4`;
`eps = 1 -> (1/16)/(4/16) = 1/4`.

---

## 12. Comparison against the frozen blind data

Produced by `scripts/verify_locc_vd_analytic.py`. The frozen files were read,
never rewritten; the 151 inputs were regenerated from the same deterministic
generator and checked against the frozen record first:

| integrity check | result |
|---|---|
| `max abs(rho_regenerated - rho_frozen)` over the 19 stored representatives | `0.0` |
| state list and order identical to the frozen summary | `True` |
| states regenerated / in frozen summary | `151 / 151` |

### Derivation ingredients, checked numerically

| check | max error |
|---|---|
| `Tr_2[(A (x) B) S] = A B` (50 random pairs) | `3.580e-15` |
| `Tr_2[S (A (x) B)] = B A` | `1.831e-15` |
| `Tr(P Q) = 4 delta_PQ` | `0.000e+00` |
| `S (rho (x) rho) S = rho (x) rho` | `0.000e+00` |
| `S R - R S = 0` for identical copies | `0.000e+00` |

### Frozen `Delta` versus analytic `Tr(O rho^2)` — 2416 comparisons

| | max | mean | RMS |
|---|---|---|---|
| frozen `Delta_6Q` vs analytic | **`6.661e-16`** | `1.021e-16` | `1.545e-16` |
| frozen `Delta_5Q` vs analytic | `4.441e-16` | `5.153e-17` | `8.331e-17` |

Worst case (both): state `pure_5`, observable `II`, frozen `+1.000000000000`,
analytic `+1.000000000000`.

Per family (6Q vs analytic):

| family | n | max | mean | RMS |
|---|---|---|---|---|
| A pure Bell | 64 | `4.441e-16` | `1.110e-16` | `2.220e-16` |
| B isotropic | 112 | `4.441e-16` | `6.360e-17` | `1.504e-16` |
| C Bell-diagonal | 320 | `4.441e-16` | `4.039e-17` | `1.043e-16` |
| D random pure | 640 | `6.661e-16` | `1.786e-16` | `2.240e-16` |
| E random full rank | 640 | `3.331e-16` | `6.646e-17` | `8.829e-17` |
| F rank-deficient | 640 | `5.551e-16` | `9.799e-17` | `1.335e-16` |

### Frozen `T_6Q` versus analytic `rho^2` — 19 representatives

| | value |
|---|---|
| max `\|\|T_6Q - rho^2\|\|_F` | **`4.785e-16`** (worst: `pure_1`) |
| max `\|\|T_5Q - rho^2\|\|_F` | `2.552e-16` |

Traces agree to all printed digits on every representative, e.g.
`isotropic eps=0.1`: `Tr T_6Q = 0.857500000000`, `Tr rho^2 = 0.857500000000`.

### Bell-isotropic fidelity, formula (12) versus frozen

| `epsilon` | frozen `F_6Q` | formula (12) | `\|diff\|` |
|---|---|---|---|
| `0.000000000000000` | `1.000000000000` | `1.000000000000` | `0.00e+00` |
| `0.010000000000001` | `0.999980965916` | `0.999980965916` | `1.11e-16` |
| `0.100000000000000` | `0.997813411079` | `0.997813411079` | `1.11e-16` |
| `0.300000000000000` | `0.972672064777` | `0.972672064777` | `1.11e-16` |
| `0.666666666666667` | `0.750000000000` | `0.750000000000` | `2.22e-16` |
| `0.800000000000000` | `0.571428571429` | `0.571428571429` | `3.33e-16` |
| `1.000000000000000` | `0.250000000000` | `0.250000000000` | `0.00e+00` |

Max `3.331e-16`.

**One discrepancy was found and is reported rather than hidden.** On the first
pass, `eps = 2/3` showed an error of `3.75e-07` — seven orders above everything
else. The cause was in the cross-check script, not in the physics or in the
frozen data: `epsilon` was being recovered by parsing the state's *name*, and
the name rounds `2/3` to six decimals (`isotropic_eps0.666667`), so formula (12)
was being evaluated `3.33e-07` away from the `epsilon` the state was actually
built with. Recovering `epsilon` from the state itself,
`eps = 4(1 - p_0)/3`, removes it. Both routes are kept as columns in
`bell_isotropic_comparison.csv`, and the metadata records that the label route
would give up to `3.750e-07`.

### Bell-diagonal specialisation — 31 inputs

| check | max error |
|---|---|
| `p_i'` from `rho^2` versus `p_i^2 / sum_j p_j^2` | `2.220e-16` |
| frozen `F_6Q` versus `p_0^2 / sum_j p_j^2` | `2.220e-16` |
| Bell off-diagonal weight of `rho^2` | `1.327e-32` |

### Section 10 — non-identical inputs (kept separate)

60 fresh `(rho, sigma)` pairs (a new seed, not the blind set) x 16 observables
= 960 comparisons, run through the frozen circuits:

| | value |
|---|---|
| max `\|Delta_6Q - (1/2)Tr[O(rho sigma + sigma rho)]\|` | `2.776e-16` |
| mean | `4.879e-17` |
| RMS | `6.412e-17` |
| max for the 5Q baseline | `1.665e-16` |
| max gap to the identical-copy formula `Tr(O rho^2)` | `9.516e-01` |

The last row is the control: the identical-copy formula is badly wrong here, so
the agreement in the rows above is not vacuous. Equation (11) holds, and the 5Q
and 6Q estimators remain equal to each other for non-identical copies too, as
section 5 predicts.

---

## 13. What is established, and what is not

**Proved analytically** (sections 2–11 above):

* the branch decomposition (1) with its `1/4`;
* that `C_XX` and `C_YY` each contain exactly four interference terms, (2) and (3);
* the sign table, and hence `X (x) X - Y (x) Y = 2(|00><11| + |11><00|)`, (4);
* cancellation of the `S_A <-> S_B` cross term and
  `Delta = (1/2) Tr[Obar (R S + S R)]`, (5), for any `R`;
* `S R = R S` for identical copies, (6);
* the SWAP identity `Tr_2[(A (x) B) S] = A B`, (7), from an explicit basis
  expansion;
* `Delta(O) = Tr(O rho^2)`, (8), and `Delta(I) = Tr(rho^2)`, (9);
* `T_6Q = rho^2`, (10), via Pauli orthogonality;
* the normalised estimator `Tr(O rho^2)/Tr(rho^2)`;
* the Bell-diagonal and Bell-isotropic specialisations, (12);
* the non-identical-copy expression (11), and the equality of the 5Q and 6Q
  estimators for arbitrary `R`.

**Numerically verified**:

* agreement with the frozen 6Q circuit data to `6.661e-16` over 2416
  state/observable comparisons spanning six input families;
* agreement with the independently simulated 5Q baseline to `4.441e-16`;
* `T_6Q = rho^2` to `4.785e-16` in Frobenius norm on all 19 frozen
  representatives;
* the Bell-isotropic fidelity formula to `3.331e-16`;
* equation (11) on a fresh non-identical test set to `2.776e-16`.

**Physical interpretation**:

* every quantum gate in the 6Q implementation is party-local — Alice touches
  only `(a, A1, A2)`, Bob only `(b, B1, B2)` (verified to `0.000e+00` in the
  blind run);
* no ancilla transmission and no quantum communication is required;
* only classical measurement records need to be combined, to form the
  correlators and then the ratio;
* for Bell fidelity the required data observables are the local Pauli products
  in `Phi+ = (II + XX - YY + ZZ)/4`.

**Not claimed**:

* that the physical state `rho^2 / Tr(rho^2)` is prepared anywhere — it is not;
  there is no postselection and the post-circuit reduced state of `(A1,B1)` is
  not that state;
* that this is conventional physical entanglement distillation — it is
  observable estimation;
* novelty of the basic LOCC Hadamard-test construction. The content here is the
  verification, not the invention of the gadget.
