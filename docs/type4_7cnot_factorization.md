# The 14-CNOT Type-4 circuit factorises into two 7-CNOT Fredkin blocks

**Status: verified fact, established numerically from the repository's own gate
list.** No analytic Type-4 recurrence is used anywhere;
`pqec_distill.analytic_exact_map` is never imported.

Reproduce with `python scripts/verify_type4_7cnot_factorization.py`
(1.5 s; exits non-zero if any stop condition triggers).

---

## 1. The gate list and its provenance

`pqec_distill.type4_7cnot_fredkin.TYPE4_5Q_OPS` holds 36 operations
(22 × U3 + 14 × CNOT).  Verified to be the same list, operation for operation
and angle for angle, as:

| source | matches |
|---|---|
| `scripts/verify_type4_qabs_circuit.py` `TYPE4_OPS` | yes |
| `scripts/verify_type4_qrel_circuit.py` `TYPE4_OPS` | yes |
| `PQEC-Operational-Threshold/pqec_resynth_noise.py` `GATES` | yes |

The U3 convention agrees with the frozen script's `u3_matrix` to `0.0` and with
an independently coded reference U3 to `0.0`.

**Wire naming.** This module uses `[c, A1, B1, A2, B2] = [0, 1, 2, 3, 4]`, the
ordering of the 5Q/6Q comparison studies. The frozen Type-4 scripts print the
same wire *indices* under the names `(a, A1, A2, B1, B2)`. Only the labels
differ; every gate and every wire index is identical. Under the ordering used
here the two local triples are `{c, A1, A2} = {0, 1, 3}` and
`{c, B1, B2} = {0, 2, 4}`, exactly as the task states.

## 2. The 14 CNOTs split 7 + 7

| block | CNOTs (time order) | support |
|---|---|---|
| A | `(1,3) (0,3) (1,3) (0,3) (0,1) (0,1) (3,1)` | `{0, 1, 3}` |
| B | `(2,4) (0,4) (2,4) (0,4) (0,2) (0,2) (4,2)` | `{0, 2, 4}` |

Both match the expected pattern, orientation and order exactly.

Positions in the frozen 36-operation list (0-based):

```
leading  H_c        : 0
block A             : 1, 3..18          (17 operations)
block B             : 2, 19..32, 34, 35 (17 operations)
trailing H_c        : 33
```

The two interleavings the task anticipated are present and are the only ones:

* operation **2**, `U3(B1; pi/2, -pi/2, pi/2)`, is block B's first gate but
  appears before block A; its support `{2}` is disjoint from A's `{0,1,3}`;
* the trailing `H_c` at position **33** precedes block B's last two operations
  (`U3(B2; pi/2, 0, -3pi/4)` and `CNOT(4,2)`), whose support `{2,4}` is
  disjoint from `{0}`.

## 3. The extracted block, and that it is a Fredkin

Each block, relabelled to local wires `(c, u, v) = (0, 1, 2)`, is **gate-for-gate
identical** to the other (max parameter difference `0.0`) and to the candidate
`G7_OPS`:

```
 1.  U3(u; +0.5000pi, -0.5000pi, +0.5000pi)
 2.  U3(v; +0.5000pi, -1.0000pi, -0.7500pi)
 3.  CNOT(u -> v)
 4.  U3(u; +0.5000pi, +0.5000pi, +0.5000pi)
 5.  U3(v; +0.2500pi, -0.7500pi, -0.5000pi)
 6.  CNOT(c -> v)
 7.  U3(v;  0,  0, +0.2500pi)
 8.  CNOT(u -> v)
 9.  U3(u;  0,  0, +0.2500pi)
10.  U3(v;  0,  0, -0.2500pi)
11.  CNOT(c -> v)
12.  CNOT(c -> u)
13.  U3(c;  0,  0, +0.2500pi)
14.  U3(u;  0,  0, -0.2500pi)
15.  CNOT(c -> u)
16.  U3(v; +0.5000pi,  0, -0.7500pi)
17.  CNOT(v -> u)
```

17 operations, **7 CNOTs**.

Fitting `U_G7 = e^{i phi} F` with `phi = arg Tr(F^dagger U_G7)` against the
independently built ideal Fredkin `F = |0><0| (x) I + |1><1| (x) SWAP`:

| | candidate `G7` | extracted A | extracted B |
|---|---|---|---|
| CNOTs | 7 | 7 | 7 |
| global phase | `-0.125 pi` | `-0.125 pi` | `-0.125 pi` |
| `|Tr(F^dag U)| / 8` | 1.0 | 1.0 | 1.0 |
| max entry residual | 7.021667e-16 | 7.021667e-16 | 7.021667e-16 |
| Frobenius residual | 1.565293e-15 | 1.565293e-15 | 1.565293e-15 |
| unitarity residual | 3.330669e-16 | 3.330669e-16 | 3.330669e-16 |
| residual **without** the phase | 0.390181 | 0.390181 | 0.390181 |
| vs independent reference unitary | 0.0 | 0.0 | 0.0 |

So

```
U_G7 = exp(-i pi/8) * Fredkin(c; u, v)
```

to `7.0e-16`. The last row matters: without the phase the residual is `0.39`, so
the phase fit is doing real work rather than hiding a mismatch.

## 4. The regrouping

### 4A. Ideal unitaries

| comparison | global phase | max residual | Frobenius |
|---|---|---|---|
| original vs regrouped `H_c G7_A G7_B H_c` | `-1.7e-18 pi` (i.e. none) | 3.510833e-16 | 1.638953e-15 |
| original vs ideal two-Fredkin SWAP test | `-0.250 pi` | 9.420555e-16 | 4.176775e-15 |
| regrouped vs ideal two-Fredkin SWAP test | `-0.250 pi` | 1.143045e-15 | 4.171725e-15 |

The two orderings agree with **no phase at all**, as they must if they are the
same gates in a different order. The `-pi/4` relative to the ideal SWAP test is
twice the `-pi/8` of one block, consistent with the `pi/4` global phase the
Qiskit transpile records. The two ideal Fredkins commute (`0.0`), and an
independently coded reference route reproduces the original unitary to `0.0`.

### 4B. Reordering certificate

Beyond matrix equality, the regrouping is certified to be a **disjoint-support
reordering**: the regrouped list is a permutation of the frozen one (same
multiset of operations, matched by exact kind/wires/angles), the permutation has
18 order inversions, and **every** inverted pair has disjoint wire support
(0 offending pairs). Since the per-CNOT channel acts on that CNOT's own two
wires, such a reordering cannot change the noisy channel either.

### 4C. Noisy full-channel equality (the decisive check)

Eight arbitrary full five-qubit inputs (3 random full-rank, 2 rank-deficient,
1 random pure, 1 `|0><0|_c (x) R`, plus their companions) propagated through
both orderings with replacement depolarizing after every CNOT, at
`q = 0, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.2, 0.5, 0.75, 1.0`:

| comparison | max entrywise difference of the final 32x32 states |
|---|---|
| original vs regrouped, main path | **8.422230e-17** |
| original vs regrouped, independent reference path | 8.375769e-17 |
| main path vs reference path | 9.080412e-17 |

Worst case: random pure input at `q = 1e-6`.

## 5. Stop conditions

All five stop conditions of the task were evaluated; **none triggered**. In
particular the repository's actual structure matches the stated candidate — the
circuit was not adjusted to make the hypothesis true.

## 6. Conclusion

**Verified fact.** The frozen 14-CNOT Type-4 sequence contains one standalone
7-CNOT block twice, acting on `(c; A1, A2)` and `(c; B1, B2)`. That block is
`exp(-i pi/8)` times the ideal Fredkin. Regrouping the sequence into
`H_c -> G7(c;A1,A2) -> G7(c;B1,B2) -> H_c` moves only operations with disjoint
support, and leaves both the ideal unitary and the noisy per-CNOT channel
unchanged at floating-point precision.

This is what makes the distributed two-ancilla circuit **6Q-L14** well defined;
see `docs/5q_6q_type4_noisy_equivalence_numerical.md`.

## 7. Files

```
src/pqec_distill/type4_7cnot_fredkin.py          gate list, G7, split, certificate
scripts/verify_type4_7cnot_factorization.py      this verification
tests/test_type4_7cnot_factorization.py          14 automated tests
results/data/type4_7cnot_factorization/
    metadata.json                                every number quoted above
    g7_sequence.csv                              the 17-operation block
    fredkin_identification.csv                   the three phase fits
    noisy_channel_equality.csv                   80 noisy-channel comparisons
```

Environment of the recorded run: Python 3.11.15, NumPy 2.4.6.
