# Verification report

Every number below was produced by executing the code in this repository.
Nothing is quoted from the manual derivation without an executed check behind
it.

**Evidence categories used throughout**

| label | meaning |
|---|---|
| **PROVED** | exact algebraic identity, derived by hand in `docs/derivation.md` AND confirmed symbolically (SymPy) or by exact Clifford propagation |
| **NUMERICALLY VERIFIED** | asserted by an executed test over a specified input set, at a stated tolerance |
| **OBSERVED NUMERICALLY** | reproducible measurement over a finite sample/grid; no proof of the general statement |
| **NOT YET TESTED** | out of scope of this repository |

Test suite at the time of writing: **2055 passed** (`pytest -q`).

---

## A. Exact circuit definition

Qubit order `|q1 q2 q3 q4>`, `q1` most significant; 0-based `q1->0 .. q4->3`.
Retained pair `A = (q1,q2)`; measured pair `B = (q3,q4)`.

```
gate order:  1.q3->q4  2.q2->q4  3.q1->q4  4.q3->q2  5.q3->q1   then H on q3

q1 ───────────────●───────────⊕────  keep (retained pair A)
q2 ─────────●─────│─────⊕─────│────  keep (retained pair A)
q3 ───●─────│─────│─────●─────●─────[H]─ measure  m3
q4 ───⊕─────⊕─────⊕───────────────────── measure  m4
```

`U_CNOT = C5 C4 C3 C2 C1`, `V = H_3 U_CNOT`.  Unitarity of both:
**NUMERICALLY VERIFIED** (`test_five_cnot_unitary.py`).  The gate list is pinned
by an equality assertion so it cannot be silently reordered or flipped.

## B. All 16 Bell-product cases

Claim: `V |B_ab>|B_cd> = |B_ab>_12 |b xor d>_3 |a xor c>_4`.

**Status: NUMERICALLY VERIFIED** — all 16 cases, max phase-aligned statevector
error **1.570e-16** (tolerance 1e-12).  Each output is supported on a *single*
`(q3,q4)` branch with probability 1.  Data: `data/bell_product_cases.{csv,json}`.

Measured `(m3,m4)`, rows = first pair, columns = second pair:

|         | Phi+ | Phi- | Psi+ | Psi- |
|---------|------|------|------|------|
| **Phi+**| 00   | 10   | 01   | 11   |
| **Phi-**| 10   | 00   | 11   | 01   |
| **Psi+**| 01   | 11   | 00   | 10   |
| **Psi-**| 11   | 01   | 10   | 00   |

This **matches the manually derived table exactly**.  The retained pair is the
first input Bell state in all 16 cases.  Success `(0,0)` occurs iff `a=c, b=d`:
**NUMERICALLY VERIFIED**.

## C. Stabilizer conjugation

Claim: `V^dag Z_3 V = XXXX`, `V^dag Z_4 V = ZZZZ`.

**Status: PROVED** (hand derivation in `docs/derivation.md` §2) **and verified by
two independent routes**:

1. dense 16x16 matrix conjugation — agreement to **< 1e-12**;
2. `pauli_propagation.py`, a rule-based Clifford tableau using only the
   CNOT/H conjugation rules and **no matrices at all** — returns exactly
   `+XXXX` and `+ZZZZ`.

The two routes are compared against each other (`test_stabilizer_mapping.py`).
The operators commute, are Hermitian and square to the identity: **NUMERICALLY
VERIFIED**.

`Pi_00 = (1/4)(I+XXXX)(I+ZZZZ)` is a rank-4 projector and equals
`V^dag (|00><00|_{q3q4}) V`: **NUMERICALLY VERIFIED** (`test_success_projector.py`).
A Bell-product state lies in the `(+1,+1)` sector iff its two Bell labels match:
**NUMERICALLY VERIFIED**.

## D. Random Bell-diagonal states

120 seeded Dirichlet Bell-diagonal probability vectors
(`numpy.random.default_rng(20260903)`), each run through the physical circuit
(`rho (x) rho -> V -> project |00> -> partial trace`) and only then compared to
the analytic target.

| quantity | tolerance | status |
|---|---|---|
| `rho_tilde_00 = rho^2` (Frobenius) | 1e-12 | **NUMERICALLY VERIFIED** |
| `rho_out = rho^2/Tr(rho^2)` | 1e-12 | **NUMERICALLY VERIFIED** |
| output populations `= p_i^2 / sum p_j^2` | 1e-12 | **NUMERICALLY VERIFIED** |
| `P_success = Tr(rho^2) = sum p_i^2` | 1e-12 | **NUMERICALLY VERIFIED** |
| output stays Bell diagonal | 1e-12 | **NUMERICALLY VERIFIED** |

## E. Bell-isotropic formulas

**Status: PROVED** (SymPy) **and NUMERICALLY VERIFIED** (circuit).

    P_success = F^2 + 3q^2 = (4 - 6 eps + 3 eps^2)/4
    F_out     = F^2/(F^2 + 3q^2)
    eps'      = eps^2/(4 - 6 eps + 3 eps^2)
    eps'      = eps^2/4 + (3/8) eps^3 + O(eps^4)

The SymPy checks (`test_epsilon_prime_formula.py`) derive `eps'` and
`P_success` from `F` and `q` symbolically and confirm the series coefficients
`0, 0, 1/4, 3/8`.

Sweep of **501** evenly spaced points on `eps in [0,1]`:
max `|circuit - analytic|` over `P_success`, `F_out` and `eps'` is
**7.772e-16**.  Data: `data/isotropic_sweep.csv`.

## F. Representative epsilon values

From `data/isotropic_reference_points.csv` (circuit-derived):

| eps | F_in | P_success | F_out | eps' | C_in | C_out |
|---|---|---|---|---|---|---|
| 0.00 | 1.000000 | 1.00000000 | 1.0000000000 | 0.000000000000 | 1.000000 | 1.000000 |
| 0.01 | 0.992500 | 0.98507500 | 0.9999809659 | 0.000025378778 | 0.985000 | 0.999962 |
| 0.10 | 0.925000 | 0.85750000 | 0.9978134111 | 0.002915451895 | 0.850000 | 0.995627 |
| 0.40 | 0.700000 | 0.52000000 | 0.9423076923 | 0.076923076923 | 0.400000 | 0.884615 |
| 2/3  | 0.500000 | 0.33333333 | 0.7500000000 | 0.333333333333 | 0.000000 | 0.500000 |
| 0.70 | 0.475000 | 0.31750000 | 0.7106299213 | 0.385826771654 | 0.000000 | 0.421260 |
| 0.80 | 0.400000 | 0.28000000 | 0.5714285714 | 0.571428571429 | 0.000000 | 0.142857 |
| 0.90 | 0.325000 | 0.25750000 | 0.4101941748 | 0.786407766990 | 0.000000 | 0.000000 |
| 1.00 | 0.250000 | 0.25000000 | 0.2500000000 | 1.000000000000 | 0.000000 | 0.000000 |

The manually supplied reference point `eps = 0.1` is reproduced exactly:
`F_in = 0.925`, `P_success = 0.8575`, `eps' = 0.002915451895`,
`F_out = 0.9978134111`.  **NUMERICALLY VERIFIED** to 1e-12.

Output density matrix in the computational basis matches the explicit
`[[1/2-e'/4,0,0,(1-e')/2],[0,e'/4,0,0],[0,0,e'/4,0],[(1-e')/2,0,0,1/2-e'/4]]`
form; Hermiticity, positive semidefiniteness, unit trace and purity all
**NUMERICALLY VERIFIED** (`test_computational_basis_output_matrix.py`).

## G. Failure-branch verification

For Bell-diagonal input and every `(mu,nu)`:

    rho_tilde_{mu,nu} = sum_ab p_ab p_{a xor nu, b xor mu} B_ab
    P_{mu,nu}         = sum_ab p_ab p_{a xor nu, b xor mu}

**Status: NUMERICALLY VERIFIED** over 30 random Bell-diagonal states x 4
branches, tolerance 1e-12.  Normalization `sum P = 1`: **NUMERICALLY VERIFIED**.
Every branch stays Bell diagonal: **NUMERICALLY VERIFIED**.

Ignoring the outcome returns the input, `sum_{mu,nu} rho_tilde_{mu,nu} = rho`:
**NUMERICALLY VERIFIED** (tolerance 1e-12).

### Discrepancy found and resolved — see §K.1

The identity in the previous paragraph requires the **retained** pair to be
Bell diagonal.  It holds for arbitrary `rho_B` (including non-Bell-diagonal,
and `rho_A != rho_B`), but **fails** when `rho_A` itself is not Bell diagonal
(20/20 random cases fail by `> 1e-6`).

Cause, established by exact Pauli propagation: `V^dag (P_A (x) II) V` is
supported on `A` alone **only** for `P_A in {XX, YY, ZZ}`, i.e. precisely on the
Bell-diagonal operator sector `S_BD = span{II,XX,YY,ZZ}`, where it acts as the
identity.  Each of the other 12 retained-pair Paulis picks up a non-identity
factor on the measured pair.  **PROVED** by rule-based Clifford propagation
(basis-complete over all 15 non-identity `P_A`).

## H. General non-Bell-diagonal behaviour

The success Kraus operator derived **from the circuit** (by extracting
`<00|_{q3q4} V`) equals the analytic
`K_00 = sum_i |B_i>_12 <B_i|_12 <B_i|_34` to `< 1e-12`: **NUMERICALLY VERIFIED**.

Consequently `(rho_tilde_00)_ij = (rho_ij)^2` in the Bell basis — an
**elementwise (Schur) square, not the matrix square**.

Over 40 seeded random density matrices (`default_rng(2718281)`), all confirmed
non-Bell-diagonal (off-diagonal norm `> 1e-3`):

| comparison | result | status |
|---|---|---|
| circuit output vs Bell-basis Schur square | agrees, `< 1e-12` | **NUMERICALLY VERIFIED** |
| circuit output vs `rho^2/Tr(rho^2)` | **differs**, `> 1e-6` in 40/40 cases | **NUMERICALLY VERIFIED** |

An explicit reproducible counterexample (a rank-4 Bell-basis state with a single
coherence `rho_01 = 0.2`) is included, and the same comparison passes once the
coherence is removed.  This is the **central limitation** of the protocol; see
`docs/limitations.md`.

## I. Repeated ideal rounds

For `eps in {0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9}` and levels 1..6, iterating the
**physical circuit**:

| check | max deviation | status |
|---|---|---|
| `F_circuit` vs recurrence `p -> p^2/sum p^2` | 5.551e-16 | **NUMERICALLY VERIFIED** |
| `rho_l` vs `rho^(2^l)/Tr[rho^(2^l)]` (Frobenius) | 2.483e-16 | **NUMERICALLY VERIFIED** |
| full-tree `P_total(l)` vs `Tr[rho^(2^l)]` | 1.443e-15 | **NUMERICALLY VERIFIED** |

Bell-isotropic inputs converge to the target Bell state; `eps = 0.9` needs ~7
rounds to reach `1 - 1e-9`, `eps = 0.05` needs 2.  `I/4` is a fixed point.
Data: `data/repeated_rounds.csv`.

### Discrepancy found and resolved — see §K.2

`P_total(l) = Tr[rho^(2^l)]` is true for the **full binary tree**,

    P_total(l) = prod_{k=1}^{l} P_k^(2^(l-k))

not for the naive one-node-per-level product `prod_k P_k`.  A guard test asserts
the two differ for `l >= 2`.

## J. Noisy-CNOT results (Phase 2 — completed)

Noise inserted after each of the five CNOTs, on the two qubits it acted on;
single-qubit gates and measurement ideal.  The **full 4x4** retained density
matrix is tracked and never projected back onto the Bell-diagonal family.
`eps in {0, 0.05, 0.1, 0.2, 0.3, 0.5}`, `p` on 41 points in `[0, 0.2]`.
Data: `data/noisy_sweep_replace.csv`, `data/noisy_sweep_pauli.csv`.

**Break-even per-CNOT noise `p*` (largest `p` with `F_out > F_in`):**

| eps | `p*` (replacement) | `p*` (Pauli) | ratio |
|---|---|---|---|
| 0.05 | 0.033697 | 0.031589 | 0.9375 |
| 0.10 | 0.061156 | 0.057332 | 0.9375 |
| 0.20 | 0.102816 | 0.096390 | 0.9375 |
| 0.30 | 0.132114 | 0.123857 | 0.9375 |
| 0.50 | 0.166546 | 0.156136 | 0.9375 |

**Status: OBSERVED NUMERICALLY** (finite `(eps, p)` grid, linear interpolation
between grid points).

At `eps = 0` the gain is 0 for all `p` — a pure Bell input cannot be improved,
only degraded — so no break-even exists there.

### J.1 The two noise conventions are the same family, reparameterized

**Status: PROVED and NUMERICALLY VERIFIED.**  Using the two-qubit Pauli-twirl
identity,

    D_Pauli,p = D_replace,(16p/15)

Verified channel-wise (`< 1e-12`) and confirmed end-to-end at circuit level:
`max |diff|` in `F_out` and `P_success` across the whole sweep is **7.772e-16**.
The constant `p*` ratio of exactly `15/16 = 0.9375` in the table above is the
visible consequence.

**This does not make the labels interchangeable** — a threshold quoted without
its convention is ambiguous by 6.7%.

### J.2 The noisy circuit does NOT leak out of the Bell-diagonal sector

Motivated by the parent project's finding that Bell-diagonal closure is
implementation-dependent under noise, this was tested rather than assumed.

For Bell-diagonal input, in **all four** measurement branches, for both noise
conventions, at `p in {0, 0.01, 0.05, 0.15, 0.4}`: the output Bell off-diagonal
norm stays `< 1e-12` (max observed over the entire sweep: **1.070e-16**).  It
also holds when the two input copies are *different* Bell-diagonal states.

A control test confirms non-vacuity: non-Bell-diagonal inputs *do* produce
off-diagonal output (10/10 cases, `> 1e-6`), so the closure is a property of the
input sector, not of the noise dephasing everything.

**Status: OBSERVED NUMERICALLY** over the tested grid and inputs.  This is not a
symbolic proof for all `p`.

### J.3 No comparison with the 16 / 14 / 12-CNOT implementations

**Status: NOT YET TESTED — deliberately.**  Those results come from a different
protocol (5-qubit SWAP-test gadget, parity-weighted read-out, no postselection)
under its own noise conventions.  No cross-comparison is made or implied.

## K. Discrepancies found

Two claims from the manual derivation needed refinement.  In both cases the
*circuit* behaved correctly and the *stated identity* was imprecise; the circuit
was never modified.

**K.1 — the no-postselection identity needs a hypothesis.**
`sum_{mu,nu} rho_tilde_{mu,nu} = rho` was stated for Bell-diagonal input
(§9 of the task) and is true there.  An initial test that extended it to
arbitrary inputs **failed**.  Investigation (not tolerance relaxation) showed
the identity requires the **retained** pair to be Bell diagonal, and gave the
exact reason: only `{XX,YY,ZZ}` survive `V^dag (P_A (x) II) V` as `A`-local
operators.  Test and documentation now state the hypothesis explicitly.

**K.2 — `P_total(l) = Tr[rho^(2^l)]` is a full-tree statement.**
An initial test read it as the product of one success probability per level and
**failed** (e.g. `0.0442` vs `2.72e-6` at `l = 4`).  The correct reading weights
level `k` by its `2^(l-k)` nodes; with that weighting the identity holds to
`1.4e-15`.  The claim is correct; the naive reading is not.

**K.3 — noise-convention naming.**  An initial docstring asserted the two
conventions "agree only at `p = 1`".  That is wrong: they are the same family
under `p_replace = 16 p_pauli/15`.  Corrected and now tested.

Nothing else disagreed with the manual derivation.  In particular, the central
five-CNOT identity, the stabilizer mapping, the `rho^2` law, `P = Tr(rho^2)`,
and every Bell-isotropic formula were confirmed exactly as supplied.

## L. Conclusion

The proposed circuit does what it was claimed to do:

1. `V |B_ab>|B_cd> = |B_ab> |b xor d>_3 |a xor c>_4` — **exact**, all 16 cases at
   1.6e-16, with the measured bits identified as the commuting stabilizers
   `XXXX` and `ZZZZ` by two independent routes.
2. For **Bell-diagonal** input, postselecting `(0,0)` physically prepares
   `rho^2/Tr(rho^2)` with probability exactly `Tr(rho^2)` — verified on 120
   random states to 1e-12.
3. The Bell-isotropic formulas, including
   `eps' = eps^2/(4-6eps+3eps^2) = eps^2/4 + O(eps^3)`, are reproduced by the
   circuit to 7.8e-16 across 501 sweep points.
4. Repeated rounds give `rho^(2^l)/Tr[rho^(2^l)]` with full-tree success
   probability `Tr[rho^(2^l)]`.
5. The `rho^2` identity is **Bell-diagonal specific**: in general the circuit
   realizes a Bell-basis Schur square, confirmed by explicit counterexamples.
6. Under per-CNOT depolarizing noise the circuit retains a finite break-even
   window and — unlike some implementations studied in the parent project —
   does not leak out of the Bell-diagonal sector.

**What this does not establish:** any novelty, priority, optimality, or
advantage over existing Bell-purification protocols; and nothing about LOCC
distillation, which this circuit is not.  See `docs/limitations.md`.
