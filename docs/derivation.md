# Derivation

Complete derivation for the 4-qubit / 5-CNOT Bell-purification circuit.
Every step here is checked numerically by the test suite; the cross-references
name the test that does it.

## 0. Conventions

Computational basis is written `|q1 q2 q3 q4>` with **q1 as the most
significant tensor factor**.  0-based indices used in the code are
`q1 -> 0, q2 -> 1, q3 -> 2, q4 -> 3`, so the index of `|b1 b2 b3 b4>` is
`8*b1 + 4*b2 + 2*b3 + b4`.  This is *not* Qiskit's little-endian ordering.

    retained pair A = (q1, q2)
    measured pair B = (q3, q4)

Bell states carry two binary labels

    |B_ab> = (1/sqrt2) sum_{r=0,1} (-1)^(b r) |r, r xor a>

with `B_00 = Phi+`, `B_01 = Phi-`, `B_10 = Psi+`, `B_11 = Psi-`, and index
`i = 2a + b`.  From the definition,

    ZZ |B_ab> = (-1)^a |B_ab>          (a = "parity"/flip label)
    XX |B_ab> = (-1)^b |B_ab>          (b = "phase" label)

*(verified: `test_bell_state_definitions.py`)*

## 1. The circuit

Five CNOTs in this exact time order, then one Hadamard:

    1. CNOT q3 -> q4
    2. CNOT q2 -> q4
    3. CNOT q1 -> q4
    4. CNOT q3 -> q2
    5. CNOT q3 -> q1
    6. H on q3

Write `U_CNOT = C5 C4 C3 C2 C1` and `V = H_3 U_CNOT`.

*(the gate list is pinned against silent edits in `test_five_cnot_unitary.py`)*

## 2. Heisenberg picture: what the two measured bits are

Measuring `q3` and `q4` in the computational basis after `V` is the same as
measuring `V^dag Z_3 V` and `V^dag Z_4 V` on the input.  Propagate with the
standard Clifford rules

    CNOT(c,t):  X_c -> X_c X_t ,  X_t -> X_t ,  Z_c -> Z_c ,  Z_t -> Z_c Z_t
    H:          X <-> Z

### q3

`V^dag Z_3 V = U_CNOT^dag (H_3 Z_3 H_3) U_CNOT = U_CNOT^dag X_3 U_CNOT`, so
peel the CNOTs off in reverse time order:

| after undoing | operator |
|---|---|
| start | `X_3` |
| C5 = CNOT(q3→q1) | `X_1 X_3` |
| C4 = CNOT(q3→q2) | `X_1 X_2 X_3` |
| C3 = CNOT(q1→q4) | `X_1 X_2 X_3 X_4` |
| C2 = CNOT(q2→q4) | `X_1 X_2 X_3` (the two `X_4` cancel) |
| C1 = CNOT(q3→q4) | `X_1 X_2 X_3 X_4` |

    V^dag Z_3 V = X1 X2 X3 X4

### q4

`H_3` does not touch `q4`, so propagate `Z_4` alone:

| after undoing | operator |
|---|---|
| start | `Z_4` |
| C5, C4 | unchanged (they do not touch q4) |
| C3 = CNOT(q1→q4) | `Z_1 Z_4` |
| C2 = CNOT(q2→q4) | `Z_1 Z_2 Z_4` |
| C1 = CNOT(q3→q4) | `Z_1 Z_2 Z_3 Z_4` |

    V^dag Z_4 V = Z1 Z2 Z3 Z4

The two operators commute (they overlap on all four qubits, so the anticommuting
single-qubit pairs come in an even number).

*(verified two independent ways — dense matrix conjugation and rule-based Pauli
propagation — in `test_stabilizer_mapping.py`)*

## 3. Pure-state action

On a Bell-product input, `XXXX` and `ZZZZ` factor across the two pairs:

    XXXX |B_ab>|B_cd> = (-1)^b (-1)^d |B_ab>|B_cd> = (-1)^(b xor d) (...)
    ZZZZ |B_ab>|B_cd> = (-1)^a (-1)^c |B_ab>|B_cd> = (-1)^(a xor c) (...)

so the measurement is deterministic with

    m3 = b xor d ,      m4 = a xor c

and the retained pair is left in the FIRST input Bell state:

    V |B_ab>|B_cd>  =  |B_ab>_12 |b xor d>_3 |a xor c>_4

*(all 16 cases verified to 1.6e-16 in `test_16_bell_product_cases.py`)*

Hence

    (m3, m4) = (0,0)   <=>   a = c  and  b = d

i.e. success exactly when the two pairs carry **identical Bell labels**.  The
circuit is a *Bell-label comparator*: it measures the XOR of the two label
pairs and reveals nothing else.

## 4. The success projector

The `(m3,m4) = (0,0)` outcome projects onto the simultaneous `+1` eigenspace of
the two commuting stabilizers:

    Pi_00 = (1/4) (I + XXXX)(I + ZZZZ)

which is a rank-4 projector, and `V^dag (|00><00|_{q3q4}) V = Pi_00`.

*(verified: `test_success_projector.py`)*

## 5. The success Kraus operator

Reading off section 3, the map from the four input qubits to the retained pair
on the success branch is

    K_00 = sum_i |B_i>_12 <B_i|_12 <B_i|_34

(a 4 x 16 matrix).  It is derived from the circuit in
`test_general_state_schur_square.py` by extracting `<00|_{q3q4} V` and compared
with this analytic form.

For two copies of an arbitrary two-qubit state `rho` with Bell-basis matrix
elements `rho_ij = <B_i|rho|B_j>`:

    K_00 (rho (x) rho) K_00^dag
      = sum_{i,j} |B_i><B_j| <B_i|rho|B_j> <B_i|rho|B_j>
      = sum_{i,j} (rho_ij)^2 |B_i><B_j|

So the unnormalized output is the **elementwise (Schur) square of rho in the
Bell basis** — *not* the matrix square in general.  See
[limitations.md](limitations.md).

## 6. Bell-diagonal input: the rho^2 identity

If `rho = sum_ab p_ab B_ab` is Bell diagonal then `rho_ij = p_i delta_ij`, so
the Schur square and the matrix square coincide:

    rho_tilde_00 = sum_i p_i^2 |B_i><B_i| = rho^2

Therefore

    P_success = Tr(rho_tilde_00) = sum_i p_i^2 = Tr(rho^2)

    rho_out   = rho_tilde_00 / P_success = rho^2 / Tr(rho^2)

which is exactly the ideal virtual-distillation / PQEC purified state — here
obtained as a genuine **physical postselected state**, not only as a
parity-weighted expectation value.

*(verified on 120 random Dirichlet Bell-diagonal states in
`test_random_bell_diagonal_rho_square.py`)*

## 7. All four measurement branches

Outcome `(m3, m4) = (mu, nu)` requires `b xor d = mu` and `a xor c = nu`, i.e.
the second pair carried labels `(c,d) = (a xor nu, b xor mu)`.  Hence

    rho_tilde_{mu,nu} = sum_{a,b} p_ab p_{a xor nu, b xor mu} B_ab

    P_{mu,nu}         = sum_{a,b} p_ab p_{a xor nu, b xor mu}

Summing over the four branches gives `sum_ab p_ab (sum_cd p_cd) = 1`, and

    sum_{mu,nu} rho_tilde_{mu,nu} = sum_ab p_ab B_ab = rho

so **ignoring the measurement outcome returns the retained pair unchanged**.

*(verified: `test_all_measurement_branches.py`,
`test_ignore_measurement_returns_input.py`)*

### When does the no-postselection identity hold?

The identity is *not* generic — it requires the **retained** pair to be Bell
diagonal.  Propagating the retained-pair Paulis through the circuit gives

| `P_A` on (q1,q2) | `V^dag (P_A (x) II) V` |
|---|---|
| `XX` | `XX II` |
| `YY` | `YY II` |
| `ZZ` | `ZZ II` |
| all 12 others | `P_A (x) (non-identity on q3q4)` |

Only `{XX, YY, ZZ}` — spanning, with `II`, exactly the Bell-diagonal operator
sector `S_BD` — come back supported on `A` alone, and there the map is the
identity.  Every other retained-pair Pauli acquires a factor on the measured
pair, so its output value gets multiplied by an expectation value taken in
`rho_B`.  If `rho_A` is Bell diagonal those 12 components vanish at the input
and the marginal is exactly preserved (for **any** `rho_B`); otherwise it is
not.

*(verified: `test_ignore_measurement_returns_input.py`)*

## 8. Bell-isotropic family

With `rho_eps = (1-eps) Phi+ + eps I/4`, write

    F = 1 - 3 eps / 4 ,     q = eps / 4 ,     p = (F, q, q, q)

Then directly from section 6:

    P_success = F^2 + 3 q^2 = (4 - 6 eps + 3 eps^2) / 4

    F_out     = F^2 / (F^2 + 3 q^2)

The output populations are `(F^2, q^2, q^2, q^2)/(F^2+3q^2)` — still of
isotropic form — so `rho_out = (1-eps') Phi+ + eps' I/4` with

    eps' = (1 - F_out) * 4/3 = eps^2 / (4 - 6 eps + 3 eps^2)

and for small eps,

    eps' = eps^2/4 + (3/8) eps^3 + O(eps^4)

so the leading O(eps) error is removed.  In the computational basis

    rho_out = [[1/2 - eps'/4, 0, 0, (1-eps')/2],
               [0, eps'/4, 0, 0],
               [0, 0, eps'/4, 0],
               [(1-eps')/2, 0, 0, 1/2 - eps'/4]]

*(verified numerically and symbolically with SymPy in
`test_bell_isotropic_formula.py`, `test_epsilon_prime_formula.py`,
`test_computational_basis_output_matrix.py`)*

## 9. Repeated ideal rounds

Each round squares and renormalizes the Bell populations, so after `l` levels

    p_i^(l) = p_i^(2^l) / sum_j p_j^(2^l)
    rho_l   = rho^(2^l) / Tr[rho^(2^l)]

A depth-`l` purification **tree** consumes `2^l` input copies; level `k`
(counted from the leaves) contains `2^(l-k)` nodes and every one must succeed:

    P_total(l) = prod_{k=1}^{l} P_k^(2^(l-k))

With `P_k = T_{2^k} / (T_{2^(k-1)})^2` and `T_m := Tr(rho^m)`, `T_1 = 1`, the
product telescopes to

    P_total(l) = Tr[rho^(2^l)]

**Note.** The naive "one node per level" product `prod_k P_k` is a different
quantity and does *not* equal `Tr[rho^(2^l)]`.

*(verified: `test_repeated_rounds.py`)*

## 10. Entanglement

For a Bell-diagonal state with largest population `F_max`,
`C = max(0, 2 F_max - 1)`.  So

- input entangled iff `F_in > 1/2` iff `eps < 2/3`;
- output entangled iff `F_out > 1/2`, which solves to
  `eps < 2 - 2 sqrt(3)/3 = 4/(3 + sqrt3) ~= 0.8452994616`.

There is therefore a window

    2/3 <= eps < 0.8452994616...

in which the **input pair is separable but the postselected output pair is
entangled**.  This is possible because the circuit is *not* an LOCC protocol —
see [limitations.md](limitations.md).

*(verified: `test_entanglement.py`)*
