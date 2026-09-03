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

## 11. Noisy circuit: exact one-round map on Bell-diagonal states

Insert the two-qubit replacement channel after every CNOT, with
no-replacement weight `qbar = 1 - p` (the Pauli convention is the same family
with `qbar = 1 - 16 p/15`, see `noise.py`).  Its Heisenberg action on a Pauli
string `P` is

    D_j^dag(P) = P          if P is the identity on both qubits of CNOT j
               = qbar P     otherwise

so a back-propagated Pauli string simply collects a factor `qbar` at every
noisy location it touches.  The success branch is read out by the four
observables `P_A (x) Pi_00` with `P_A in {II, XX, YY, ZZ}` and
`Pi_00 = 1/4 (I + Z_3)(I + Z_4)`; expand `Pi_00` into its four Pauli terms,
propagate each backwards through `H_3` and then, for `j = 5..1`, apply the
noise weight and conjugate by CNOT `j`.  Evaluating on `rho (x) rho` with
`rho = 1/4(II + xXX + yYY + zZZ)` (so only `<II>=1, <XX>=x, <YY>=y, <ZZ>=z`
survive) gives the unnormalized retained operator
`tau = 1/4 [Q II + X XX + Y YY + Z ZZ]` with

    Q = 1/4 [ 1 + qbar^5 (x^2 + y^2) + qbar^3 z^2 ]          (= P_success)
    X = 1/4 qbar^3 [ (1+qbar) x - 2 qbar^2 y z ]
    Y = 1/4 qbar^4 [ (1+qbar) y - 2 qbar   x z ]
    Z = 1/4 qbar^4 [ (1+qbar) z - 2 qbar   x y ]

and therefore the **exact noisy one-round map**

    D  = 1 + qbar^5 (x^2 + y^2) + qbar^3 z^2
    x' = qbar^3 [(1+qbar) x - 2 qbar^2 y z] / D
    y' = qbar^4 [(1+qbar) y - 2 qbar   x z] / D
    z' = qbar^4 [(1+qbar) z - 2 qbar   x y] / D

At `qbar = 1` this is `x' = 2(x - yz)/(1+x^2+y^2+z^2)` etc., i.e. exactly
`rho -> rho^2/Tr(rho^2)`.  The map keeps Bell-diagonal states Bell diagonal
(consistent with section J.2 of the report) but is *not* symmetric between
`XX` and `YY`: the phase-type correlator `x` is damped by `qbar^3`, the other
two by `qbar^4`, because the CNOTs feed `XXXX` and `ZZZZ` to the two measured
qubits asymmetrically.

*(derived by weighted Pauli propagation and verified against the dense noisy
simulator to 4.4e-16 for both conventions in `test_noisy_closed_form.py`)*

### 11.1 Invariant plane and reduced map

From the formulas, `y = -z` implies `y' = -z'` for every `qbar`, so the plane
`y = -z` is invariant.  (The plane `y = -x` used in the 5-qubit SWAP-test
study is *not* invariant here for `qbar < 1`.)  A Bell-isotropic input
`(x, y, z) = (eb, -eb, eb)` therefore stays on the two-parameter family

    rho(u, v) = 1/4 [ II + u XX + v (ZZ - YY) ],   u = <XX>,  v = <ZZ> = -<YY>

with Bell populations (from `<XX> = (-1)^b`, `<ZZ> = (-1)^a`,
`<YY> = -(-1)^(a+b)` on `|B_ab>`, so `p_ab = 1/4 [1 + (-1)^b x - (-1)^(a+b) y + (-1)^a z]`,
with `x = u, y = -v, z = v`)

    p_Phi+ = (1 + u + 2v)/4,      p_Psi+ = (1 + u - 2v)/4,
    p_Phi- = p_Psi- = (1 - u)/4.

The two "wrong-`XX`-sign" Bell states are always equally populated on this
plane, while `Psi+` (right `XX` sign, wrong `ZZ`/`YY` signs) is not — e.g. the
`p = 0.05` fixed point has populations `(0.9415, 0.0149, 0.0286, 0.0149)`.

Hence `F = <Phi+|rho|Phi+> = (1 + u + 2v)/4`, and the reduced map is

    D(u, v) = 1 + qbar^5 u^2 + (qbar^5 + qbar^3) v^2
    u' = qbar^3 [ (1+qbar) u + 2 qbar^2 v^2 ] / D
    v' = qbar^4 v [ (1+qbar) + 2 qbar u ] / D

## 12. One-round operational threshold

For a pure Bell input (`u = v = 1`)

    F_1(1, p) = 1 - p - (5/4) p^2 + O(p^3)

and in general `F_1(eb, p) = F_ideal(eb) - K(eb) p + O(p^2)` with

    K(eb) = eb (12 eb^3 - 3 eb^2 + 30 eb + 25) / (4 (3 eb^2 + 1)^2),   K(1) = 1

(both symbolic, SymPy).  The operational threshold is the largest `p` at which
one round still improves the Bell fidelity, `F_1(eb, p*) = F_in(eb)`; clearing
denominators this is a quintic in `p` whose physical root is found by
bracketing.  It reproduces the earlier 41-point grid break-even values to
`< 5e-6` and is **not monotone in eps**: it peaks at `p* = 0.1779` near
`eps ~ 0.71` and decreases for noisier inputs (a very mixed input has little
to gain, and the noisy circuit costs `~ 1 - qbar^3` of what remains).

*(`test_noisy_threshold.py`; data `results/data/noisy_threshold.csv`)*

## 13. Repeated noisy rounds: fixed points

Feeding two copies of the postselected output into the next round defines
`rho_{n+1} = M_p(rho_n)`.  On the invariant plane, a fixed point with
`v* != 0` satisfies, from the `v` equation,

    D* = qbar^4 [ (1+qbar) + 2 qbar u* ]

which fixes `v*^2` in terms of `u*`; substituting into the `u` equation
leaves a **quadratic in `u*`**.  The branch continuously connected to `Phi+`
has the weak-noise expansion

    u* = 1 -       p - (13/4) p^2 + O(p^3)
    v* = 1 - (3/2) p - (33/8) p^2 + O(p^3)
    F* = 1 -       p - (23/8) p^2 + O(p^3)

so, as for the 5-qubit gadgets, the first-order asymptotic loss equals the
first-order one-round loss (here both equal `1`).  The branch ends at a
saddle node where the quadratic's discriminant vanishes,

    p_SN  = 0.180669725979

and its fidelity crosses `1/2` slightly earlier,

    p_ent = 0.179815332614

so in the narrow window `p_ent < p < p_SN` a fixed point exists but is
separable.  Beyond `p_SN` every Bell-isotropic input decays to `I/4`
(`F -> 1/4`).

Other fixed-point families on the plane: the second root of the quadratic is
a low-fidelity **unstable** point (Bell-sector eigenvalue `1.77` at
`p = 0.05`) which exists for `p < 0.143` and again for `p > 0.176` — in
between its `v*^2` is negative and it has merged with the `v = 0` family
`1/4 (II + u0 XX)`, `u0^2 = (qbar^3(1+qbar) - 1)/qbar^5`, of classically
correlated separable states.  These organise the basin boundary; they are not
reached from Bell-isotropic inputs below `p_SN`.

By the Pauli symmetry of the noise model each of the four Bell states carries
its own copy of the attracting branch (same `F*`, `C*`; `(x, y, z)` signs
flipped).  A generic input is purified toward whichever Bell state dominates
it.

*(`test_repeated_noisy.py`; data `results/data/repeated_fixed_points.csv`)*

## 14. Full-state stability: the fixed point is an attractor

Bell-sector attraction does not by itself imply full-state attraction — in
the 5-qubit SWAP-test study the corresponding fixed points are saddles with an
unstable direction `(ZI + IZ)/sqrt2`.  Here the exact 15x15 Jacobian in Pauli
coordinates is computed without finite differences from the bilinearity of
the unnormalized map, `d tau/d r_j = B(P_j/4, rho*) + B(rho*, P_j/4)`.

Result: its spectral radius equals the Bell-sector (2x2) value — `0.0193` at
`p = 0.01`, `0.119` at `p = 0.05`, `0.608` at `p = 0.15` — and **all twelve
off-Bell eigenvalues are zero** (`< 1e-16`).  The reason is section 5: the
success branch is a Bell-basis Schur square, `(rho_tilde)_ij = (rho_ij)^2`, so
an off-Bell coherence `delta` returns as `delta^2` and its linearization at
any Bell-diagonal point vanishes.  The noise does not spoil this: it is Pauli
and hence Bell-basis covariant.  Seeding the fixed point with
`eta (ZI + IZ)/(4 sqrt2)` confirms the quadratic contraction,
`1e-2 -> 3.6e-5 -> 4.6e-10 -> 0`.

The fixed point is therefore a genuine **full-state attractor** (superattracting
off the Bell sector), and the same structural fact that restricts the exact
`rho^2` identity to Bell-diagonal inputs is what makes the repeated noisy
dynamics robust.

*(`test_repeated_noisy.py`; figures `repeated_stability.png`,
`repeated_offbell_decay.png`)*
