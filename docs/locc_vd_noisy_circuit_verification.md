# Noisy 6Q-L16 — circuit-level numerical study (one round only)

First noisy study of the six-qubit distributed LOCC virtual-distillation
circuit. Everything below comes from the full 64×64 density-matrix circuit. No
analytic noisy recurrence was derived or used, no Bell-diagonal projection or
twirl was applied, and no threshold formula was assumed or fitted.

Produced by `scripts/verify_locc_vd_noisy.py`; raw numbers in
`results/data/locc_vd_noisy/`, figures in `results/figures/locc_vd_noisy/`.

---

## GIVEN

### Architecture

The 6Q LOCC architecture of `docs/locc_vd_circuit_verification.md`, wire
ordering unchanged:

```
[ a , b , A1 , B1 , A2 , B2 ]  ->  wires 0, 1, 2, 3, 4, 5
Alice = (a, A1, A2) = (0, 2, 4)        Bob = (b, B1, B2) = (1, 3, 5)
```

Input `|+><+|_a (x) |+><+|_b (x) rho_(A1B1) (x) rho_(A2B2)` with two identical
copies.

### 6Q-L16: the decomposition

Each ideal Fredkin is replaced by the **canonical Type-3 Fredkin
decomposition already used by the 5Q Type-3 study**:
`_c2(b,a); _tof(q,a,b); _c2(b,a)` with the Clifford+T Toffoli — 8 CNOTs per
Fredkin — transcribed from
`PQEC-Operational-Threshold/verify_analytic_decomposed.py` (`_c2`/`_tof`/
`_fred`, commit `a4969947`, sha256 `dd8de89220cb0b53…`), the same sequence
`scripts/verify_type3_qabs_circuit.py` uses. A test asserts the two sequences
are identical rather than merely similar.

The decomposition is remapped **independently** onto Alice's triple and Bob's
triple, so no gate crosses the party cut.

### Noise

Two-qubit **replacement** depolarizing after every CNOT, on that CNOT's two
wires, taken from this repository's own
`pqec_distill.noise.replacement_depolarizing` rather than reimplemented.
Noise is applied **only** after CNOTs: single-qubit gates, the `|+>`
preparation and the read-out basis rotations are ideal.

Because the noise lives entirely inside the local Fredkin decompositions, the
noisy quantum core is **identical** for the XX and YY settings; those differ
only in the ideal ancilla observable, so no extra noise enters the read-out.

### The reconstructed operator

For the sixteen two-qubit Pauli products `P` on `A1B1`,

```
d_P(q, rho) = <X_a X_b (x) P> - <Y_a Y_b (x) P>
T_q(rho)    = (1/4) sum_P d_P P
D_q(rho)    = Tr T_q = d_II
R_q(rho)    = T_q / D_q            (when |D_q| is safely nonzero)
```

**`T_q` is not called `rho^2`.** At `q = 0` it must reproduce the frozen ideal
result; at `q > 0` it is whatever the circuit produces, and its hermiticity,
trace, spectrum and Bell off-diagonal content are measured.

---

## NUMERICALLY OBSERVED

### Structure

| check | value |
|---|---|
| CNOTs in Alice's block | **8** |
| CNOTs in Bob's block | **8** |
| total | **16** |
| two-qubit gates crossing the Alice/Bob cut | **none** (empty list) |
| Alice's 8-CNOT block vs the ideal Fredkin | `3.140e-16` |
| Bob's 8-CNOT block vs the ideal Fredkin | `3.140e-16` |
| unitarity of the full 16-CNOT network | `8.882e-16` |
| commutator of the two local blocks | `0.000e+00` |

### The noise parameter is the existing one

| check | max abs difference |
|---|---|
| vs the parent repository's eps2 Kraus channel `global_depol_kraus`, embedded on a wire pair | `2.429e-17` |
| vs the parent repository's own `replacement_depol` used by the iterated Type-3 analysis (5 wires) | `6.939e-18` |
| channel trace preservation | `4.441e-16` |
| channel hermiticity preservation | `0.000e+00` |
| channel positivity (min eigenvalue) | `0.000e+00` |
| `q = 0` is the identity channel | `0.000e+00` |

So the `q` used here is the same parameter as in the existing Type-3 / Type-4
studies.

Final six-qubit state (eps = 0.3): `|tr - 1| <= 2.44e-15`, hermiticity
`<= 1.42e-16`, min eigenvalue `>= -8.02e-17`.

### q = 0 reproduces the frozen ideal 6Q result

25 inputs (four Bell states, five isotropic, six random Bell-diagonal, ten
general two-qubit) × 16 Pauli products:

| | value |
|---|---|
| max `\|d_P(q=0) - d_P^{ideal}\|` | **`8.882e-16`** (worst: `isotropic_0.1`, `ZZ`) |
| max `\|\|T_0 - T_0^{ideal}\|\|_F` | `9.544e-16` |

### Study A — Bell-diagonality survives

48 Bell-diagonal inputs (4 vertices, 12 edge points, 4 face centres, 4
asymmetric, 24 Dirichlet interior points) × 11 values of `q` = 528
reconstructions.

| | value |
|---|---|
| **max `C_Bell` of `R_q` over everything** | **`2.126e-16`** |
| worst input | `dirichlet_4` (interior) |
| worst `q` | `0.2` |
| round-off scale | `1e-12` |
| verdict | **consistent with round-off** |

`C_Bell` by `q` never exceeds `2.2e-16`:

| q | 0 | 1e-4 | 1e-3 | 0.005 | 0.01 | 0.02 | 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| max `C_Bell` | 1.4e-16 | 1.7e-16 | 1.6e-16 | 1.6e-16 | 2.0e-16 | 1.2e-16 | 1.7e-16 | 1.3e-16 | 1.4e-16 | 2.1e-16 | 1.3e-16 |

Reconstructions with an eigenvalue below `-1e-10`: **0** (most negative
eigenvalue seen anywhere: `-3.5e-17`). Smallest `|D_q|` over all 528
reconstructions: `7.06e-03`, so the normalisation never came close to
vanishing on this grid.

The metric is not trivially zero: a test plants a `0.01` Bell off-diagonal
term and checks the same metric returns `> 1e-3`.

**Numerically observed, for arbitrary Bell-diagonal inputs across the tested
grid: `R_q(rho)` is Bell diagonal to round-off.** This is not proved.

### Study B — the Bell-isotropic family does NOT survive

12 values of `eps` × 11 values of `q`.

| | value |
|---|---|
| **max `delta_iso`** | **`2.940e-02`** at `eps = 0`, `q = 0.2` |
| max `\|p_Phi- - p_Psi+\|` | `2.940e-02` |
| **max `\|p_Psi+ - p_Psi-\|`** | **`2.290e-16`** |
| max `\|sum_i p_i - 1\|` | `4.441e-16` |

So isotropy breaks at any `q > 0`, but one relation survives to round-off:

```
p_Psi+ = p_Psi-        (measured, not imposed)
p_Phi- != p_Psi+       (broken, by up to 2.9e-02)
```

`eps = 0.1`:

| q | `p_Phi+` | `p_Phi-` | `p_Psi+` | `p_Psi-` | `delta_iso` |
|---|---|---|---|---|---|
| 0 | 0.99781341 | 0.00072886 | 0.00072886 | 0.00072886 | 1.9e-17 |
| 1e-4 | 0.99758911 | 0.00082024 | 0.00079532 | 0.00079532 | 2.5e-05 |
| 1e-3 | 0.99557139 | 0.00164202 | 0.00139329 | 0.00139329 | 2.5e-04 |
| 0.005 | 0.98662462 | 0.00528036 | 0.00404751 | 0.00404751 | 1.2e-03 |
| 0.01 | 0.97548948 | 0.00979592 | 0.00735730 | 0.00735730 | 2.4e-03 |
| 0.02 | 0.95338315 | 0.01871808 | 0.01394938 | 0.01394938 | 4.8e-03 |
| 0.05 | 0.88843254 | 0.04459127 | 0.03348809 | 0.03348809 | 1.1e-02 |
| 0.10 | 0.78516452 | 0.08458551 | 0.06512499 | 0.06512499 | 2.0e-02 |
| 0.20 | 0.60102800 | 0.15171216 | 0.12362992 | 0.12362992 | 2.8e-02 |
| 0.30 | 0.45419091 | 0.20008667 | 0.17286121 | 0.17286121 | 2.7e-02 |

`eps = 0.3`:

| q | `p_Phi+` | `p_Phi-` | `p_Psi+` | `p_Psi-` | `delta_iso` |
|---|---|---|---|---|---|
| 0 | 0.97267206 | 0.00910931 | 0.00910931 | 0.00910931 | 2.2e-16 |
| 0.01 | 0.94789309 | 0.01893288 | 0.01658702 | 0.01658702 | 2.4e-03 |
| 0.02 | 0.92346646 | 0.02855509 | 0.02398922 | 0.02398922 | 4.6e-03 |
| 0.05 | 0.85239132 | 0.05618714 | 0.04571077 | 0.04571077 | 1.1e-02 |
| 0.10 | 0.74183278 | 0.09797896 | 0.08009413 | 0.08009413 | 1.8e-02 |
| 0.30 | 0.41630638 | 0.20934733 | 0.18717315 | 0.18717315 | 2.2e-02 |

`delta_iso` grows roughly linearly in `q` at small `q` and turns over near
`q ~ 0.2`. The `Psi+ / Psi-` equality is measured independently at every
point; a test confirms the code is not forcing it by showing that a
non-symmetric Bell-diagonal input gives `|p_Psi+ - p_Psi-| > 1e-3`.

**Numerically observed: the isotropic family is not preserved; the
lower-dimensional pattern that is visible is `p_Psi+ = p_Psi-`.** Not proved.

### Study C — one-round Phi+ fidelity

`F_in(eps) = 1 - 3 eps / 4`, `F_out = <Phi+| R_q |Phi+>`,
`Delta F = F_out - F_in`.

| eps | `F_in` | `dF(q=0)` | `dF(0.02)` | `dF(0.05)` | `dF(0.1)` |
|---|---|---|---|---|---|
| 0.00 | 1.000000 | -0.00000000 | -0.04214431 | -0.10396890 | -0.20293687 |
| 0.01 | 0.992500 | +0.00748097 | -0.03488321 | -0.09701212 | -0.19640934 |
| 0.05 | 0.962500 | +0.03699427 | -0.00626984 | -0.06963659 | -0.17075754 |
| 0.10 | 0.925000 | +0.07281341 | +0.02838315 | -0.03656746 | -0.13983548 |
| 0.20 | 0.850000 | +0.13972603 | +0.09287787 | +0.02473440 | -0.08260212 |
| 0.30 | 0.775000 | +0.19767206 | +0.14846646 | +0.07739132 | -0.03316722 |
| 0.40 | 0.700000 | +0.24230769 | +0.19120781 | +0.11808883 | +0.00619354 |
| 0.50 | 0.625000 | +0.26785714 | +0.21607252 | +0.14287392 | +0.03311810 |
| 2/3 | 0.500000 | +0.25000000 | +0.20322388 | +0.13880003 | +0.04609830 |
| 0.75 | 0.437500 | +0.20723684 | +0.16771814 | +0.11404498 | +0.03843227 |
| 0.90 | 0.325000 | +0.08519417 | +0.06771215 | +0.04443568 | +0.01258763 |
| 1.00 | 0.250000 | -0.00000000 | -0.00000000 | -0.00000000 | -0.00000000 |

**There is a region of CNOT noise where one noisy round raises the Phi+
fidelity**, and it closes as `q` grows. Bisecting `Delta F = 0` on the circuit
(tolerance `1e-10`, scan `q in [0, 0.5]` on 51 points):

| eps | `q_gain_num` | unique sign change? |
|---|---|---|
| 0.00 | none — no gain at any tested q | — |
| 0.01 | **0.003506958000** | yes |
| 0.05 | **0.017078783251** | yes |
| 0.10 | **0.032990942709** | yes |
| 0.20 | **0.061194333248** | yes |
| 0.30 | **0.084498672374** | yes |
| 0.40 | **0.102948667146** | yes |
| 0.50 | **0.116656604521** | yes |
| 2/3 | **0.129173711650** | yes |
| 0.75 | **0.130526772104** | yes |
| 0.90 | **0.123989108689** | yes |
| 1.00 | none — degenerate | — |

The sign change is unique on the scanned interval at every `eps` where a root
exists. `q_gain_num(eps)` rises steeply from `eps = 0`, peaks near
`eps ~ 0.75` at `q ~ 0.131`, and falls again as the input approaches the
maximally mixed state.

**Two degenerate endpoints are reported as such rather than as roots.** At
`eps = 0` the input is already pure `Phi+`, so `Delta F(q=0) = 0` and `Delta F`
is negative for every `q > 0` — there is nothing to gain. At `eps = 1` the
input is maximally mixed and `Delta F = 0` to round-off at every `q`
(`max |Delta F| = 5.6e-17` over the scan); an earlier version of the script
bisected that round-off noise and reported a spurious root at `q = 0.22`, which
is why a degeneracy guard was added and the case is now flagged
`degenerate: Delta_F = 0 to round-off at every q`.

### Independent implementation check

Route A is the module's merged-program path plus
`pqec_distill.noise.replacement_depolarizing`. Route B applies each gate from
its own matrix (CNOT built as a basis permutation) and implements the channel
with its own einsum partial trace, sharing no helper with route A.

7 states × `q in {0, 0.02, 0.10}`:

| | value |
|---|---|
| max `\|rho_6Q(A) - rho_6Q(B)\|` | `2.931e-17` |
| max `\|d_P(A) - d_P(B)\|` | `1.388e-16` |

---

## Metadata for a later 5Q-vs-6Q comparison (no comparison drawn here)

| | 6Q-L16 |
|---|---|
| total CNOTs | 16 (measured from the circuit) |
| CNOTs on Alice's side | 8 |
| CNOTs on Bob's side | 8 |
| all CNOTs party-local | yes |
| ancillas | 2 |
| measurement settings required | XX and YY |
| noise convention | per-CNOT two-qubit replacement depolarizing, parameter `q` |

For reference only: the existing 5Q Type-3 baseline also uses 16 CNOTs under
the same per-CNOT replacement-depolarizing convention. **No performance
comparison is drawn**; the two use different numbers of ancillas and different
measurement settings, and the data have not been compared.

---

## NOT YET PROVED

Everything in the previous section is numerical sampling on a finite grid.
None of it is a theorem. In particular these are **not** established:

* any closed-form noisy map for the 6Q-L16 circuit;
* exact Bell-diagonal invariance — only that `C_Bell <= 2.2e-16` on 528
  reconstructions over 48 inputs and 11 noise values;
* an exact invariant plane — only that `|p_Psi+ - p_Psi-| <= 2.3e-16` on the
  isotropic inputs tested;
* any analytic threshold formula. `q_gain_num(eps)` is a table of bisected
  circuit points, deliberately not fitted or smoothed;
* anything about repeated noisy rounds. **This study is one round only.** The
  6Q LOCC-VD circuit is an observable-estimation protocol, not a state-output
  protocol; `R_q` is never fed back into the circuit, and nothing here licenses
  calling such an iteration a physical purification trajectory;
* any noisy fixed point.

---

## Reproducing

```
python scripts/verify_locc_vd_noisy.py
python -m pytest tests/test_locc_vd_noisy.py -q
```

Deterministic: one seeded generator (`seed = 20261007`) used only for the
random sanity checks and the Dirichlet test points; no randomness enters the
circuits themselves. Two consecutive runs produced byte-identical output.
