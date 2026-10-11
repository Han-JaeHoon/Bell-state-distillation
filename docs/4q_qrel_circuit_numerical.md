# 4Q relative Bell-error-suppression threshold `q_rel(epsilon)` — circuit-level numerical study

**Status: circuit-level numerical observation. This is NOT an analytic
derivation.**

The primary path uses no analytic 4Q Bell-diagonal recurrence, no closed-form
noisy map, and no previously stored threshold — in particular the frozen
`q_abs` values are never used as bracketing hints.
`scripts/verify_4q_qrel_circuit.py` imports nothing from `pqec_distill` at all.
The comparison with `q_abs` happens in a single clearly separated SECONDARY
step after the primary table is complete.

Reproduce with `python scripts/verify_4q_qrel_circuit.py` (174.5 s).

---

## 1. Definition tested

Target state `Phi+`. From the FULL conditional two-qubit density matrix of
round `n`,

```
p_B^(n) = <B| rho_n |B>,        B in {Phi+, Phi-, Psi+, Psi-}
R_i^(n) = p_i^(n) / p_Phi+^(n), i in {Phi-, Psi+, Psi-}

q_rel(eps) = sup { q : R_i^(n+1) < R_i^(n) for every non-target i
                       and every n >= 0 }
```

`q_rel` does **not** require the absolute error populations to fall. A round in
which `p_i` rises while `R_i / p_Phi+` falls still passes; such events are
searched for explicitly (§8 below).

Nothing was assumed about which component is limiting, whether the first round
binds, how `q_rel` relates to `q_abs`, whether the output stays Bell diagonal,
or any invariant plane. All of those are outputs.

## 2. Frozen physical 4Q circuit

Source of truth: `src/pqec_distill/circuit.py` (`CNOT_SEQUENCE` + module
docstring). Verified at run time, by textual extraction rather than import, to
be identical to that file **and** to `CNOT_SEQUENCE` in
`scripts/verify_4q_qabs_circuit.py`: both checks returned `True`.

Ordering `|q1 q2 q3 q4>`, `q1` most significant, 0-based `q1->0 ... q4->3`;
retained pair `(q1,q2)`, measured/discarded pair `(q3,q4)`.

```
  1.  CNOT(q3 -> q4)   wires (2 -> 3)   + D_q on (2,3)
  2.  CNOT(q2 -> q4)   wires (1 -> 3)   + D_q on (1,3)
  3.  CNOT(q1 -> q4)   wires (0 -> 3)   + D_q on (0,3)
  4.  CNOT(q3 -> q2)   wires (2 -> 1)   + D_q on (2,1)
  5.  CNOT(q3 -> q1)   wires (2 -> 0)   + D_q on (2,0)
  6.  H(q3)                             [ideal, no noise]
  7.  measure (q3,q4); success = (0,0); retain (q1,q2)
```

Unitarity residual `2.220446e-16`. The directions matter: reversing any single
CNOT changes the unitary by more than `1e-3` (checked for all five).

## 3. Noise model

After **every** CNOT on wires `(i,j)`:

```
D_q^(ij)(sigma) = (1-q) sigma + q [ I_ij/4 (x) Tr_ij(sigma) ]
```

the repository's two-qubit **replacement** depolarizing channel — *not* the
15-Pauli convention. `H`, state preparation, measurement and classical
postselection are ideal.

Measured: 5 channel insertions, whose wire pairs equal the CNOT list exactly;
`q = 0` is the exact identity channel (`0.0`); at `q = 1` the pair's reduced
state is exactly `I/4` while the other two qubits are untouched (`0.0`); the
channel agrees with an independently coded one to `0.0`; the four measurement
branch probabilities sum to 1 within `5.551115e-16`.

## 4. Repeated-round rule

```
rho_n (x) rho_n  ->  noisy 5-CNOT circuit  ->  H(q3)
                 ->  postselect (q3,q4) = (0,0)  ->  trace out (q3,q4)
                 ->  normalise by P_succ  =  rho_(n+1)
```

Two copies of the **actual** full 4x4 conditional output feed the next round.
No Bell projection, twirling, re-isotropisation, symmetrisation, Pauli
truncation, or analytic Bell-diagonal substitution anywhere.

Recorded every round: the four Bell populations, the three ratios, the success
probability, the trace error, the hermiticity residual, the minimum
eigenvalue, and the Bell-basis off-diagonal weight.

## 5. Tolerances (all recorded, none used to hide a case)

| name | value | role |
|---|---|---|
| `STEP_TOL` | `1e-14` | stop once `max|rho_(n+1) - rho_n|` falls below it: the orbit is at the numerical fixed point |
| `VIOL_TOL` | `1e-12` | a round violates only if some non-target `Delta R_i > +VIOL_TOL` |
| `N_MAX` | `5000` | maximum rounds; the number actually used is reported (max 84) |
| `OFFDIAG_TRUST` | `1e-18` | round-off contamination guard on the Bell off-diagonal weight |
| `RATIO_FLOOR` | `1e-14` | below this `p_Phi+` the ratios are flagged, not silently used |
| bisection | `1e-10` | achieved bracket width `7.451e-11` at every epsilon |
| coarse scan | 91 points on `q in [0, 0.9]` | sign changes counted, never assumed |

Measured round-off floor at the fixed point (`eps = 0.15`, `q = 0.02`):
`max |Delta R| = 0.0`, `max |Delta p| = 0.0`, `max step = 0.0` — the orbit
settles to a bit-exact fixed point, so `VIOL_TOL = 1e-12` is far above the
floor.

Violation-tolerance sensitivity (`1e-10 / 1e-12 / 1e-14`):

| eps | `1e-10` | `1e-12` | `1e-14` | spread |
|---|---|---|---|---|
| 0.05 | 0.023507999629 | 0.023507999480 | 0.023507999480 | 1.49e-10 |
| 0.15 | 0.062666225955 | 0.062666225806 | 0.062666225806 | 1.49e-10 |
| 0.40 | 0.126513294503 | 0.126513294354 | 0.126513294354 | 1.49e-10 |
| 2/3  | 0.156264485121 | 0.156264485046 | 0.156264485046 | 7.45e-11 |

The spread is at the bisection width, i.e. the answer is insensitive to the
tolerance choice over four orders of magnitude.

## 6. Independent implementation routes

| | route A | route B |
|---|---|---|
| CNOT | `\|0><0\| (x) I + \|1><1\| (x) X` by Kronecker product | the computational-basis permutation it induces |
| H | Kronecker product | explicit index arithmetic |
| channel | einsum partial trace + identity re-insertion | independently coded einsum, own index letters/order |
| postselection | diagonal sub-block | explicit 16x16 projector, then einsum partial trace |
| Bell populations | `<B\|rho\|B>` expectation values | `Tr(\|B><B\| rho)` with projectors built from the stabilizer form `(I+s1 XX)(I+s2 ZZ)/4` |
| propagation | merged unitary program | gate by gate |

Worst disagreement over 7 `(eps, q)` points x 5 rounds each:

| quantity | max difference |
|---|---|
| full retained `rho_(n+1)` | 3.469447e-18 |
| Bell populations | 2.220446e-16 |
| success probability | **0.0** |
| relative ratios | 2.220446e-16 |
| projector vs expectation populations on the same state | 2.220446e-16 |

## 7. Primary result table

Input `rho_0(eps) = (1-eps)|Phi+><Phi+| + eps I/4`, so
`F_in = p_Phi+^(0) = 1 - 3 eps/4`.

| eps | `F_in` | `q_rel_full` | `q_upper_violating` | `q_rel_first` | full − first | limiting | binding round | `Delta R` at upper | `P_succ` at that step | max `C_off` | abs-rise/rel-fall events | rounds at `q_rel` | converged | sign changes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 0.9925 | **0.0049380875** | 0.0049380876 | 0.0049380875 | 0.0 | Psi+ | 0 | 2.565e-11 | 0.969482 | 5.3e-33 | 1 | 7 | yes | 1 |
| 0.02 | 0.9850 | **0.0097546519** | 0.0097546520 | 0.0097546519 | 0.0 | Psi+ | 0 | 1.711e-11 | 0.940374 | 1.1e-32 | 1 | 8 | yes | 1 |
| 0.05 | 0.9625 | **0.0235079995** | 0.0235079996 | 0.0235079995 | 0.0 | Psi+ | 0 | 2.000e-11 | 0.860730 | 1.1e-32 | 2 | 11 | yes | 1 |
| 0.10 | 0.9250 | **0.0442869429** | 0.0442869430 | 0.0442869429 | 0.0 | Psi+ | 0 | 1.210e-11 | 0.749688 | 1.1e-32 | 2 | 14 | yes | 1 |
| 0.15 | 0.8875 | **0.0626662258** | 0.0626662259 | 0.0626662258 | 0.0 | Psi+ | 0 | 1.240e-11 | 0.660135 | 8.9e-33 | 2 | 17 | yes | 1 |
| 0.20 | 0.8500 | **0.0789160329** | 0.0789160330 | 0.0789160329 | 0.0 | Psi+ | 0 | 5.181e-11 | 0.587182 | 5.3e-33 | 2 | 20 | yes | 1 |
| 0.30 | 0.7750 | **0.1058770252** | 0.1058770253 | 0.1058770252 | 0.0 | Psi+ | 0 | 4.912e-11 | 0.477572 | 6.0e-33 | 2 | 28 | yes | 1 |
| 0.40 | 0.7000 | **0.1265132944** | 0.1265132944 | 0.1265132944 | 0.0 | Psi+ | 0 | 5.558e-12 | 0.401509 | 1.2e-32 | 2 | 38 | yes | 1 |
| 0.50 | 0.6250 | **0.1417109164** | 0.1417109165 | 0.1417109164 | 0.0 | Psi+ | 0 | 3.184e-11 | 0.347737 | 1.0e-32 | 2 | 50 | yes | 1 |
| 0.60 | 0.5500 | **0.1520034142** | 0.1520034143 | 0.1520034142 | 0.0 | Psi+ | 0 | 8.811e-11 | 0.309472 | 1.1e-32 | 5 | 65 | yes | 1 |
| 2/3  | 0.5000 | **0.1562644850** | 0.1562644851 | 0.1562644850 | 0.0 | Psi+ | 0 | 5.858e-11 | 0.290440 | 5.6e-33 | 5 | 75 | yes | 1 |
| 0.75 | 0.4375 | **0.1586566045** | 0.1586566046 | 0.1586566045 | 0.0 | Psi+ | 0 | 1.047e-10 | 0.272479 | 8.1e-33 | 7 | 84 | yes | 1 |
| 0.90 | 0.3250 | **0.1540639677** | 0.1540639678 | 0.1540639677 | 0.0 | Psi+ | 0 | 9.450e-12 | 0.253679 | 1.6e-32 | 17 | 82 | yes | 1 |

Every trajectory at `q_rel` converged (`step < 1e-14`) and was trusted (the
off-diagonal guard never fired). Minimum eigenvalue along those trajectories
was between `1.26e-03` and `8.77e-02`; maximum trace error `2.22e-16`; minimum
`p_Phi+` between 0.325 and 0.9925, so the ratio floor was never approached.

Secondary dense sweep: 48 points at `eps = 0.02, 0.04, ... 0.96`, **every one
with exactly one sign change**. The curve is non-monotone in `eps`, peaking
near `eps ~ 0.76` at `q_rel ~ 0.158719` and falling again by `eps = 0.96`.

### Component-resolved thresholds

| eps | `Phi-` first / all | `Psi+` first / all | `Psi-` first / all |
|---|---|---|---|
| 0.01 | 0.0096633921 / 0.0094808429 | **0.0049380875 / 0.0049380875** | 0.0096633921 / 0.0094808498 |
| 0.05 | 0.0426790667 / 0.0389800964 | **0.0235079995 / 0.0235079995** | 0.0426790667 / 0.0389800985 |
| 0.15 | 0.0996848948 / 0.0890431964 | **0.0626662258 / 0.0626662258** | 0.0996848948 / 0.0890431931 |
| 0.40 | 0.1687605765 / 0.1559521251 | **0.1265132944 / 0.1265132944** | 0.1687605765 / 0.1559521251 |
| 2/3  | 0.1893685880 / 0.1800057866 | **0.1562644850 / 0.1562644850** | 0.1893685880 / 0.1800057866 |
| 0.90 | 0.1765107667 / 0.1758829889 | **0.1540639677 / 0.1540639677** | 0.1765107667 / 0.1758829889 |

Three things are worth separating here:

* `Psi+` is the **smallest** component threshold at every epsilon, so it is
  what `q_rel` equals — the limiting component, detected from the data
  (arg-max of `Delta R` just above threshold) and not hard-coded.
* For `Psi+` the first-round and all-round component thresholds coincide
  exactly (difference `0.0` at every epsilon). For `Phi-` and `Psi-` they do
  **not**: the all-round value is lower by up to `1.31e-02` (at `eps = 0.40`).
  So later rounds genuinely do bind for those two components — the first round
  is not universally binding, it simply binds for the component that happens to
  be limiting.
* `Phi-` and `Psi-` first-round thresholds are **bit-identical** (difference
  `0.0`); their all-round thresholds agree to `6.93e-09` at `eps = 0.01` and
  exactly from `eps = 0.30` upward. The small residual is a round-off-driven
  tie-break in a long single-component trajectory, not a physical asymmetry;
  the two populations are computed as separate expectation values, so the
  equality is measured, not imposed.

## 8. "Absolute rises but relative falls"

**50 such events were found**, at every epsilon, already in the first round at
`q` at or just below `q_rel`. Concrete example (`results/4q_qrel/4q_qrel_abs_rise_rel_fall.csv`):

```
eps        = 0.60
q          = 0.15200341418385502      (= q_rel at this eps)
round      = 0
component  = Psi+
p_Psi+     : 0.149999999999999972  ->  0.158302464241787820   (Delta p = +8.302464e-03)
p_Phi+     : 0.549999999999999822  ->  0.580442368943470100
R_Psi+     : 0.272727272727272763  ->  0.272727272700530650   (Delta R = -2.674211e-11)
```

The `Psi+` population rises by `+8.3e-03` while its ratio to the target falls.
This is exactly the regime in which the relative criterion is satisfied and the
absolute criterion is not, and it is the mechanism by which `q_rel` can be the
less restrictive threshold.

The count grows with epsilon (1 event at `eps <= 0.02`, 17 at `eps = 0.90`),
which is consistent with the target population having more room to grow when
the input is noisier.

## 9. Bell diagonality and the transverse direction — NUMERICAL OBSERVATION ONLY

The Bell-basis off-diagonal weight
`C_off = sum_{i != j} |<B_i| rho_n |B_j>|^2` stays at round-off along every
trajectory: the largest value seen anywhere in the study is
**`1.603866e-32`** (i.e. an amplitude of `~1.3e-16`). The input is built exactly
Bell diagonal, so this says the circuit does not generate Bell coherences from
a Bell-diagonal state at double precision.

Why that is trustworthy here, measured rather than assumed: a deliberate
Hermitian Bell-off-diagonal perturbation of amplitude `2.0e-08` added to
`rho_0(0.15)` is mapped in **one** round to amplitude `2.120343e-16`, a ratio of
`1.060171e-08` — i.e. the output transverse amplitude is of the order of the
*square* of the input. A two-amplitude fit gives an amplitude-scaling exponent
of `2.16` (the smaller probe, input `2.0e-09 -> 1.46e-18`, is already at the
double-precision floor, which inflates the fitted exponent above 2). So the map
annihilates the transverse direction to first order rather than amplifying it,
and accumulated round-off cannot manufacture a ratio increase. Subsequent
"growth ratios" in the sequence (up to 2.44) are pure round-off at the `1e-17`
level and must not be read as growth; the study reports them separately for
that reason. The `OFFDIAG_TRUST = 1e-18` guard never fired.

This is reported as a numerical observation. It is **not** used as an analytic
assumption anywhere in this task.

## 10. Degenerate endpoints

Reported separately, never forced into an ordinary threshold root.

**`eps = 0`.** `p^(0) = (1, 0, 0, 0)`, so `R_i^(0) = 0` for all non-target `i`
(the stored value `5.98e-34` for `Phi-` is the round-off of the exact
construction). Since `R_i >= 0`, no `q` can lower them further. At `q = 0` the
state is a fixed point and `Delta R = 0` exactly, which under the tolerance
policy is not a violation; at every `q > 0` the circuit creates error
populations from nothing and `Psi+` violates already in round 0
(`Delta R = +5.00e-07` at `q = 1e-6`, rising to `+9.99e-01` at `q = 0.9`). The
scan therefore returns the degenerate root `q_rel(0) = 0`.

**`eps = 1`.** `rho_0 = I/4`, `p^(0) = (1/4, 1/4, 1/4, 1/4)`, so
`R_i^(0) = 1` for all `i`. The maximally mixed state is a fixed point of the
postselected circuit at every `q` (`P_succ = 1/4`), `Delta R = 0.0` exactly at
all eight probed `q` values, and the coarse scan finds **zero** sign changes:
there is no root in `q in [0, 0.9]` and the criterion is never violated. The
strict inequality `R_i^(n+1) < R_i^(n)` is never actually achieved either — the
ratios simply do not move.

## 11. Negative / sensitivity controls

Baseline `q_rel_full(eps = 0.15) = 0.06266622580587862`. All four controls
changed the state, and all four were detected. They are guards only and never
enter the `q_rel` computation.

| control | perturbation | `max \|Delta rho\|` | `max \|Delta R\|` | `q_rel` shift | root in scan |
|---|---|---|---|---|---|
| A | noise after CNOT #3 set to `q + 1e-3` | 1.058e-04 | 1.724e-04 | **−2.4946e-04** | yes |
| B | depolarizing channel after CNOT #2 removed | 5.118e-03 | 8.251e-03 | **+1.9713e-02** | yes |
| C | CNOT #4 reversed, `(2,1) -> (1,2)` | 2.847e-01 | 9.667e-01 | — | **no** (violates at every scanned `q`) |
| D | postselect `(q3,q4) = (0,1)` instead of `(0,0)` | 2.355e-01 | 7.827e-01 | — | **no** (violates at every scanned `q`) |

Controls A and B shift the threshold in the expected directions (more noise on
one CNOT lowers it; removing a channel raises it). Controls C and D destroy
relative suppression entirely, so the structural specification is not
interchangeable.

## 12. Near-threshold trajectories

For `eps in {0.05, 0.15, 0.40, 2/3}` at `q_rel - 5e-4`, `q_rel`, `q_rel + 5e-4`
(`small_delta = 5e-4`, five orders of magnitude above the bisection width),
40 rounds each, tabulated in `results/4q_qrel/4q_qrel_trajectories.csv` and
plotted in `4q_qrel_trajectories_R.*` (ratios) and `4q_qrel_trajectories_p.*`
(absolute populations):

| eps | `q_below` | `q_near` | `q_above` | first violation above |
|---|---|---|---|---|
| 0.05 | 0.0230079995 (none) | 0.0235079995 (none) | 0.0240079995 | round 0, `Psi+`, `Delta R = +2.844e-04`, `Delta p = +4.188e-04` |
| 0.15 | 0.0621662258 (none) | 0.0626662258 (none) | 0.0631662258 | round 0, `Psi+`, `Delta R = +3.603e-04`, `Delta p = +1.509e-03` |
| 0.40 | 0.1260132944 (none) | 0.1265132944 (none) | 0.1270132944 | round 0, `Psi+`, `Delta R = +5.874e-04`, `Delta p = +6.223e-03` |
| 2/3  | 0.1557644850 (none) | 0.1562644850 (none) | 0.1567644850 | round 0, `Psi+`, `Delta R = +8.146e-04`, `Delta p = +8.501e-03` |

In every case the violation is immediate (round 0) and in `Psi+`; the ratios
are monotone decreasing below threshold, and no late-round or non-monotone
relative behaviour was observed on the limiting component.

## 13. SECONDARY: comparison with the frozen `q_abs`

Read once, after the table above was complete, from
`results/4q_qabs/4q_qabs_circuit.csv` (which this study never writes to; a test
checks its hash is unchanged). Nine epsilon values are common to both studies.

| eps | `q_abs` (frozen) | `q_rel` (this study) | `q_rel - q_abs` |
|---|---|---|---|
| 0.02 | 0.0097070588 | 0.0097546519 | +0.0000475931 |
| 0.05 | 0.0232310922 | 0.0235079995 | +0.0002769072 |
| 0.10 | 0.0432979403 | 0.0442869429 | +0.0009890026 |
| 0.15 | 0.0606654173 | 0.0626662258 | +0.0020008085 |
| 0.20 | 0.0756973614 | 0.0789160329 | +0.0032186716 |
| 0.30 | 0.0998446590 | 0.1058770252 | +0.0060323662 |
| 0.40 | 0.1173995783 | 0.1265132944 | +0.0091137160 |
| 0.50 | 0.1294002390 | 0.1417109164 | +0.0123106774 |
| 0.60 | 0.1364550036 | 0.1520034142 | +0.0155484105 |

`q_abs <= q_rel` holds at all nine epsilon values, with the gap growing
monotonically from `4.76e-05` to `1.55e-02` over the tested range. Labelled as
a numerical observation consistent with the expected logical relationship (the
relative criterion is weaker than the absolute one), not as a newly proven
theorem.

## 14. Conclusions

### VERIFIED IMPLEMENTATION FACT

* The gate list used here is the repository's own frozen 4Q specification:
  `CNOT q3->q4, q2->q4, q1->q4, q3->q2, q3->q1`, then `H(q3)`, measure
  `(q3,q4)`, postselect `(0,0)`, retain `(q1,q2)`. Verified textually against
  both `src/pqec_distill/circuit.py` and `scripts/verify_4q_qabs_circuit.py`.
* The noise is the repository's two-qubit replacement depolarizing channel
  applied after each of the five CNOTs on exactly that CNOT's wires; `q = 0`
  is the exact identity and `q = 1` replaces the pair by `I/4` exactly.
* The postselected conditional state is normalised, Hermitian and positive at
  every round tested (trace error `<= 2.22e-16`, minimum eigenvalue
  `>= 1.26e-03` along the threshold trajectories).
* Two numerically independent implementation routes agree on the full retained
  state to `3.47e-18` and on every derived quantity to `2.22e-16`, with the
  success probability agreeing exactly.
* The experiment is sensitive to the specification: all four negative controls
  change the result, two of them destroying suppression entirely.

### NUMERICAL OBSERVATION

* `q_rel(eps)` over the mandatory epsilon set is the table in §7, bracketed to
  `7.451e-11` with exactly one predicate sign change at every epsilon (and at
  all 48 dense-sweep points).
* `q_rel_full = q_rel_first` to `0.0` at every epsilon tested. The binding
  round is round 0 and the limiting component is `Psi+` at every epsilon.
* The first round is binding only for the limiting component. For `Phi-` and
  `Psi-` the all-round component thresholds are strictly lower than their
  first-round values (by up to `1.31e-02`), so "the first round binds" is a
  property of `Psi+` here, not of the circuit in general.
* `p_Phi- = p_Psi-` along the trajectories to `< 1e-14`, and their first-round
  component thresholds are bit-identical.
* Bell diagonality survives to round-off (max `C_off = 1.60e-32`), and the
  transverse direction is annihilated to first order (one round takes a
  `2.0e-08` amplitude to `2.12e-16`; fitted scaling exponent `2.16`, inflated
  above 2 by the double-precision floor at the smaller probe).
* Rounds in which `Delta p_i > 0` while `Delta R_i < 0` do occur — 50 of them,
  with the `eps = 0.60` example in §8 showing `Delta p = +8.30e-03` against
  `Delta R = -2.67e-11`.
* `q_abs <= q_rel` at all nine shared epsilon values, with the gap growing from
  `4.76e-05` to `1.55e-02`.
* `eps = 0` gives the degenerate root `q_rel = 0`; `eps = 1` is a fixed point at
  which the ratios never move and no root exists in `q in [0, 0.9]`.

### NOT YET PROVED ANALYTICALLY

* **The exact analytic formula, or exact closed-form value, of `q_rel(eps)` for
  the 4Q protocol.** Every number in §7 is a bisection result from the full
  noisy circuit, accurate to `~7.5e-11` in `q`, and nothing more.
* That `Psi+` is the limiting component at every epsilon, rather than only on
  the range scanned.
* That the first round binds the overall threshold for all epsilon.
* That `q_rel_full = q_rel_first` exactly, as opposed to agreeing to the
  bisection width.
* That Bell diagonality is preserved exactly by the noisy 4Q map, and that the
  transverse amplitude-scaling exponent is exactly 2.
* That `q_abs(eps) <= q_rel(eps)` holds as an inequality for all epsilon.
* The location and value of the maximum of `q_rel(eps)` near `eps ~ 0.76`.

These will be derived separately; the numbers above are frozen for that
comparison.

## 15. Files

```
scripts/verify_4q_qrel_circuit.py          the study (self-contained)
tests/test_4q_qrel_circuit.py              42 automated tests
docs/4q_qrel_circuit_numerical.md          this document
results/4q_qrel/
    4q_qrel_circuit.csv                    the main table (§7)
    4q_qrel_dense.csv                      48-point secondary epsilon sweep
    4q_qrel_trajectories.csv               near-threshold round-by-round records
    4q_qrel_abs_vs_rel.csv                 SECONDARY q_abs comparison
    4q_qrel_verification.json              every number quoted above
    4q_qrel_component_thresholds.csv       per-component, first and all rounds
    4q_qrel_degenerate_endpoints.csv       eps = 0 and eps = 1 probes
    4q_qrel_abs_rise_rel_fall.csv          the 50 events of §8
    4q_qrel_route_comparison.csv           route A vs route B
    4q_qrel_negative_controls.csv          the four controls
    4q_qrel_tolerance_sensitivity.csv      VIOL_TOL study
    4q_qrel_curve.{pdf,png}                q_rel vs epsilon
    4q_qrel_first_vs_full.{pdf,png}        all-round vs first-round predicate
    4q_qrel_components.{pdf,png}           component thresholds
    4q_qrel_trajectories_R.{pdf,png}       ratios near threshold
    4q_qrel_trajectories_p.{pdf,png}       absolute populations near threshold
    4q_qrel_abs_vs_rel.{pdf,png}           SECONDARY q_abs comparison
```

The frozen `results/4q_qabs/` directory is read once and never written.

Environment of the recorded run: Python 3.11.15, NumPy 2.4.6, 174.5 s.
