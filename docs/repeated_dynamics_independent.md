# Repeated dynamics, derived independently from the verified one-round map

This note re-derives the repeated-application dynamics of the Type 3, Type 4
and 4Q circuits starting **only** from the one-round exact maps that were
validated row-by-row against the frozen circuit-only reference
(`results/data/circuit_reference/circuit_reference.csv`).

Nothing in this note was taken from an existing repeated-dynamics document,
module or result file.  The exclusion list is recorded in
`results/data/repeated_dynamics_independent/metadata.json`.

**Disclosure.** This derivation was produced inside a working session in
which earlier repeated-dynamics numbers for these circuits had already been
computed, so those numbers were present in the analyst's context.  No stored
result file, module or document from the exclusion list was read while
producing this note, and every equation below is derived here from the
one-round map; but this is an *independent derivation path*, not a blind
replication.

---

## 1. Starting point

The one-round maps (verified, `src/pqec_distill/analytic_exact_map.py`) act on
the Bell-diagonal Pauli coordinates defined by

```
rho = (1/4) ( II + x XX + y YY + z ZZ )
```

with `p_Phi+ = (1+x-y+z)/4`, `p_Phi- = (1-x+y+z)/4`,
`p_Psi+ = (1+x+y-z)/4`, `p_Psi- = (1-x-y-z)/4`, and `qbar = 1-q`.

**Type 3 (k = 4) and Type 4 (k = 2):**

```
N  = 1 + qbar^k (x^2 + y^2 + z^2)
x' = 2 qbar^(k+2) (x - y z) / N
y' = 2 qbar^(k+2) (y - x z) / N
z' =   qbar^(k+1) (1 + qbar) (z - x y) / N
```

**4Q:**

```
N  = 1 + qbar^5 (x^2 + y^2) + qbar^3 z^2
x' = qbar^3 [ (1+qbar) x - 2 qbar^2 y z ] / N
y' = qbar^4 [ (1+qbar) y - 2 qbar   x z ] / N
z' = qbar^4 [ (1+qbar) z - 2 qbar   x y ] / N
```

For 4Q, `P_succ = N/4` is the physical per-round postselection probability.
For Type 3/4 the analogous quantity is a *trace weight*, not a success
probability, and is not used here.

## 2. The Bell-isotropic initial line

```
rho_0(eps) = (1-eps) Phi+ + eps I/4
p^(0)      = ( 1 - 3eps/4,  eps/4,  eps/4,  eps/4 )
```

In Pauli coordinates, with `t = 1 - eps`,

```
(x_0, y_0, z_0) = ( t, -t, t ).
```

So the initial line lies simultaneously on `y = -x` **and** on `y = -z`.
Which of the two matters depends on the circuit (section 3).

## 3. Invariant relations, proved from the map

These were first noticed in the numerical trajectories (the relation holds to
machine precision at every round) and then proved symbolically.

**Type 3 / Type 4 — the plane `y = -x` is invariant.**

```
x' + y' = 2 qbar^(k+2) [ (x - y z) + (y - x z) ] / N
        = 2 qbar^(k+2) (x + y)(1 - z) / N
```

so `x + y = 0` implies `x' + y' = 0`, for every `z` and every `q`.
(The same algebra gives `x' - y' = 2 qbar^(k+2)(x-y)(1+z)/N`, so `y = +x` is
invariant too; the Bell-isotropic line is on the `y = -x` sheet.)

**4Q — the plane `y = -z` is invariant.**

```
y' + z' = qbar^4 (y + z) [ (1+qbar) - 2 qbar x ] / N
```

so `y + z = 0` implies `y' + z' = 0`.

**Scope.** These statements are about the *invariance of those two-dimensional
manifolds*.  They are **not** a claim that an arbitrary Bell-diagonal state
satisfies any of these relations, and not a claim that the other plane is
invariant for the other circuit — it is not (the corresponding combination is
generically non-zero).

In Bell-population language, `y = -x` is `p_Psi+ = p_Psi-` (Type 3/4) and
`y = -z` is `p_Phi- = p_Psi-` (4Q).  Consequently the dominant non-target
component is `Phi-` for Type 3/4 and `Psi+` for 4Q.

## 4. Reduced maps

**Type 3 / Type 4**, with `u = x = -y`, `v = z`, `F = (1 + 2u + v)/4`:

Substituting `y = -u`, `x = u`, `z = v` gives
`x^2+y^2+z^2 = 2u^2+v^2`, `x - yz = u(1+v)`, `z - xy = v + u^2`, hence

```
D  = 1 + qbar^k (2 u^2 + v^2)
u' = 2 qbar^(k+2) u (1 + v) / D
v' =   qbar^(k+1) (1 + qbar) (u^2 + v) / D
```

with `k = 4` (Type 3) and `k = 2` (Type 4).  Note the two circuits differ
*only* through `k`: the Type 3 / Type 4 dynamics are one two-parameter family.

**4Q**, with `a = x`, `b = z = -y`, `F = (1 + a + 2b)/4`:

Substituting `y = -b`, `z = b` gives `x^2+y^2 = a^2+b^2`, `z^2 = b^2`,
`yz = -b^2`, `xy = -ab`, hence

```
D  = 1 + qbar^5 a^2 + (qbar^5 + qbar^3) b^2
a' = qbar^3 [ (1+qbar) a + 2 qbar^2 b^2 ] / D
b' = qbar^4 b [ (1+qbar) + 2 qbar a ] / D
```

The Bell-isotropic start is `(u,v) = (t,t)` and `(a,b) = (t,t)` respectively.

## 5. Fixed points

### 5.1 Type 3 / Type 4

*Branch with `u = 0`.*  Then `D = 1 + qbar^k v^2` and the second equation gives
`v [ 1 + qbar^k v^2 - qbar^(k+1)(1+qbar) ] = 0`, i.e. either

- `v = 0` — the **maximally mixed** fixed point, `F = 1/4`, present for all `q`; or
- the **off-plane branch**

  ```
  u = 0,   v^2 = [ qbar^(k+1) (1 + qbar) - 1 ] / qbar^k ,
  ```

  which exists only while `qbar^(k+1)(1+qbar) > 1`.  Its fidelity is
  `F = (1+v)/4`, i.e. a state with `x = y = 0` and only a `ZZ` correlation.

*Branch with `u != 0`.*  The first equation forces `D = 2 qbar^(k+2)(1+v)`.
Feeding that into the second gives `2 qbar v (1+v) = (1+qbar)(u^2+v)`, i.e.

```
u^2 = v ( 2 qbar v + qbar - 1 ) / (1 + qbar).
```

Substituting back into `D = 2 qbar^(k+2)(1+v)` and clearing `(1+qbar)`:

```
qbar^k (5 qbar + 1) v^2
  - 2 qbar^k ( qbar^3 + qbar^2 - qbar + 1 ) v
  - (qbar + 1) ( 2 qbar^(k+2) - 1 )  =  0.
```

Of its two roots, only those with `u^2 >= 0` are physical.  The larger-`F`
root is the branch continuously connected to `Phi+` as `q -> 0`; it is the
**target branch**.  The other root is its **competing (saddle) partner**.

### 5.2 4Q

*Branch with `b = 0`.*  `a [ 1 + qbar^5 a^2 ] = qbar^3 (1+qbar) a` gives

- `a = 0` — maximally mixed; or
- the **off-plane branch**

  ```
  b = 0,   a^2 = [ qbar^3 (1 + qbar) - 1 ] / qbar^5 ,
  ```

  existing while `qbar^3(1+qbar) > 1`, with `F = (1+a)/4`, i.e. only an `XX`
  correlation.

*Branch with `b != 0`.*  The second equation forces
`D = qbar^4 [ (1+qbar) + 2 qbar a ]`.  The first then gives

```
b^2 = a ( 2 qbar^2 a + qbar^2 - 1 ) / (2 qbar^2),
```

and substituting back,

```
2 qbar^3 (2 qbar^2 + 1) a^2  -  qbar ( 3 qbar^4 + 1 ) a
  -  2 ( qbar^5 + qbar^4 - 1 )  =  0.
```

Same reading: larger-`F` root = target branch, other root = competing saddle.

## 6. Local stability

Linearising at the maximally mixed point (`D -> 1`, the maps become diagonal):

| circuit | eigenvalues at `F = 1/4` |
|---|---|
| Type 3 / Type 4 | `2 qbar^(k+2)` and `qbar^(k+1) (1 + qbar)` |
| 4Q | `qbar^3 (1 + qbar)` and `qbar^4 (1 + qbar)` |

Away from the origin the 2x2 Jacobian is evaluated by high-precision central
differences (`mpmath`, dps = 60) in
`scripts/analyze_repeated_dynamics.py::jacobian`, and its two eigenvalues come
from the trace/determinant formula.  A branch is called stable when its
spectral radius is `< 1`.

**On the off-plane branch the Jacobian is triangular and both eigenvalues are
closed-form.**  The vanishing coordinate enters its own update equation only
linearly (`u'` is proportional to `u`; `b'` is proportional to `b`), so the
two diagonal derivatives *are* the eigenvalues.  Writing

```
M = qbar^(k+1) (1 + qbar)   (Type 3/4)        M = qbar^3 (1 + qbar)   (4Q)
```

the branch exists exactly when `M > 1`, its fixed-point condition reads
`D = M`, and

```
lambda_parallel = (2 - M) / M                       (both families)
lambda_perp     = 2 qbar (1 + v*) / (1 + qbar)      (Type 3/4)
lambda_perp     = qbar + 2 qbar^5 a* / M            (4Q)
```

Checked against the numerical Jacobian in
`results/data/repeated_dynamics_independent/offplane_eigenvalues.csv`:
worst disagreement `2.0e-40` at dps = 60.

Two consequences follow, and both are independently confirmed there:

1. `M` is *also* one of the two closed-form eigenvalues of the maximally mixed
   fixed point (agreement exactly `0`).  So at `M = 1` the off-plane branch is
   born precisely where the origin loses stability in that direction, and
   `lambda_parallel = 1` there.  **This is a transcritical bifurcation**, not a
   saddle node: the branch does not appear out of nothing, it passes through
   the origin and exchanges stability with it.
2. Type 4 has `k = 2`, so its `M` is *identically* 4Q's `M`.  The two circuits
   share the same off-plane existence condition and the same
   `lambda_parallel` at every `q` (difference exactly `0` at all tested `q`),
   although the branch coordinate and `lambda_perp` differ.

## 7. Critical `q` values

Each critical `q` is obtained by root-solving an **analytic condition**, not by
grid search (`mpmath.findroot`, anderson, dps = 60):

1. `q_saddle_node_discriminant_zero` — discriminant of the fixed-point
   quadratic of section 5 vanishes (target branch and its competing partner
   merge and disappear).
2. `q_offplane_branch_vanishes` — `qbar^(k+1)(1+qbar) = 1` (Type 3/4) or
   `qbar^3(1+qbar) = 1` (4Q); the off-plane branch collides with the origin.
3. `q_origin_lambda{1,2}_eq_1` — each closed-form origin eigenvalue crosses 1.
4. `q_offplane_branch_spectral_radius_eq_1` — the off-plane branch changes
   stability.
5. `q_target_branch_fidelity_eq_half` — the target branch crosses `F = 1/2`
   (the separability point of a Bell-diagonal state).

A condition `q_target_branch_spectral_radius_eq_1` was also tested for and
**no root was found** for any of the three circuits: the target branch never
destabilises on its own — it is destroyed by the saddle node while still
attracting.

## 8. What the numerics add

`scripts/broad_scan.py` runs the exploratory grid; `scripts/analyze_basins_precision.py`,
`scripts/check_separatrix.py`, `scripts/transient_scaling.py` and
`scripts/verify_branch_eigenvalues.py` complement the algebra with

- a scan along the Bell-isotropic line at fixed `q`, classifying the endpoint
  of each orbit **against the analytically computed fixed points** — an orbit
  is only given an attractor label when it is within `1e-6` of a fixed point,
  the residual of the *full four-component* one-round map there is below
  `1e-9`, **and** that fixed point's analytic spectral radius is `< 1`.  A
  small step size alone never earns a label; anything else is reported as
  `unresolved`;
- bisection of the `eps` at which the label changes, at fixed `q`;
- a float64 vs `mpmath` (dps = 60) orbit comparison at the slow / near-critical
  parameter points;
- an independent integration of both sides of every bisected boundary at
  dps = 60, long enough for the two sides to separate unambiguously;
- a measurement of the saddle-node passage time as `q -> q_SN`, and its
  log-log slope;
- a check of the closed-form off-plane eigenvalues against the numerical
  Jacobian.

## 9. What the numbers show

### 9.1 No cycles, and one invariant plane per circuit

Over the exploratory grid (3 circuits x 18 `q` x 20 `eps`, 4000 rounds,
`broad_scan.csv`):

- **no orbit has a tail cycle of period 2..8** — every orbit settles on a
  fixed point; there is no oscillatory or period-doubling behaviour anywhere
  in the scanned range;
- the preserved combination stays at `<= 4.4e-16` for every orbit, while the
  *other* combination — which also vanishes at `n = 0`, because the
  Bell-isotropic initial state sits on both planes — grows to `5.9e-2`
  (Type 3), `9.9e-2` (Type 4) and `2.8e-1` (4Q).  This is the sharpest
  evidence that exactly one of the two planes is invariant per circuit, and
  that the other is not;
- the fidelity history is monotone for 1073 of 1080 orbits.  The only orbit
  with a genuine interior maximum above its start *and* above its end is
  Type 4 at `q = 0.17`, `eps = 0.55`: `F` rises from `0.587500` to `0.589942`
  at round 6 and then falls back to the fixed point `0.589903`.  The
  non-monotone 4Q cases are the opposite shape — `F` first drops and then
  climbs to a *higher* fixed point (e.g. `q = 0.17`, `eps = 0.9`:
  `0.325 -> 0.614101` reached at round 190).

### 9.2 Bistability and basins (`basin_scan.csv`, `basin_boundaries.csv`)

Attractors are assigned by matching the endpoint to the analytic fixed points
(rule in section 8), never by step size alone.  Concrete bistable examples,
same `q`, different `eps`, different asymptotic state:

| circuit | `q` | `eps` | endpoint | `eps` | endpoint |
|---|---|---|---|---|---|
| Type 3 | 0.130 | 0.50 | `F = 0.478636230374` | 0.90 | `F = 0.25` (maximally mixed) |
| Type 4 | 0.180 | 0.50 | `F = 0.535946533813` | 0.96 | `F = 0.268010423976` (off-plane) |
| Type 4 | 0.185 | 0.50 | `F = 0.497556590744` | 0.95 | `F = 0.25` |
| 4Q | 0.178 | 0.50 | `F = 0.535172257231` | 0.95 | `F = 0.294631996395` (off-plane) |
| 4Q | 0.1805 | 0.50 | `F = 0.474264607325` | 0.95 | `F = 0.265276172816` (off-plane) |

The competing attractor is **not the same object in all three circuits**:

- Type 3 in its bistable window (`q >= 0.12`) has already lost its off-plane
  branch (that branch dies at `q = 0.11872853836643040559`), so the competitor
  is the maximally mixed state itself;
- Type 4 has a narrow window (`q ~ 0.178..0.1808`) where the competitor is the
  **off-plane branch** (`x = y = 0`, only a `ZZ` correlation), and above
  `q = 0.1808` that branch has merged into the origin and the competitor
  becomes `F = 1/4`;
- 4Q's competitor over its whole bistable window is the **off-plane branch**
  (`y = z = 0`, only an `XX` correlation, `p_Phi+ = p_Psi+`).  Just past its
  saddle node, at `q = 0.1807`, the off-plane branch is the *global* attractor:
  every `eps` in `(0,1)` ends at `F = 0.259535604402`.

Basin boundaries along the Bell-isotropic line, bisected to a bracket below
one float64 `ulp` (`basin_boundaries.csv`), move to smaller `eps` as `q` grows:

```
Type 3   q = 0.12    eps* = 0.947755962860843
         q = 0.125   eps* = 0.891937076670015
         q = 0.129   eps* = 0.823587115843650
         q = 0.13057 eps* = 0.750504832456704
Type 4   q = 0.18    eps* = 0.936780856880702
         q = 0.185   eps* = 0.884623406012601
         q = 0.18941 eps* = 0.783661273288113
4Q       q = 0.178   eps* = 0.836667803783961
         q = 0.180   eps* = 0.777356234194224
         q = 0.18066 eps* = 0.728016805676009
```

`eps = 1` is a special case and is reported as `unresolved`, deliberately: it
*is* the maximally mixed state, an exact fixed point, so the orbit has step
size exactly `0` from round 1 — while the analytic spectral radius there is
`> 1` at every `q` below the stabilisation point (e.g. `1.508873` for Type 3
at `q = 0.05`).  A converged-looking orbit sitting on an unstable fixed point
is exactly the misclassification the labelling rule is designed to refuse.

### 9.3 Long transients at the saddle node (`transient_scaling.csv`)

Fixing `eps = 0.1` and approaching the saddle-node `q` from either side, the
number of rounds diverges.  Above it, where the target branch no longer
exists but its bottleneck ("ghost") remains, the passage time obeys

```
N  ~  C (q - q_SN)^p
```

with the local log-log slope converging to `-1/2` — the saddle-node signature:

| `q - q_SN` | Type 3 | Type 4 | 4Q |
|---|---|---|---|
| `1e-3` | 116 | 151 | 171 |
| `1e-4` | 394 | 512 | 557 |
| `1e-5` | 1262 | 1640 | 1743 |
| `1e-6` | 4005 | 5205 | 5486 |
| local slope, last decade | `-0.5009` | `-0.5009` | `-0.4984` |

Below `q_SN` the settling time diverges the same way (14038 / 17832 / 18901
rounds at `q_SN - 1e-6`), and the endpoint fidelity creeps down to the
saddle-node value.  Re-running the `q_SN + 1e-5` case at dps = 60 gives the
**same round count to the round** (1262 / 1640 / 1743, difference 0), so the
crawl is a property of the map, not of double precision.

### 9.4 Double vs high precision (`precision_check.csv`, `separatrix_check.csv`)

- 4000-round orbits at the 12 most sensitive `(circuit, eps, q)` points:
  float64 and dps = 60 differ by at most `1.4e-14` in fidelity and
  `1.9e-14` in the reduced coordinates.
- Every basin boundary, stepped `+-1e-3` and `+-1e-6` and integrated for
  20 000 rounds at dps = 60, lands on a fixed point that is **stable by the
  analytic criterion**, with distance `0` to it in almost every case and
  `<= 5.8e-13` in the worst; float64 agrees with dps = 60 to `<= 4e-14`.
- Every endpoint recorded in the basin scan was re-checked against the full
  four-component map (`endpoint_verification.csv`): the largest residual among
  labelled endpoints is `2.2e-16`, all population vectors sum to 1 exactly and
  have no negative entry, and the only `unresolved` rows are the 15 `eps = 1`
  rows discussed above.

**Conclusion on reliability.** Double precision is sufficient everywhere that
was tested; the danger in this problem is not round-off but *interpretation* —
a small step size near the separatrix or at an unstable fixed point, which is
why every attractor label here is backed by an analytic fixed point and its
analytic spectral radius.

## 10. Frozen outputs

All under `results/data/repeated_dynamics_independent/`:

| file | contents |
|---|---|
| `fixed_points.csv` | every branch vs `q`, with both Jacobian eigenvalues, spectral radius, stability flag and map residual |
| `critical_points.json` | the critical `q` values of section 7, 20 significant digits |
| `representative_trajectories.csv` | round-by-round Bell populations, Pauli coordinates, step size, fidelity, non-target/target ratios and (4Q) `P_succ` |
| `basin_scan.csv` | endpoint attractor along the Bell-isotropic line at each scanned `q` |
| `basin_boundaries.csv` | bisected `eps` separating basins at fixed `q` |
| `precision_check.csv` | float64 vs dps-60 orbits at the sensitive points |
| `endpoint_verification.csv` | every distinct endpoint re-checked against the full map and the analytic fixed points |
| `broad_scan.csv` | the exploratory grid: convergence, tail cycles, monotonicity, overshoot, which Bell component grows, invariant-plane defects |
| `offplane_eigenvalues.csv` | closed-form vs numerical Jacobian eigenvalues on the off-plane branch, and `M` vs the origin eigenvalues |
| `separatrix_check.csv` | both sides of every basin boundary, integrated at dps = 60 and in float64 |
| `transient_scaling.csv`, `transient_scaling_fit.json` | saddle-node passage / settling times vs `q - q_SN`, and the fitted exponent |
| `metadata.json`, `basins_precision_metadata.json` | code commit, precision, tolerances, scan ranges, root solver, exclusion list |

These files are **frozen**: they are the independent result to be compared
against the separately held analytic calculation, and must not be regenerated
after that comparison begins.
