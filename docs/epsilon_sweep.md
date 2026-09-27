# Repeated dynamics along the Bell-isotropic initial line

Type 3, Type 4 and 4Q: how the Bell populations move under repeated rounds, and
how the answer depends on the strength `eps` of the input mixing.  Reference
run: `q = 0.02`, `eps in {0.02, 0.05, 0.10, 0.15, 0.20, 0.30}`, `n = 0..50`.

```
python scripts/epsilon_sweep.py                       # all three protocols
python scripts/epsilon_sweep.py --protocol 4Q --q 0.05
python scripts/epsilon_sweep.py --q 0.12 --eps 0.05 0.2 0.5
```

Type 5 is excluded: its noisy dynamics does not stay Bell-diagonal, so there is
no exact Bell-diagonal map for it.

## 1. What was used

The dynamics come from the **verified exact Bell-diagonal maps** in
`pqec_distill.analytic_exact_map` (`analytic_map_type3`, `analytic_map_type4`,
`analytic_map_4q`), which were validated row-by-row against the frozen
circuit-only reference.  No map was re-derived or guessed.  A full
density-matrix simulation of the actual circuit runs alongside as a cross-check
at every `eps`.

For 4Q the round is a physical postselection: the normalised `(q3,q4) = (0,0)`
branch feeds the next round, and the branch probability is recorded as
`P_succ`.  `P_succ^(n)` is the probability of the round that **produced**
`rho_n`, so it is undefined at `n = 0`.

`q` is the two-qubit **replacement** depolarizing strength applied after every
CNOT on that CNOT's two qubits — the same convention, at the same numerical
value, as the parent repository's `eps2`.

## 2. Checks

| check | Type 3 | Type 4 | 4Q |
|---|---|---|---|
| CNOT count | **16** | **14** | **5** |
| `\|sum_i p_i - 1\|` | `<= 2.2e-16` | `<= 2.2e-16` | `<= 2.2e-16` |
| negative populations | none | none | none |
| invariant pair | `p_Psi+ = p_Psi-` | `p_Psi+ = p_Psi-` | `p_Phi- = p_Psi-` |
| worst deviation of that pair | `2.8e-17` | `2.8e-17` | **`0.0`** |
| exact map vs full density matrix, 50 rounds | `<= 1e-15` | `<= 6.7e-16` | `<= 1e-15` |
| Bell off-diagonal weight in the full simulation | `<= 4e-30` | `<= 9.5e-28` | `<= 1.5e-33` |
| `q` = parent `eps2` (2-qubit / 5-qubit) | `4.4e-16` / `6.9e-18` | same | same |

## 3. Comparison across protocols (`q = 0.02`)

| | Type 3 | Type 4 | 4Q |
|---|---|---|---|
| CNOTs | 16 | 14 | 5 |
| `F_inf` | 0.955802561784 | 0.964381114877 | **0.978775759863** |
| converged `(Phi+, Phi-, Psi+, Psi-)` | (0.955803, 0.018029, 0.013084, 0.013084) | (0.964381, 0.015188, 0.010215, 0.010215) | (0.978776, 0.005348, 0.010529, 0.005348) |
| same fixed point for every `eps`? | yes, spread `0.0` | yes, spread `0.0` | yes, spread `0.0` |
| invariant pair | `Psi+ = Psi-` | `Psi+ = Psi-` | `Phi- = Psi-` |
| dominant error component | `Phi-` | `Phi-` | `Psi+` |
| asymptotic contraction / round | 0.0651409 | 0.0426488 | 0.0416654 |
| independent Jacobian spectral radius | 0.0651424 | 0.0427601 | 0.0408291 |
| rounds to `1e-12` | 9–10 | 7–9 | 7–9 |
| break-even `eps` (`F_0 = F_inf`) | 0.058929917622 | 0.047491846831 | 0.028298986849 |
| `eps` in the set whose fidelity falls | 0.02, 0.05 | 0.02 | 0.02 |
| `P_succ` at the fixed point | — | — | 0.898865082648 |

The last two rows of the contraction block are two estimates of the same thing:
the per-round residual ratio measured here, and the Jacobian spectral radius
found earlier by the separate repeated-dynamics analysis.  The measured ratio
is still approaching that value when round-off stops the sequence, so it sits a
fraction of a percent (Type 3/4) to about 2% (4Q) above it.

## 4. Per-protocol tables

### Type 3 (16 CNOT)

| `eps` | `F_0` | `F_1` | `F_2` | `F_5` | `F_10` | `F_50` | `F_inf` | `F_inf - F_0` |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.9850000000 | 0.9573365617 | 0.9559013371 | 0.9558025890 | 0.9558025618 | 0.9558025618 | 0.955802561784 | **−0.0291974382** |
| 0.05 | 0.9625000000 | 0.9562301643 | 0.9558310470 | 0.9558025698 | 0.9558025618 | 0.9558025618 | 0.955802561784 | **−0.0066974382** |
| 0.10 | 0.9250000000 | 0.9533831502 | 0.9556451171 | 0.9558025189 | 0.9558025618 | 0.9558025618 | 0.955802561784 | +0.0308025618 |
| 0.15 | 0.8875000000 | 0.9490413223 | 0.9553473282 | 0.9558024372 | 0.9558025618 | 0.9558025618 | 0.955802561784 | +0.0683025618 |
| 0.20 | 0.8500000000 | 0.9428778733 | 0.9548943470 | 0.9558023121 | 0.9558025618 | 0.9558025618 | 0.955802561784 | +0.1058025618 |
| 0.30 | 0.7750000000 | 0.9234664642 | 0.9532228559 | 0.9558018440 | 0.9558025618 | 0.9558025618 | 0.955802561784 | +0.1808025618 |

### Type 4 (14 CNOT)

| `eps` | `F_0` | `F_1` | `F_2` | `F_5` | `F_10` | `F_50` | `F_inf` | `F_inf - F_0` |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.9850000000 | 0.9650787098 | 0.9644105150 | 0.9643811172 | 0.9643811149 | 0.9643811149 | 0.964381114877 | **−0.0206188851** |
| 0.05 | 0.9625000000 | 0.9643183787 | 0.9643793783 | 0.9643811148 | 0.9643811149 | 0.9643811149 | 0.964381114877 | +0.0018811149 |
| 0.10 | 0.9250000000 | 0.9620827465 | 0.9642850504 | 0.9643811078 | 0.9643811149 | 0.9643811149 | 0.964381114877 | +0.0393811149 |
| 0.15 | 0.8875000000 | 0.9583945214 | 0.9641202649 | 0.9643810955 | 0.9643811149 | 0.9643811149 | 0.964381114877 | +0.0768811149 |
| 0.20 | 0.8500000000 | 0.9529234534 | 0.9638543399 | 0.9643810756 | 0.9643811149 | 0.9643811149 | 0.964381114877 | +0.1143811149 |
| 0.30 | 0.7750000000 | 0.9349802959 | 0.9627926487 | 0.9643809953 | 0.9643811149 | 0.9643811149 | 0.964381114877 | +0.1893811149 |

### 4Q (5 CNOT)

| `eps` | `F_0` | `F_1` | `F_2` | `F_5` | `F_10` | `F_50` | `F_inf` | `F_inf - F_0` |
|---|---|---|---|---|---|---|---|---|
| 0.02 | 0.9850000000 | 0.9789881433 | 0.9787841817 | 0.9787757604 | 0.9787757599 | 0.9787757599 | 0.978775759863 | **−0.0062242401** |
| 0.05 | 0.9625000000 | 0.9778838034 | 0.9787348654 | 0.9787757565 | 0.9787757599 | 0.9787757599 | 0.978775759863 | +0.0162757599 |
| 0.10 | 0.9250000000 | 0.9750077034 | 0.9786012527 | 0.9787757458 | 0.9787757599 | 0.9787757599 | 0.978775759863 | +0.0537757599 |
| 0.15 | 0.8875000000 | 0.9705881195 | 0.9783819971 | 0.9787757280 | 0.9787757599 | 0.9787757599 | 0.978775759863 | +0.0912757599 |
| 0.20 | 0.8500000000 | 0.9642874516 | 0.9780400769 | 0.9787757001 | 0.9787757599 | 0.9787757599 | 0.978775759863 | +0.1287757599 |
| 0.30 | 0.7750000000 | 0.9443705049 | 0.9767212046 | 0.9787755896 | 0.9787757599 | 0.9787757599 | 0.978775759863 | +0.2037757599 |

4Q per-round success probability (undefined at `n = 0`):

| `eps` | `P_succ^(1)` | `P_succ^(2)` | `P_succ^(5)` | `P_succ^(10)` | `P_succ^(inf)` |
|---|---|---|---|---|---|
| 0.02 | 0.910042966 | 0.899244306 | 0.898865107 | 0.898865083 | 0.898865083 |
| 0.05 | 0.870250705 | 0.897278270 | 0.898864942 | 0.898865083 | 0.898865083 |
| 0.10 | 0.806679303 | 0.892172408 | 0.898864494 | 0.898865083 | 0.898865083 |
| 0.15 | 0.746544193 | 0.884366244 | 0.898863752 | 0.898865083 | 0.898865083 |
| 0.20 | 0.689845375 | 0.873320443 | 0.898862584 | 0.898865083 | 0.898865083 |
| 0.30 | 0.586756615 | 0.839042654 | 0.898857977 | 0.898865083 | 0.898865083 |

## 5. Observations

1. **Every `eps` reaches the same fixed point, in all three protocols.**  The
   spread of the converged populations across the six runs is `0.0` — the same
   floating-point numbers — for Type 3, Type 4 and 4Q alike.
2. **The common fixed points** are listed in §3.  `F_inf` is an attribute of the
   protocol and `q` alone; it does not depend on `eps`.
3. **Convergence rate barely depends on `eps`.**  The residual `|F_n - F_inf|`
   falls geometrically at a rate set by the protocol (0.065 / 0.043 / 0.042),
   the same to within a percent across the six `eps`.  What `eps` changes is the
   starting residual, hence one or two extra rounds: 9–10 rounds (Type 3) and
   7–9 (Type 4, 4Q) to reach `1e-12`.
4. **Yes, some `eps` lose fidelity.**  Where `F_0 > F_inf` the trajectory falls
   monotonically: `eps = 0.02` and `0.05` for Type 3, `eps = 0.02` for Type 4
   and for 4Q.  All other `eps` in the set rise monotonically.
5. **Break-even `eps`** (where `F_0 = F_inf`, i.e. `eps = 4(1 - F_inf)/3`):
   Type 3 `0.058929917622`, Type 4 `0.047491846831`, 4Q `0.028298986849`.
   This is *not* being called a threshold here; it is simply where the input
   crosses the attractor.
6. **Invariant relations differ between the families.**  Type 3 and Type 4 keep
   `p_Psi+ = p_Psi-` (the Pauli-coordinate plane `y = -x`), and the error weight
   piles up in `Phi-`.  4Q keeps `p_Phi- = p_Psi-` (the plane `y = -z`), and the
   error weight piles up in `Psi+`.  In all three the initial threefold equality
   `Phi- = Psi+ = Psi-` breaks after a single round.  For 4Q the surviving pair
   agrees to the last bit (`0.0`); for Type 3/4 to `2.8e-17`.
7. **4Q's success probability converges with the state**, to `0.898865083`,
   again the same for every `eps`.  At `eps = 0.02` it starts *above* that value
   (0.910 at `n = 1`) and comes down; for the noisier inputs it starts lower
   (0.587 at `eps = 0.30`) and rises.  `P_succ^(n)` tracks the state, so it
   settles on the same round the populations do.

Nothing here defines a threshold, and no theoretical claim is made beyond what
the numbers show at these `(eps, q)` points.

## 6. Files

Figures in `results/figures/`, with `<tag>` in `{type3, type4, 4q}`:

| file | content |
|---|---|
| `<tag>_eps_<eps>_stacked.png` | Figure A, per `eps`: stacked populations, `Phi+` at the bottom so the bottom band's height is `F_n`; second panel zooms on the error bands |
| `<tag>_eps_<eps>_lines.png` | Figure B, per `eps`: the four populations as lines; second panel isolates the error components.  The second member of the protocol's invariant pair is drawn dashed on top of the first, so their exact coincidence reads as two curves rather than one missing curve |
| `<tag>_F_overlay_q0p02.png` | Figure C: `F_n` for all six `eps`; second panel shows `\|F_n - F_inf\|` on a log axis |
| `<tag>_summary_vs_eps_q0p02.png` | Figure D: `F_0`, `F_1`, `F_50` against `eps`, with `F_inf` as a reference line |
| `4q_Psucc_overlay_q0p02.png` | 4Q only: `P_succ^(n)` for all six `eps` |

Data in `results/data/epsilon_sweep/`: one CSV per (protocol, `eps`), a
`<tag>_summary.csv` / `<tag>_summary.json` per protocol, and `comparison.csv` /
`comparison.json` with the three protocols side by side.

Colour note: `eps` is a magnitude, so the overlays use one hue light-to-dark
rather than categorical hues.  The documented blue ramp holds five steps that
clear the ordinal lightness-gap gate on a light surface; six `eps` values need a
sixth step, so the darkest pair is closer than that gate wants.  Every curve
therefore carries its own marker and a direct label, and no two are separated by
lightness alone.
