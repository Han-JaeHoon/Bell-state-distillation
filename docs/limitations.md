# Limitations and scope

Read this before quoting any result from this repository.

## 1. The `rho^2` equivalence is Bell-diagonal specific

The circuit's success branch implements the Kraus operator

    K_00 = sum_i |B_i>_12 <B_i|_12 <B_i|_34

whose action on two copies of an arbitrary two-qubit state is the **elementwise
(Schur) square in the Bell basis**:

    (rho_tilde_00)_ij = (rho_ij)^2

This equals the matrix square `rho^2` **only when `rho` is Bell diagonal**
(all `rho_ij` with `i != j` vanish, so the Schur and matrix squares coincide).

For a general input the circuit output and `rho^2 / Tr(rho^2)` are genuinely
different states — verified on 40 random density matrices, where the Frobenius
distance between them is `> 1e-6` in every case, and on an explicit reproducible
counterexample (`test_general_state_not_matrix_square.py`).

The Bell-isotropic family studied here **is** Bell diagonal, so the exact
identity applies to it.  Do not extrapolate the identity beyond that sector.

## 2. The protocol is postselective

The purified state appears only in the `(m3,m4) = (0,0)` branch.  For a mixed
input the success probability is

    P_success = Tr(rho^2) < 1

and it falls quickly with input noise (e.g. `0.8575` at `eps = 0.1`, `0.52` at
`eps = 0.4`, `0.25` at `eps = 1`).  Under repetition the full-tree cost is
`Tr[rho^(2^l)]`, which decays doubly exponentially: at `eps = 0.3` the depth-6
tree succeeds with probability `~2.9e-6` while consuming 64 input copies.

This is a different accounting from virtual distillation, which pays a sampling
overhead instead of discarding runs.  Conversely, VD/PQEC does **not** in general
prepare `rho^2 / Tr(rho^2)` in each shot — it accesses it through
parity-weighted statistics.  This circuit does produce the state physically, on
the branch that succeeds.

## 3. The other three branches are not failures in the trivial sense

Every branch returns a valid (unnormalized) Bell-diagonal operator

    rho_tilde_{mu,nu} = sum_ab p_ab p_{a xor nu, b xor mu} B_ab

and the four branches sum back to the input.  Nothing here investigates whether
the failure branches can be recycled; they are simply characterized.

## 4. This is not an LOCC entanglement-distillation protocol

If the two Bell pairs are read as distributed Alice–Bob pairs

    pair 1 = q1 (Alice) — q2 (Bob)
    pair 2 = q3 (Alice) — q4 (Bob)

then Alice holds `{q1, q3}` and Bob holds `{q2, q4}`.  Of the five CNOTs,

| # | gate | crosses the Alice/Bob cut? |
|---|---|---|
| 1 | `q3 -> q4` | **yes** (Alice → Bob) |
| 2 | `q2 -> q4` | no (Bob → Bob) |
| 3 | `q1 -> q4` | **yes** (Alice → Bob) |
| 4 | `q3 -> q2` | **yes** (Alice → Bob) |
| 5 | `q3 -> q1` | no (Alice → Alice) |

Three of the five CNOTs are **global entangling gates across the partition**.
The protocol therefore consumes entanglement across the cut and is not
implementable by local operations and classical communication.

That is exactly why the re-entangling window of section 10 of
[derivation.md](derivation.md) is not a contradiction: LOCC cannot create
entanglement from a separable state, but this circuit is not LOCC.

**The correct reading is: physical Bell-state purification of two pairs held in
one quantum processor — not network LOCC Bell-pair distillation.**  Do not
present it as the latter.

## 5. Circuit noise changes the result

All ideal identities above assume perfect gates.  With per-CNOT noise the
output fidelity degrades and the purification gain vanishes at a finite noise
strength (see `results/data/noisy_sweep_*.csv`).  The break-even values quoted
there are specific to:

- this exact five-CNOT gate list and ordering,
- noise inserted after each CNOT on the two qubits it acted on,
- ideal single-qubit gates and ideal measurement,
- a specific noise convention (the two supported conventions are related by
  `p_replace = 16 p_pauli / 15`, so a threshold is meaningless without its
  convention label).

Measurement error, idling/decoherence, crosstalk and state-preparation error
are **not** modelled.

## 6. No novelty claim is established here

This repository verifies internal mathematical consistency of a circuit and its
associated identities.  It does **not** constitute:

- a literature search,
- a priority or originality claim,
- a claim of optimality in CNOT count or depth,
- a resource comparison against the 16 / 14 / 12-CNOT SWAP-test implementations
  from the parent project.

Bell-measurement–based purification and Bell-label comparison are long-standing
ideas (Bennett et al. 1996; Deutsch et al. 1996; and the whole
entanglement-pumping literature).  Whether this particular four-qubit
five-CNOT arrangement is new, and how it compares to known constructions, is an
open question that requires a proper literature review — not answered by any
calculation in this repository.

## 7. No cross-comparison with the parent project's Step 3/4/5 numbers

The parent project's 16-CNOT / 14-CNOT / 12-CNOT figures come from a
**different protocol** (a 5-qubit SWAP-test gadget read out through a
parity-weighted correlator, with no postselection) under its own noise
conventions.  The numbers in `results/data/noisy_sweep_*.csv` are **not**
comparable to those thresholds and no such comparison is made here.  Doing it
properly requires importing those exact circuits and noise definitions and
re-running both under one common convention.
