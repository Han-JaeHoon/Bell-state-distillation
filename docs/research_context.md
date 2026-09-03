# Research context

## The wider programme

This repository is a self-contained side investigation within a project on
entanglement purification, which draws on:

- **Virtual Distillation (VD)** — estimating `Tr(O rho^2)/Tr(rho^2)` from two
  copies without ever preparing the purified state;
- **Purification Quantum Error Correction (PQEC)** — the same `rho -> rho^2`
  mechanism used as an error-correction primitive;
- **Bell-state purification** — improving the fidelity of noisy entangled pairs;
- **noisy implementations of the purification circuit itself** — the actual
  research question of the parent project.

## The input state

The family studied throughout is the Bell-isotropic (global-depolarized) state

    rho_eps = (1 - eps) |Phi+><Phi+| + eps I/4

with Bell-basis populations `(1 - 3eps/4, eps/4, eps/4, eps/4)`.  Writing
`F = 1 - 3eps/4` and `q = eps/4`,

    rho_eps = F Phi+ + q (Phi- + Psi+ + Psi-)

For ideal purification `P(rho) = rho^2 / Tr(rho^2)` this gives

    F_VD = F^2 / (F^2 + 3 q^2),      eps' = eps^2 / (4 - 6 eps + 3 eps^2)

so the leading `O(eps)` error is removed.

## The conceptual gap this circuit addresses

VD and the postselection-free PQEC formulation do **not** necessarily prepare
`rho^2 / Tr(rho^2)` in each experimental shot; the purified state is accessed
through parity-weighted measurement statistics.  The circuit verified here
*does* produce the state physically, on a postselected branch — at the cost of
a success probability `Tr(rho^2) < 1`, and only for Bell-diagonal inputs.

## The 5-qubit SWAP-test baseline it is compared against

Standard SWAP-test PQEC for a 2-qubit Bell state uses **five qubits**: one
ancilla plus two 2-qubit copies of `rho`.  The controlled-SWAP of the two
2-qubit registers needs two Fredkin gates.  The parent project studied several
CNOT decompositions of that same ideal unitary:

| label | implementation | CNOT count |
|---|---|---|
| Step 3 | textbook Fredkin decomposition | 16 |
| Step 4 | compiler-resynthesized | 14 |
| Step 5 | learned and pruned | 14 (the task brief said 12; the parent repository's circuit has 14) |

The ideal unitaries agree, but once gate-local noise is inserted after
individual CNOTs the resulting **channels differ**, and the repeated map

    rho_{n+1} = M_p(rho_n)

can have different fixed points, different convergence and — importantly —
different invariant manifolds.  A key lesson from that work is that a reduced
Bell-diagonal description must be *checked*, because some noisy implementations
leave the Bell-diagonal manifold.

## Motivation for the 4-qubit circuit

In the SWAP-test construction the second Bell pair is discarded after
purification anyway.  The idea tested here is to:

1. drop the separate SWAP-test ancilla entirely, and
2. use the second Bell pair **itself** as the syndrome / parity register.

That leaves only the four data qubits and — as verified — five CNOTs plus one
Hadamard, at the cost of making the protocol postselective and restricting the
exact `rho^2` identity to Bell-diagonal inputs.

## What this repository checked about the noisy version

Because the parent project found that Bell-diagonal closure is
implementation-dependent under noise, the same question was asked here rather
than assumed.  Result: with per-CNOT depolarizing noise (either convention),
a Bell-diagonal input stays Bell diagonal in **all four** measurement branches,
to `~1e-16`, for every noise strength tested — the noisy 4-qubit circuit does
**not** leak out of the Bell-diagonal sector.  A control test confirms this is a
property of the input sector rather than an artifact of the noise dephasing
everything.

This is a numerical observation for this circuit and these noise models; it is
not a symbolic proof for all `p`.
