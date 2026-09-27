"""Bell-basis population dynamics under repeated noisy purification.

Purpose: see, round by round, what happens to the four Bell populations of the
retained pair when a noisy purification round is applied over and over.  Every
entry point takes ``eps`` and ``q`` so the same code can be swept later.

FOUR PROTOCOLS
--------------
============  ======  =========================================  ==============
name          CNOTs   circuit                                    Bell-diagonal?
============  ======  =========================================  ==============
``Type3``     16      textbook SWAP-test gadget                  preserved
``Type4``     14      resynthesised SWAP-test gadget             preserved
``Type5``     14      learned + pruned SWAP-test gadget          NOT preserved
``4Q``         5      4-qubit postselected comparator            preserved
============  ======  =========================================  ==============

"Bell-diagonal preserved" is a statement about *these* maps acting on a
Bell-DIAGONAL input; it is verified numerically by this module (the Bell
off-diagonal weight is recorded at every round for all four protocols, so the
claim is visible rather than assumed).  Type 5 is different in kind: its learned
gate sequence reproduces the ideal gadget unitary only to ``~2e-7`` in Frobenius
norm, and that residual is a coherent error which the round map feeds back into
itself.  Type 5 therefore MUST be run as a full density matrix; there is no
Bell-diagonal reduced map for it, and none is used here.

NOISE MODEL -- one convention, one symbol
-----------------------------------------
After every CNOT, on the two qubits that CNOT acted on, the two-qubit
REPLACEMENT depolarizing channel

    D_q(rho) = (1-q) rho + q [ I_ij/4  (x)  Tr_ij(rho) ]

The parent repository writes this channel's strength as ``eps2`` and implements
it through ``noisy_bell_state.global_depol_kraus``; this package writes it as
``p`` with ``convention="replace"``.  The two are the SAME channel at the SAME
numerical value -- :func:`verify_noise_convention` checks that identity on
random states rather than taking it on trust.  Do not confuse it with the Pauli
depolarizing convention of :mod:`pqec_distill.noise`, which is the same family
only after ``p_replace = 16 p_pauli / 15``.

WHAT ONE ROUND MEANS
--------------------
Both families consume two identical copies of the current state and return a
two-qubit operator on the retained pair, which is normalised to give the next
iterate:

* SWAP-test family (Type 3/4/5), five qubits:
  ``tau_A = Tr_{a,B}[(Z_a (x) I) E_q(|0><0|_a (x) rho (x) rho)]``, then
  ``rho -> tau_A / Tr tau_A``.  ``tau_A`` is parity-weighted, not the
  unconditional output state, so iterating it is an *effective* map.
* 4Q family, four qubits: run the noisy 5-CNOT + H circuit on ``rho (x) rho``,
  project (q3,q4) onto ``(0,0)``, trace them out, normalise.  Here the trace of
  the unnormalised operator IS the physical per-round success probability.

SOURCES OF TRUTH (nothing is re-derived here)
---------------------------------------------
* Type 3/4/5 circuits and the five-qubit round: the parent repository's
  ``iterated_noisy_pqec`` (see :mod:`pqec_distill.swap_test_source`).
* 4Q circuit + noise + postselection: ``pqec_distill.circuit``,
  ``pqec_distill.noise``, ``pqec_distill.measurement``.
* Verified Bell-diagonal exact maps for Type 3/4/4Q:
  ``pqec_distill.analytic_exact_map`` (validated row-by-row against the frozen
  circuit-only reference).  Used as an independent second path, never as the
  only path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .analytic_exact_map import (
    analytic_map_4q, analytic_map_type3, analytic_map_type4, analytic_success_4q,
)
from .bell_states import BELL_NAMES, bell_diagonal_state, to_bell_basis
from .measurement import postselect_branch
from .noise import noisy_full_channel_dm

__all__ = [
    "PROTOCOLS", "DIAGNOSTIC_PROTOCOLS", "PROTOCOL_INFO", "BELL_NAMES",
    "INVARIANT_PAIR",
    "isotropic_dm", "isotropic_populations",
    "bell_populations", "bell_offdiagonal_c", "state_diagnostics",
    "one_round", "run_dynamics", "cross_check_exact_vs_dm",
    "verify_noise_convention", "verify_cnot_counts", "compilation_residuals",
    "plateau_and_escape",
]

PROTOCOLS = ("Type3", "Type4", "Type5", "4Q")

#: Not one of the four protocols: the Type 5 gate sequence with the learned
#: circuit's residual mis-compilation of the ideal gadget unitary removed.  Same
#: gates, same noise model, same q, same noise locations.  It exists only to
#: answer "is the Type 5 behaviour caused by the 2e-7 compilation error or by the
#: noise?" -- run it next to Type 5 and compare.
DIAGNOSTIC_PROTOCOLS = ("Type5cal",)

#: ``swap_test`` circuits are named ``stepN`` in the parent repository.
_PARENT_CIRCUIT = {"Type3": "step3", "Type4": "step4", "Type5": "step5",
                   "Type5cal": "step5cal"}

PROTOCOL_INFO: dict[str, dict] = {
    "Type3": dict(label="Type 3 - textbook 16-CNOT SWAP-test gadget",
                  n_cnot=16, family="swap_test", has_exact_map=True,
                  bell_diagonal_preserved=True),
    "Type4": dict(label="Type 4 - resynthesised 14-CNOT SWAP-test gadget",
                  n_cnot=14, family="swap_test", has_exact_map=True,
                  bell_diagonal_preserved=True),
    "Type5": dict(label="Type 5 - learned 14-CNOT SWAP-test gadget",
                  n_cnot=14, family="swap_test", has_exact_map=False,
                  bell_diagonal_preserved=False),
    "4Q": dict(label="4Q - 4-qubit 5-CNOT postselected comparator",
               n_cnot=5, family="postselect", has_exact_map=True,
               bell_diagonal_preserved=True),
    "Type5cal": dict(label="Type 5 (exact-compilation calibrated) - diagnostic only",
                     n_cnot=14, family="swap_test", has_exact_map=False,
                     bell_diagonal_preserved=False),
}

_ALL_PROTOCOLS = PROTOCOLS + DIAGNOSTIC_PROTOCOLS

#: The pair of Bell populations each protocol keeps exactly equal on a
#: Bell-diagonal trajectory.  Type 3/4 preserve the Pauli-coordinate plane
#: y = -x, which reads p_Psi+ = p_Psi-; 4Q preserves y = -z, which reads
#: p_Phi- = p_Psi-.  Type 5 preserves neither, so it has no entry: its noisy
#: dynamics does not stay Bell-diagonal at all.
INVARIANT_PAIR = {"Type3": ("Psi+", "Psi-"),
                  "Type4": ("Psi+", "Psi-"),
                  "4Q": ("Phi-", "Psi-")}


# ---------------------------------------------------------------------------
# states and observables
# ---------------------------------------------------------------------------

def isotropic_populations(eps: float) -> np.ndarray:
    """Bell populations of rho_0(eps) = (1-eps) Phi+ + eps I/4."""
    e = float(eps)
    return np.array([1.0 - 0.75 * e, 0.25 * e, 0.25 * e, 0.25 * e])


def isotropic_dm(eps: float) -> np.ndarray:
    """rho_0(eps) as a 4x4 density matrix in the computational basis."""
    return bell_diagonal_state(isotropic_populations(eps))


def bell_populations(rho: np.ndarray) -> np.ndarray:
    """(p_Phi+, p_Phi-, p_Psi+, p_Psi-) = the Bell-basis diagonal of rho."""
    return np.real(np.diag(to_bell_basis(np.asarray(rho))))


def bell_offdiagonal_c(rho: np.ndarray) -> float:
    """C_Bell = sum_{i != j} |<B_i| rho |B_j>|^2, the Bell off-diagonal weight."""
    r = to_bell_basis(np.asarray(rho))
    off = r - np.diag(np.diag(r))
    return float(np.sum(np.abs(off) ** 2))


def state_diagnostics(rho: np.ndarray) -> dict:
    """Everything needed to decide whether an iterate is still a valid state."""
    rho = np.asarray(rho)
    herm = float(np.linalg.norm(rho - rho.conj().T))
    eig = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
    pops = bell_populations(rho)
    return {
        "trace_err": float(abs(np.trace(rho) - 1.0)),
        "herm_err": herm,
        "min_eig": float(eig.min()),
        "max_eig": float(eig.max()),
        "pop_sum_err": float(abs(pops.sum() - 1.0)),
        "purity": float(np.real(np.trace(rho @ rho))),
    }


# ---------------------------------------------------------------------------
# the one-round maps
# ---------------------------------------------------------------------------

def _step_swap_test(rho: np.ndarray, q: float, protocol: str):
    """One SWAP-test round as a full 4x4 density matrix (parent repository)."""
    from .swap_test_source import load_parent_module

    mod = load_parent_module()
    rho_next, info = mod.one_round_effective_map(
        np.asarray(rho, dtype=complex), float(q), _PARENT_CIRCUIT[protocol])
    if rho_next is None:
        raise FloatingPointError(
            f"{protocol}: the parity weight Tr(tau_A) vanished (Q={info['Q']!r})")
    # tau_A is Hermitian in exact arithmetic; the parent projects and reports the
    # discarded anti-Hermitian norm, which we pass through as a diagnostic.
    return rho_next, {"weight": float(info["Q"]), "herm_discarded": float(info["herm_err"])}


def _step_4q(rho: np.ndarray, q: float, convention: str = "replace"):
    """One 4Q round: noisy 5-CNOT + H, postselect (q3,q4) = (0,0), normalise."""
    out4 = noisy_full_channel_dm(np.kron(np.asarray(rho), np.asarray(rho)),
                                 float(q), convention)
    tau, prob = postselect_branch(out4, 0, 0)
    if prob <= 0.0:
        raise FloatingPointError(f"4Q: postselection probability vanished ({prob!r})")
    return tau / prob, {"weight": prob, "herm_discarded": 0.0}


def one_round(rho: np.ndarray, q: float, protocol: str):
    """(rho_next, info) for one round of ``protocol`` on the 4x4 state ``rho``.

    ``info["weight"]`` is Tr of the unnormalised retained operator: the physical
    per-round postselection probability for ``4Q``, and the parity weight
    Tr(tau_A) -- NOT a probability -- for the SWAP-test family.
    """
    if protocol not in _ALL_PROTOCOLS:
        raise KeyError(f"unknown protocol {protocol!r}; choose from {_ALL_PROTOCOLS}")
    if PROTOCOL_INFO[protocol]["family"] == "postselect":
        return _step_4q(rho, q)
    return _step_swap_test(rho, q, protocol)


def _exact_map(protocol: str) -> Callable[[np.ndarray, float], np.ndarray]:
    return {"Type3": analytic_map_type3,
            "Type4": analytic_map_type4,
            "4Q": analytic_map_4q}[protocol]


# ---------------------------------------------------------------------------
# iteration
# ---------------------------------------------------------------------------

@dataclass
class Dynamics:
    """Round-by-round record of one (protocol, eps, q) run."""

    protocol: str
    eps: float
    q: float
    backend: str
    n: np.ndarray
    pops: np.ndarray                     # (n_rounds+1, 4)
    fidelity: np.ndarray
    c_bell: np.ndarray
    weight: np.ndarray                   # NaN at n = 0
    diagnostics: list[dict] = field(default_factory=list)
    final_state: np.ndarray | None = None

    @property
    def label(self) -> str:
        return PROTOCOL_INFO[self.protocol]["label"]

    def at(self, n: int) -> np.ndarray:
        return self.pops[int(n)]

    def rows(self) -> list[dict]:
        out = []
        for i, n in enumerate(self.n):
            row = {"protocol": self.protocol, "eps": self.eps, "q": self.q,
                   "backend": self.backend, "n": int(n),
                   "p_PhiP": self.pops[i, 0], "p_PhiM": self.pops[i, 1],
                   "p_PsiP": self.pops[i, 2], "p_PsiM": self.pops[i, 3],
                   "pop_sum": float(self.pops[i].sum()),
                   "F": self.fidelity[i], "C_Bell": self.c_bell[i],
                   "weight": self.weight[i]}
            if i < len(self.diagnostics):
                row.update({f"diag_{k}": v for k, v in self.diagnostics[i].items()})
            out.append(row)
        return out


def run_dynamics(protocol: str, eps: float, q: float, n_rounds: int = 50,
                 backend: str = "auto", record_every: int = 1) -> Dynamics:
    """Iterate ``rho_{n+1} = M_q(rho_n)`` from the Bell-isotropic initial state.

    ``backend``:
      * ``"dm"``      -- full density matrix (the only option for Type 5)
      * ``"exact"``   -- the verified Bell-diagonal exact map (Type 3/4/4Q only);
                         cannot represent Bell off-diagonals, so ``C_Bell`` is
                         identically 0 by construction and is reported as such
      * ``"auto"``    -- ``"dm"`` for Type 5, ``"dm"`` otherwise as well; the
                         exact map is reserved for the cross-check and for long
                         runs requested explicitly
    """
    if protocol not in _ALL_PROTOCOLS:
        raise KeyError(f"unknown protocol {protocol!r}; choose from {_ALL_PROTOCOLS}")
    if backend == "auto":
        backend = "dm"
    if backend == "exact" and not PROTOCOL_INFO[protocol]["has_exact_map"]:
        raise ValueError(
            f"{protocol} has no verified Bell-diagonal exact map "
            "(its noisy dynamics leaves the Bell-diagonal manifold); use backend='dm'")
    if backend not in ("dm", "exact"):
        raise ValueError(f"unknown backend {backend!r}")

    ns, pops, fid, cb, wts, diags = [], [], [], [], [], []

    if backend == "exact":
        step = _exact_map(protocol)
        p = isotropic_populations(eps)
        # weight[n] is the weight of the round that PRODUCED p_n, so it is NaN
        # at n = 0 -- the same indexing the density-matrix backend uses.
        weight = float("nan")
        for n in range(n_rounds + 1):
            if n % record_every == 0 or n == n_rounds:
                ns.append(n)
                pops.append(p.copy())
                fid.append(float(p[0]))
                cb.append(0.0)
                wts.append(weight)
                diags.append({"pop_sum_err": float(abs(p.sum() - 1.0)),
                              "min_pop": float(p.min())})
            if n < n_rounds:
                if protocol == "4Q":
                    weight = float(analytic_success_4q(p, q))
                p = step(p, q)
        final = bell_diagonal_state(p)
    else:
        rho = isotropic_dm(eps)
        weight = float("nan")
        for n in range(n_rounds + 1):
            if n % record_every == 0 or n == n_rounds:
                ns.append(n)
                pk = bell_populations(rho)
                pops.append(pk)
                fid.append(float(pk[0]))          # F_n = p_Phi+ = <Phi+|rho_n|Phi+>
                cb.append(bell_offdiagonal_c(rho))
                wts.append(weight)
                diags.append(state_diagnostics(rho))
            if n < n_rounds:
                rho, info = one_round(rho, q, protocol)
                weight = info["weight"]
        final = rho

    return Dynamics(protocol=protocol, eps=float(eps), q=float(q), backend=backend,
                    n=np.array(ns), pops=np.array(pops), fidelity=np.array(fid),
                    c_bell=np.array(cb), weight=np.array(wts),
                    diagnostics=diags, final_state=final)


def cross_check_exact_vs_dm(protocol: str, eps: float, q: float,
                            n_rounds: int = 50) -> dict:
    """Run both paths from the same start and report the largest disagreement."""
    if not PROTOCOL_INFO[protocol]["has_exact_map"]:
        raise ValueError(f"{protocol} has no exact Bell-diagonal map to compare against")
    dm = run_dynamics(protocol, eps, q, n_rounds, backend="dm")
    ex = run_dynamics(protocol, eps, q, n_rounds, backend="exact")
    dpop = np.max(np.abs(dm.pops - ex.pops), axis=1)
    return {
        "protocol": protocol, "eps": float(eps), "q": float(q), "n_rounds": n_rounds,
        "max_abs_pop_diff": float(dpop.max()),
        "argmax_round": int(dm.n[int(dpop.argmax())]),
        "max_abs_F_diff": float(np.max(np.abs(dm.fidelity - ex.fidelity))),
        "max_C_Bell_dm": float(dm.c_bell.max()),
        "per_round_max_diff": dpop,
    }


def plateau_and_escape(dyn: "Dynamics", drop: float = 0.01,
                       settle_from: int = 2) -> dict:
    """Describe a plateau-then-escape fidelity history, without assuming one.

    ``plateau_F`` is the fidelity at ``settle_from``; ``n_escape`` is the first
    round after that at which ``F`` has fallen ``drop`` below it (``None`` if it
    never does).  ``c_bell_at_escape`` is the Bell off-diagonal weight there, and
    ``c_bell_growth_per_round`` is fitted on the pre-escape stretch where the
    off-diagonal weight is growing but the fidelity has not yet moved -- which is
    what makes the plateau metastable rather than stable.
    """
    n, F, C = dyn.n, dyn.fidelity, dyn.c_bell
    idx = int(np.searchsorted(n, settle_from))
    if idx >= len(n):
        return {"plateau_F": None, "n_escape": None}
    plateau = float(F[idx])
    below = np.where(F[idx:] < plateau - drop)[0]
    if below.size == 0:
        return {"plateau_F": plateau, "n_escape": None,
                "F_end": float(F[-1]), "c_bell_end": float(C[-1]),
                "c_bell_growth_per_round": None}
    j = idx + int(below[0])
    mask = (np.arange(len(n)) >= idx) & (np.arange(len(n)) < j) & (C > 0)
    growth = None
    if mask.sum() >= 4:
        slope = np.polyfit(n[mask], np.log(C[mask]), 1)[0]
        growth = float(np.exp(slope))
    return {"plateau_F": plateau, "n_escape": int(n[j]),
            "F_at_escape": float(F[j]), "c_bell_at_escape": float(C[j]),
            "c_bell_growth_per_round": growth,
            "F_end": float(F[-1]), "c_bell_end": float(C[-1]),
            "F_max": float(F.max()), "n_F_max": int(n[int(F.argmax())])}


# ---------------------------------------------------------------------------
# convention / circuit checks
# ---------------------------------------------------------------------------

def verify_noise_convention(n_trials: int = 6, seed: int = 20260927) -> dict:
    """Check that ``q`` means the same channel on both sides.

    Two independent comparisons, on random (generally non-Bell-diagonal) states:

    1. two qubits -- the parent repository's Kraus set
       ``noisy_bell_state.global_depol_kraus(q)`` against this package's
       ``replacement_depolarizing(..., q, n_qubits=2)``;
    2. five qubits -- the parent's ``iterated_noisy_pqec.replacement_depol``,
       which is what the round map actually calls after every CNOT, against the
       same function of this package on the same wire pair.

    Both must agree at machine precision for every ``q``, otherwise the value
    ``q = 0.02`` would not mean the same thing in the two families.
    """
    import importlib

    from .noise import replacement_depolarizing
    from .swap_test_source import load_parent_module

    mod = load_parent_module()
    nbs = importlib.import_module("noisy_bell_state")
    rng = np.random.default_rng(seed)

    def random_dm(dim):
        a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        rho = a @ a.conj().T
        return rho / np.trace(rho)

    worst_kraus = 0.0
    worst_parent = 0.0
    for q in (0.0, 0.002, 0.02, 0.2, 0.7, 1.0):
        ks = nbs.global_depol_kraus(q)
        for _ in range(n_trials):
            rho2 = random_dm(4)
            mine = replacement_depolarizing(rho2, (0, 1), q, n_qubits=2)
            theirs = sum(k @ rho2 @ k.conj().T for k in ks)
            worst_kraus = max(worst_kraus, float(np.max(np.abs(mine - theirs))))

            rho5 = random_dm(32)
            for pair in ((0, 1), (1, 3), (2, 4), (3, 4)):
                a = replacement_depolarizing(rho5, pair, q, n_qubits=5)
                b = mod.replacement_depol(rho5, pair, q)
                worst_parent = max(worst_parent, float(np.max(np.abs(a - b))))
    return {"max_abs_diff_2q_vs_parent_kraus": worst_kraus,
            "max_abs_diff_5q_vs_parent_replacement": worst_parent}


def verify_cnot_counts() -> dict:
    """CNOT count of every protocol, read off the actual gate sequences."""
    from .circuit import CNOT_SEQUENCE
    from .swap_test_source import load_parent_module

    mod = load_parent_module()
    counts = {p: int(mod.n_cnots(_PARENT_CIRCUIT[p])) for p in ("Type3", "Type4", "Type5")}
    counts["4Q"] = len(CNOT_SEQUENCE)
    return counts


def compilation_residuals() -> dict:
    """Frobenius distance between each SWAP-test circuit's noiseless unitary and
    the exact ideal gadget unitary (phase-aligned).  Non-zero only for Type 5."""
    from .swap_test_source import load_parent_module

    mod = load_parent_module()
    return {p: mod.compilation_residual(_PARENT_CIRCUIT[p])
            for p in ("Type3", "Type4", "Type5")}
