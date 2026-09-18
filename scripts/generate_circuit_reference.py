"""Independent CIRCUIT-LEVEL reference data for Type 3 / Type 4 / 4Q.

PURPOSE AND SCOPE
-----------------
This script produces a numerical input->output reference for arbitrary
Bell-diagonal inputs, computed ONLY by simulating the actual noisy circuits.
It is intended as a fixed ground truth against which a separately derived
analytic map can later be compared.

It therefore deliberately does NOT import, read or reuse any closed-form
input-output expression that exists in either repository.  In particular it
never touches:

    pqec_distill.noisy_analytics      (closed-form noisy Bell-diagonal map)
    pqec_distill.swap_test_reference  (closed-form Step 3/4 (u,v) recursions)
    pqec_distill.repeated_noisy       (built on the above)
    pqec_distill.analytics            (ideal-case closed forms)
    iterated_noisy_pqec.solve_fixed_point / .jacobian / closed-form validators

Only circuit construction, gate application, noise channels, partial traces,
projectors and Bell-basis rotations are used.

EXECUTION BACKENDS (two independent paths per circuit, cross-checked)
---------------------------------------------------------------------
Type 3 / Type 4 (5 qubits, ordering (a, A1, A2, B1, B2) = (0,1,2,3,4)):
  path A  dense 32x32 propagation -- iterated_noisy_pqec.build_program /
          .one_round_tau, whose gate sequences are captured verbatim from
          verify_analytic_decomposed._fred (Type 3) and
          pqec_resynth_noise._apply / .GATES (Type 4)
  path B  PennyLane default.mixed execution of the same qfuncs

4Q (4 qubits, ordering (q1,q2,q3,q4) = (0,1,2,3), q1 most significant):
  path A  pqec_distill numpy dense 16x16 -- noise.noisy_full_channel_dm +
          measurement.postselect_branch
  path B  PennyLane default.mixed, reading out <P_A (x) |00><00|_{q3q4}>

EFFECTIVE OUTPUT DEFINITIONS (unchanged from the existing repositories)
-----------------------------------------------------------------------
Type 3 / 4: the parity-weighted PQEC operator
    tau_A = Tr_{a,B}[ (Z_a (x) I) sigma_out ],   rho_out = tau_A / Tr(tau_A)
  NOTE: tau_A is an effective (parity-weighted) operator, not a postselected
  physical state, so Tr(tau_A) is a parity VISIBILITY, not a probability, and
  its Bell populations are not guaranteed non-negative.  Reported as-is.

4Q: genuine postselection of (m3, m4) = (0, 0)
    rho_tilde = Tr_{q3q4}[ Pi_00 sigma_out Pi_00 ],  P_success = Tr(rho_tilde)
    rho_out = rho_tilde / P_success

NOISE (identical convention in both repositories, verified below)
------------------------------------------------------------------
    D_q^(ij)(rho) = (1-q) rho + q [ I_ij/4 (x) Tr_ij(rho) ]
applied immediately after every CNOT, on the two qubits that CNOT acted on.
Single-qubit gates and the final H / measurement are ideal.

Usage
-----
    python scripts/generate_circuit_reference.py [--parent-repo PATH] [--quick]

Type 3 / Type 4 rows require the parent repository (PQEC-Operational-Threshold)
and PennyLane; if either is unavailable those rows are skipped and the 4Q rows
are still produced.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _bootstrap import DATA_DIR  # noqa: E402

# --- circuit-only imports from this repository -----------------------------
from pqec_distill.bell_states import (  # noqa: E402
    BELL_NAMES, bell_basis_matrix, bell_diagonal_state,
)
from pqec_distill.gates import I2, PAULI_X, PAULI_Y, PAULI_Z, kron_list  # noqa: E402
from pqec_distill.measurement import postselect_branch  # noqa: E402
from pqec_distill.noise import noisy_full_channel_dm  # noqa: E402

OUT_DIR = DATA_DIR / "circuit_reference"
NOISE_CONVENTION = "replace"
Q_VALUES = [0.0, 1e-4, 1e-3, 5e-3, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
Q_VALUES_QUICK = [0.0, 0.01, 0.05, 0.20]
SEED = 20260918

_PAULI1 = {"I": I2, "X": PAULI_X, "Y": PAULI_Y, "Z": PAULI_Z}
PAULI_LABELS_2Q = [a + b for a in "IXYZ" for b in "IXYZ"]
PAULI_2Q = {lab: kron_list([_PAULI1[lab[0]], _PAULI1[lab[1]]]) for lab in PAULI_LABELS_2Q}


# ===========================================================================
# Bell-diagonal input construction
# ===========================================================================

def isotropic_pops(eps: float) -> np.ndarray:
    """(p_Phi+, p_Phi-, p_Psi+, p_Psi-) of (1-eps)|Phi+><Phi+| + eps I/4."""
    return np.array([1.0 - 3.0 * eps / 4.0, eps / 4.0, eps / 4.0, eps / 4.0])


def deterministic_cases() -> list[tuple[str, np.ndarray]]:
    """Named Bell-population vectors covering the requested regimes."""
    cases: list[tuple[str, np.ndarray]] = []
    for eps in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 2.0 / 3.0, 0.8, 1.0):
        cases.append((f"isotropic_eps{eps:.4f}", isotropic_pops(eps)))
    cases += [
        ("near_phi_plus", np.array([0.99, 0.005, 0.003, 0.002])),
        ("near_phi_plus_tight", np.array([0.9999, 0.00005, 0.00003, 0.00002])),
        ("asym_large_phi_minus", np.array([0.60, 0.35, 0.03, 0.02])),
        ("asym_large_psi_plus", np.array([0.60, 0.02, 0.35, 0.03])),
        ("asym_large_psi_minus", np.array([0.60, 0.02, 0.03, 0.35])),
        ("asym_dominant_psi_plus", np.array([0.20, 0.10, 0.65, 0.05])),
        ("strongly_mixed_uniform", np.array([0.25, 0.25, 0.25, 0.25])),
        ("strongly_mixed_tilted", np.array([0.30, 0.25, 0.25, 0.20])),
        ("boundary_pure_phi_plus", np.array([1.0, 0.0, 0.0, 0.0])),
        ("boundary_pure_phi_minus", np.array([0.0, 1.0, 0.0, 0.0])),
        ("boundary_pure_psi_plus", np.array([0.0, 0.0, 1.0, 0.0])),
        ("boundary_pure_psi_minus", np.array([0.0, 0.0, 0.0, 1.0])),
        ("boundary_edge_phi", np.array([0.5, 0.5, 0.0, 0.0])),
        ("boundary_edge_phiplus_psiplus", np.array([0.5, 0.0, 0.5, 0.0])),
        ("boundary_face_three", np.array([0.34, 0.33, 0.33, 0.0])),
        ("boundary_rank2_asym", np.array([0.70, 0.30, 0.0, 0.0])),
        ("boundary_rank2_tight", np.array([0.999, 0.001, 0.0, 0.0])),
        ("separable_threshold", np.array([0.5, 0.2, 0.2, 0.1])),
    ]
    return cases


def random_cases(n_uniform: int = 40, n_boundary: int = 20) -> list[tuple[str, np.ndarray]]:
    """Seeded Dirichlet samples: uniform on the simplex, plus boundary-biased."""
    rng = np.random.default_rng(SEED)
    out = []
    for i, p in enumerate(rng.dirichlet(np.ones(4), size=n_uniform)):
        out.append((f"random_uniform_{i:02d}", p))
    for i, p in enumerate(rng.dirichlet(0.25 * np.ones(4), size=n_boundary)):
        out.append((f"random_boundary_{i:02d}", p))
    return out


# ===========================================================================
# Bell-basis analysis of an output operator (no analytic map involved)
# ===========================================================================

def bell_analysis(rho: np.ndarray) -> tuple[np.ndarray, float]:
    """Bell populations <B|rho|B> and max_{B!=C} |<B|rho|C>|."""
    u = bell_basis_matrix()                      # columns = Bell vectors
    rb = u.conj().T @ rho @ u
    pops = np.real(np.diag(rb))
    off = np.abs(rb - np.diag(np.diag(rb)))
    return pops, float(off.max())


# ===========================================================================
# 4Q  --  path A: this repository's numpy dense simulator
# ===========================================================================

def run_4q_numpy(rho: np.ndarray, q: float):
    """Returns (rho_tilde, P_success) for the (0,0) postselected branch."""
    out4 = noisy_full_channel_dm(np.kron(rho, rho), q, NOISE_CONVENTION)
    return postselect_branch(out4, 0, 0)


# ===========================================================================
# 4Q  --  path B: independent PennyLane execution
# ===========================================================================

def _replacement_kraus(q: float):
    """Kraus operators of D_q on a two-qubit pair (16-Pauli form)."""
    ks = [np.sqrt(1 - q + q / 16) * PAULI_2Q["II"]]
    ks += [np.sqrt(q / 16) * PAULI_2Q[lab] for lab in PAULI_LABELS_2Q[1:]]
    return ks


def run_4q_pennylane(rho: np.ndarray, q: float):
    """Same circuit in PennyLane; tau reconstructed from <P_A (x) |00><00|>."""
    import pennylane as qml
    from pqec_distill.circuit import CNOT_SEQUENCE, Q3

    proj00 = np.zeros((4, 4), dtype=complex)
    proj00[0, 0] = 1.0
    dev = qml.device("default.mixed", wires=4)

    @qml.qnode(dev)
    def circ():
        qml.QubitDensityMatrix(np.kron(rho, rho), wires=[0, 1, 2, 3])
        for c, t in CNOT_SEQUENCE:
            qml.CNOT(wires=[c, t])
            if q > 0:
                qml.QubitChannel(_replacement_kraus(q), wires=[c, t])
        qml.Hadamard(Q3)
        return tuple(
            qml.expval(qml.Hermitian(np.kron(PAULI_2Q[lab], proj00), wires=[0, 1, 2, 3]))
            for lab in PAULI_LABELS_2Q)

    vals = circ()
    tau = np.zeros((4, 4), dtype=complex)
    for lab, v in zip(PAULI_LABELS_2Q, vals):
        tau += float(v) * PAULI_2Q[lab]
    tau /= 4.0
    return tau, float(np.real(np.trace(tau)))


# ===========================================================================
# Type 3 / Type 4  --  parent-repository circuits
# ===========================================================================

class ParentCircuits:
    """Loads the parent repository's Type 3 / Type 4 circuit implementations.

    Only circuit-simulation entry points are used:
      build_program / one_round_tau   (dense 32x32 propagation)
      CIRCUITS[...] qfuncs            (PennyLane execution of the same gates)
    No closed-form / fixed-point / Jacobian helper is imported.
    """

    def __init__(self, parent_repo: Path, scratch: Path):
        self.parent = Path(parent_repo).resolve()
        src = self.parent / "iterated_noisy_pqec.py"
        if not src.exists():
            # only present on the iterated-noisy-pqec branch -> extract it
            scratch.mkdir(parents=True, exist_ok=True)
            src = scratch / "_parent_iterated_noisy_pqec.py"
            blob = subprocess.run(
                ["git", "-C", str(self.parent), "show",
                 "origin/iterated-noisy-pqec:iterated_noisy_pqec.py"],
                capture_output=True, text=True, check=True).stdout
            src.write_text(blob)
            self.source_note = ("origin/iterated-noisy-pqec:iterated_noisy_pqec.py "
                                "(extracted; not on the checked-out branch)")
        else:
            self.source_note = "iterated_noisy_pqec.py (working tree)"
        for p in (str(self.parent), str(src.parent)):
            if p not in sys.path:
                sys.path.insert(0, p)
        self.mod = __import__(src.stem)
        self.commit = subprocess.run(
            ["git", "-C", str(self.parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()

    def n_cnots(self, name: str) -> int:
        return self.mod.build_program(name, 0.0)[1]

    def compilation_residual(self, name: str):
        return self.mod.compilation_residual(name)

    def tau_dense(self, rho: np.ndarray, q: float, name: str) -> np.ndarray:
        return self.mod.one_round_tau(rho, float(q), name)

    def tau_pennylane(self, rho: np.ndarray, q: float, name: str) -> np.ndarray:
        """Independent PennyLane path: <Z_a (x) P_A (x) I_B> for the 16 Paulis."""
        import pennylane as qml
        label, qfunc = self.mod.CIRCUITS[name]
        dev = qml.device("default.mixed", wires=5)

        @qml.qnode(dev)
        def circ():
            qml.QubitDensityMatrix(np.kron(rho, rho), wires=[1, 2, 3, 4])
            qfunc(float(q))
            return tuple(
                qml.expval(qml.PauliZ(0) @ qml.Hermitian(PAULI_2Q[lab], wires=[1, 2]))
                for lab in PAULI_LABELS_2Q)

        vals = circ()
        tau = np.zeros((4, 4), dtype=complex)
        for lab, v in zip(PAULI_LABELS_2Q, vals):
            tau += float(v) * PAULI_2Q[lab]
        return tau / 4.0


# ===========================================================================
# driver
# ===========================================================================

FIELDS = [
    "circuit", "n_cnot", "case", "case_kind",
    "p_in_PhiP", "p_in_PhiM", "p_in_PsiP", "p_in_PsiM", "q",
    "p_out_PhiP", "p_out_PhiM", "p_out_PsiP", "p_out_PsiM",
    "max_bell_offdiag", "trace_weight", "trace_weight_meaning",
    "norm_err", "min_bell_pop", "backend_cross_check",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-repo", default="/home/user/PQEC-Operational-Threshold")
    ap.add_argument("--scratch", default="/tmp/claude-0/-home-user-PQEC-Operational-Threshold/"
                                         "7cd40898-5d53-5f43-854d-6feeff8e5cd6/scratchpad")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--cross-check-every", type=int, default=0,
                    help="run the second backend on every Nth row (0 = deterministic cases only)")
    args = ap.parse_args()

    q_values = Q_VALUES_QUICK if args.quick else Q_VALUES
    det = deterministic_cases()
    rnd = random_cases(6, 4) if args.quick else random_cases()
    cases = [(n, p, "deterministic") for n, p in det] + [(n, p, "random") for n, p in rnd]

    # ---- parent circuits (optional) ---------------------------------------
    parent = None
    parent_err = None
    try:
        parent = ParentCircuits(Path(args.parent_repo), Path(args.scratch))
    except Exception as exc:                                   # noqa: BLE001
        parent_err = f"{type(exc).__name__}: {exc}"
        print(f"WARNING: Type 3 / Type 4 unavailable -- {parent_err}")

    have_pl = True
    try:
        import pennylane  # noqa: F401
    except Exception as exc:                                   # noqa: BLE001
        have_pl = False
        print(f"WARNING: PennyLane cross-check unavailable -- {type(exc).__name__}: {exc}")

    specs = [("4Q", 5)]
    if parent is not None:
        specs = [("Type3", parent.n_cnots("step3")),
                 ("Type4", parent.n_cnots("step4"))] + specs
    print("circuits:", ", ".join(f"{n} ({c} CNOT)" for n, c in specs))
    print(f"{len(cases)} inputs x {len(q_values)} q values x {len(specs)} circuits "
          f"= {len(cases)*len(q_values)*len(specs)} rows\n")

    rows = []
    worst_cross = {name: 0.0 for name, _ in specs}
    for circ_name, ncx in specs:
        for case_name, pops_in, kind in cases:
            rho = bell_diagonal_state(pops_in)
            for q in q_values:
                cross = ""
                if circ_name == "4Q":
                    tau, weight = run_4q_numpy(rho, q)
                    meaning = "P_success"
                else:
                    key = "step3" if circ_name == "Type3" else "step4"
                    tau = parent.tau_dense(rho, q, key)
                    weight = float(np.real(np.trace(tau)))
                    meaning = "parity_visibility"

                do_cross = have_pl and (
                    kind == "deterministic" if args.cross_check_every == 0
                    else len(rows) % args.cross_check_every == 0)
                if do_cross:
                    if circ_name == "4Q":
                        tau_b, _ = run_4q_pennylane(rho, q)
                    else:
                        key = "step3" if circ_name == "Type3" else "step4"
                        tau_b = parent.tau_pennylane(rho, q, key)
                    d = float(np.max(np.abs(tau - tau_b)))
                    worst_cross[circ_name] = max(worst_cross[circ_name], d)
                    cross = f"{d:.3e}"

                if weight > 1e-14:
                    rho_out = tau / weight
                    pops_out, offdiag = bell_analysis(rho_out)
                else:
                    pops_out = np.full(4, np.nan)
                    offdiag = float("nan")

                rows.append({
                    "circuit": circ_name, "n_cnot": ncx,
                    "case": case_name, "case_kind": kind,
                    "p_in_PhiP": pops_in[0], "p_in_PhiM": pops_in[1],
                    "p_in_PsiP": pops_in[2], "p_in_PsiM": pops_in[3], "q": q,
                    "p_out_PhiP": pops_out[0], "p_out_PhiM": pops_out[1],
                    "p_out_PsiP": pops_out[2], "p_out_PsiM": pops_out[3],
                    "max_bell_offdiag": offdiag,
                    "trace_weight": weight, "trace_weight_meaning": meaning,
                    "norm_err": abs(float(np.nansum(pops_out)) - 1.0),
                    "min_bell_pop": float(np.nanmin(pops_out)),
                    "backend_cross_check": cross,
                })
        print(f"  {circ_name}: {sum(1 for r in rows if r['circuit']==circ_name)} rows done")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "circuit_reference.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    meta = {
        "purpose": "circuit-only input->output reference; no analytic map used",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "this_repo_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True).stdout.strip(),
        "noise": {"convention": NOISE_CONVENTION,
                  "definition": "D_q(rho) = (1-q) rho + q [I_ij/4 (x) Tr_ij rho], after every CNOT",
                  "single_qubit_gates": "ideal"},
        "bell_order": BELL_NAMES,
        "seed": SEED, "q_values": q_values,
        "n_cases": len(cases), "n_rows": len(rows),
        "circuits": {},
        "backend_cross_check_max_abs_diff": worst_cross,
        "excluded_modules": ["pqec_distill.noisy_analytics", "pqec_distill.swap_test_reference",
                             "pqec_distill.repeated_noisy", "pqec_distill.analytics",
                             "iterated_noisy_pqec closed-form/fixed-point helpers"],
    }
    meta["circuits"]["4Q"] = {
        "n_cnot": 5, "qubit_order": "(q1,q2,q3,q4)=(0,1,2,3), q1 most significant",
        "gates": "pqec_distill.circuit.CNOT_SEQUENCE then H on q3",
        "output_definition": "postselect (m3,m4)=(0,0); rho_out = rho_tilde/P_success",
        "backends": ["pqec_distill numpy dense 16x16", "PennyLane default.mixed"],
    }
    if parent is not None:
        for name, key in (("Type3", "step3"), ("Type4", "step4")):
            res = parent.compilation_residual(key)
            meta["circuits"][name] = {
                "n_cnot": parent.n_cnots(key),
                "qubit_order": "(a,A1,A2,B1,B2)=(0,1,2,3,4)",
                "gates": ("verify_analytic_decomposed._fred x2 (textbook CSWAP)"
                          if key == "step3" else "pqec_resynth_noise.GATES"),
                "output_definition": ("tau_A = Tr_{a,B}[(Z_a (x) I) sigma_out]; "
                                      "rho_out = tau_A/Tr(tau_A) (parity-weighted, "
                                      "NOT a postselected physical state)"),
                "backends": ["dense 32x32 (iterated_noisy_pqec)", "PennyLane default.mixed"],
                "source": parent.source_note,
                "parent_commit": parent.commit,
                "compilation_residual_vs_U_PQEC": {"max_abs": res[0], "frobenius": res[1]},
            }
    else:
        meta["circuits"]["Type3"] = meta["circuits"]["Type4"] = {"skipped": parent_err}

    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"\nwrote {csv_path}  ({len(rows)} rows)")
    print(f"wrote {OUT_DIR / 'metadata.json'}")
    print("backend cross-check, max |tau_A - tau_B|:",
          {k: f"{v:.2e}" for k, v in worst_cross.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
