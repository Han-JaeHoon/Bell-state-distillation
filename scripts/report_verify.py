"""Verification stage for the report figures.  Run before anything is plotted.

Three independent implementations are compared:

  NOTE  the recurrence as written in the report note (scripts/report_recurrence)
  MAP   the repository's verified exact Bell-diagonal maps
        (pqec_distill.analytic_exact_map), validated row-by-row against the
        frozen circuit-only reference
  SIM   a full two-qubit density-matrix simulation of the actual circuit --
        the parent repository's five-qubit SWAP-test round for Type 3/4/5, and
        this package's noisy 5-CNOT + postselection for 4Q

Nothing is twirled or re-projected at any round: each round's exact output is
the next round's input.  That the Bell-diagonal representation loses nothing for
Type 3/4/4Q is DEMONSTRATED here (the full simulation's Bell off-diagonal weight
stays at round-off over the whole trajectory), not assumed.  Type 5 is carried
as a full density matrix throughout, because its map does not preserve the
Bell-diagonal subspace.
"""

from __future__ import annotations

import json

import numpy as np

from _bootstrap import DATA_DIR  # noqa: E402
from pqec_distill.analytic_exact_map import (  # noqa: E402
    analytic_map_4q, analytic_map_type3, analytic_map_type4, analytic_success_4q,
)
from pqec_distill.bell_population_dynamics import (  # noqa: E402
    INVARIANT_PAIR, bell_offdiagonal_c, bell_populations, isotropic_dm,
    one_round, verify_cnot_counts, verify_noise_convention,
)
from report_recurrence import (  # noqa: E402
    BELL_ORDER, NOTE_PROTOCOLS, isotropic_pops, note_fidelity_in_plane,
    note_step, note_success, pops_to_xyz,
)

MAP = {"Type3": analytic_map_type3, "Type4": analytic_map_type4,
       "4Q": analytic_map_4q}
EXPECTED_CNOT = {"Type3": 16, "Type4": 14, "Type5": 14, "4Q": 5}


def random_bell_diagonal(rng, n):
    """Random VALID Bell-diagonal states -- generic, not Bell-isotropic."""
    return rng.dirichlet(np.ones(4), size=n)


# ---------------------------------------------------------------------------

def print_setup(eps: float, q: float, n_rounds: int, n_long: int) -> dict:
    s = 1.0 - q
    setup = {
        "initial_state": "rho_0(eps) = (1-eps)|Phi+><Phi+| + eps I/4",
        "bell_order": list(BELL_ORDER),
        "eps_main": eps, "q": q, "s = 1-q": s,
        "p_0": isotropic_pops(eps).tolist(),
        "rounds_main": n_rounds, "rounds_type5": n_long,
        "noise_model": ("two-qubit replacement depolarizing "
                        "D_q(sigma) = (1-q) sigma + q [I_ij/4 (x) Tr_ij(sigma)], "
                        "applied after every CNOT on that CNOT's two qubits"),
        "backends": {
            "Type3": "exact Bell-diagonal recurrence (k=4); full 5-qubit density-matrix simulation as cross-check",
            "Type4": "exact Bell-diagonal recurrence (k=2); full 5-qubit density-matrix simulation as cross-check",
            "4Q": "exact Bell-diagonal recurrence + P_succ = D/4; full 4-qubit simulation with (q3,q4)->(0,0) postselection as cross-check",
            "Type5": "FULL two-qubit density matrix only -- the noisy map does not preserve the Bell-diagonal subspace",
        },
        "iteration_rule": ("each round's exact output state is the next round's "
                           "input; no twirling, no re-projection onto "
                           "Bell-isotropic or Werner form at any round"),
        "bell_isotropic_assumption": "used for the initial state only",
    }
    print("=" * 78)
    print("SETUP")
    print("=" * 78)
    print(f"  initial state      {setup['initial_state']}")
    print(f"  Bell ordering      {', '.join(BELL_ORDER)}")
    print(f"  main comparison    eps = {eps}, q = {q}  (s = 1 - q = {s})")
    print(f"  p_0                {np.array2string(np.array(setup['p_0']), precision=6)}")
    print(f"  noise              {setup['noise_model']}")
    print(f"  iteration          {setup['iteration_rule']}")
    print("  backends")
    for k, v in setup["backends"].items():
        print(f"    {k:<7} {v}")
    print()
    return setup


# ---------------------------------------------------------------------------

def verify(eps: float, q: float, n_rounds: int, seed: int = 20260929) -> dict:
    rng = np.random.default_rng(seed)
    out: dict = {}
    print("=" * 78)
    print("VERIFICATION")
    print("=" * 78)

    # ---- circuits and noise convention --------------------------------
    counts = verify_cnot_counts()
    out["cnot_counts"] = {"measured": counts, "expected": EXPECTED_CNOT,
                          "match": all(counts[k] == v for k, v in EXPECTED_CNOT.items())}
    print(f"  CNOT counts        {counts}  -> match {out['cnot_counts']['match']}")
    nc = verify_noise_convention(n_trials=3)
    out["noise_convention"] = nc
    print(f"  q == parent eps2   {nc['max_abs_diff_2q_vs_parent_kraus']:.2e} (2-qubit Kraus), "
          f"{nc['max_abs_diff_5q_vs_parent_replacement']:.2e} (5-qubit channel)")

    # ---- NOTE vs MAP on generic Bell-diagonal states -------------------
    print("\n  (1) note recurrence vs repository exact map, 400 random "
          "Bell-diagonal states (not Bell-isotropic)")
    states = random_bell_diagonal(rng, 400)
    note_vs_map = {}
    for protocol in NOTE_PROTOCOLS:
        worst = 0.0
        for p in states:
            for qq in (0.0, 0.005, 0.02, 0.08, 0.2):
                worst = max(worst, float(np.max(np.abs(
                    note_step(p, qq, protocol) - MAP[protocol](p, qq)))))
        note_vs_map[protocol] = worst
        print(f"      {protocol:<7} max |note - map| = {worst:.3e}")
    out["note_vs_map_random"] = note_vs_map

    # 4Q success probability, note vs repository
    worst_ps = 0.0
    for p in states:
        for qq in (0.0, 0.02, 0.2):
            worst_ps = max(worst_ps, abs(note_success(p, qq)
                                         - float(analytic_success_4q(p, qq))))
    out["note_vs_map_success_4q"] = worst_ps
    print(f"      4Q      max |note P_succ - map P_succ| = {worst_ps:.3e}")

    # ---- NOTE/MAP vs full simulation -----------------------------------
    print("\n  (2) exact map vs FULL density-matrix simulation of the circuit, "
          "40 random Bell-diagonal states")
    sim = {}
    for protocol in NOTE_PROTOCOLS:
        worst_p, worst_off = 0.0, 0.0
        for p in states[:40]:
            from pqec_distill.bell_states import bell_diagonal_state
            rho = bell_diagonal_state(p)
            for qq in (0.0, 0.02, 0.08):
                rho_next, _ = one_round(rho, qq, protocol)
                worst_p = max(worst_p, float(np.max(np.abs(
                    bell_populations(rho_next) - MAP[protocol](p, qq)))))
                worst_off = max(worst_off, bell_offdiagonal_c(rho_next))
        sim[protocol] = {"max_abs_population_diff": worst_p,
                         "max_C_Bell_of_simulated_output": worst_off}
        print(f"      {protocol:<7} max |map - simulation| = {worst_p:.3e}   "
              f"max C_Bell of the simulated output = {worst_off:.3e}")
    out["map_vs_simulation_random"] = sim

    # ---- along the actual trajectory ------------------------------------
    print(f"\n  (3) along the reported trajectory, eps = {eps}, q = {q}, "
          f"n = 0..{n_rounds}")
    traj = {}
    for protocol in NOTE_PROTOCOLS:
        p_note = isotropic_pops(eps)
        rho = isotropic_dm(eps)
        d_note_map = d_map_sim = d_fid = plane = off = 0.0
        sum_err, min_pop = 0.0, np.inf
        inv_a, inv_b = INVARIANT_PAIR[protocol]
        i_a, i_b = BELL_ORDER.index(inv_a), BELL_ORDER.index(inv_b)
        inv_err = 0.0
        for n in range(n_rounds + 1):
            p_sim = bell_populations(rho)
            d_map_sim = max(d_map_sim, float(np.max(np.abs(p_note - p_sim))))
            off = max(off, bell_offdiagonal_c(rho))
            sum_err = max(sum_err, abs(float(p_note.sum()) - 1.0))
            min_pop = min(min_pop, float(p_note.min()))
            inv_err = max(inv_err, abs(float(p_note[i_a] - p_note[i_b])))
            x, y, z = pops_to_xyz(p_note)
            plane = max(plane, abs(x + y) if protocol != "4Q" else abs(y + z))
            d_fid = max(d_fid, abs(note_fidelity_in_plane(p_note, protocol)
                                   - float(p_note[0])))
            if n < n_rounds:
                nxt = MAP[protocol](p_note, q)
                d_note_map = max(d_note_map, float(np.max(np.abs(
                    note_step(p_note, q, protocol) - nxt))))
                p_note = nxt
                rho, _ = one_round(rho, q, protocol)
        traj[protocol] = {
            "max_abs_note_minus_map": d_note_map,
            "max_abs_map_minus_simulation": d_map_sim,
            "max_C_Bell_in_full_simulation": off,
            "max_abs_population_sum_minus_1": sum_err,
            "min_population": min_pop,
            "invariant_pair": f"p_{inv_a} = p_{inv_b}",
            "invariant_pair_max_abs_diff": inv_err,
            "invariant_plane": "y = -x" if protocol != "4Q" else "y = -z",
            "invariant_plane_max_abs_residual": plane,
            "fidelity_formula": ("(1 + x + 2z)/4" if protocol == "4Q"
                                 else "(1 + 2x + z)/4"),
            "max_abs_fidelity_formula_minus_p_PhiPlus": d_fid,
        }
        t = traj[protocol]
        print(f"      {protocol:<7} note-map {t['max_abs_note_minus_map']:.2e}  "
              f"map-sim {t['max_abs_map_minus_simulation']:.2e}  "
              f"C_Bell(sim) {t['max_C_Bell_in_full_simulation']:.2e}  "
              f"sum-1 {t['max_abs_population_sum_minus_1']:.2e}  "
              f"min p {t['min_population']:.6f}")
        print(f"              {t['invariant_pair']:<19} {t['invariant_pair_max_abs_diff']:.2e}   "
              f"plane {t['invariant_plane']}  {t['invariant_plane_max_abs_residual']:.2e}   "
              f"F = {t['fidelity_formula']}  vs p_Phi+  {t['max_abs_fidelity_formula_minus_p_PhiPlus']:.2e}")
    out["trajectory"] = traj

    # ---- Type 5 -----------------------------------------------------------
    print("\n  (4) Type 5, full two-qubit density matrix")
    rho = isotropic_dm(eps)
    sum_err, off_max, min_eig, herm = 0.0, 0.0, np.inf, 0.0
    for n in range(n_rounds + 1):
        p = bell_populations(rho)
        sum_err = max(sum_err, abs(float(p.sum()) - 1.0))
        off_max = max(off_max, bell_offdiagonal_c(rho))
        ev = np.linalg.eigvalsh(0.5 * (rho + rho.conj().T))
        min_eig = min(min_eig, float(ev.min()))
        herm = max(herm, float(np.linalg.norm(rho - rho.conj().T)))
        if n < n_rounds:
            rho, _ = one_round(rho, q, "Type5")
    out["type5"] = {
        "max_abs_population_sum_minus_1": sum_err,
        "max_C_Bell": off_max,
        "min_eigenvalue": min_eig,
        "max_anti_hermitian_norm": herm,
        "bell_populations_alone_describe_the_state": bool(off_max < 1e-20),
    }
    print(f"      populations sum to 1 to {sum_err:.2e}, min eigenvalue {min_eig:.3e}, "
          f"anti-Hermitian {herm:.1e}")
    print(f"      but C_Bell reaches {off_max:.3e} by n = {n_rounds}: the four Bell "
          "populations do NOT describe the Type 5 state")

    out["all_agreements_at_machine_precision"] = bool(
        max(list(note_vs_map.values())
            + [worst_ps]
            + [v["max_abs_population_diff"] for v in sim.values()]
            + [t["max_abs_note_minus_map"] for t in traj.values()]
            + [t["max_abs_map_minus_simulation"] for t in traj.values()]) < 1e-12)
    print(f"\n  all three implementations agree at machine precision: "
          f"{out['all_agreements_at_machine_precision']}")
    return out


def main() -> int:
    eps, q, n_rounds, n_long = 0.15, 0.02, 50, 5000
    setup = print_setup(eps, q, n_rounds, n_long)
    checks = verify(eps, q, n_rounds)
    path = DATA_DIR / "report_figures" / "verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"setup": setup, "checks": checks}, indent=2))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
