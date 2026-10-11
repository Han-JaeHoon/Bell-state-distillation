#!/usr/bin/env python3
"""Does the frozen 14-CNOT Type-4 circuit factorise into two identical
standalone 7-CNOT Fredkin blocks?

This script answers that question from the repository's own gate list.  It
checks, in order:

1. that the gate list used here is byte-identical to the frozen Type-4 list in
   ``scripts/verify_type4_qabs_circuit.py`` and
   ``scripts/verify_type4_qrel_circuit.py``, and to the parent repository's
   ``pqec_resynth_noise.py`` ``GATES`` when that file is reachable;
2. that the 14 CNOTs split 7 + 7 over the supports ``{c,A1,A2} = {0,1,3}`` and
   ``{c,B1,B2} = {0,2,4}``, matching the expected orientation and order;
3. that the two blocks extracted from the frozen sequence are gate-for-gate
   identical after relabelling their local wires to ``(c,u,v)``, and identical
   to the candidate block ``G7``;
4. that each block's 8x8 unitary equals the ideal Fredkin up to a global phase;
5. that regrouping the frozen sequence into ``H_c G7_A G7_B H_c`` moves only
   operations with disjoint support (a certificate), and that the ideal 32x32
   unitaries agree;
6. that the NOISY 32x32 channels of the two orderings agree, on arbitrary full
   five-qubit density matrices and over a wide q grid -- the decisive check,
   since the decomposition and not merely the unitary is what the noise sees.

The expected answers are treated as hypotheses.  No analytic Type-4 recurrence
is used; ``pqec_distill.analytic_exact_map`` is never imported.
"""

from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pqec_distill import type4_7cnot_fredkin as T4                 # noqa: E402
from pqec_distill.noise import replacement_depolarizing            # noqa: E402

OUT = ROOT / "results" / "data" / "type4_7cnot_factorization"

TOL_SANITY = 1e-12
Q_GRID = (0.0, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.2, 0.5, 0.75, 1.0)

#: the CNOT pattern the task states the frozen sequence should have
EXPECTED_CNOTS_A = [(1, 3), (0, 3), (1, 3), (0, 3), (0, 1), (0, 1), (3, 1)]
EXPECTED_CNOTS_B = [(2, 4), (0, 4), (2, 4), (0, 4), (0, 2), (0, 2), (4, 2)]


# ===========================================================================
# independent reference route: permutation CNOTs, U3 from its matrix, own
# replacement-depolarizing channel, gate-by-gate, no merged unitaries
# ===========================================================================

_I2 = np.eye(2, dtype=complex)
_ROW, _COL = "abcdefgh", "ijklmnop"


def ref_u3(theta, phi, lam) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -(np.cos(lam) + 1j * np.sin(lam)) * s],
                     [(np.cos(phi) + 1j * np.sin(phi)) * s,
                      (np.cos(phi + lam) + 1j * np.sin(phi + lam)) * c]],
                    dtype=complex)


def ref_gate(kind, arg, n: int) -> np.ndarray:
    d = 2 ** n
    if kind == "CNOT":
        c, tgt = arg
        m = np.zeros((d, d), dtype=complex)
        for idx in range(d):
            bits = [(idx >> (n - 1 - w)) & 1 for w in range(n)]
            if bits[c] == 1:
                bits[tgt] ^= 1
            m[sum(b << (n - 1 - w) for w, b in enumerate(bits)), idx] = 1.0
        return m
    w, t, p, l = arg
    out = np.array([[1.0 + 0j]])
    for k in range(n):
        out = np.kron(out, ref_u3(t, p, l) if k == w else _I2)
    return out


def ref_depol(sigma, pair, q, n) -> np.ndarray:
    """``(1-q) sigma + q [I_ij/4 (x) Tr_ij sigma]`` as two einsum contractions."""
    if q == 0.0:
        return sigma
    i, j = pair
    rest = [w for w in range(n) if w not in (i, j)]
    t = sigma.reshape([2] * (2 * n))
    row, col = list(_ROW[:n]), list(_COL[:n])
    for w in (i, j):
        col[w] = row[w]
    sub = "".join(row) + "".join(col)
    out = "".join(row[w] for w in rest) + "".join(col[w] for w in rest)
    reduced = np.einsum(f"{sub}->{out}", t)
    target = _ROW[:n] + _COL[:n]
    rebuilt = np.einsum(f"{out},{_ROW[i]}{_COL[i]},{_ROW[j]}{_COL[j]}->{target}",
                        reduced, _I2, _I2)
    d = 2 ** n
    return (1 - q) * sigma + q * 0.25 * rebuilt.reshape(d, d)


def ref_run(sigma, ops, q, n, q_schedule=None, skip_noise_after=()):
    k = 0
    for kind, arg in ops:
        g = ref_gate(kind, arg, n)
        sigma = g @ sigma @ g.conj().T
        if kind == "CNOT":
            k += 1
            qk = q if q_schedule is None else q_schedule[k - 1]
            if k not in skip_noise_after:
                sigma = ref_depol(sigma, tuple(arg), qk, n)
    return sigma


def ref_unitary(ops, n) -> np.ndarray:
    u = np.eye(2 ** n, dtype=complex)
    for kind, arg in ops:
        u = ref_gate(kind, arg, n) @ u
    return u


# ===========================================================================
# 1. provenance of the gate list
# ===========================================================================

def _extract_list(path: Path, name: str):
    text = path.read_text()
    start = text.index(f"{name} = [")
    end = text.index("\n]", start) + 2
    ns = {"np": np}
    exec(text[start:end], ns)                                     # noqa: S102
    return ns[name]


def _normalise(ops):
    out = []
    for item in ops:
        if len(item) == 3:                      # parent repo ('u', [w], [..])
            kind, wires, params = item
            if kind == "cx":
                out.append(("CNOT", tuple(int(w) for w in wires)))
            elif kind == "u":
                out.append(("U3", (int(wires[0]),) + tuple(float(x) for x in params)))
            else:
                raise ValueError(f"unexpected parent gate {kind!r}")
        else:
            kind, arg = item
            out.append((kind, tuple(arg) if kind == "CNOT"
                        else (int(arg[0]),) + tuple(float(x) for x in arg[1:])))
    return out


def provenance_checks() -> dict:
    mine = _normalise(T4.TYPE4_5Q_OPS)
    res = {"n_operations": len(mine), "n_cnot": len(T4.ops_cnot_pairs(T4.TYPE4_5Q_OPS))}
    for label, path, name in (
            ("verify_type4_qabs_circuit.py", ROOT / "scripts" / "verify_type4_qabs_circuit.py", "TYPE4_OPS"),
            ("verify_type4_qrel_circuit.py", ROOT / "scripts" / "verify_type4_qrel_circuit.py", "TYPE4_OPS")):
        try:
            res[f"matches_{label}"] = _normalise(_extract_list(path, name)) == mine
        except Exception as exc:                                  # pragma: no cover
            res[f"matches_{label}"] = f"unavailable: {exc}"
    parent = ROOT.parent / "PQEC-Operational-Threshold" / "pqec_resynth_noise.py"
    if parent.exists():
        try:
            res["matches_parent_pqec_resynth_noise_GATES"] = (
                _normalise(_extract_list(parent, "GATES")) == mine)
        except Exception as exc:                                  # pragma: no cover
            res["matches_parent_pqec_resynth_noise_GATES"] = f"unavailable: {exc}"
    else:
        res["matches_parent_pqec_resynth_noise_GATES"] = "parent file not present"
    # the module's own U3 must agree with the frozen script's U3
    frozen_u3 = None
    try:
        text = (ROOT / "scripts" / "verify_type4_qabs_circuit.py").read_text()
        start = text.index("def u3_matrix(")
        end = text.index("\ndef ", start + 1)
        ns = {"np": np}
        exec(text[start:end], ns)                                 # noqa: S102
        frozen_u3 = ns["u3_matrix"]
    except Exception as exc:                                      # pragma: no cover
        res["u3_vs_frozen_u3"] = f"unavailable: {exc}"
    if frozen_u3 is not None:
        rng = np.random.default_rng(1)
        worst = 0.0
        for _ in range(200):
            t, p, l = rng.uniform(-2 * np.pi, 2 * np.pi, size=3)
            worst = max(worst, float(np.max(np.abs(
                T4.u3_matrix(t, p, l) - frozen_u3(t, p, l)))))
        res["u3_vs_frozen_u3"] = worst
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(200):
        t, p, l = rng.uniform(-2 * np.pi, 2 * np.pi, size=3)
        worst = max(worst, float(np.max(np.abs(
            T4.u3_matrix(t, p, l) - ref_u3(t, p, l)))))
    res["u3_vs_reference_u3"] = worst
    return res


# ===========================================================================
# 2-3. CNOT split, block extraction, Fredkin identification
# ===========================================================================

def split_checks() -> dict:
    split = T4.split_type4_sequence()
    a_pairs = T4.ops_cnot_pairs(split["alice_ops"])
    b_pairs = T4.ops_cnot_pairs(split["bob_ops"])
    key = lambda ops: [(k, tuple(a)) for k, a in ops]             # noqa: E731
    res = {
        "leading_h_position": split["leading_h_position"],
        "trailing_h_position": split["trailing_h_position"],
        "alice_positions": split["alice_positions"],
        "bob_positions": split["bob_positions"],
        "alice_cnots": [list(p) for p in a_pairs],
        "bob_cnots": [list(p) for p in b_pairs],
        "alice_cnots_match_expected": a_pairs == EXPECTED_CNOTS_A,
        "bob_cnots_match_expected": b_pairs == EXPECTED_CNOTS_B,
        "n_cnot_alice": len(a_pairs),
        "n_cnot_bob": len(b_pairs),
        "n_ops_alice": len(split["alice_ops"]),
        "n_ops_bob": len(split["bob_ops"]),
        "alice_local_equals_bob_local": key(split["alice_local"]) == key(split["bob_local"]),
        "alice_local_equals_candidate_G7": key(split["alice_local"]) == key(T4.G7_OPS),
        "bob_local_equals_candidate_G7": key(split["bob_local"]) == key(T4.G7_OPS),
        "n_cnot_candidate_G7": T4.N_CNOT_G7,
        "alice_support": sorted({w for k, a in split["alice_ops"]
                                 for w in T4.ops_support(k, a)}),
        "bob_support": sorted({w for k, a in split["bob_ops"]
                               for w in T4.ops_support(k, a)}),
    }
    # gate-for-gate numeric residual between the two extracted local blocks
    worst = 0.0
    for (k1, a1), (k2, a2) in zip(split["alice_local"], split["bob_local"]):
        if k1 != k2 or len(a1) != len(a2):
            worst = float("inf")
            break
        worst = max(worst, max(abs(float(x) - float(y)) for x, y in zip(a1, a2)))
    res["alice_vs_bob_local_max_parameter_diff"] = worst
    res["G7_sequence"] = [(k, list(a)) for k, a in T4.G7_OPS]
    return res, split


def fredkin_checks(split) -> dict:
    fred = T4.fredkin_matrix()
    res = {}
    for name, ops_local in (("candidate_G7", T4.G7_OPS),
                            ("extracted_alice", split["alice_local"]),
                            ("extracted_bob", split["bob_local"])):
        u = T4.block_unitary(ops_local)
        res[name] = T4.global_phase_fit(u, fred)
        res[name]["n_cnot"] = len(T4.ops_cnot_pairs(ops_local))
        # the reference route must build the same 8x8 unitary
        res[name]["vs_reference_unitary"] = float(np.max(np.abs(
            u - ref_unitary(ops_local, 3))))
    return res


# ===========================================================================
# 4A. ideal full-unitary check
# ===========================================================================

def ideal_unitary_checks() -> dict:
    u_orig = T4.ops_unitary(T4.TYPE4_5Q_OPS, 5)
    u_regr = T4.ops_unitary(T4.REGROUPED_5Q_OPS, 5)
    cert = T4.commutation_certificate(T4.TYPE4_5Q_OPS, T4.REGROUPED_5Q_OPS)

    # the ideal target: H_c . Fredkin(c;A1,A2) . Fredkin(c;B1,B2) . H_c
    h = T4.gate_matrix(*T4.H_C_OP, 5)
    fred8 = T4.fredkin_matrix()

    def embed3(gate8, wires):
        rest = [w for w in range(5) if w not in wires]
        big = np.kron(gate8, np.eye(2 ** len(rest), dtype=complex))
        t = big.reshape([2] * 10)
        order = list(wires) + rest
        perm = np.argsort(order)
        axes = list(perm) + [5 + p for p in perm]
        return t.transpose(axes).reshape(32, 32)

    f_a = embed3(fred8, T4.ALICE_TRIPLE_5Q)
    f_b = embed3(fred8, T4.BOB_TRIPLE_5Q)
    target = h @ f_b @ f_a @ h

    res = {
        "original_vs_regrouped": T4.global_phase_fit(u_orig, u_regr),
        "original_vs_ideal_swap_test": T4.global_phase_fit(u_orig, target),
        "regrouped_vs_ideal_swap_test": T4.global_phase_fit(u_regr, target),
        "original_vs_reference_route": float(np.max(np.abs(
            u_orig - ref_unitary(T4.TYPE4_5Q_OPS, 5)))),
        "fredkins_commute": float(np.max(np.abs(f_a @ f_b - f_b @ f_a))),
        "commutation_certificate": {
            k: cert[k] for k in ("same_length", "same_multiset", "n_inversions",
                                 "is_certified")},
        "commutation_offending_pairs": cert["offending_pairs"],
        "commutation_permutation": cert["permutation"],
    }
    return res


# ===========================================================================
# 4B. noisy full-channel check
# ===========================================================================

def _random_dm(rng, dim):
    a = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
    m = a @ a.conj().T
    return m / np.trace(m).real


def _random_dm_rank(rng, dim, rank):
    a = rng.normal(size=(dim, rank)) + 1j * rng.normal(size=(dim, rank))
    m = a @ a.conj().T
    return m / np.trace(m).real


def noisy_channel_checks(n_states=8) -> dict:
    """Original vs regrouped NOISY channel, on arbitrary full 5Q states."""
    rng = np.random.default_rng(20261011)
    states = [("random_full_rank", _random_dm(rng, 32)) for _ in range(n_states // 2)]
    states += [("rank_deficient", _random_dm_rank(rng, 32, 5))
               for _ in range(max(1, n_states // 4))]
    v = rng.normal(size=32) + 1j * rng.normal(size=32)
    v = v / np.linalg.norm(v)
    states.append(("random_pure", np.outer(v, v.conj())))
    # also the physically relevant preparation |0><0|_c (x) R
    r = _random_dm(rng, 16)
    states.append(("ancilla0_tensor_R", np.kron(
        np.array([[1, 0], [0, 0]], dtype=complex), r)))

    prog_o = T4.build_program(T4.TYPE4_5Q_OPS, 5)
    prog_r = T4.build_program(T4.REGROUPED_5Q_OPS, 5)

    def channel(s, pair, q, n):
        return replacement_depolarizing(s, pair, q, n_qubits=n)

    rows = []
    worst = {"main": 0.0, "reference": 0.0, "main_vs_reference": 0.0}
    worst_tag = None
    for label, sigma in states:
        for q in Q_GRID:
            a = T4.run_noisy_ops(sigma, T4.TYPE4_5Q_OPS, q, 5,
                                 channel=channel, program=prog_o)
            b = T4.run_noisy_ops(sigma, T4.REGROUPED_5Q_OPS, q, 5,
                                 channel=channel, program=prog_r)
            ra = ref_run(sigma, T4.TYPE4_5Q_OPS, q, 5)
            rb = ref_run(sigma, T4.REGROUPED_5Q_OPS, q, 5)
            d_main = float(np.max(np.abs(a - b)))
            d_ref = float(np.max(np.abs(ra - rb)))
            d_cross = max(float(np.max(np.abs(a - ra))),
                          float(np.max(np.abs(b - rb))))
            rows.append({"state": label, "q": q,
                         "max_abs_diff_main": d_main,
                         "frobenius_diff_main": float(np.linalg.norm(a - b)),
                         "max_abs_diff_reference": d_ref,
                         "main_vs_reference": d_cross,
                         "trace_original": float(np.real(np.trace(a))),
                         "trace_regrouped": float(np.real(np.trace(b)))})
            if d_main > worst["main"]:
                worst["main"], worst_tag = d_main, (label, q)
            worst["reference"] = max(worst["reference"], d_ref)
            worst["main_vs_reference"] = max(worst["main_vs_reference"], d_cross)
    return {"rows": rows, "worst": worst, "worst_tag": worst_tag,
            "n_states": len(states), "q_grid": list(Q_GRID)}


# ===========================================================================
# output
# ===========================================================================

def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _git(*args) -> str:
    try:
        return subprocess.run(["git", "-C", str(ROOT)] + list(args),
                              capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return ""


def main() -> int:
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 74)
    print("Does the 14-CNOT Type-4 circuit factorise into two 7-CNOT Fredkins?")
    print("=" * 74)

    print("\n[1] provenance of the gate list")
    prov = provenance_checks()
    for k, v in prov.items():
        print(f"      {k:46s} {v}")

    print("\n[2] CNOT split and block extraction")
    split_res, split = split_checks()
    for k in ("leading_h_position", "trailing_h_position", "n_cnot_alice",
              "n_cnot_bob", "n_ops_alice", "n_ops_bob", "alice_support",
              "bob_support", "alice_cnots_match_expected",
              "bob_cnots_match_expected", "alice_local_equals_bob_local",
              "alice_local_equals_candidate_G7", "bob_local_equals_candidate_G7",
              "n_cnot_candidate_G7", "alice_vs_bob_local_max_parameter_diff"):
        print(f"      {k:46s} {split_res[k]}")
    print("      Alice positions in the frozen list:", split_res["alice_positions"])
    print("      Bob   positions in the frozen list:", split_res["bob_positions"])

    print("\n[3] is each block an ideal Fredkin up to a global phase?")
    fred = fredkin_checks(split)
    for name, info in fred.items():
        print(f"      {name}")
        for k in ("n_cnot", "global_phase", "global_phase_over_pi",
                  "trace_overlap_abs_over_dim", "max_entry_residual",
                  "frobenius_residual", "unitarity_residual",
                  "residual_without_phase", "vs_reference_unitary"):
            print(f"          {k:34s} {info[k]}")

    print("\n[4A] ideal full-unitary regrouping check")
    ideal = ideal_unitary_checks()
    for name in ("original_vs_regrouped", "original_vs_ideal_swap_test",
                 "regrouped_vs_ideal_swap_test"):
        info = ideal[name]
        print(f"      {name}: phase = {info['global_phase_over_pi']:+.6f} pi, "
              f"max = {info['max_entry_residual']:.6e}, "
              f"fro = {info['frobenius_residual']:.6e}")
    print(f"      original_vs_reference_route      "
          f"{ideal['original_vs_reference_route']:.6e}")
    print(f"      fredkins_commute                 {ideal['fredkins_commute']:.6e}")
    print(f"      commutation certificate          {ideal['commutation_certificate']}")
    print(f"      offending (support-overlapping) reorderings: "
          f"{len(ideal['commutation_offending_pairs'])}")

    print("\n[4B] NOISY full-channel regrouping check")
    noisy = noisy_channel_checks()
    print(f"      states {noisy['n_states']}, q grid {noisy['q_grid']}")
    print(f"      max |original - regrouped| (main path)      "
          f"{noisy['worst']['main']:.6e}   worst {noisy['worst_tag']}")
    print(f"      max |original - regrouped| (reference path) "
          f"{noisy['worst']['reference']:.6e}")
    print(f"      max |main - reference|                      "
          f"{noisy['worst']['main_vs_reference']:.6e}")

    stop = []
    if not split_res["alice_cnots_match_expected"] or not split_res["bob_cnots_match_expected"]:
        stop.append("CNOT pattern does not match the stated candidate")
    if not (split_res["alice_local_equals_candidate_G7"]
            and split_res["bob_local_equals_candidate_G7"]):
        stop.append("extracted blocks differ from the candidate G7")
    if not split_res["alice_local_equals_bob_local"]:
        stop.append("the two extracted blocks are not identical after relabelling")
    for name in ("candidate_G7", "extracted_alice", "extracted_bob"):
        if fred[name]["max_entry_residual"] > TOL_SANITY:
            stop.append(f"{name} is not an ideal Fredkin up to a global phase")
    if ideal["original_vs_regrouped"]["max_entry_residual"] > TOL_SANITY:
        stop.append("ideal unitaries of the two orderings disagree")
    if not ideal["commutation_certificate"]["is_certified"]:
        stop.append("regrouping is not a disjoint-support reordering")
    if noisy["worst"]["main"] > TOL_SANITY:
        stop.append("noisy channels of the two orderings disagree")

    print("\n[5] stop conditions")
    if stop:
        for s in stop:
            print(f"      STOP: {s}")
    else:
        print("      none triggered")

    write_csv(OUT / "noisy_channel_equality.csv", noisy["rows"])
    write_csv(OUT / "g7_sequence.csv",
              [{"index": i + 1, "kind": k,
                "wires": " ".join(str(x) for x in (a if k == "CNOT" else a[:1])),
                "theta_over_pi": "" if k == "CNOT" else a[1] / np.pi,
                "phi_over_pi": "" if k == "CNOT" else a[2] / np.pi,
                "lam_over_pi": "" if k == "CNOT" else a[3] / np.pi}
               for i, (k, a) in enumerate(T4.G7_OPS)])
    write_csv(OUT / "fredkin_identification.csv",
              [dict(block=name, **{k: v for k, v in info.items()})
               for name, info in fred.items()])

    meta = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/verify_type4_7cnot_factorization.py",
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_head": _git("rev-parse", "HEAD"),
        "python": sys.version.split()[0], "numpy": np.__version__,
        "platform": platform.platform(),
        "type4_source": T4.TYPE4_SOURCE,
        "wires_5q": list(T4.WIRES_5Q),
        "alice_triple_5q": list(T4.ALICE_TRIPLE_5Q),
        "bob_triple_5q": list(T4.BOB_TRIPLE_5Q),
        "tolerance_sanity": TOL_SANITY,
        "not_used": ["analytic Type-4 Bell-diagonal recurrence",
                     "pqec_distill.analytic_exact_map",
                     "closed-form noisy map",
                     "previously derived Type-4 thresholds"],
        "provenance": prov,
        "split": split_res,
        "fredkin_identification": fred,
        "ideal_unitary": ideal,
        "noisy_channel": {"worst": noisy["worst"], "worst_tag": noisy["worst_tag"],
                          "n_states": noisy["n_states"],
                          "q_grid": noisy["q_grid"]},
        "stop_conditions_triggered": stop,
        "total_seconds": time.time() - t0,
    }
    with (OUT / "metadata.json").open("w") as fh:
        json.dump(meta, fh, indent=2, default=str)
    print(f"\n  written to {OUT.relative_to(ROOT)}   ({time.time() - t0:.1f} s)")
    return 1 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
