"""Access to the parent repository's verified SWAP-test round map.

The Type 3 / Type 4 / Type 5 circuits and the five-qubit round
``tau_A = Tr_{a,B}[(Z_a (x) I) E_q(|0><0| (x) rho (x) rho)]`` are NOT
reimplemented in this package.  They live in the parent research repository

    PQEC-Operational-Threshold

as ``iterated_noisy_pqec.py`` (on the branch that introduced it) together with
the circuit definitions it captures:

    verify_analytic_decomposed._fred / _tof / _c2   -> Type 3, textbook 16-CNOT
    pqec_resynth_noise.GATES                        -> Type 4, resynthesised 14-CNOT
    pqc_ring_threshold._apply                       -> Type 5, learned 14-CNOT
        (pqc_ring_prune.ansatz_masked + pqc_ring_pruned.json + ..._params.npy)
    noisy_bell_state.global_depol_kraus             -> the per-CNOT noise channel

This module locates that checkout, materialises ``iterated_noisy_pqec.py`` from
a pinned revision WITHOUT touching the parent working tree (``git show``, no
checkout, no branch switch), imports it, and records provenance: the parent
commit, the revision the round map came from, and a SHA-256 for every file whose
contents entered the calculation.  Nothing in the parent repository is modified.

Set ``PQEC_PARENT_REPO`` to override the search.  If no checkout is found,
:func:`load_parent_module` raises with a message naming exactly what is missing;
Type 5 then cannot be analysed at all, and Type 3/4 are still available through
``pqec_distill.analytic_exact_map``.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import os
import subprocess
import sys
import tempfile
from pathlib import Path

__all__ = [
    "PARENT_REPO_NAME", "ROUND_MAP_FILE", "ROUND_MAP_REV", "CIRCUIT_FILES",
    "find_parent_repo", "load_parent_module", "provenance",
]

PARENT_REPO_NAME = "PQEC-Operational-Threshold"
ROUND_MAP_FILE = "iterated_noisy_pqec.py"

#: Revisions to try, in order, for the round-map file.  The first that exists
#: wins; the one actually used is recorded in :func:`provenance`.
ROUND_MAP_REV = ("HEAD", "origin/iterated-noisy-pqec", "iterated-noisy-pqec")

#: Files in the parent working tree whose contents enter the calculation.
CIRCUIT_FILES = (
    "verify_analytic_decomposed.py",
    "pqec_resynth_noise.py",
    "noisy_bell_state.py",
    "pqc_ring_threshold.py",
    "pqc_ring_prune.py",
    "pqc_common.py",
    "pqc_ring_pruned.json",
    "pqc_ring_pruned_params.npy",
)

_CACHE: dict[str, object] = {}


def _candidates() -> list[Path]:
    env = os.environ.get("PQEC_PARENT_REPO")
    here = Path(__file__).resolve()
    repo_root = here.parents[2]
    out = []
    if env:
        out.append(Path(env).expanduser())
    out += [repo_root.parent / PARENT_REPO_NAME,
            Path.home() / PARENT_REPO_NAME,
            Path("/home/user") / PARENT_REPO_NAME]
    return out


def find_parent_repo() -> Path:
    """Path of the parent checkout, or raise with the places that were tried."""
    tried = []
    for cand in _candidates():
        tried.append(str(cand))
        if (cand / ".git").exists() and (cand / "pqc_ring_threshold.py").exists():
            return cand.resolve()
    raise FileNotFoundError(
        "the parent repository " + PARENT_REPO_NAME + " was not found, so the "
        "Type 3 / Type 4 / Type 5 SWAP-test round map is unavailable.\n"
        "Tried: " + ", ".join(tried) + "\n"
        "Set PQEC_PARENT_REPO to a checkout that contains "
        + ROUND_MAP_FILE + " (on HEAD or on the iterated-noisy-pqec branch) "
        "together with " + ", ".join(CIRCUIT_FILES) + ".")


def _git(root: Path, *args: str) -> str:
    return subprocess.run(("git", "-C", str(root)) + args,
                          capture_output=True, text=True, check=True).stdout


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch_round_map(root: Path) -> tuple[bytes, str]:
    """(file bytes, revision used).  Prefers the working tree if the file is there."""
    local = root / ROUND_MAP_FILE
    if local.exists():
        return local.read_bytes(), "working tree"
    errors = []
    for rev in ROUND_MAP_REV:
        try:
            out = subprocess.run(("git", "-C", str(root), "show", f"{rev}:{ROUND_MAP_FILE}"),
                                 capture_output=True, check=True)
            return out.stdout, rev
        except subprocess.CalledProcessError as exc:  # pragma: no cover - env dependent
            errors.append(f"{rev}: {exc.stderr.decode(errors='replace').strip()}")
    raise FileNotFoundError(
        f"{ROUND_MAP_FILE} is not in {root} nor in any of {ROUND_MAP_REV}.\n"
        + "\n".join(errors))


@contextlib.contextmanager
def _in_dir(path: Path):
    """Temporarily chdir.  ``pqc_ring_threshold`` opens its learned parameters by
    relative path, so the parent root must be the working directory while it is
    imported."""
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def load_parent_module():
    """Import and return the parent repository's ``iterated_noisy_pqec`` module."""
    if "module" in _CACHE:
        return _CACHE["module"]

    root = find_parent_repo()
    source, rev = _fetch_round_map(root)
    digest = _sha256_bytes(source)

    cache_dir = Path(tempfile.mkdtemp(prefix="pqec_swap_test_"))
    target = cache_dir / ROUND_MAP_FILE
    target.write_bytes(source)

    added = []
    for p in (str(cache_dir), str(root)):
        if p not in sys.path:
            sys.path.insert(0, p)
            added.append(p)

    spec = importlib.util.spec_from_file_location("iterated_noisy_pqec", target)
    module = importlib.util.module_from_spec(spec)
    sys.modules["iterated_noisy_pqec"] = module
    with _in_dir(root):
        spec.loader.exec_module(module)
        # The round map imports the circuit modules lazily, inside the qfunc for
        # each circuit.  ``pqc_ring_threshold`` loads the learned parameters by
        # RELATIVE path, so import them all now, while the parent root is still
        # the working directory; afterwards they are in ``sys.modules`` and the
        # lazy imports are no-ops from any directory.
        for name in ("verify_analytic_decomposed", "pqec_resynth_noise",
                     "noisy_bell_state", "pqc_ring_threshold"):
            importlib.import_module(name)

    files = {}
    for name in CIRCUIT_FILES:
        path = root / name
        files[name] = _sha256_bytes(path.read_bytes()) if path.exists() else None

    _CACHE["module"] = module
    _CACHE["provenance"] = {
        "parent_repo": str(root),
        "parent_head_commit": _git(root, "rev-parse", "HEAD").strip(),
        "parent_head_branch": _git(root, "rev-parse", "--abbrev-ref", "HEAD").strip(),
        "parent_worktree_clean": _git(root, "status", "--porcelain").strip() == "",
        "round_map_file": ROUND_MAP_FILE,
        "round_map_revision": rev,
        "round_map_sha256": digest,
        "circuit_file_sha256": files,
        "note": ("the parent repository is only READ: the round-map file is "
                 "extracted with `git show` into a temporary directory, and no "
                 "checkout, branch switch or write is performed"),
    }
    return module


def provenance() -> dict:
    """Provenance of the parent code actually used (loads it if needed)."""
    load_parent_module()
    return dict(_CACHE["provenance"])
