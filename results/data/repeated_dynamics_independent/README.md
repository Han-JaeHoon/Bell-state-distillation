# Independent repeated-dynamics results — FROZEN

These files are the output of an analysis that started from the verified
one-round exact maps (`src/pqec_distill/analytic_exact_map.py`) and derived the
repeated dynamics from scratch.  They exist so that a separately held analytic
calculation can be compared against them.

**Do not regenerate these files.**  Once the comparison starts, regenerating
them would destroy the only thing they are for.  If a script here needs to
change, write the new output to a different directory.

## Generating scripts

| script | writes |
|---|---|
| `scripts/broad_scan.py` | `broad_scan.csv` |
| `scripts/analyze_repeated_dynamics.py` | `fixed_points.csv`, `critical_points.json`, `representative_trajectories.csv`, `metadata.json` |
| `scripts/analyze_basins_precision.py` | `basin_scan.csv`, `basin_boundaries.csv`, `precision_check.csv`, `endpoint_verification.csv`, `basins_precision_metadata.json` |
| `scripts/check_separatrix.py` | `separatrix_check.csv` |
| `scripts/transient_scaling.py` | `transient_scaling.csv`, `transient_scaling_fit.json` |
| `scripts/verify_branch_eigenvalues.py` | `offplane_eigenvalues.csv` |

The derivation behind them is written up in
`docs/repeated_dynamics_independent.md`; the guard tests are in
`tests/test_repeated_dynamics_independent.py`.

## Excluded on purpose

The exclusion list is in `metadata.json`.  None of the modules, scripts,
result files or document sections listed there was read while producing these
results; only the one-round map was reused.

`metadata.json` also carries an explicit disclosure: earlier numbers for these
circuits were present in the analyst's working context, so this is an
independent *derivation path*, not a blind replication.
