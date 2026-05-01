# Serie BB26 Prompt 7: Two-region guard/contract consistency line

Status: **in progress hardening complete** for the bounded two-region baseline (Region 1 + Region 2 only).

## Canonical two-region consistency map

The repo now pins a canonical two-region consistency map with exactly six classes:

1. `canonical region-1 path`
2. `canonical region-2 path`
3. `bounded inter-region relation path`
4. `caveated two-region path`
5. `blocked/insufficient two-region path`
6. `non-canonical/internal-only two-region path`

This is intentionally narrow and does **not** introduce a general inter-region orchestration platform.

## Cross-region guard line (unchanged authority boundaries)

Across Region 1, Region 2, and their bounded relation, the no-direct boundaries remain hard:

- no direct action trigger,
- no direct execution trigger,
- no direct retry trigger,
- no direct memory commit,
- no direct compute invocation,
- no safety override,
- no third region class.

## Contract/diagnostics consistency (runtime/selection/reference)

Runtime, selection, and reference now read the same bounded semantics across both regions:

- `advisory-only` stays advisory-only,
- `caveated` stays caveated,
- `deferred` stays deferred,
- `blocked` stays blocked,
- `insufficient` stays insufficient,
- `diagnostic-only` stays diagnostic-only,
- `reference-only` stays reference-only,
- `non-canonical/internal-only` stays excluded from operational paths.

## Bounded relation remains bounded

The Region-1↔Region-2 coupling remains a bounded relation only:

- shared reference mediation is informational and non-authoritative,
- no direct region-to-region decision authority,
- no implicit orchestration, retry, planner, memory, or compute platform.

## Scope intentionally not expanded

Out of scope and still blocked in this step:

- third region class,
- broad inter-region platform,
- planner/agent/retry orchestration,
- memory persistence authority,
- compute-core expansion,
- HH production integration.
