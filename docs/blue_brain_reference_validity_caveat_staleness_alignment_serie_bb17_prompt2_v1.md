# BlueBrain Reference Validity / Caveat / Staleness Alignment — Serie BB17 Prompt 2 (v1)

Status: runtime-selection-dynamics-execution alignment on one canonical reference-validity basis.

## Canonical reference-validity states

All BB17 reference consumers now use the same canonical states:

- `current`
- `caveated`
- `stale`
- `invalidated`
- `blocked`
- `insufficient`
- `reference_only`
- `non_canonical_internal_only_path`

This is a bounded semantics alignment, not a global lifecycle/truth platform.

## Memory and execution mapping on one line

- Memory:
  - `RetrievedReferenceOnly` -> `current`
  - `RetrievedWithCaveat` -> `caveated`
  - `RetrievedStale` -> `stale`
  - `RetrievedInvalidated` -> `invalidated`
  - `Blocked` -> `blocked`
  - `Missing` / `Unavailable` -> `insufficient`
- Execution references:
  - successful completed result -> `current` (or `caveated` when explicit caveat markers exist)
  - failed / cancelled -> `caveated` (not strong current basis)
  - blocked -> `blocked`
  - unavailable / unsupported / placeholder-only / not-observed -> `insufficient`

`reference_only` remains explicit for diagnostic/reference-only lanes and is never treated as strong current basis.

## Runtime / Selection / Dynamics / Execution alignment

- Runtime evidence ingestion classifies reference kind + canonical validity before execution-feedback bucketing.
- Selection and dynamics consume the same bounded evidence buckets (`current` strong basis vs caveated/stale/blocked/insufficient weakened basis).
- Combined retrieval now exposes one canonical validity state in addition to memory and execution basis diagnostics.

## Boundary clarity (no implicit collapse)

- `caveated != stale`
- `stale != invalidated`
- `blocked != invalidated`
- `insufficient != blocked`
- `reference_only != current`

## no-direct-* boundaries (unchanged)

- no direct compute invocation
- no direct action execution
- no implicit memory persistence
- no policy/authority escalation
- no non-canonical path normalization into canonical authority
