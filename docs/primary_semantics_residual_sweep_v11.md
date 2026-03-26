# Primary Semantics Residual Sweep v11

`primary-semantics-residual-sweep` is the final v11 proof that canonical governance/readiness/bundle/review/export/interop/gate surfaces no longer derive primary blocking/remediation semantics from local precedence hints.

## What the sweep proves

The sweep enforces that canonical primary semantics come from final authoritative inputs only:

1. canonical condition model
2. canonical remediation registry
3. `CanonicalPrimarySemanticsAuthorityV1`
4. `FinalPrimarySemanticsConsumerAuthorityV1`

It emits `FinalPrimarySemanticsResidualSweepV1` containing:

- canonical governance/readiness/bundle digest prefixes
- canonical primary semantics authority digest prefix
- final primary semantics consumer authority digest prefix
- covered surface count
- residual path count
- `PASS | FAIL | LEGACY_PRESENT`
- deterministic sweep digest

## Covered canonical surfaces

- governance entry / scope-authority consumers
- readiness spine consumers
- bundle spine / roundtrip / export normalize consumers
- operator signoff / review packet / workflow consumers
- interop consistency matrix consumers
- canonical gate helper surfaces represented in final primary-semantics sweep

## Why residual reconstruction is disallowed

Canonical surfaces must fail closed when final primary-semantics inputs are missing, stale, or contradictory.
Any residual path (local main-reason ordering, local action-hint ordering, ad-hoc condition→remediation translation) is treated as a blocked residual path, not as canonical authority.

## Command

```bash
cargo run -p ucf-ops -- primary-semantics-residual-sweep --out ./out/primary_semantics_residual_sweep.json
```

## v12 update

v12 (`residual-free-primary-semantics-sweep`) finalizes the canonical consumer cleanup and blocks remaining historical/implicit/local primary-semantics reconstruction in canonical flows.

v13 (`primary-semantics-absolute-sweep`) additionally enforces absolute residual-free final primary-semantics input reuse on all covered canonical surfaces and blocks any surviving historical/local lineage as canonical authority.

## v14 terminal note

v14 (`primary-semantics-terminal-sweep`) completes the terminal consumer sweep so canonical flows cannot use priority echoes, embedded summaries, cache residue, or override paths as primary truth.

