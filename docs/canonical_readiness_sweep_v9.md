# Canonical Readiness Sweep v9

`readiness-spine-sweep` is the final consolidation check that canonical operator/review/export/gate surfaces derive readiness/reviewability from `CanonicalReadinessSpineV1`.

## What it proves
- Canonical readiness authority is represented by `CanonicalReadinessAuthorityV2`.
- Covered surfaces are checked for spine usage and scope/governance alignment.
- Secondary readiness derivations are classified and blocked from canonical authority.

## Covered surfaces
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `CanonicalReadinessSpine`

## Why secondary derivation is disallowed
Canonical flows must not produce parallel readiness truths. Any path that skips canonical spine authority is reported as mismatch and fails closed.

## Command
```bash
cargo run -p ucf-ops -- readiness-spine-sweep --out ./out/readiness_spine_sweep.json
```

For final canonical primary blocking/remediation authority across all covered surfaces, run `ucf-ops primary-semantics-sweep` (see `docs/primary_semantics_sweep_v9.md`).
