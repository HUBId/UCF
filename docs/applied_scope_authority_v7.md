# Applied Scope Authority v7

Applied scope authority means canonical governance/export/review/gate surfaces must use the already-applied `AppliedSupportedSetContextV1` scope as the single source of truth.

## Covered surfaces

- backend evidence snapshot
- active review snapshot
- operator review packet
- operator signoff
- interop consistency matrix
- v6/v7 governance gate checks

Legacy/ad-hoc scope inference is unsafe because it can silently widen or narrow slot scope and create drift across governance artifacts.

## Command

```bash
cargo run -p ucf-ops -- scope authority-check --out ./out/scope_authority_check.json
```

The check is deterministic and fail-closed on missing/legacy applied-scope context.
