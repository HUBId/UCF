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


## Supported-scope expansion in v7

Scope expansion is permitted only through reevaluation under applied-scope authority:

```bash
cargo run -p ucf-ops -- models supported-scope-reevaluate --out ./out/supported_scope_reeval.json
```

If reevaluation returns `REAFFIRM_FREEZE`, scope remains unchanged explicitly.
If reevaluation returns `EXECUTE_EXPAND_BY_ONE`, apply may add exactly one slot via `supported-set-apply`.


See also canonical entrypoint rule: docs/canonical_governance_entry_v8.md
