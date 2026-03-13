# Applied Supported Scope v6

`AppliedSupportedSetContextV1` is the authoritative, bounded slot scope context for governance review surfaces.

## Scope vocabulary
- **Reviewed policy scope**: `SupportedRealSlotSetPolicyV2` from `models supported-set-review`.
- **Applied supported set**: `SupportedRealSlotSetV2` from `models supported-set-apply`.
- **Applied review/signoff scope**: `AppliedSupportedSetContextV1` derived from the applied set and consumed by:
  - `AggregatedActiveReviewSnapshotV1`
  - `OperatorSignoffDecisionV1`
  - `OperatorReviewPacketV1`
  - `GovernancePrimarySurfacesV1` validation

Only the applied context is allowed to define in-scope slots for these surfaces.

## Deterministic behavior
- Extra evidence outside applied slots is deterministically blocked/fail-closed.
- Missing in-scope evidence is explicit and never silently ignored.
- Legacy pre-v2 scope artifacts are handled explicitly via:
  - `LEGACY_SCOPE_ARTIFACT`
  - `LEGACY_SCOPE_TRANSLATED`
  - `LEGACY_SCOPE_REJECTED`

## Command
```bash
cargo run -p ucf-ops -- models applied-scope-check --out ./out/applied_scope_check.json
```

Mismatch categories are bounded and emitted in `mismatch_categories` with remediation codes.
