# Remediation Consistency v5

`ucf-ops remediation-consistency-check` proves that a covered canonical condition resolves to the same **primary canonical remediation code** across governance/reporting surfaces.

## What is proven

- One canonical condition ⇒ one canonical primary remediation code.
- Comparison is deterministic and offline.
- Missing surface mappings are explicit (`MISSING`) and unsupported surfaces are explicit (`SKIP`).
- Legacy translation drift is classified as `FAIL` with mismatch kind.

## Covered surfaces

- strict check
- eligibility report
- operator report
- operator signoff
- v4 gate family
- enriched export manifest reason/remediation normalization (where meaningful)

## Status semantics

- `PASS`: all participating surfaces agree on the primary canonical remediation code.
- `FAIL`: at least one participating surface emits a different primary code.
- `MISSING`: at least one required surface has no mapping for that condition.
- `SKIP`: condition is not representable on that surface and is explicitly skipped.

## Command

```bash
cargo run -p ucf-ops -- remediation-consistency-check --out ./out/remediation_consistency.json
```

The report includes per-condition check rows plus an overall summary with mismatch categories and remediation suggestions.


## v7 strengthened proof

For cross-surface review/gate/export/interop harmonization, run:

```bash
cargo run -p ucf-ops -- remediation-interop-check --out ./out/remediation_interop_check.json
```

See `docs/remediation_interop_consistency_v7.md` for the expanded surface set and mismatch normalization rules.
