# Final Primary Semantics Sweep (v10)

`ucf-ops final-primary-semantics-sweep` is the v10 proof that canonical primary blocking/remediation semantics are consumed from the same authority inputs across canonical governance/readiness/bundle/review/export/interop/gate surfaces.

## Command

```bash
cargo run -p ucf-ops -- final-primary-semantics-sweep --out ./out/final_primary_semantics_sweep.json
```

## What it proves

- Canonical primary semantics are fail-closed on:
  - canonical condition model
  - canonical remediation registry
  - `CanonicalPrimarySemanticsAuthorityV1`
- A compact consumer proof is emitted as `FinalPrimarySemanticsConsumerAuthorityV1`.
- Covered canonical surfaces are checked for primary-semantics mismatches and legacy primary input usage.

## Covered canonical surfaces

- governance entry / scope authority lineage
- readiness spine lineage
- bundle spine lineage
- operator signoff / review packet / workflow
- export normalize / export roundtrip
- interop consistency matrix
- gate family lineage (via canonical gate checks)

## Why local primary semantics are no longer allowed

Local top-level reason ordering, surface-specific action-hint precedence, and ad-hoc legacy mappings can diverge and produce inconsistent primary language. v10 enforces one canonical primary blocking/remediation authority for canonical flows and demotes local inputs to secondary diagnostics only.
