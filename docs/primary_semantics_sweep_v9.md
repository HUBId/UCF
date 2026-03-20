# Primary Semantics Sweep v9

`ucf-ops primary-semantics-sweep` is the final v9 proof surface for canonical **primary blocking/remediation semantics** across governance, readiness, bundle, review, export, interop, and gate families.

## Command

```bash
cargo run -p ucf-ops -- primary-semantics-sweep --out ./out/primary_semantics_sweep.json
```

## What it proves

- The same canonical condition yields the same primary blocking code and primary remediation code across the covered canonical surfaces.
- Surface-local reason/action details are carried only as secondary diagnostics (`secondary_diagnostic_codes`, `secondary_surface_reason_codes`).
- Missing/unsupported surfaces are explicit (`MISSING`/`SKIP`) and never silent.
- The compact authority summary is emitted as `CanonicalPrimarySemanticsAuthorityV1`.

## Covered surfaces

- `AppliedScopeAuthority`
- `CanonicalGovernanceEntry`
- `CanonicalReadinessSpine`
- `CanonicalBundleSpine`
- `BundleSpineCheck`
- `ExportRoundTrip`
- `ExportNormalizeCheck`
- `InteropMatrix`
- `OperatorExportAuthorityChain`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflow`
- `GateV4` / `GateV5` / `GateV6` / `GateV7` / `GateV8`

## Mismatch categories

- `PRIMARY_BLOCKING_MISMATCH`
- `PRIMARY_REMEDIATION_MISMATCH`
- `CANONICAL_CONDITION_MISMATCH`
- `LEGACY_PRIMARY_SEMANTICS_PRESENT`
- `REQUIRED_SURFACE_MISSING`


## v10 finalization

Universal canonical consumer enforcement is finalized in `docs/final_primary_semantics_sweep_v10.md` via `ucf-ops final-primary-semantics-sweep`.
