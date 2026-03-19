# Remediation & Interop Consistency v7

`ucf-ops remediation-interop-check` is the strengthened cross-surface proof that the **same canonical condition** maps to the **same primary canonical remediation code** across review, gate, export, and interop surfaces.

## Command

```bash
cargo run -p ucf-ops -- remediation-interop-check --out ./out/remediation_interop_check.json
```

## Covered surfaces

- Strict
- Eligibility
- ActiveReviewSnapshot
- OperatorReport
- OperatorSignoff
- OperatorReviewPacket
- GateV3 / GateV4 / GateV5 / GateV6 / GateV7
- ExportNormalizeCheck
- ExportRoundTripCheck
- InteropMatrix

## Covered canonical condition family

The bounded condition set includes:

- `ScopeMismatch`
- `PolicyMismatch`
- `ManifestMismatch`
- `HashMismatch`
- `EvidenceMissingProbe`
- `EvidenceMissingCompare`
- `EvidenceStaleCompare`
- `DriftSevere`
- `StrictFail`
- `OptionalBackendClosedUnsupported`
- `ExportLayoutMismatch`
- `ExportRoundTripMismatch`
- `InteropMatrixMismatch`
- `AppliedScopeMismatch`

## Export/Interop normalization

Surface-specific mismatch categories are mapped back into the canonical condition family, and then resolved through the canonical remediation registry.

- Export normalize mismatch categories map to canonical condition codes.
- Export roundtrip mismatch codes map to canonical condition codes.
- Interop mismatch categories map to canonical condition codes.

Unknown mappings are fail-closed as `UNKNOWN_CONDITION_MAPPING`; unsupported/missing surfaces are explicit `SKIP`/`MISSING` (never silent).

## What v7 detects beyond v5

Compared to `remediation-consistency-check`, v7 additionally detects drift between:

- export normalize/roundtrip mismatch surfaces and canonical remediation semantics
- interop matrix mismatch categories and canonical remediation semantics
- extended gate/review/packet/snapshot surface families

All checks remain offline, deterministic, bounded, and read-only.

See `docs/remediation_spine_consistency_v8.md` for the stronger v8 spine-level cross-surface proof and `docs/primary_semantics_sweep_v9.md` for the final v9 canonical primary semantics authority.
