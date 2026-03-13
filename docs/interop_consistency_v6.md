# Interop Consistency v6

`CrossSurfaceContextMatrixV1` ist die kanonische, deterministische Cross-Surface-Konsistenzmatrix für Governance- und Export-Surfaces.

## Beteiligte Surfaces

- `V5Gate`
- `StrictEvidence`
- `BackendEvidenceSnapshot`
- `ActiveReviewSnapshot`
- `OperatorReport`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `ReproPackManifest`
- `BugKitManifest`

## Was die Matrix prüft

`CrossSurfaceMatchRulesV1` bewertet zentral (keine verstreuten Einzelregeln):

1. Scope-Match gegen `AppliedSupportedSetContextV1`
2. Policy-Digest-Match
3. Manifest-Digest-Match
4. Snapshot-Referenz-Match (Backend/Active/Signoff/Review)
5. Remediation-/Blocking-Konsistenz (primäre Codes)
6. Export-Ref-Konsistenz (kanonische Export-Refs)
7. Legacy-/Missing-Surface-Sichtbarkeit

## Mismatch-Kategorien

- `SCOPE_MISMATCH`
- `POLICY_MISMATCH`
- `MANIFEST_MISMATCH`
- `SNAPSHOT_REFERENCE_MISMATCH`
- `REMEDIATION_MISMATCH`
- `EXPORT_REF_MISMATCH`
- `LEGACY_SURFACE_PRESENT`
- `REQUIRED_SURFACE_MISSING`

## Legacy-Behandlung

Legacy-Layouts werden explizit als Legacy markiert (inkl. `LEGACY_SURFACE_PRESENT`, `LEGACY_SURFACE_TRANSLATED`, `LEGACY_SURFACE_UNSUPPORTED`) und nie stillschweigend als modern interpretiert.

## Command

```bash
cargo run -p ucf-ops -- interop consistency-matrix --out ./out/interop_consistency_matrix.json
```

Der Command ist read-only und liefert die primäre v6-Interop-Proof-Oberfläche.
