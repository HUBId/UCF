# Export Bundle Normalization v6

This document defines the canonical export artifact model shared by Repro Pack, BugKit, BackendEvidenceSnapshotV1, AggregatedActiveReviewSnapshotV1, OperatorReviewPacketV1, and OperatorSignoffDecisionV1.

Consumption/verification round-trip semantics are defined in `docs/export_roundtrip_v7.md` and enforced by `ucf-ops exports roundtrip-check`.

## CanonicalExportArtifactRefV1

Fields:
- `artifact_kind`
- `relative_path`
- `included_state` (`INCLUDED`, `MISSING`, `EXCLUDED`, `SKIP`)
- `sha256` (optional unless included)
- `schema_version` (optional)
- `artifact_digest` prefix (optional)
- `reason_code` (optional)
- `ref_digest`

Semantics:
- `INCLUDED`: Artifact is present in the bundle and hash-verifiable.
- `MISSING`: Optional artifact was expected but not present.
- `EXCLUDED`: Artifact intentionally excluded by policy/scope/context mismatch.
- `SKIP`: Artifact was intentionally skipped and is neither missing nor excluded.

## CanonicalExportContextV1

Fields:
- `supported_slot_set_digest_prefix`
- `policy_graph_digest_prefix`
- `manifest_digest_prefix`
- `run_id` (optional)
- `operator_signoff_digest_prefix` (optional)
- `backend_evidence_snapshot_digest_prefix` (optional)
- `active_review_snapshot_digest_prefix` (optional)
- `context_digest`

## Normalized path conventions

Canonical bundle-relative paths:
- `artifacts/backend_evidence_snapshot.json`
- `artifacts/active_review_snapshot.json`
- `artifacts/operator_signoff.json`
- `artifacts/operator_review_packet.json`
- `artifacts/backend_resolution.json`

Legacy `evidence/*` paths are compatibility-accepted only via explicit compatibility status.

## Command

Run normalization governance check:

```bash
cargo run -p ucf-ops -- exports normalize-check --out ./out/export_normalize_check.json
```

Mismatch categories are bounded:
- `PATH_NAMING_DRIFT`
- `CONTEXT_FIELD_DRIFT`
- `INCLUDED_STATE_DRIFT`
- `DIGEST_FIELD_DRIFT`
- `LEGACY_EXPORT_LAYOUT`


## Cross-surface interop proof (v6)

Export normalization is additionally verified as part of the cross-surface interop matrix:

```bash
cargo run -p ucf-ops -- interop consistency-matrix --out ./out/interop_consistency_matrix.json
```

This enforces shared context/scope/digest/reference compatibility across gate, strict, snapshot, operator, and export surfaces.


## Operator workflow chain integration

`exports normalize-check` ist ein verpflichtender Input in der top-level Workflow-Orchestrierung:

```bash
cargo run -p ucf-ops -- operator workflow --out ./out/operator_workflow_chain.json
```

Wenn Normalisierung fehlschlägt, ist die Workflow-Stage fail-closed (`WORKFLOW_BLOCKED`) und Export-Readiness bleibt `false`. Siehe `docs/operator_workflow_chain_v6.md`.

## Nachgelagerte End-to-End-Prüfung (v8)

Die Normalisierung bleibt Grundlage; die Bundle-Endprüfung erfolgt mit:

```bash
cargo run -p ucf-ops -- exports bundle-spine-check --in <bundle.zip> --out ./out/bundle_spine_check.json
```



## v9 update
Canonical export normalization now assumes bundle authority is finalized through `CanonicalBundleSpineV1` + `CanonicalBundleAuthorityV2` (`exports bundle-spine-sweep`).


## v10 finalization

v10 finalizes universal bundle-input authority for canonical export consumers via `ucf-ops final-bundle-consumer-sweep`.
