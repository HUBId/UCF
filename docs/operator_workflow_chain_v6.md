# Operator Workflow Chain v6

`OperatorWorkflowChainV1` ist die deterministische, read-only Top-Level-Orchestrierung für den Operator-Review/Export-Flow.

Es ersetzt **nicht** bestehende Artefakte, sondern verbindet sie als kanonische Workflow-Spine:

1. Governance Surfaces validieren
2. Applied Supported Scope prüfen
3. Review Packet + Signoff prüfen
4. Export-Readiness (repro/bugkit) deterministisch ausweisen

## Command

```bash
cargo run -p ucf-ops -- operator workflow --out ./out/operator_workflow_chain.json
```

Optional:

- `--run <id>`
- `--latest`
- `--text`

## Felder (`OperatorWorkflowChainV1`)

- `schema_version`
- `workflow_stage` (`OperatorWorkflowStageV2`)
- `governance_surfaces_digest_prefix`
- `applied_supported_scope_digest_prefix`
- `operator_review_packet_digest_prefix`
- `operator_signoff_digest_prefix`
- `interop_matrix_digest_prefix`
- `export_normalize_check_digest_prefix`
- `export_targets.repro_ready`
- `export_targets.bugkit_ready`
- `blocking_codes` (bounded)
- `remediation_codes` (bounded)
- `chain_digest`

## Stage-Interpretation

- `WORKFLOW_BLOCKED`
  - Pflichtvoraussetzungen fehlen oder sind inkonsistent:
    - Governance surfaces ungültig/fehlend
    - Applied-scope mismatch
    - Review/Signoff blockiert
    - Interop fail
    - Export normalize-check fail
  - Fail-closed.

- `WORKFLOW_REVIEW_READY`
  - Review Packet + Signoff sind kohärent.
  - Export-Kette ist noch nicht vollständig ready (z. B. repro/bugkit Manifest fehlt oder repro verify nicht erwartbar PASS).

- `WORKFLOW_EXPORT_READY`
  - Review kohärent und nicht blockiert.
  - Export-Kontext normalisiert.
  - repro/bugkit readiness ist deterministisch ableitbar (`repro_ready=true`, `bugkit_ready=true`).

## Abgrenzung zu Review Packet / Signoff

- `OperatorReviewPacketV1`: fokussierte Review-Reduktion.
- `OperatorSignoffDecisionV1`: Signoff-Entscheidung.
- `OperatorWorkflowChainV1`: Top-Level-Reihenfolge + Gesamtstatus über Governance, Scope, Review, Signoff, Interop und Export-Normalisierung.

## Invarianten

- Read-only (keine Runtime-/Config-Mutation).
- Offline-first.
- Deterministisch, bounded Codes.
- Reuse bestehender kanonischer Checks/Artefakte statt ad-hoc Recompute von Business-Logik.

See also: `docs/operator_export_authority_chain_v7.md` and `ucf-ops operator export-chain-check` for applied-scope authority validation across review/signoff/workflow/export chain.


See also canonical entrypoint rule: docs/canonical_governance_entry_v8.md

See also: readiness spine canon (`docs/readiness_spine_v8.md`).

## v8 continuity
`OperatorWorkflowChainV1` is now consumed by `CanonicalRoundTripChainV1` as part of end-to-end operator->bundle continuity proof.


## v9 note
`OperatorWorkflowChainV1` now carries canonical readiness authority references and is expected to align with spine-only readiness consumption validated by `readiness-spine-sweep`.

## Final continuity authority (v9)
Use `cargo run -p ucf-ops -- continuity-authority-check --bundle <path> --out ./out/continuity_authority_check.json` as final top-level proof that this surface aligns with canonical governance/readiness/operator/export/bundle continuity.


## v10 Continuity Position

`OperatorWorkflowChainV1` ist weiterhin deterministische Workflow-Stufe, aber **nicht** mehr top-level continuity proof.
Top-level continuity authority wird ausschließlich über `final-continuity-sweep` festgestellt.


## v11 residual-free alignment

`OperatorWorkflowChainV1` remains a workflow-stage artifact and references residual sweep lineage.
Top-level continuity truth is delegated exclusively to `ResidualFreeContinuityAuthorityV1`.
