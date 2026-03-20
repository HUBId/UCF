# Operator Export Authority Chain v7

`OperatorExportAuthorityChainV1` proves that operator review packet, signoff decision,
workflow chain, and optional export context are all bound to the same applied scope authority and
reviewability reduction basis.

## Participating artifacts
- `AppliedSupportedSetContextV1`
- `OperatorReviewPacketV1`
- `OperatorSignoffDecisionV1`
- `OperatorWorkflowChainV1`
- optional export manifest context (`repro_pack_manifest.json` / `bugkit_manifest.json`)

## Command
```bash
cargo run -p ucf-ops -- operator export-chain-check --out ./out/operator_export_chain_check.json
```

## Mismatch categories
- `REVIEW_PACKET_SCOPE_MISMATCH`
- `SIGNOFF_SCOPE_MISMATCH`
- `WORKFLOW_SCOPE_MISMATCH`
- `EXPORT_CONTEXT_SCOPE_MISMATCH`
- `REVIEWABILITY_BASIS_MISMATCH`
- `APPLIED_SCOPE_MISSING`

Unlike interop matrix and export roundtrip checks, this command is a direct end-to-end authority
proof for applied-scope alignment across operator and export readiness surfaces.

Export build/readiness now fail-closed when the authority chain is not `PASS`.


See also canonical entrypoint rule: docs/canonical_governance_entry_v8.md

## v8 continuity
See `docs/roundtrip_chain_v8.md` for the top-level operator->export->bundle continuity proof command and mismatch semantics.


## v9 note
Operator/export authority continuity now includes canonical readiness spine/authority references and is checked by `readiness-spine-sweep` for spine-only readiness consumption.

## Final continuity authority (v9)
Use `cargo run -p ucf-ops -- continuity-authority-check --bundle <path> --out ./out/continuity_authority_check.json` as final top-level proof that this surface aligns with canonical governance/readiness/operator/export/bundle continuity.


## v10 Continuity Position

`operator export-chain-check`/Roundtrip-nahe Surfaces liefern weiterhin Diagnostik, sind aber keine konkurrierenden top-level continuity proofs mehr.
Sole top-level continuity proof: `ucf-ops final-continuity-sweep`.
