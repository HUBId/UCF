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
