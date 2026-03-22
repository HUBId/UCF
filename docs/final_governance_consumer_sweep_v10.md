# Final Governance Consumer Sweep v10

`ucf-ops final-governance-consumer-sweep` is the v10 proof that canonical consumers share one final governance authority chain:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`

The sweep emits:
- `FinalGovernanceConsumerAuthorityV1`
- per-consumer statuses and mismatch categories

Covered canonical consumers:
- ActiveReviewSnapshot
- OperatorSignoff
- OperatorReviewPacket
- OperatorWorkflowChain
- InteropConsistencyMatrix

Legacy governance inputs are not allowed as primary truth in canonical flows. Remaining legacy paths are surfaced as deterministic fail-closed mismatches.

`models supported-scope-execute-v5` consumes `FinalGovernanceConsumerAuthorityV1` directly; expansion must reaffirm freeze unless this authority is `PASS` under current applied scope.

## Command

```bash
cargo run -p ucf-ops -- final-governance-consumer-sweep --out ./out/final_governance_consumer_sweep.json
```

v11 extends this with `governance-residual-sweep`, which removes/blocks remaining residual governance reconstruction paths from canonical consumers.

## v12 update

Canonical consumer flows now additionally embed `residual_free_governance_authority_digest_prefix`, proving residual-free governance authority lineage in v12.

## v13 follow-up

v13 completes the absolute last consumer sweep: canonical consumers now embed/reference residual-free governance authority plus governance absolute sweep digest evidence.

