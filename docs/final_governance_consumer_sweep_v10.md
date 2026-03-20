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
