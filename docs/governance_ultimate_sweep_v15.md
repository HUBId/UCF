# Governance Ultimate Sweep v15

`governance-ultimate-sweep` provides the terminal proof that canonical governance consumers derive authority only from the final residual-free governance input chain.

## What this proves

`TerminalGovernanceUltimateSweepV1` binds and verifies:

1. `AppliedSupportedSetContextV1`
2. `CanonicalGovernanceEntryV1`
3. `CanonicalGovernanceEntryAuthorityV2`
4. `FinalGovernanceConsumerAuthorityV1`
5. `FinalGovernanceResidualSweepV1`
6. `ResidualFreeGovernanceConsumerAuthorityV1`
7. `ResidualFreeGovernanceAbsoluteSweepV1`
8. `AbsoluteFinalGovernanceTerminalSweepV1`

The sweep records `covered_consumer_count`, `residual_path_count`, `sweep_status`, and `sweep_digest`.

## Covered canonical consumers

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `V15PrepGateHelper` (`v14_gate_report`)

## Why caches/mirrors/snapshots are blocked

Canonical consumers must carry the full terminal governance digest chain and the new `governance_ultimate_sweep_digest_prefix`.
If any consumer tries to rely on governance cache/mirror/snapshot residue as primary truth, the sweep detects it and fails closed (`FAIL` / `LEGACY_PRESENT`).

## Command

```bash
cargo run -p ucf-ops -- governance-ultimate-sweep --out ./out/governance_ultimate_sweep.json
```
