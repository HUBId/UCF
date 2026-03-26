# Readiness Absolute Sweep v13

## Purpose

`readiness-absolute-sweep` provides a deterministic proof artifact (`ResidualFreeReadinessAbsoluteSweepV1`) that canonical readiness consumers no longer reconstruct readiness from historical stage lineage, aggregate snapshot memory, embedded hints, or raw evidence-first shortcuts.

The sweep is authoritative only when these inputs align and pass:

- `SlotReviewabilityTruthV1`
- `ReviewabilityReductionV1`
- `CanonicalReadinessSpineV1`
- `CanonicalReadinessAuthorityV2`
- `FinalReadinessConsumerAuthorityV1`
- `FinalReadinessResidualSweepV1`
- `ResidualFreeReadinessConsumerAuthorityV1`

## Covered canonical consumers

- `ActiveReviewSnapshot`
- `OperatorSignoff`
- `OperatorReviewPacket`
- `OperatorWorkflowChain`
- `InteropConsistencyMatrix`
- `V13PrepGateHelper` (`v12_gate_report`)

## Why lineage/aggregate reconstruction is blocked

Canonical flows must fail closed when residual-free final readiness inputs are missing, stale, or contradictory. Historical/lineage/aggregate readiness reconstruction is therefore treated as non-canonical and reported via mismatch categories:

- `CONSUMER_SKIPPED_ABSOLUTE_READINESS_INPUTS`
- `CONSUMER_USED_HISTORICAL_READINESS_LINEAGE`
- `READINESS_INPUT_SCOPE_MISMATCH`
- `READINESS_INPUT_SPINE_MISMATCH`
- `HISTORICAL_READINESS_LINEAGE_PRESENT`

## Command

```bash
cargo run -p ucf-ops -- readiness-absolute-sweep --out ./out/readiness_absolute_sweep.json
```

## v14 follow-up

v14 introduces `readiness-terminal-sweep` to remove terminal consumer dependence on readiness echoes/summaries/aggregate-memory traces in canonical flows.

