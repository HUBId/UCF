# Evolution Engine v0 (Structural Delta Suggestions)

`structural_delta` is operational in v0 as a **suggestion-only** channel.

## Safety model

- No self-modification: deltas are never applied automatically.
- No tool/actions: the evolution engine only computes deterministic candidate deltas and scores.
- Deterministic/offline: no external model calls, no network dependency.
- Bounded execution:
  - max 8 candidates/window
  - max 8 ops/delta
  - bounded score audit lists
  - bounded ESS record payloads
- Auditability: proposal/evaluation/recommendation records are persisted to ESS with evidence-chain linkage.

## Structural delta schema (v1)

Targets:
- `FepWeights`
- `PolicyThresholds`
- `ComputeBudgetHints`
- `CoherenceGating`
- `BiophysGating`

Ops:
- `Set { key, value }`
- `Add { key, delta }`
- `Clamp { key, min, max }`

Keys are fixed enum entries (e.g. policy risk beta, coherence thresholds, structure cap).

## Runtime wiring

The orchestrator can run one evolution step per consolidation window and persist:
- `DeltaProposal`
- `DeltaEvaluation`
- `DeltaRecommendation` (with `requires_human_apply=true`)

Application is explicitly out of scope for v0.

## Enable/inspect

Environment flags:

- `UCF_ENABLE_EVOLUTION=1` (default: disabled)
- `UCF_EVOLUTION_WINDOW_TICKS=64`

Inspect by reading ESS records and filtering:
- `ExperienceKind::DeltaProposal`
- `ExperienceKind::DeltaEvaluation`
- `ExperienceKind::DeltaRecommendation`

## Manual apply workflow (out of scope automation)

1. Review accepted recommendations in ESS.
2. Validate reason codes, penalties, and evidence-chain digest.
3. Export proposal as an ops artifact/patch in a separate controlled step.
4. Apply through human/ops gated config deployment.
