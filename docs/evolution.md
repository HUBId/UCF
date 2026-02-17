# Evolution Engine v0 (Structural Delta Suggestions)

`structural_delta` is operational in v0 as a **suggestion-only** channel.

## Safety model

- No self-modification: deltas are never applied automatically.
- No tool/actions: the evolution engine only computes deterministic candidate deltas and scores.
- Deterministic/offline: no external model calls, no network dependency.
- Bounded execution:
  - max 4 candidates/window
  - max 2 ops/delta
  - per-key magnitude clamp: `|delta| <= 0.05`
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

## Liquid-window context

Each evolution window now carries deterministic, quantized `LiquidWindowStats`:
- window: `[t0, t1]`
- means/maxima for uncertainty, pressure, surprise, risk
- optional coherence mean
- trend terms:
  - `delta_mean_uncertainty`
  - `delta_mean_pressure`
- `window_stats_digest` for replay-safe proposal/evaluation linkage

`EvolutionContext` also includes:
- `governor_tier_mean`, `governor_tier_max`
- `emergency_active`
- `backend_pack_digest`
- `nsr_available`

## Proposal rules (deterministic heuristic)

Examples of generated **safe suggestions**:
- high uncertainty + high risk:
  - increase `beta_policy_risk`
  - tighten coherence-risk inhibit thresholds
- high pressure + low risk:
  - tighten memory/structure budget hints
- high surprise + low confidence:
  - damp coupling/drift-sensitive knobs

Loosening is intentionally constrained to very stable low-risk windows and tiny steps.

## Suppression / safety gates

Before proposals are emitted:
- emergency active ⇒ engine suppressed (no proposals)
- NSR unavailable ⇒ conservative mode only (safety-tightening)
- governor tier max >= 3 ⇒ only safety-tightening allowed

Suppression is tracked via telemetry and persisted bounded notes:
- `evolution_engine_suppressed:emergency`
- `evolution_engine_suppressed:nsr_unavailable`

## Runtime wiring

The orchestrator can run one evolution step per consolidation window and persist:
- `DeltaProposal`
- `DeltaEvaluation`
- `DeltaRecommendation` (with `requires_human_apply=true`)

Application is explicitly out of scope for v0.

## Telemetry

- `ucf_structural_delta_proposals_total`
- `ucf_structural_delta_suppressed_total{reason=...}`
- `ucf_structural_delta_recommended_total`
- `ucf_structural_delta_rejected_total{reason=...}`

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
