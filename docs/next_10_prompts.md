# Next 10 Prompts (v9 Anchor)

Anchor: `Real Compute Onboarding v9` (precondition: `ucf-ops v8 gate` PASS at Prompt 258)

> Guardrail: This queue is capped to **10** entries unless an explicit request expands it.

## Prompt 260 (MUST) — Canonical governance entry and supported-set context unification
Objective: Unify every remaining canonical consumer around `CanonicalGovernanceEntry` plus `AppliedSupportedSetContext` as the sole governance entry path.
- Acceptance:
  - All active governance/review/export/gate consumers read governance authority from `CanonicalGovernanceEntry` plus `AppliedSupportedSetContext`.
  - Legacy or parallel governance-entry reads are removed or fail closed with deterministic diagnostics.
  - Behavior remains hardware-neutral, offline-first, and runtime-semantics preserving.
- Dependencies: 258, 259.

## Prompt 261 (MUST) — Supported-scope reevaluation-controlled expansion or reaffirmed freeze
Objective: Re-execute supported-scope expansion only if current execution plus canonical governance state still justify it; otherwise reaffirm freeze.
- Acceptance:
  - Scope expansion requires explicit canonical governance authority, current supported-scope execution support, and evidence continuity.
  - If justification is insufficient, supported scope remains frozen with explicit fail-closed reporting.
  - Probe-first and shadow-first progression remains explicit on all affected paths.
- Dependencies: 260.

## Prompt 262 (MUST) — Canonical readiness-spine consumption deepening
Objective: Deepen canonical readiness-spine consumption across all remaining operator/gate/export consumers.
- Acceptance:
  - All remaining consumers use one canonical readiness spine contract and ordering.
  - Missing or stale readiness prerequisites fail closed with bounded, deterministic diagnostics.
  - No runtime capability widening is introduced.
- Dependencies: 260, 261.

## Prompt 263 (MUST) — Canonical bundle spine build/verify/inspect normalization
Objective: Normalize bundle build/verify/inspect semantics completely around `CanonicalBundleSpine` and round-trip continuity.
- Acceptance:
  - Build/verify/inspect paths consume one canonical bundle spine contract.
  - Operator→export→bundle round-trip checks remain deterministic and offline-reproducible.
  - Governance/readiness evidence is reused from canonical surfaces instead of re-derived when available.
- Dependencies: 260, 262.

## Prompt 264 (MUST) — Canonical remediation and interop continuity hardening
Objective: Harden canonical remediation/interop semantics across governance entry, readiness spine, bundle spine, and gate family.
- Acceptance:
  - Equivalent evidence states produce consistent remediation/status outcomes across all covered surfaces.
  - Divergence is detected deterministically and handled fail closed.
  - Interop semantics remain bounded, hardware-neutral, and evidence-bound.
- Dependencies: 262, 263.

## Prompt 265 (MUST) — v9 schema snapshot refresh for governance/scope/readiness/bundle/workflow artifacts
Objective: Refresh schema snapshots for v9 governance/scope/readiness/bundle/workflow artifacts after normalization and hardening.
- Acceptance:
  - Updated snapshots reflect current canonical contracts with deterministic ordering.
  - Versioning and compatibility notes are documented for changed artifacts.
  - Snapshot outputs remain reproducible offline.
- Dependencies: 263, 264.

## Prompt 266 (NICE) — Portability and operator-doc refresh for v9 surfaces
Objective: Refresh portability and operator-facing documentation for v9 governance/scope/readiness/bundle surfaces.
- Acceptance:
  - Operator docs reflect conservative v9 path expectations and evidence requirements.
  - Portability guidance remains hardware-neutral and offline-first.
  - Documentation updates avoid behavior-changing policy semantics.
- Dependencies: 265.

## Prompt 267 (MUST) — Operator workflow and export-chain continuity hardening
Objective: Harden operator workflow/export chain behavior with stronger canonical continuity requirements.
- Acceptance:
  - Workflow/export continuity checks enforce canonical invariants across active v9 surfaces.
  - Failures emit bounded, actionable, deterministic diagnostics.
  - Probe-first/shadow-first/fail-closed guarantees remain explicit and preserved.
- Dependencies: 264, 265, 266.

## Prompt 268 (MUST) — v9 gate schema and orchestration
Objective: Define and wire v9 gate schema and orchestration for consolidated governance/scope/readiness/bundle continuity assurances.
- Acceptance:
  - v9 gate verifies required v9 surfaces with deterministic ordering and normalized status semantics.
  - Required-surface mismatches fail closed with explicit evidence references.
  - Gate semantics remain conservative and do not widen runtime behavior.
- Dependencies: 267.

## Prompt 269 (MUST) — v9 wrap and next-anchor governance
Objective: Close v9 planning loop and prepare next-anchor governance artifacts after v9 gate readiness.
- Acceptance:
  - v9 wrap artifacts record closure state and next-anchor handoff points.
  - Next prompt queue remains capped to 10 entries unless explicitly expanded.
  - Wrap output remains hardware-neutral, offline-first, probe-first, shadow-first, fail-closed, and evidence-bound.
- Dependencies: 268.
