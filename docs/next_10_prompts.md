# Next 10 Prompts (v11 Anchor)

Anchor: `Real Compute Onboarding v11` (precondition: `ucf-ops v10 gate` PASS at Prompt 278)

> Guardrail: This queue is capped to **10** entries unless an explicit request expands it.

## Prompt 280 (MUST) — Final governance-consumer authority unification completion
Objective: Unify every remaining canonical consumer around final governance-consumer authority and applied-scope authority as sole governance inputs.
- Acceptance:
  - All active governance/review/export/gate consumers read only final governance-consumer authority and applied-scope authority from canonical sources.
  - Any residual legacy/parallel governance input path is removed or fails closed with deterministic diagnostics.
  - Behavior remains hardware-neutral, offline-first, and runtime-semantics preserving.
- Dependencies: 278, 279.

## Prompt 281 (MUST) — Supported-scope expansion re-execution or freeze reaffirmation
Objective: Re-execute supported-scope expansion only if final consumer authorities plus current scope execution evidence justify it; otherwise reaffirm freeze.
- Acceptance:
  - Expansion requires final consumer-authority alignment, current scope-execution coherence, and explicit evidence.
  - If evidence is insufficient or inconsistent, scope remains frozen and reports fail closed deterministically.
  - Probe-first and shadow-first progression remains explicit and unchanged.
- Dependencies: 280.

## Prompt 282 (MUST) — Final readiness-consumer authority deepening
Objective: Deepen final readiness-consumer authority consumption across all remaining operator/export/gate consumers.
- Acceptance:
  - Remaining consumers adopt one canonical final readiness-consumer authority contract.
  - Missing, stale, or conflicting readiness prerequisites fail closed with bounded deterministic diagnostics.
  - No runtime capability widening is introduced.
- Dependencies: 280, 281.

## Prompt 283 (MUST) — Final bundle authority and sole continuity-proof normalization
Objective: Normalize bundle build/verify/inspect semantics around final bundle-consumer authority and sole top-level continuity proof.
- Acceptance:
  - Build/verify/inspect flows consume only final bundle-consumer authority and the sole top-level continuity proof.
  - Governance/readiness evidence reuse is canonical and deterministic across bundle surfaces.
  - Operator/export/bundle round-trip remains offline reproducible.
- Dependencies: 280, 282.

## Prompt 284 (MUST) — Final primary blocking/remediation hardening
Objective: Harden final primary blocking/remediation semantics across governance/readiness/bundle/continuity/gate consumers.
- Acceptance:
  - Equivalent evidence states produce consistent blocking/remediation outcomes across covered consumers.
  - Divergence is detected deterministically and handled fail closed.
  - Semantics remain conservative, hardware-neutral, and evidence-bound.
- Dependencies: 282, 283.

## Prompt 285 (MUST) — v11 schema snapshot refresh
Objective: Refresh schema snapshots for v11 final consumer-authority, execution, readiness, bundle, and continuity artifacts.
- Acceptance:
  - Snapshots reflect current canonical contracts with deterministic ordering.
  - Compatibility/version notes are documented for any changed artifacts.
  - Snapshot outputs remain reproducible offline.
- Dependencies: 283, 284.

## Prompt 286 (NICE) — v11 portability and docs refresh
Objective: Refresh portability and operator documentation for v11 governance/readiness/bundle/continuity surfaces.
- Acceptance:
  - Documentation reflects conservative v11 path expectations and evidence requirements.
  - Guidance remains hardware-neutral and offline-first.
  - Documentation-only updates do not change policy/runtime semantics.
- Dependencies: 285.

## Prompt 287 (MUST) — Operator workflow/export chain continuity hardening
Objective: Harden operator workflow/export chains with sole top-level continuity proof as the only continuity authority.
- Acceptance:
  - Workflow/export continuity checks enforce the sole top-level continuity proof across active v11 surfaces.
  - Failures emit bounded, actionable, deterministic diagnostics.
  - Probe-first/shadow-first/fail-closed guarantees remain explicit and preserved.
- Dependencies: 284, 285, 286.

## Prompt 288 (MUST) — v11 gate schema and orchestration
Objective: Define and wire v11 gate schema/orchestration for consolidated governance/execution/readiness/bundle/continuity assurances.
- Acceptance:
  - v11 gate verifies required v11 surfaces with deterministic ordering and normalized status semantics.
  - Required-surface mismatches fail closed with explicit evidence references.
  - Gate semantics remain conservative and do not widen runtime behavior.
- Dependencies: 287.

## Prompt 289 (MUST) — v11 wrap and next-anchor governance
Objective: Close the v11 planning loop and prepare the next anchor governance artifacts.
- Acceptance:
  - v11 wrap artifacts record closure state and next-anchor handoff points.
  - Next prompt queue remains capped to 10 entries unless explicitly expanded.
  - Wrap outputs remain hardware-neutral, offline-first, probe-first, shadow-first, fail-closed, and evidence-bound.
- Dependencies: 288.
