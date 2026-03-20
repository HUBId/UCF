# Next 10 Prompts (v10 Anchor)

Anchor: `Real Compute Onboarding v10` (precondition: `ucf-ops v9 gate` PASS at Prompt 268)

> Guardrail: This queue is capped to **10** entries unless an explicit request expands it.

## Prompt 270 (MUST) — Final governance-entry and applied-scope authority unification
Objective: Unify every remaining canonical consumer around final governance-entry authority plus applied-scope authority as the sole inputs.
- Acceptance:
  - All active governance/review/export/gate consumers read final governance-entry authority and applied-scope authority from canonical sources only.
  - Parallel or legacy authority reads are removed or fail closed with deterministic diagnostics.
  - Behavior remains hardware-neutral, offline-first, and runtime-semantics preserving.
- Dependencies: 268, 269.

## Prompt 271 (MUST) — Supported-scope reevaluation with authority/evidence guardrails
Objective: Re-execute supported-scope expansion only if final governance authority and current execution evidence still justify it; otherwise reaffirm freeze.
- Acceptance:
  - Expansion requires explicit final governance authority, current supported-scope execution support, and coherent evidence.
  - If justification is insufficient, scope remains frozen with explicit fail-closed reporting.
  - Probe-first and shadow-first progression remains explicit.
- Dependencies: 270.

## Prompt 272 (MUST) — Final readiness-authority consumption deepening
Objective: Deepen final readiness-authority consumption across all remaining operator/export/gate consumers.
- Acceptance:
  - Remaining consumers use one canonical final readiness authority contract and ordering.
  - Missing or stale readiness prerequisites fail closed with bounded deterministic diagnostics.
  - No runtime capability widening is introduced.
- Dependencies: 270, 271.

## Prompt 273 (MUST) — Final bundle authority normalization and round-trip continuity
Objective: Normalize bundle build/verify/inspect semantics around final bundle authority and canonical round-trip continuity.
- Acceptance:
  - Build/verify/inspect paths consume a single final bundle authority contract.
  - Operator→export→bundle round-trip checks remain deterministic and offline reproducible.
  - Governance/readiness evidence is reused from canonical surfaces when available.
- Dependencies: 270, 272.

## Prompt 274 (MUST) — Final blocking/remediation semantics hardening
Objective: Harden final primary blocking/remediation semantics across governance/readiness/bundle/continuity/gate surfaces.
- Acceptance:
  - Equivalent evidence states produce consistent blocking/remediation outcomes across covered surfaces.
  - Divergence is detected deterministically and handled fail closed.
  - Semantics remain conservative, hardware-neutral, and evidence-bound.
- Dependencies: 272, 273.

## Prompt 275 (MUST) — v10 schema snapshot refresh for final-authority chains
Objective: Refresh schema snapshots for v10 final-authority, execution, readiness, bundle, and continuity artifacts.
- Acceptance:
  - Snapshots reflect current canonical contracts with deterministic ordering.
  - Versioning and compatibility notes are documented for changed artifacts.
  - Snapshot outputs remain reproducible offline.
- Dependencies: 273, 274.

## Prompt 276 (NICE) — v10 portability and operator-doc refresh
Objective: Refresh portability and operator documentation for v10 governance/readiness/bundle/continuity surfaces.
- Acceptance:
  - Operator docs reflect conservative v10 path expectations and evidence requirements.
  - Portability guidance remains hardware-neutral and offline-first.
  - Documentation updates avoid behavior-changing policy semantics.
- Dependencies: 275.

## Prompt 277 (MUST) — Operator workflow/export chain hardening via final continuity authority
Objective: Harden workflow/export chains with final continuity authority as the sole top-level continuity proof.
- Acceptance:
  - Workflow/export continuity checks enforce final continuity authority across active v10 surfaces.
  - Failures emit bounded, actionable, deterministic diagnostics.
  - Probe-first/shadow-first/fail-closed guarantees remain explicit and preserved.
- Dependencies: 274, 275, 276.

## Prompt 278 (MUST) — v10 gate schema and orchestration
Objective: Define and wire v10 gate schema and orchestration for consolidated governance/execution/readiness/bundle/continuity assurances.
- Acceptance:
  - v10 gate verifies required v10 surfaces with deterministic ordering and normalized status semantics.
  - Required-surface mismatches fail closed with explicit evidence references.
  - Gate semantics remain conservative and do not widen runtime behavior.
- Dependencies: 277.

## Prompt 279 (MUST) — v10 wrap and next-anchor governance
Objective: Close the v10 planning loop and prepare next-anchor governance artifacts after v10 gate readiness.
- Acceptance:
  - v10 wrap artifacts record closure state and next-anchor handoff points.
  - Next prompt queue remains capped to 10 entries unless explicitly expanded.
  - Wrap outputs remain hardware-neutral, offline-first, probe-first, shadow-first, fail-closed, and evidence-bound.
- Dependencies: 278.
