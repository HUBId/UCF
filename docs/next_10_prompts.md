# Next 10 Prompts (v17 Anchor)

Anchor: `Real Compute Onboarding v17` (precondition: `ucf-ops v16 gate` PASS at Prompt 338)

> Guardrail: This queue is capped to **10** entries unless an explicit request expands it.

## Prompt 340 (MUST) — Canonical governance chain as sole truth for remaining consumers
Objective: Unify every remaining canonical consumer around the converged canonical governance chain as the sole governance truth.
- Acceptance:
  - Active governance/review/export/gate consumers read only converged canonical governance inputs and aligned applied-scope authority.
  - Parallel governance-truth paths are removed or fail closed with deterministic diagnostics.
  - Behavior remains hardware-neutral, offline-first, probe-first, shadow-first, and fail-closed.
- Dependencies: 338, 339.

## Prompt 341 (MUST) — Supported-scope expansion re-check or freeze reaffirmation
Objective: Re-execute supported-scope expansion only when converged canonical governance plus current scope execution evidence still justify it; otherwise reaffirm freeze.
- Acceptance:
  - Expansion requires coherent converged canonical governance, current supported-scope execution evidence, and explicit bounded justification.
  - Missing or inconsistent evidence keeps scope frozen with deterministic fail-closed reporting.
  - Runtime semantics remain conservative and hardware-neutral.
- Dependencies: 340.

## Prompt 342 (MUST) — Readiness-input convergence completion for remaining consumers
Objective: Deepen converged canonical readiness-input consumption across all remaining operator/export/gate consumers.
- Acceptance:
  - Remaining consumers adopt one canonical converged readiness-input contract.
  - Missing, stale, or conflicting readiness prerequisites fail closed with bounded deterministic diagnostics.
  - No runtime-capability widening is introduced.
- Dependencies: 340, 341.

## Prompt 343 (MUST) — Bundle semantics fully normalized to converged chain
Objective: Normalize bundle build/verify/inspect semantics around the converged canonical bundle chain and sole top-level continuity proof.
- Acceptance:
  - Build/verify/inspect flows consume only converged canonical bundle inputs and the sole converged top-level continuity proof.
  - Governance/readiness evidence reuse is canonical and deterministic across bundle surfaces.
  - Operator/export/bundle round-trip remains reproducible offline.
- Dependencies: 340, 342.

## Prompt 344 (MUST) — Primary blocking/remediation convergence hardening
Objective: Harden converged canonical primary blocking/remediation semantics across governance/readiness/bundle/continuity/gate consumers.
- Acceptance:
  - Equivalent evidence states yield consistent blocking/remediation outcomes across covered consumers.
  - Divergence is detected deterministically and handled fail closed.
  - Semantics stay conservative, evidence-bound, and hardware-neutral.
- Dependencies: 342, 343.

## Prompt 345 (MUST) — v17 schema snapshot refresh
Objective: Refresh schema snapshots for v17 convergence, execution, readiness, bundle, and continuity artifacts.
- Acceptance:
  - Snapshots reflect canonical v17 contracts with deterministic ordering.
  - Compatibility/version notes are documented for changed artifacts.
  - Snapshot outputs remain reproducible offline.
- Dependencies: 343, 344.

## Prompt 346 (NICE) — v17 portability and docs refresh
Objective: Refresh portability and operator documentation for v17 governance/readiness/bundle/continuity surfaces.
- Acceptance:
  - Documentation reflects conservative v17 path expectations and evidence requirements.
  - Guidance remains hardware-neutral and offline-first.
  - Documentation-only updates do not change policy/runtime semantics.
- Dependencies: 345.

## Prompt 347 (MUST) — Workflow/export continuity authority hardening
Objective: Harden operator workflow/export chains with the sole converged canonical top-level continuity proof as the only continuity authority.
- Acceptance:
  - Workflow/export continuity checks enforce the sole converged canonical top-level continuity proof across active v17 surfaces.
  - Failures emit bounded, actionable, deterministic diagnostics.
  - Probe-first/shadow-first/fail-closed guarantees remain explicit and preserved.
- Dependencies: 344, 345, 346.

## Prompt 348 (MUST) — v17 gate schema and orchestration
Objective: Define and wire v17 gate schema/orchestration for consolidated governance/execution/readiness/bundle/continuity assurances.
- Acceptance:
  - v17 gate verifies required v17 surfaces with deterministic ordering and normalized status semantics.
  - Required-surface mismatches fail closed with explicit evidence references.
  - Gate semantics remain conservative and do not widen runtime behavior.
- Dependencies: 347.

## Prompt 349 (MUST) — v17 wrap and next-anchor governance
Objective: Close the v17 planning loop and prepare next-anchor governance artifacts.
- Acceptance:
  - v17 wrap artifacts record closure state and next-anchor handoff points.
  - Next prompt queue remains capped to 10 entries unless explicitly expanded.
  - Wrap outputs remain hardware-neutral, offline-first, probe-first, shadow-first, fail-closed, and evidence-bound.
- Dependencies: 348.
