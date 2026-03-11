# Next 10 Prompts (v4 Anchor)

Anchor: `Real Compute Onboarding v4` (precondition: `ucf-ops v3 gate` PASS)

> Guardrail: This queue is capped to **10** entries unless an explicit request expands it.

## Prompt 210 (MUST) — Extend Active evidence/signoff consistency for supported real-slot set
Objective: Tighten Active-path evidence and signoff consistency across the declared supported real-slot set without broadening runtime semantics.
- Acceptance:
  - Active evidence/signoff expectations are documented consistently for each supported slot.
  - Missing/inconsistent evidence remains fail-closed with deterministic remediation guidance.
  - Probe-first and shadow-first precedence remain explicit.
- Dependencies: v3 gate PASS (Prompt 208), Prompt 209 wrap complete.

## Prompt 211 (NICE) — Optional parity extension for chosen second slot backend (if scaffold exists)
Objective: Define an optional second backend parity extension for the chosen second slot only if compatible scaffolding is already present.
- Acceptance:
  - Scope remains conditional and does not create new mandatory runtime paths.
  - Parity expectations remain bounded to supported slots and existing evidence pipelines.
  - Hardware-neutral/offline-first language is preserved.
- Dependencies: 210.

## Prompt 212 (MUST) — Unified backend evidence snapshot/spec export refresh
Objective: Refresh the unified evidence snapshot/spec export so supported backend paths produce one coherent, deterministic documentation baseline.
- Acceptance:
  - Snapshot/spec export references supported backend evidence in stable order.
  - Export artifacts remain reproducible offline with canonical formatting.
  - No new runtime behavior is introduced.
- Dependencies: 210, 211 (if 211 is exercised).

## Prompt 213 (MUST) — Stricter operator signoff automation from consolidated reports + gates
Objective: Strengthen operator signoff automation by consolidating report and gate outcomes into deterministic signoff criteria.
- Acceptance:
  - Signoff automation rules are documented with explicit fail-closed outcomes.
  - Consolidated reports/gates map to stable operator decisions.
  - No expansion of runtime privileges or side effects.
- Dependencies: 210, 212.

## Prompt 214 (MUST) — Normalized remediation-code registry across reports/gates
Objective: Normalize remediation-code usage so reports and gates share one bounded, deterministic remediation registry.
- Acceptance:
  - Remediation codes are consistent across relevant report/gate outputs.
  - Registry semantics and ownership are documented for operator use.
  - Unknown/unsupported conditions remain deny-by-default.
- Dependencies: 213.

## Prompt 215 (MUST) — Report/schema snapshot checks for v4 artifacts
Objective: Add/refresh snapshot checks that lock v4 report/schema artifacts to deterministic, reviewable expectations.
- Acceptance:
  - Snapshot checks cover the v4 artifacts introduced/refreshed by the queue.
  - Drift is surfaced through deterministic diffs/remediation guidance.
  - Offline reproducibility remains explicit in docs.
- Dependencies: 212, 214.

## Prompt 216 (NICE) — Portability/docs refresh for expanded evidence paths
Objective: Refresh portability and runbook documentation to reflect expanded, still-bounded evidence paths in v4.
- Acceptance:
  - Portability notes remain hardware-neutral and device-profile agnostic.
  - Runbook/docs updates align with probe-first/shadow-first constraints.
  - Documentation changes avoid introducing new runtime obligations.
- Dependencies: 212, 215.

## Prompt 217 (MUST) — v4 strict-mode/operator interplay hardening
Objective: Harden strict-mode and operator-interaction expectations so evidence failures are handled consistently and safely.
- Acceptance:
  - Strict-mode/operator interplay is documented with deterministic fail-closed behavior.
  - Operator decision boundaries and remediation flow are unambiguous.
  - Safety/determinism invariants remain unchanged.
- Dependencies: 213, 214, 216.

## Prompt 218 (MUST) — v4 gate schema and orchestration
Objective: Define v4 gate schema and orchestration expectations that verify evidence/signoff hardening outcomes.
- Acceptance:
  - Gate schema documents required inputs/outputs and deterministic ordering.
  - Orchestration rules preserve offline-first execution and bounded runtime behavior.
  - PASS/FAIL semantics are explicit and auditable.
- Dependencies: 215, 217.

## Prompt 219 (MUST) — v4 wrap and next-anchor governance
Objective: Close v4 planning/governance and re-anchor the series state for the subsequent phase without speculative scope creep.
- Acceptance:
  - Series snapshot, anchor, and queue docs are aligned at wrap time.
  - Next-anchor proposal remains bounded by evidence and policy invariants.
  - Queue cap and MUST/NICE/DEFERRED governance remain intact.
- Dependencies: 218.
