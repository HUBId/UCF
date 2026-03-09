# Next 10 Prompts (v3 Anchor)

Anchor: `Real Compute Onboarding v3` (precondition: `ucf-ops v2 gate` PASS)

> Guardrail: This queue is capped to **10** entries unless an explicit request expands it.

## Prompt 200 (MUST) — Extend Active evidence from one slot to supported real slots set
Objective: Expand Active evidence requirements from single-slot assumptions to the supported real-slot set while preserving deterministic, offline-first enforcement.
- Acceptance:
  - Active evidence schema/validation covers each supported real slot with stable field semantics.
  - Missing slot evidence remains deny-by-default and reports deterministic remediation hints.
  - Replay compatibility and canonical digest linkage remain intact.
- Dependencies: v2 gate PASS (Prompt 198), Prompt 199 wrap complete.

## Prompt 201 (MUST) — Unified Eligibility Report for Probe/Shadow/Active readiness
Objective: Define one bounded eligibility report that unifies Probe, Shadow, and Active readiness outcomes across supported real slots.
- Acceptance:
  - A single report schema emits per-slot readiness verdicts in deterministic order.
  - Parity/drift and evidence checks feed one eligibility decision pipeline.
  - Offline reproducibility is documented for report generation and verification.
- Dependencies: 200.

## Prompt 202 (MUST) — Candle adapter extension for chosen second slot beyond fixture smoke
Objective: Extend Candle adapter parity for the chosen second slot beyond fixture-smoke scope under probe-first constraints.
- Acceptance:
  - Probe-path adapter behavior for the second slot is specified beyond load-only smoke.
  - Stub fallback behavior remains deterministic when optional backend paths are unavailable.
  - No hardware-specific assumptions are introduced in adapter docs/spec.
- Dependencies: 200, 201.

## Prompt 203 (NICE) — Burn parity extension or second-slot backend parity (if scaffolded)
Objective: Add bounded parity planning for Burn or equivalent second-slot backend path only where scaffolding already exists.
- Acceptance:
  - Scope is conditional and explicitly limited to pre-existing scaffolded backend paths.
  - Parity outputs align with unified eligibility inputs without introducing new runtime requirements.
  - Probe-first and shadow-first constraints remain unchanged.
- Dependencies: 201, 202.

## Prompt 204 (MUST) — Real-slot compare window normalization across World + second slot
Objective: Normalize compare-window definitions so World and second-slot parity/drift checks share consistent evidence windows.
- Acceptance:
  - Compare-window terms and boundaries are harmonized across both slots.
  - Normalization preserves deterministic replay outcomes for equivalent inputs.
  - Drift verdict generation remains bounded and auditable.
- Dependencies: 201, 202.

## Prompt 205 (MUST) — v3 strict-mode updates for broader real-slot evidence
Objective: Update strict-mode docs/gates for broader multi-slot real-backend evidence expectations in v3.
- Acceptance:
  - Strict-mode checks reference expanded evidence requirements for supported slots.
  - Failure modes include stable remediation codes/messages for missing or inconsistent evidence.
  - Offline-first and hardware-neutral constraints are explicitly preserved.
- Dependencies: 200, 201, 204.

## Prompt 206 (MUST) — v3 operator report and signoff consolidation
Objective: Consolidate operator-facing readiness and signoff reporting for multi-slot real backend workflows.
- Acceptance:
  - Operator report sections for probe/shadow/active readiness are unified and bounded.
  - Signoff summary fields remain deterministic and easy to audit.
  - Runbook guidance stays non-interactive and offline-first.
- Dependencies: 201, 205.

## Prompt 207 (NICE) — v3 portability and docs checks refresh
Objective: Refresh portability-focused docs checks and governance docs to reflect v3 multi-slot evidence/reporting scope.
- Acceptance:
  - Docs checks clearly cover updated v3 anchor and queue artifacts.
  - Hardware-neutral wording checks continue to guard core planning docs.
  - Updated docs references remain consistent with prompt rulebook constraints.
- Dependencies: 205, 206.

## Prompt 208 (MUST) — v3 gate schema and orchestration
Objective: Define v3 gate schema/orchestration so eligibility, evidence, and reporting checks are evaluated in one deterministic signoff flow.
- Acceptance:
  - Gate schema lists ordered checks and PASS/FAIL semantics for v3 scope.
  - Orchestration integrates unified eligibility outputs and multi-slot evidence checks.
  - Artifacts/output paths follow repository conventions and remain offline reproducible.
- Dependencies: 205, 206, 207.

## Prompt 209 (MUST) — v3 wrap and next anchor governance
Objective: Close v3 planning/execution cycle with explicit wrap criteria and next-anchor governance updates.
- Acceptance:
  - v3 completion conditions and transition rules are documented without speculative scope expansion.
  - Next immediate queue remains capped to 10 prompts unless explicitly expanded.
  - Anchor/state snapshot docs are updated with deterministic resume metadata.
- Dependencies: 208.
