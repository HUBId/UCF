# Next 10 Prompts (v5 Anchor)

Anchor: `Real Compute Onboarding v5` (precondition: `ucf-ops v4 gate` PASS at Prompt 218)

> Guardrail: This queue is capped to **10** entries unless an explicit request expands it.

## Prompt 220 (MUST) — Cautious supported real-slot governance expansion
Objective: Extend supported real-slot detection/governance only within a cautiously expanded scope when current repository evidence justifies it.
- Acceptance:
  - Expansion, if any, is explicitly evidence-gated and remains fail-closed.
  - Probe-first and shadow-first precedence remains explicit for all supported slots.
  - Hardware-neutral and offline-first wording is preserved in all updated docs/spec surfaces.
- Dependencies: v4 gate PASS (Prompt 218), Prompt 219 wrap complete.

## Prompt 221 (MUST) — Unify Active-review evidence export across supported real slots
Objective: Normalize Active-review evidence export so supported real slots share one deterministic and auditable export pattern.
- Acceptance:
  - Evidence export fields and ordering are consistent across supported slots.
  - Signoff-relevant evidence is derivable from one unified export surface.
  - Missing required Active-review evidence fails closed with deterministic remediation hints.
- Dependencies: 220.

## Prompt 222 (NICE) — Optional second-slot Burn parity completion or supported-state closure
Objective: Either complete bounded Burn parity for the optional second slot where scaffolding exists or explicitly close the supported-state without parity expansion.
- Acceptance:
  - Outcome is explicit: parity-complete path or documented supported-state closure.
  - No new mandatory runtime path is introduced.
  - Any optional path remains probe-first/shadow-first and evidence-bound.
- Dependencies: 220, 221.

## Prompt 223 (MUST) — Reuse backend evidence snapshot + signoff in repro/bugkit exports
Objective: Improve artifact/export ergonomics by reusing backend evidence snapshot and signoff surfaces inside reproducibility/bugkit flows.
- Acceptance:
  - Repro/bugkit export references existing evidence/signoff artifacts without schema drift.
  - Export outputs remain deterministic and offline reproducible.
  - Runtime behavior remains unchanged; documentation/governance surfaces only are expanded.
- Dependencies: 221, 222 (if optional parity path is exercised).

## Prompt 224 (MUST) — Harden gate/report remediation consistency for v5 artifacts
Objective: Ensure remediation code semantics remain consistent across all active v5 gate/report artifacts.
- Acceptance:
  - Remediation mappings are aligned across affected v5 reports/gates.
  - Inconsistent or missing remediation surfaces fail closed with bounded diagnostics.
  - Operator-facing guidance remains deterministic and hardware-neutral.
- Dependencies: 223.

## Prompt 225 (MUST) — Refresh schema snapshots for v5 artifacts/export surfaces
Objective: Refresh and lock schema snapshots for newly stabilized v5 artifact and export surfaces.
- Acceptance:
  - Snapshot outputs reflect current v5 schemas with canonical ordering.
  - Snapshot drift is detectable through deterministic checks.
  - Documentation references match the refreshed schema snapshots.
- Dependencies: 223, 224.

## Prompt 226 (NICE) — Portability/docs refresh for v5 evidence and export paths
Objective: Update portability and documentation guidance to reflect v5 evidence/export flow consolidation.
- Acceptance:
  - Portability docs describe supported evidence/export paths without hardware assumptions.
  - Offline-first and fail-closed constraints remain explicit.
  - Cross-doc references to v5 artifacts are coherent and current.
- Dependencies: 225.

## Prompt 227 (MUST) — Read-only operator review workflow hardening from evidence + signoff + gates
Objective: Harden read-only operator review workflow composition from evidence snapshots, signoff, and gate outputs.
- Acceptance:
  - Operator review steps are deterministic and derived from existing artifacts.
  - Review workflow remains read-only and does not introduce runtime control expansion.
  - Missing required inputs produce explicit fail-closed outcomes.
- Dependencies: 224, 225, 226.

## Prompt 228 (MUST) — v5 gate schema and orchestration
Objective: Define and stabilize the v5 gate schema/orchestration for conservative evidence/signoff completion.
- Acceptance:
  - v5 gate schema/check set is documented with deterministic PASS/FAIL semantics.
  - Required checks cover supported-slot governance, evidence export, and signoff consistency.
  - Optional checks remain explicitly bounded and do not weaken fail-closed guarantees.
- Dependencies: 227.

## Prompt 229 (MUST) — v5 wrap and next-anchor governance
Objective: Close v5 governance and prepare the subsequent anchor with bounded, evidence-backed planning.
- Acceptance:
  - v5 completion status and next anchor decision are documented in series-state artifacts.
  - Next queue handoff remains capped to 10 prompts unless explicitly expanded.
  - Wrap language remains hardware-neutral, offline-first, probe-first/shadow-first, and fail-closed.
- Dependencies: 228.
