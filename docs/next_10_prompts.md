# Next 10 Prompts (v7 Anchor)

Anchor: `Real Compute Onboarding v7` (precondition: `ucf-ops v6 gate` PASS at Prompt 238)

> Guardrail: This queue is capped to **10** entries unless an explicit request expands it.

## Prompt 240 (MUST) — Applied-scope governance/export surface unification
Objective: Unify applied-scope consumption across every remaining governance/export surface using existing primary review/evidence artifacts.
- Acceptance:
  - Remaining governance/export surfaces consume applied-scope authority from shared canonical artifacts rather than ad hoc derivations.
  - Inconsistencies are surfaced deterministically with fail-closed remediation semantics.
  - Behavior remains offline-first, hardware-neutral, and runtime-semantics preserving.
- Dependencies: 238, 239.

## Prompt 241 (MUST) — Supported-scope expansion execution only if justified
Objective: Execute supported-scope expansion only when v6 applied policy and evidence still explicitly justify the change.
- Acceptance:
  - Expansion is applied only with explicit Supported-Set policy approval and evidence references that remain valid.
  - Without sufficient justification, applied scope remains unchanged with explicit fail-closed reporting.
  - Probe-first and shadow-first progression remains explicit for all affected paths.
- Dependencies: 240.

## Prompt 242 (MUST) — Active review/signoff/review-packet consistency deepening
Objective: Deepen consistency between active-review outputs, signoff artifacts, and review packets for the applied scope.
- Acceptance:
  - Review/signoff/packet outputs are deterministic and semantically aligned for the applied scope.
  - Missing prerequisites or mismatched authority inputs fail closed with bounded diagnostics.
  - No runtime capability widening is introduced.
- Dependencies: 240, 241.

## Prompt 243 (MUST) — Governance-primary export bundle normalization end-to-end
Objective: Normalize enriched export-bundle consumption of governance primary surfaces end-to-end.
- Acceptance:
  - Export bundles use canonical structure, ordering, and references to governance primary surfaces.
  - End-to-end export consumption remains offline-reproducible and deterministic.
  - Existing governance/evidence surfaces are reused rather than re-derived where possible.
- Dependencies: 240, 242.

## Prompt 244 (MUST) — Remediation/interoperability consistency across export/review/gate chains
Objective: Harden remediation and interoperability consistency across export, review, and gate chains.
- Acceptance:
  - Equivalent evidence states produce consistent remediation and status semantics across chains.
  - Divergence is detected deterministically and handled fail-closed.
  - Interop behavior remains bounded, hardware-neutral, and evidence-bound.
- Dependencies: 242, 243.

## Prompt 245 (MUST) — v7 schema snapshot refresh for governance/export/workflow artifacts
Objective: Refresh schema snapshots for v7 governance/export/workflow artifacts after normalization and interoperability hardening.
- Acceptance:
  - Snapshot artifacts capture current schemas with canonical encoding and stable ordering.
  - Schema drift is detectable via deterministic checks.
  - Documentation references align with refreshed schema artifacts.
- Dependencies: 243, 244.

## Prompt 246 (NICE) — v7 portability and operator docs refresh
Objective: Refresh portability and operator-facing documentation for v7 governance/export/review surfaces.
- Acceptance:
  - Docs reflect v7 surfaces with hardware-neutral and offline-first language.
  - Probe-first, shadow-first, and fail-closed guidance remains explicit and consistent.
  - Cross-links between policy, review, export, and gate docs are coherent.
- Dependencies: 245.

## Prompt 247 (MUST) — Operator workflow/export-chain hardening with applied-scope authority
Objective: Harden operator workflow and export-chain steps so applied-scope authority is explicit throughout execution.
- Acceptance:
  - Workflow steps consume applied-scope authority consistently from canonical governance artifacts.
  - Missing authority/evidence prerequisites fail closed with bounded remediation hints.
  - Workflow behavior remains deterministic, read-only where required, and offline-first.
- Dependencies: 244, 245, 246.

## Prompt 248 (MUST) — v7 gate schema and orchestration
Objective: Define and stabilize v7 gate schema and orchestration over v7 governance/export/workflow invariants.
- Acceptance:
  - v7 gate checks map directly to applied-scope governance, interoperability, and workflow invariants.
  - PASS/FAIL behavior is deterministic with explicit bounded remediation guidance.
  - Optional paths remain conservative and do not weaken fail-closed behavior.
- Dependencies: 247.

## Prompt 249 (MUST) — v7 wrap and next-anchor governance
Objective: Close v7 governance work and prepare the subsequent anchor with a bounded evidence-backed handoff queue.
- Acceptance:
  - v7 completion state and next-anchor decision are documented in series-state artifacts.
  - Immediate prompt queue remains capped to 10 unless explicitly requested otherwise.
  - Wrap language remains hardware-neutral, offline-first, probe-first, shadow-first, and fail-closed.
- Dependencies: 248.
