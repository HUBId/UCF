# Next 10 Prompts (v6 Anchor)

Anchor: `Real Compute Onboarding v6` (precondition: `ucf-ops v5 gate` PASS at Prompt 228)

> Guardrail: This queue is capped to **10** entries unless an explicit request expands it.

## Prompt 230 (MUST) — Primary governance surfaces reuse unification
Objective: Unify review/export/gate consumption around `BackendEvidenceSnapshotV1` and `AggregatedActiveReviewSnapshotV1` as primary governance surfaces.
- Acceptance:
  - Review/export/gate paths prefer the two primary snapshots over direct lower-level evidence collection where reuse is possible.
  - Snapshot mismatch is explicit and fail-closed with stable remediation semantics.
  - Behavior remains read-only, offline-first, deterministic, and hardware-neutral.
- Dependencies: 228, 229.

## Prompt 231 (MUST) — Supported-slot expansion execution only if explicitly justified
Objective: Execute a formal supported-slot expansion only when Supported-Set-Review and evidence explicitly justify it.
- Acceptance:
  - Expansion is implemented only with explicit Supported-Set-Review justification and evidence references.
  - If justification is missing, scope remains unchanged with clear fail-closed reporting.
  - Probe-first/shadow-first precedence remains explicit for every affected slot.
- Dependencies: 220, 230.

## Prompt 232 (MUST) — Expanded-set active review/signoff consistency deepening
Objective: Deepen active-review/signoff consistency for the supported set active in v6, including any explicitly approved expansion.
- Acceptance:
  - Active-review and signoff outputs remain deterministic and semantically aligned for the entire supported set.
  - Missing or inconsistent inputs fail closed with bounded diagnostics.
  - No runtime execution semantics are widened.
- Dependencies: 230, 231.

## Prompt 233 (MUST) — Export bundle normalization across repro/bugkit/review/signoff artifacts
Objective: Normalize export bundle structure and cross-references across repro, bugkit, review packet, and signoff artifacts.
- Acceptance:
  - Export bundles share canonical ordering and normalized governance references.
  - Repro/bugkit/review/signoff artifacts remain offline-reproducible and deterministic.
  - Existing evidence surfaces are reused instead of re-derived where feasible.
- Dependencies: 230, 232.

## Prompt 234 (MUST) — Gate/remediation/report interoperability hardening
Objective: Harden interoperability between gates, remediation mapping, and reports across active v6 governance surfaces.
- Acceptance:
  - Gate/report/remediation semantics are consistent for equivalent evidence states.
  - Inconsistencies are detected deterministically and fail closed.
  - Interoperability remains hardware-neutral and bounded.
- Dependencies: 232, 233.

## Prompt 235 (MUST) — v6 schema snapshot refresh for governance artifacts
Objective: Refresh schema snapshots for v6 export/review/gate artifacts after normalization and interoperability hardening.
- Acceptance:
  - Snapshot outputs capture current v6 schemas with canonical encoding and stable ordering.
  - Schema drift is detectable via deterministic checks.
  - Docs references are updated to match refreshed snapshots.
- Dependencies: 233, 234.

## Prompt 236 (NICE) — v6 portability and operator docs refresh
Objective: Refresh portability and operator-facing docs for v6 review/export/gate surfaces.
- Acceptance:
  - Documentation reflects v6 governance surfaces without hardware or cluster assumptions.
  - Offline-first, probe-first, shadow-first, and fail-closed guidance remains explicit.
  - Cross-doc links and operator flow references are coherent.
- Dependencies: 235.

## Prompt 237 (MUST) — Operator workflow hardening for review/export/signoff chain
Objective: Harden the operator workflow chain spanning review, export, and signoff using normalized v6 governance surfaces.
- Acceptance:
  - Workflow composition remains deterministic, read-only, and evidence-bound.
  - Required artifact prerequisites are explicit with fail-closed handling on missing inputs.
  - Operator guidance reuses normalized artifacts consistently.
- Dependencies: 234, 235, 236.

## Prompt 238 (MUST) — v6 gate schema and orchestration
Objective: Define and stabilize the v6 gate schema/orchestration for the v6 governance hardening scope.
- Acceptance:
  - v6 gate checks map directly to v6 governance surfaces and interoperability invariants.
  - PASS/FAIL behavior is deterministic with bounded and explicit remediation guidance.
  - Optional paths remain explicitly bounded and do not weaken fail-closed behavior.
- Dependencies: 237.

## Prompt 239 (MUST) — v6 wrap and next-anchor governance
Objective: Close v6 governance work and prepare the next anchor with a bounded, evidence-backed handoff queue.
- Acceptance:
  - v6 completion state and next-anchor decision are documented in series-state artifacts.
  - Next prompt queue remains capped to 10 entries unless explicitly requested otherwise.
  - Wrap language remains hardware-neutral, offline-first, probe-first, shadow-first, and fail-closed.
- Dependencies: 238.
