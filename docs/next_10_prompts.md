# Next 10 Prompts (MUST-only, v0 Anchor)

Anchor: `Real Compute Onboarding v0` (`docs/roadmap_anchor_v0.md`)

> Guardrail: This queue is capped to **10** entries unless an explicit request expands it.

## Prompt 129 — Backend trait contract freeze for v0 stubs
- Class: **MUST**
- Acceptance:
  - Backend trait interfaces are explicitly frozen for v0 scope.
  - No hardware-specific assumptions are introduced.
  - Contract docs and compile checks remain deterministic.

## Prompt 130 — Deterministic CPU backend stub conformance tests
- Class: **MUST**
- Acceptance:
  - CPU stubs return deterministic outputs for identical fixtures.
  - Conformance tests cover success and bounded-failure paths.
  - Replay of fixtures yields byte-stable evidence fields.

## Prompt 131 — JEPA mock deterministic signal mapping
- Class: **MUST**
- Acceptance:
  - JEPA mock emits deterministic `spikes/surprise/pressure`-related fields.
  - Signal bounds are validated in tests.
  - Outputs are explainable from fixture inputs.

## Prompt 132 — SAE mock deterministic signal mapping
- Class: **MUST**
- Acceptance:
  - SAE mock emits deterministic feature/surprise-related fields.
  - Canonical ordering is preserved in persisted outputs.
  - Determinism checks pass across repeated runs.

## Prompt 133 — SSM mock deterministic state update mapping
- Class: **MUST**
- Acceptance:
  - SSM mock state updates are deterministic and bounded.
  - No nondeterministic iteration leaks to persisted/output-facing artifacts.
  - State transition evidence is replay-compatible.

## Prompt 134 — Compute summary wiring for spikes/surprise/pressure
- Class: **MUST**
- Acceptance:
  - Compute summary includes required onboarding signals.
  - Wiring is covered by deterministic tests.
  - Evidence digests remain chain-consistent.

## Prompt 135 — E2E ControlFrame->Decision->ESS canonical fixture pass
- Class: **MUST**
- Acceptance:
  - E2E fixture verifies `ControlFrame -> Decision -> ESS append`.
  - Ordering and linkage invariants hold for all records.
  - Offline execution path passes without network dependency.

## Prompt 136 — Policy gate assertion: no decision, no action
- Class: **MUST**
- Acceptance:
  - Runtime denies actions when no decision exists.
  - Negative-path tests assert deny-by-default behavior.
  - Safety invariant is documented in acceptance evidence.

## Prompt 137 — Minimal explain-tick observability acceptance checks
- Class: **MUST**
- Acceptance:
  - Explain-tick minimum output exists for onboarding ticks.
  - Key fields required for audit/debug are populated deterministically.
  - Documentation and tests reference the same observable fields.

## Prompt 138 — v0 onboarding signoff bundle (must-only evidence)
- Class: **MUST**
- Acceptance:
  - Signoff bundle includes tests, docs, and gate artifacts for MUST scope.
  - NICE/DEFERRED items are explicitly excluded from signoff requirements.
  - Snapshot/index/state docs are updated consistently.
