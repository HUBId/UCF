# Next 10 Prompts (v2 Anchor)

Anchor: `Real Compute Onboarding v2` (precondition: `ucf-ops v1 gate` PASS)

> Guardrail: This queue is capped to **10** entries unless an explicit request expands it.

## Prompt 189 — Adapter trait hardening for optional Candle/Burn slots
- Objective: Refine backend adapter trait boundaries so optional Candle/Burn integrations stay deterministic and hardware-neutral.
- Acceptance:
  - Trait contracts document slot boundaries and deterministic fallback behavior.
  - Optional backend flags are explicit and default-off.
  - No vendor- or hardware-specific assumptions are required.
- Dependencies: v1 gate PASS.

## Prompt 190 — Optional feature wiring for one or two backend-enabled slots
- Objective: Define feature wiring for at most one or two slots to enable optional real backend paths.
- Acceptance:
  - Slot scope is limited to one or two named slots.
  - Disabled-feature behavior remains stub-only and deterministic.
  - Build/runtime docs specify optional activation paths.
- Dependencies: 189.

## Prompt 191 — Tiny real-weights fixture contract for one slot (probe-first)
- Objective: Specify a tiny fixture contract for one slot so probe workflows can validate real backend loading behavior.
- Acceptance:
  - Fixture contract includes digest, size cap, and canonical metadata fields.
  - Probe-first usage is required; production activation is excluded.
  - Baseline workflows do not require downloading real weights.
- Dependencies: 190.

## Prompt 192 — Probe-only execution flow for optional real backend fixture
- Objective: Define deterministic probe execution and reporting for the tiny real-backend fixture.
- Acceptance:
  - Probe run produces deterministic PASS/FAIL outputs for identical inputs.
  - Evidence payload format is canonical and bounded.
  - Failure remediation steps are documented for offline runs.
- Dependencies: 191.

## Prompt 193 — Shadow-only rollout guard for optional real backend
- Objective: Introduce rollout guardrails that keep optional real backend operation in shadow mode only.
- Acceptance:
  - Shadow mode is enforced as the only allowed mode for optional real backend paths.
  - Decision-impact parity checks remain mandatory.
  - Safety invariant `no decision, no action` is unchanged.
- Dependencies: 192.

## Prompt 194 — Drift/evidence parity checks for stub vs optional backend
- Objective: Define parity checks between deterministic stubs and optional backend shadow outputs.
- Acceptance:
  - Drift budget fields and thresholds are documented for parity checks.
  - Evidence records include stable digests and comparison verdicts.
  - Checks are reproducible in offline fixture runs.
- Dependencies: 193.

## Prompt 195 — Operator runbook update for optional backend probe workflows
- Objective: Update operator docs for staging, probing, and shadow monitoring of optional backend slots.
- Acceptance:
  - Runbook steps are non-interactive and deterministic.
  - Required artifacts and report paths are listed explicitly.
  - Rollback/disable path for optional backend features is documented.
- Dependencies: 190, 192, 193, 194.

## Prompt 196 — Deterministic performance bench envelope for optional backend paths
- Objective: Add optional benchmark planning for backend probe/shadow paths with fixed budget envelopes.
- Acceptance:
  - Bench definitions use budget envelopes, not hardware-specific targets.
  - Output schema and ordering are deterministic.
  - Bench scope is clearly marked optional (NICE).
- Dependencies: 192, 193.

## Prompt 197 — Extended drift dashboard docs for probe/shadow comparison
- Objective: Expand documentation for drift dashboard views covering probe and shadow parity trends.
- Acceptance:
  - Dashboard inputs and derived metrics are documented with stable field names.
  - Data retention and boundedness constraints are explicit.
  - Work is clearly marked optional (NICE).
- Dependencies: 194.

## Prompt 198 — v2 phase-1 signoff gate for optional real-backend readiness
- Objective: Define v2 phase-1 gate criteria for optional real-backend probe/shadow readiness.
- Acceptance:
  - Gate criteria cover adapters, fixture contract, probe flow, shadow guard, and parity evidence.
  - PASS/FAIL semantics are deterministic and CI-friendly.
  - Training, remote compute, and GPU lanes remain out of scope.
- Dependencies: 189-195.
