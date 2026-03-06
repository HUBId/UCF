# Next 10 Prompts (MUST-only, v1 Anchor)

Anchor: `Real Compute Onboarding v1` transition plan (v0 completion prerequisite: `ucf-ops v0 gate` PASS)

> Guardrail: This queue is capped to **10** entries unless an explicit request expands it.

## Prompt 178 — Weights lifecycle scaffold (staging/promoted, no real weights required)
- Objective: Define deterministic v1 lifecycle metadata and transitions for `staging` and `promoted` slots without requiring real weight artifacts.
- Acceptance:
  - Lifecycle schema documents `staging -> promoted` transitions with deterministic state encoding.
  - Real weight blobs are optional in v1 unless a probe fixture explicitly references them.
  - Offline fixture path validates empty/dummy payload handling.
- Dependencies: v0 gate PASS; prompt rulebook 10-prompt cap (Prompt 171).

## Prompt 179 — Hardware-neutral backend adapter traits (Candle/Burn optional, no real compute)
- Objective: Specify backend adapter trait interfaces that allow Candle/Burn integration later while keeping v1 behavior hardware-neutral.
- Acceptance:
  - Trait contract defines adapter boundaries independent of vendor/device assumptions.
  - Candle/Burn adapters remain optional and feature-gated for v1.
  - Deterministic stub behavior is specified for no-backend environments.
- Dependencies: 178.

## Prompt 180 — Probe infrastructure per model slot with dummy fixtures
- Objective: Add slot-scoped probe fixture planning so v1 can validate interfaces without real weights.
- Acceptance:
  - Each slot has a deterministic dummy probe fixture contract.
  - Probe outputs are canonicalized for repeatable compare runs.
  - Real weights are only required when a probe fixture explicitly opts in.
- Dependencies: 178, 179.

## Prompt 181 — Slot-level rollout state machine (shadow/compare/active)
- Objective: Define deterministic rollout semantics per slot using `shadow`, `compare`, and `active` states.
- Acceptance:
  - State machine transitions are explicit, bounded, and auditable.
  - `shadow` and `compare` do not force production activation paths.
  - Canonical transition evidence fields are listed for docs/tests.
- Dependencies: 178, 180.

## Prompt 182 — Drift budget schema and evaluator for shadow outputs
- Objective: Specify minimal drift budget fields and deterministic evaluation flow for shadow-vs-reference outputs.
- Acceptance:
  - Drift budget schema defines threshold fields with stable encoding.
  - Evaluator behavior is deterministic for identical fixture inputs.
  - Budget verdicts are consumable by rollout state transitions.
- Dependencies: 180, 181.

## Prompt 183 — Minimal alerts rules and report format
- Objective: Define minimal alerting rules and deterministic reporting for drift and rollout anomalies.
- Acceptance:
  - Alert rule set covers at least budget breach and transition violation signals.
  - Report schema is bounded and deterministic in ordering/content.
  - Offline report generation is documented for CI artifacts.
- Dependencies: 182.

## Prompt 184 — Portability gate integration (Linux/Windows lanes)
- Objective: Integrate v1 scaffolding checks into existing portability gate lanes without changing hardware assumptions.
- Acceptance:
  - Linux and Windows lane expectations are documented for v1 scaffolding checks.
  - Determinism is required within each OS lane for identical fixtures.
  - Gate output artifact paths follow repository conventions.
- Dependencies: 179, 180, 183.

## Prompt 185 — Strict-mode wiring for v1 scaffold features
- Objective: Define strict-mode behavior that enforces v1 scaffold invariants at runtime/docs gates.
- Acceptance:
  - Strict mode fails on missing required v1 scaffold metadata.
  - Non-strict mode preserves backward-compatible behavior for staged rollout.
  - Failure reasons are deterministic and operator-readable.
- Dependencies: 181, 182, 183.

## Prompt 186 — Operator docs and end-state update for v1 onboarding
- Objective: Update operator-facing documentation to reflect v1 onboarding semantics and end-state expectations.
- Acceptance:
  - Docs describe lifecycle, rollout states, drift budgets, and alerts in one coherent flow.
  - End-state wording remains hardware-neutral and probe-first.
  - Prompt index/state references are synchronized with v1 queue.
- Dependencies: 178-185.

## Prompt 187 — v1 scaffolding signoff gate (PASS/FAIL)
- Objective: Define the v1 gate criteria that mark scaffolding completeness without requiring real production weights by default.
- Acceptance:
  - Gate criteria cover lifecycle, probe, rollout, drift, alerts, portability, and strict-mode checks.
  - PASS/FAIL semantics are deterministic and CI-friendly.
  - Real weights remain optional unless explicitly required by probe fixtures.
- Dependencies: 178-186.
