# Roadmap Anchor v0 — Real Compute Onboarding (Hardware-Neutral)

> Status update: v0 and v1 onboarding phases are complete; active planning moved to `docs/roadmap_anchor_v2.md`.

## A) Current target milestone
**Real Compute Onboarding v0** is the active anchor for prompt-series execution and docs governance.

This anchor is hardware-neutral, offline-first, and limited to deterministic runtime onboarding scope.

## B) MUST (v0 required)
1. **Backend traits + deterministic CPU stubs**
   - Trait-level backend contract is present and wired.
   - Deterministic stub implementations are available for local/offline execution.
2. **JEPA/SAE/SSM mock implementations producing core signals**
   - Mock outputs include bounded and deterministic `spikes`, `surprise`, and `pressure`.
3. **E2E integration path**
   - Deterministic E2E coverage for `ControlFrame -> Decision -> ESS append`.
4. **Policy gating invariant**
   - `no decision, no action` remains enforced in onboarding flows.
5. **Basic observability**
   - Minimal explain-tick/operator-facing trace is available for onboarding verification.

## C) NICE (v0 optional)
1. Drift dashboards and richer trend visualization.
2. Additional benchmarks/perf characterization beyond minimum acceptance checks.
3. Advanced docs lint expansion beyond baseline governance gates.

## D) DEFERRED (later)
1. Real model weights integration.
2. GPU lanes and specialized acceleration paths.
3. Remote/cluster execution topologies.
4. Training pipelines.

## E) Acceptance criteria summary for v0
- MUST scope is implemented, testable, and documented with deterministic outputs.
- Onboarding evidence demonstrates the full `ControlFrame -> Decision -> ESS` path.
- Safety invariant `no decision, no action` is preserved.
- Explain-tick minimum observability remains available.
- NICE/DEFERRED items are explicitly tracked but not required for v0 signoff.
