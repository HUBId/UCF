# Roadmap Anchor v2 — Optional Real Backend Onboarding (Probe-First)

## A) Current target milestone
**Real Compute Onboarding v2** is the active anchor after v1 signoff.

This anchor remains hardware-neutral and offline-first. Real backend use is optional and constrained to probe/shadow pathways.

## B) MUST (v2 required)
1. **Candle/Burn adapter layer (optional feature) for 1–2 slots**
   - Adapter layer is defined behind optional feature flags.
   - At most one or two slots are in scope for the first v2 phase.
   - Default behavior remains deterministic stubs when optional features are off.
2. **Minimal real-weights fixture (tiny) for one slot, probe-first**
   - A tiny fixture is documented for one slot only.
   - Probe flow validates fixture loading without requiring production-sized weights.
   - No mandatory download requirement is introduced in baseline flows.
3. **Shadow-only rollout for real backend**
   - Real backend path is restricted to shadow mode in initial v2 phase.
   - No direct production activation path is allowed from probe-only evidence.
   - Safety invariant `no decision, no action` remains enforced.

## C) NICE (v2 optional)
1. Performance benchmark documentation and reproducible harness updates.
2. Deeper drift dashboard coverage for stub vs optional real backend comparisons.

## D) DEFERRED (later)
1. Training pipelines.
2. Remote compute/distributed execution.
3. GPU-specific execution lanes.

## E) Acceptance criteria summary for v2 phase-1
- v1 gate PASS evidence exists before v2 execution starts.
- MUST scope is documented with deterministic, hardware-neutral constraints.
- Optional real backend pathways are probe-first and shadow-only.
- NICE and DEFERRED scopes are explicitly separated from MUST.
