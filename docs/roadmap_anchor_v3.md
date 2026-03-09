# Roadmap Anchor v3 — Real Backend Expansion (Probe-First, Shadow-First)

## A) Current target milestone
**Real Compute Onboarding v3** is the active anchor after v2 gate PASS.

This anchor remains hardware-neutral and offline-first. v3 scope extends real backend readiness carefully without introducing hardware-specific assumptions or remote-compute dependencies.

## B) MUST (v3 required)
1. **Extend real backend plumbing beyond minimal two-slot support (probe-first)**
   - Broaden adapter parity across the supported real-slot set in bounded steps.
   - Keep probe-first gating and deterministic fallback behavior intact.
   - Maintain shadow-first activation posture for decision-impact safety.
2. **Harden Active evidence model beyond shadow-ready baseline**
   - Expand Active evidence requirements from single-slot assumptions to supported real slots.
   - Preserve canonical evidence encoding, digest linkage, and replay compatibility.
   - Keep deny-by-default behavior for missing or invalid evidence.
3. **Unify parity/drift evidence into one eligibility pipeline**
   - Consolidate probe/shadow/active readiness signals into a single deterministic eligibility flow.
   - Keep thresholds, verdicts, and remediation hints stable and auditable.
   - Ensure offline reproducibility for generated eligibility artifacts.
4. **Add richer but bounded operator/signoff reporting for real slots**
   - Provide clearer multi-slot readiness summaries and signoff outputs.
   - Retain bounded report schemas and deterministic ordering.
   - Keep operator workflows non-interactive and offline-first.

## C) NICE (v3 optional)
1. Better benchmark coverage for supported real slots under fixed budget envelopes.
2. Richer spec snapshots for backend adapters and eligibility/report schema evolution.
3. Broader docs and runbook polish for multi-slot real backend operations.

## D) DEFERRED (later)
1. Training pipelines.
2. Remote compute or distributed execution.
3. GPU-mandatory pathways.
4. Hardware-specific optimization assumptions.

## E) Acceptance criteria summary for v3 planning
- v3 MUST scope is documented with hardware-neutral and offline-first language.
- Probe-first and shadow-first constraints remain explicit.
- NICE and DEFERRED items are clearly separated from MUST scope.
- `docs/next_10_prompts.md` defines exactly prompts `200-209` for immediate execution.
