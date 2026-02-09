# AI Stack Roadmap (T91)

## Current state (implemented now)
- **ONN v1 phase bus** — core routing + phase bus backbone in place.
- **SNN v1 spike bus** — spike transport + bus interfaces wired.
- **SSM v1 selective scan** — selective scan core implemented.
- **JEPA placeholder** — placeholder wired for downstream integration.
- **TCF auto lock-window** — lock-window scheduling in place.
- **NSR phase/spike facts + stabilize gating** — facts + gating implemented.
- **CDE observation binding** — observation hooks wired into the commit flow.
- **ThoughtOnly non-leak enforcement + coherence gate tests** — guardrails and tests in place.

## Near-term (next 5 milestones)
1. Replace JEPA placeholder with a **world latent provider trait**.
2. Replace mock SAE/Lens producers with a **real feature extraction adapter trait** (still no model).
3. Introduce **ModelHost** abstraction (Candle/Burn later) for LFM/RLM hooks.
4. Implement **CDE v1 causal discovery** over observation commits (graph skeleton + interventions API).
5. Implement **IIT monitor v1** over commit dependencies + PLV/lag metrics.

## Mid-term (real compute)
- Candle/Burn integration points.
- SAE training pipeline (offline).
- Lens extraction points.
- BlueBrain bridge (FFI, streaming).

## Long-term
- RSA/OpenEvolve safe sandboxed optimization + budgets.
- memristor/photonic hardware abstraction (drivers behind traits).
