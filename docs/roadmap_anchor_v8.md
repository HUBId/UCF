# Roadmap Anchor v8

v8 continues conservative governance hardening after v7 gate PASS, with no runtime capability widening assumptions.

## MUST (v8)
- Deepen applied-scope governance reuse across all active review/export/gate surfaces.
- Harden end-to-end export/review/signoff round-trip semantics further without widening runtime semantics.
- Consider supported-scope expansion only where applied-scope authority plus reevaluation plus current evidence explicitly justify it.
- Keep the Active path conservative, probe-first, shadow-first, fail-closed, and explicitly evidence-bound.
- Preserve hardware-neutral and offline-first execution assumptions across all v8 deliverables.

## NICE (v8)
- Improve bounded export/review ergonomics while preserving determinism and fail-closed behavior.
- Further normalize schema/report/export surfaces already used in governance/review/export flows.
- Improve operator documentation and runbook clarity for existing v8 surfaces.

## DEFERRED (v8)
- Training flows.
- Remote compute dependencies.
- GPU-required execution paths.
- Hardware-specific optimization assumptions.
- Broad scope expansion without explicit applied authority + reevaluation + evidence scaffolding.
