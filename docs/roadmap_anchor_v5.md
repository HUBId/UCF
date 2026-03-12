# Real Compute Onboarding v5 — Roadmap Anchor

## Scope and posture
v5 continues the conservative governance trajectory after v4 gate PASS.
The phase remains hardware-neutral, offline-first, probe-first, shadow-first, and fail-closed.
No broad runtime expansion is implied by this anchor.

## MUST for v5
- Extend supported real-slot governance/evidence/signoff in a bounded manner without weakening fail-closed guarantees.
- Optionally expand real-backend parity or lifecycle depth only where probe/shadow evidence already exists and is reproducible.
- Improve artifact/export/operator flows by reusing existing evidence surfaces rather than introducing risky new runtime paths.
- Keep Active-path handling conservative and explicitly evidence-bound for each supported slot.

## NICE for v5
- Richer bounded snapshot/export ergonomics for operator and audit workflows.
- More unified report reuse across gate, signoff, and export surfaces.
- Additional schema and portability hygiene for evidence/export artifacts.

## DEFERRED beyond v5
- Training workflows.
- Remote compute dependencies.
- GPU-required flows.
- Hardware-specific optimization assumptions.
- Broad slot expansion without prior evidence scaffolding.

## v5 completion framing
v5 completion is expected to culminate in a dedicated v5 gate prompt, followed by wrap governance that sets the next anchor only after evidence/signoff/doc checks are green.
