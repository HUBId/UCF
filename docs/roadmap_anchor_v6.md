# Roadmap Anchor v6 — Governance Surface Reuse and Interoperability Hardening

## Scope statement
v6 continues the conservative governance track after v5 gate PASS. The anchor is documentation/governance-first, offline-first, hardware-neutral, probe-first, shadow-first, and fail-closed.

## MUST for v6
- Deepen evidence/signoff reuse across the currently supported real-slot set, with deterministic and canonical artifact reuse.
- Harden export/review/gate interoperability without widening runtime semantics or introducing new runtime activation behavior.
- Consider supported-slot expansion only when Supported-Set-Review explicitly justifies it with evidence.
- Keep the Active path conservative, explicitly evidence-bound, and fail-closed for missing or inconsistent prerequisites.

## NICE for v6
- Add more bounded automation around review/export bundle assembly and verification.
- Introduce additional schema/report normalization where it reduces semantic drift and duplicate orchestration.
- Improve operator docs and ergonomics for review/export/signoff workflows while preserving deterministic behavior.

## DEFERRED beyond v6
- Training-related workflow expansion.
- Remote-compute requirements or orchestration dependencies.
- GPU-required operational flows.
- Hardware-specific optimization assumptions in core governance requirements.
- Broad supported-slot expansion without established evidence scaffolding and explicit Supported-Set-Review justification.
