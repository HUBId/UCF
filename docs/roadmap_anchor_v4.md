# Roadmap Anchor v4 — Conservative Evidence & Automation Hardening

## Scope intent
v4 continues from v3 gate PASS with a documentation/governance-first plan that remains hardware-neutral, offline-first, probe-first, shadow-first, and fail-closed.

## A) MUST for v4
1. Deepen supported real-slot evidence and signoff consistency.
   - Keep scope bounded to declared supported slots.
   - Preserve deterministic, canonical evidence semantics and remediation outputs.
2. Extend backend parity carefully to remaining supported paths only.
   - Allow optional second-slot parity extension only when a clean scaffold already exists.
   - Avoid introducing new mandatory runtime capabilities.
3. Harden operator/gate automation without broadening runtime semantics.
   - Tighten report-to-signoff linkage and gate consistency checks.
   - Keep policy-first, deny-by-default behavior unchanged.
4. Keep Active path conservative and evidence-bound.
   - No active-by-default expansion.
   - Require explicit evidence sufficiency and stable failure behavior.

## B) NICE for v4
1. Richer bounded reporting for operators and audits.
2. Additional portability/schema checks for v4 artifacts.
3. Improved docs/runbook ergonomics for supported evidence paths.

## C) DEFERRED
1. Training-related scope.
2. Remote compute scope.
3. GPU-required flows.
4. Hardware-specific optimization assumptions.

## Exit expectation
v4 is complete when v4-gated evidence/signoff automation is coherent for supported slots, with no expansion of runtime risk envelope.
