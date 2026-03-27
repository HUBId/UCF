# v14 Signoff Gate

## Purpose
`ucf-ops v14 gate` is the v14 terminal absolute residual-free final-input and sole-top-level-continuity hardening gate.
It is offline-first, deterministic, hardware-neutral, bounded, and scope-authoritative.

## Command
```bash
cargo run -p ucf-ops -- v14 gate --out ./out/v14_gate_report.json
```

## Exit codes
- `0`: PASS
- `2`: FAIL

## PASS guarantees
PASS certifies required v14 terminal absolute residual-free final-input surfaces are coherent and enforced:
- terminal absolute residual-free final governance inputs are PASS across canonical consumers
- current `SupportedScopeExecutionV9` artifact is present and consistent with the current applied scope context
- terminal absolute residual-free final readiness inputs are PASS across canonical consumers
- terminal absolute residual-free final bundle inputs are PASS across canonical export consumers
- terminal absolute residual-free final primary-semantics inputs are PASS across canonical consumers
- exactly one terminal absolute residual-free final-input authoritative top-level continuity proof is active and PASS for canonical operator/export flow
- artifact schema snapshot checks PASS
- portability and docs checks PASS

## PASS does not guarantee
PASS does **not** imply:
- broader runtime capability
- new slots/backends beyond currently applied supported scope
- automatic activation of any slot/backend
- GPU readiness, remote compute readiness, training readiness, or large real-model readiness

## Status interpretation
- **PASS**: required check succeeded.
- **FAIL**: required check failed or required artifact/surface missing.
- **SKIP**: optional check only, used for unsupported optional backend paths.

## Authority note
The current applied supported scope from authoritative `supported_scope_execute_v9` + applied-scope artifacts is the only valid scope for this gate.
No implicit expansion, legacy scope inference, or unsupported-slot widening is allowed.

## Phase note
v14 is a terminal absolute residual-free final-input / sole-top-level-continuity hardening phase.
This gate is governance/scope/readiness/bundle/primary-semantics/continuity hygiene, not a compute capability gate.
