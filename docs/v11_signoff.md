# v11 Signoff Gate

## Purpose
`ucf-ops v11 gate` is the final v11 residual-cleanup and residual-free continuity hardening gate.
It is offline-first, deterministic, hardware-neutral, and scope-authoritative.

## Command
```bash
cargo run -p ucf-ops -- v11 gate --out ./out/v11_gate_report.json
```

## Exit codes
- `0`: PASS
- `2`: FAIL

## PASS guarantees
PASS certifies all required v11 residual-hardening surfaces are coherent and enforced:
- final governance residual cleanup is PASS across canonical consumers
- current `SupportedScopeExecutionV6` artifact is present and consistent with current applied scope
- final readiness residual cleanup is PASS
- final bundle residual cleanup is PASS
- final primary-semantics residual cleanup is PASS
- exactly one residual-free authoritative top-level continuity proof is active and PASS for canonical operator/export flow
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
The current applied supported scope from authoritative `supported_scope_execute_v6` + applied-scope artifacts is the only valid scope for this gate.
No implicit expansion or legacy scope inference is allowed.

## Post-PASS continuation note
After `ucf-ops v11 gate` PASS, continue at Prompt 290 via `docs/next_10_prompts.md`.
