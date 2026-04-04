# v20 Signoff

`ucf-ops v20 gate` is the v20 seal hardening gate for governance/scope/readiness/bundle/primary-semantics/continuity.

## Command

```bash
cargo run -p ucf-ops -- v20 gate --out ./out/v20_gate_report.json
```

## Exit codes

- `0`: overall PASS
- `2`: overall FAIL

## PASS guarantees

PASS means all required checks succeeded under offline, hardware-neutral, bounded execution:

- v0..v19 gates are PASS.
- Governance seal sweep is PASS.
- Current `SupportedScopeExecutionV15` artifact is present and consistent with applied scope and governance seal chain.
- Readiness seal sweep is PASS.
- Bundle seal sweep is PASS.
- Primary-semantics seal sweep is PASS.
- Exactly one seal-complete canonical authoritative top-level continuity proof (`canonical-seal-continuity-sweep`) is PASS.
- Artifact schema snapshot checks PASS.
- Portability/docs checks PASS.

Optional backend/legacy translation checks may be `SKIP` only where the path is explicitly unsupported/unconfigured.

## PASS does not guarantee

- No broader runtime capability expansion.
- No new slot/backend production readiness.
- No automatic activation of any slot.
- No GPU/remote/training/large-model readiness claim.

## Status interpretation

- `PASS`: check is satisfied.
- `FAIL`: required condition missing/inconsistent; gate fails closed.
- `SKIP`: only allowed for explicitly optional unsupported/unconfigured paths.

v20 is a seal and sole-top-level-continuity hardening phase. The current applied supported scope (from authoritative applied-scope + supported-scope execution artifacts) is the only authoritative scope for this gate.
