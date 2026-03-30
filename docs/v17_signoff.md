# v17 Signoff

`ucf-ops v17 gate` is the v17 stabilization gate for governance/readiness/bundle/primary-semantics hardening, current supported-scope execution v12 coherence, and sole stabilized canonical top-level continuity proof enforcement.

## Command

```bash
cargo run -p ucf-ops -- v17 gate --out ./out/v17_gate_report.json
```

## Exit codes

- `0`: PASS
- `2`: FAIL

## PASS guarantees

A PASS certifies all required v17 stabilization conditions simultaneously:

- governance stabilization is enforced for canonical consumers
- current `SupportedScopeExecutionV12` artifact is explicit, present, and coherent with applied scope + stabilized governance chain
- readiness stabilization is enforced for canonical consumers
- bundle stabilization is enforced for canonical export consumers
- primary-semantics stabilization is enforced for canonical surfaces
- exactly one stabilized canonical authoritative top-level continuity proof (`canonical-stabilization-continuity-sweep`) is active and PASS
- artifact schema snapshot checks pass
- refreshed portability + docs checks pass

## PASS does not guarantee

- broader runtime capability
- additional production-ready slots/backends
- automatic activation of any slot
- GPU/remote compute/training readiness

## Check semantics

- Required checks: `PASS` required, otherwise overall FAIL.
- Optional checks: may be `SKIP` only when unsupported/unconfigured under current applied supported scope.
- `FAIL`: missing required artifact/surface or inconsistent/stale stabilization evidence.

## Scope authority note

The gate is scope-conservative: the **current applied supported scope** from authoritative applied-scope + supported-scope execution v12 artifacts is the only scope used for v17 decisions.

## Phase intent

v17 is a stabilization and sole-top-level-continuity hardening phase, not a compute-capability expansion phase.

## Continuation note

After `ucf-ops v17 gate` PASS, continue at Prompt 350 via `docs/next_10_prompts.md`.
