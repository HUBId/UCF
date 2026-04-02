# v18 Signoff

`ucf-ops v18 gate` is the v18 final-consolidation gate for governance/current-supported-scope-execution/readiness/bundle/primary-semantics hardening and sole canonical top-level continuity proof enforcement.

## Command

```bash
cargo run -p ucf-ops -- v18 gate --out ./out/v18_gate_report.json
```

## Exit codes

- `0`: PASS
- `2`: FAIL

## PASS guarantees

A PASS certifies all required v18 final-consolidation conditions simultaneously:

- governance final consolidation is enforced for canonical consumers
- current `SupportedScopeExecutionV13` artifact is explicit, present, and coherent with applied scope + final-consolidated governance chain
- readiness final consolidation is enforced for canonical consumers
- bundle final consolidation is enforced for canonical export consumers
- primary-semantics final consolidation is enforced for canonical surfaces
- exactly one final-consolidated canonical authoritative top-level continuity proof (`canonical-final-consolidation-continuity-sweep`) is active and PASS
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
- `FAIL`: missing required artifact/surface or inconsistent/stale final-consolidation evidence.

## Scope authority note

The gate is scope-conservative: the **current applied supported scope** from authoritative applied-scope + supported-scope execution v13 artifacts is the only scope used for v18 decisions.

## Phase intent

v18 is a final-consolidation and sole-top-level-continuity hardening phase, not a compute-capability expansion phase.

## Post-v18 continuation

After v18 gate PASS, continue at Prompt 360 via `docs/next_10_prompts.md`.
