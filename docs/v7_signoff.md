# v7 Signoff

`ucf-ops v7 gate` is the v7 closure gate for applied-scope authority hardening, reviewability truth consistency, canonical export round-trip verification, remediation/interop proof, and operator/export authority-chain coherence.

The gate is explicitly:
- offline-first
- hardware-neutral
- deterministic
- conservative/fail-closed for required surfaces

It validates consistency over the currently applied authoritative scope (`SupportedRealSlotSetV2` + `AppliedSupportedSetContextV1`).

## Command

```bash
cargo run -p ucf-ops -- v7 gate --out ./out/v7_gate_report.json
```

Exit codes:
- `0` => overall `PASS`
- `2` => overall `FAIL`

## PASS guarantees

A `PASS` means all required v7 governance/review/export/interop surfaces are coherent for the current applied supported scope:
- applied-scope authority is enforced on canonical surfaces
- supported-scope reevaluation artifact is present and consistent with current applied context
- reviewability truth is consistent across review/signoff surfaces
- export bundle round-trip check passes canonically
- remediation/interop proof passes
- operator/export authority chain is coherent
- artifact schema snapshot checks pass
- portability/docs checks pass

## PASS does not guarantee

A `PASS` does **not** mean:
- broader runtime capability
- additional slots/backends are production-ready
- any slot is auto-activated
- GPU/remote/training/large-model readiness

## Check normalization

The report schema is `V7GateReportV1` with fixed check ordering and normalized statuses.

Required checks fail-closed:
- missing required surface/artifact => `FAIL`
- applied-scope mismatch/incoherence => `FAIL`
- review/export/interop/operator chain mismatch => `FAIL`

Optional checks:
- unsupported optional path => `SKIP`
- optional legacy translation check without legacy surface evidence => `SKIP`

## Interpretation

- `PASS`: v7 hardening closure is satisfied for the applied scope.
- `FAIL`: one or more required governance/review/export/interop invariants are missing or inconsistent.
- `SKIP`: optional, explicitly unsupported/unconfigured path only.

v7 is an applied-scope/review/export/interop hardening phase, not a compute capability expansion phase.


After `ucf-ops v7 gate` PASS, continue at Prompt 250 via `docs/next_10_prompts.md`.
