# v3 Signoff Gate (Offline, Hardware-Neutral)

## Purpose
The v3 gate is a bounded **integration readiness** gate for the currently supported real-slot set.

`PASS` certifies:
- supported real-slot scope is coherent (`world_jepa` + exactly one second slot)
- probes and shadow-ready evidence are present for both supported slots
- no-impact shadow behavior is preserved for both supported slots
- unified compare-window semantics are normalized and evidenced
- unified eligibility, strict v3, operator report, and portability/docs checks align

`PASS` does **not** certify:
- active-by-default or general active approval
- support beyond the declared two-slot scope
- GPU readiness, remote compute readiness, training readiness, or large-model production readiness

## Command
```bash
cargo run -p ucf-ops -- v3 gate --out ./out/v3_gate_report.json
```

## Exit codes
- `0`: overall `PASS`
- `2`: overall `FAIL`

## Check status interpretation
- `PASS`: required evidence exists and satisfies v3 constraints.
- `FAIL`: required evidence is missing/malformed/stale or a required invariant is violated.
- `SKIP`: optional path is unsupported/unconfigured (only used for optional burn world parity path).

## Supported-slot rule for v3
v3 scope is strictly limited to:
- `world_jepa`
- plus exactly one second supported slot (`sae` or `ssm`) declared in `docs/series_state_snapshot.md`

Ambiguous or missing second-slot declaration is a hard gate failure.

## Post-v3 continuation
After v3 gate PASS, continue at Prompt 210 via `docs/next_10_prompts.md`.
