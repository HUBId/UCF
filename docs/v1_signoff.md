# v1 Sign-off Gate (Scaffolding, Offline)

The v1 gate provides a **single PASS/FAIL sign-off** for the v1 scaffolding layer without requiring real model execution.

## Guarantees

`ucf-ops v1 gate` checks, in fixed order:

1. `v0_gate_pass`
2. `models_manifest_verify` (SKIP only when `models/` is absent)
3. `probes_dummy_pass` (deterministic probe set: `world_jepa`, `sae`, `ssm`)
4. `shadow_no_decision_impact` (baseline vs shadow decision digest parity)
5. `drift_budget_present_if_shadow`
6. `alerts_present`
7. `strict_check_v1`
8. `portability_scans` (optional SKIP if scan unavailable)

Each check emits bounded evidence digest prefixes and a remediation hint.

## Run

```bash
cargo run -p ucf-ops -- v1 gate --out ./out/v1_gate_report.json
```

- Exit code `0`: overall PASS.
- Exit code `2`: overall FAIL.

## Scope and Non-Goals

Included:
- Offline-only scaffolding validation.
- Deterministic JSON schema and check ordering.
- Conservative failure behavior for required checks.

Not included:
- Real-model quality/performance evaluation.
- Online/internet-dependent checks.
- Production shadow window tuning.
