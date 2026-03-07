# DriftBudget v1

DriftBudget v1 defines deterministic drift envelopes per slot. Budgets are loaded **only** from policy packs (`drift_budget.toml`) and merged via the policy graph.

## Schema

`drift_budget.toml` (`schema_version = 1`) contains ordered `entries` with:

- `slot_id`
- `window_size_ticks`
- `scalar_delta_max_q` (map of scalar-name -> UQ0_16 delta cap)
- `invalid_rate_max_q`
- `latency_p95_max_ms`
- `digest_mismatch_rate_max_q` (optional)
- `severity.severe_fields`
- `action_on_severe` (`disable_shadow` or `none`)

The policy graph computes and carries `drift_budget_digest` as canonical digest for DriftBudget content.

## Drift evaluation

At compare-window close, the runtime evaluates each `SlotCompareWindowRecordV1` against the slot budget:

- invalid rate from `invalid_shadow_count / sample_count`
- digest mismatch rate from `digest_mismatch_count / sample_count`
- scalar deltas from quantized primary/shadow scalar deltas (e.g. risk mean/p95)

For each breached window, runtime emits `DriftAlarmRecordV1` into ESS with:

- slot/window id
- breached fields
- observed quantized values
- severity (`warn`/`severe`)
- `action_taken`

## Tightening-only actions

v1 is conservative and tightening-only:

- severe + `action_on_severe = disable_shadow` => shadow is auto-disabled for the slot and `ShadowDisableRecord` is emitted
- rollback is recommendation-only in ops report
- no auto-promote or auto-rollback

`UCF_SHADOW_DISABLE_TO_OFF=1` may additionally force slot mode to `Off`; default keeps conservative behavior.

## Ops drift report

`ucf-ops drift report --run <id> --windows <n> --out ./out/drift_report.json`

Output is bounded and deterministic:

- per-slot status (`OK`, `WARN`, `SEVERE`)
- bounded window slice (by `--windows` and budget `window_size_ticks`)
- last alarms (max 20)
- deterministic recommended actions (`disable_shadow`, `recommend_rollback`, `none`)

No payload data is included in drift report records.
