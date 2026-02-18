# Capability Economics v1

This runtime overlay introduces deterministic compute-token accounting to reduce DoS risk from "free" inference.

## Cost schedule

Costs are fixed integer charges (no floating estimators) in `runtime/ucf-runtime/src/compute_economics.rs`:

- `LLM = LLM_BASE + LLM_PER_IN_TOKEN*in + LLM_PER_OUT_TOKEN*out`
- `JEPA = JEPA_BASE + JEPA_PER_DIM*dim`
- `SAE = SAE_BASE + SAE_PER_FEATURE*features`
- `SSM = SSM_BASE + SSM_PER_DIM*dim`
- `LFM = LFM_BASE + LFM_PER_DIM*dim`
- `GOVERNOR = GOVERNOR_BASE`
- `TOOL = TOOL_BASE`

## Budgets and pools

Per-session pools:

- `primary_compute_budget`
- `shadow_compute_budget`

Sub-buckets:

- `llm_window_budget` / `llm_per_tick_budget`
- `others_window_budget`

When budgets are exhausted, expensive stages are denied and runtime degrades safely.

## Governor arbitration

Per-tick caps are scaled by governor tier (higher tier => lower budget), and emergency mode forces near-minimal budgets.

## Shadow mode

Shadow consumes a dedicated shadow pool (`BudgetPool::Shadow`) so primary budget integrity is preserved.

## Persistence and replayability

ESS audit emits:

- `ComputeBudgetWindowRecord` once per window
- `ComputeBudgetViolationRecord` on denials/anomalies

These records include stage spend, balances, governor tier stats, and policy hash prefix, enabling deterministic replay of budget evolution.

## Telemetry

- `ucf_compute_tokens_spent_total{stage=...}`
- `ucf_compute_tokens_remaining{pool=primary|shadow}`
- `ucf_budget_denials_total{stage=...}`

## Tuning for NUC Ultra 7

1. Start with small budgets in dev (`UCF_PRIMARY_COMPUTE_BUDGET`, `UCF_LLM_COMPUTE_BUDGET`).
2. Replay adversarial prompt bursts and verify safe refusals.
3. Increase only after observing stable p95 latency and no budget starvation on critical safety paths.
