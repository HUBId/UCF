# ucf-runtime

## Optional Consolidation/Geist hooks (v0)

The runtime orchestrator keeps ESS append as source-of-truth and can optionally run derived hooks after decision append.

### Flags

- `UCF_ENABLE_CONSOLIDATION_HOOK=true` enables bounded compute milestone aggregation.
- `UCF_ENABLE_GEIST_HOOK=true` enables conservative Geist macro-baseline updates from milestones.

Both hooks are disabled by default.

### Behavior

- Consolidation hook consumes decision ESS records and projects bounded `ComputeSignalSummaryView` values.
- Milestones are bounded aggregates (windowed means/counters/top-k digests only).
- Geist hook consumes milestones and persists only accepted stable updates (drift/degraded gating).
- Hook failures are counted and skipped without affecting the control loop.

## WASM sandbox runtime (feature `sandbox-wasm`)

`ucf-runtime` includes a v0 WASM backend behind `--features sandbox-wasm`.

- Runtime selection: `UCF_ISOLATION_RUNTIME=inproc|wasm` (default: `inproc`).
- WASM modules are embedded fixtures (`wasm.echo`, `wasm.tool_probe`) compiled from local WAT strings.
- Determinism guards: canonical envelope bytes, fixed memory page cap, fuel budget mapping from `SandboxBudget.work_units`, bounded input/output sizes.
- Hostcalls are explicit (`host_log`, `host_tool_request`) and deny by default when capability mapping is missing.
