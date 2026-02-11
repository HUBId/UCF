# Sandboxing v1 Prep

This change introduces a deterministic isolation interface that keeps the default runtime **in-process** while preparing feature-gated backends for future WASM and process isolation.

## Default mode (in-proc)

No runtime flags are needed; `ucf-runtime` uses the in-process isolation runtime by default.

## Optional future runtime stubs

`ucf-runtime` now exposes two feature flags for non-default backends:

- `sandbox-wasm` (stub: returns backend disabled)
- `sandbox-proc` (stub: not implemented)

Example:

```bash
cargo test -p ucf-runtime --features sandbox-wasm
cargo test -p ucf-runtime --features sandbox-proc
```

## Capability flow in sandbox calls

1. Orchestrator issues policy capabilities from decision context.
2. A `ToolRequest` is built, then transformed into a sandbox call spec (`module/op/input`).
3. `InProcIsolationRuntime` authorizes through `ToolGate` and dispatches the handler.
4. Canonical call/reply digests are produced.
5. ESS audit chain persists `SandboxCall` and `SandboxReply` records linked to existing tool audit records.

This gives a single choke-point for tool execution that can later be swapped to WASM/process isolation without changing orchestrator-level contracts.
