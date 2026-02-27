# Tool Plugins v1 (local only)

`ToolExecute` now supports a local-only plugin adapter in the host runtime.

## Safety properties

- **Local only**: v1 uses `ToolPluginRegistry::with_builtin_stubs()` and does not load network or remote plugins.
- **Deterministic interface**: requests are bounded (`args <= 4KB`), stable (`request_id`, `plan_digest`), and responses are bounded (`preview <= 128B`).
- **Capability-token plan binding**: host checks `CapabilityTokenBinding.plan_digest == ToolExecRequest.plan_digest` before invoking a plugin.
- **Host-owned persistence**: plugins return `ToolExecResponse`; host persists `ToolRequest`/`ToolAuth`/`ToolExecution`/`Sandbox*` records.
- **Redaction-safe result storage**: host stores compact `result_digest + preview` note in `ToolExecution.error_code` (bounded summary, no raw payload dump).

## Interface

```rust
pub trait ToolPlugin {
    fn tool_id(&self) -> ToolId;
    fn tool_class(&self) -> ToolClassId;
    fn execute(
        &self,
        req: ToolExecRequest,
        caps: &CapabilityToken,
        sandbox: &SandboxEnv,
    ) -> ToolExecResponse;
}
```

`ToolExecRequest` fields:
- `request_id`
- `plan_digest`
- `args` (canonical, bounded)
- `allowed_roots`
- `max_bytes_out`
- `timeout_ms`

`ToolExecResponse` fields:
- `status` (`Ok | Denied | Timeout | Error`)
- `result_digest`
- `preview`
- `bytes_out`
- `error_code`

## Built-in deterministic stubs

- `EchoTool` (`external_api` / `external_output`): returns digest/preview based on args hash.
- `FileReadTool` (`file_read` / `memory_write`): requires allowed root id, reads via `SandboxFs`, returns digest + bounded preview.
- `MathTool` (`internal_thought` / `internal`): deterministic integer sum from canonical args.

## Adding a new plugin safely

1. Implement `ToolPlugin` in `runtime/ucf-runtime/src/tool_plugins.rs` (or sibling module).
2. Register it in `ToolPluginRegistry::with_builtin_stubs()`.
3. Keep behavior local and deterministic:
   - no network calls,
   - stable digesting,
   - bounded outputs,
   - no direct ESS writes.
4. Add tests for:
   - plan/token binding,
   - bounds,
   - deterministic repeatability.

## End-to-end (stub)

A tool intent now goes through:

1. `ToolPlan` digest creation.
2. `ToolIssue` with issued `CapabilityToken`.
3. Host binds token + plan digest and calls `run_plugin_tool(...)`.
4. Host writes `ToolAuth`, `ToolExecution`, and sandbox audit records.
