# Sandboxing v1 Prep

This change keeps **in-process isolation** as default and adds a first **process-isolated runtime (`proc`)** behind `sandbox-proc`.

## Runtime selection

`UCF_ISOLATION_RUNTIME` supports:

- `inproc` (default)
- `wasm` (feature-gated)
- `proc` (feature-gated)

For process mode:

- Host spawns `ucf-sandbox-worker`.
- IPC uses deterministic binary frames: `len:u32_le || envelope_bytes`.
- Envelope payloads are canonical bytes plus a blake3 payload digest.
- Worker sends a startup heartbeat with schema version + build tag.

## Why tool I/O is host-proxied

In `proc` mode the worker cannot authorize or execute tools directly. Instead:

1. Worker emits `ToolRequest` over IPC.
2. Host evaluates via `ToolGate`.
3. Host executes approved effects through the host adapter.
4. Host sends `ToolReply` summary back.

This keeps capability enforcement in one place (host), deny-by-default.

## Crash containment

If the worker dies or IPC fails mid-call, host returns a failed sandbox reply (`WORKER_CRASH`) instead of panicking.

## Useful commands

```bash
cargo test -p ucf-runtime
cargo test -p ucf-runtime --features sandbox-proc
cargo clippy -p ucf-runtime --features sandbox-proc -- -D warnings
```
