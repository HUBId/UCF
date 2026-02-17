# Process Isolation v1

`worker_v1` is an optional runtime mode that isolates compute pillars into separate OS processes.

## Enable

- Backend pack mode: `UCF_BACKEND_PACK=worker_v1`
- Compute backend mode: `UCF_COMPUTE_BACKEND=worker`
- Optional worker binary path override: `UCF_WORKER_BIN=/path/to/ucf-worker`
- Optional memory cap (Linux, best effort): `UCF_WORKER_MEMORY_LIMIT_MB=512`

## IPC framing

Worker IPC uses deterministic framed msgpack:

- 4-byte LE payload length
- payload (`rmp-serde` with schema-versioned structs)
- 32-byte SHA-256 checksum over payload

The host rejects frames with length/checksum mismatch and enforces bounded payload sizes.

## Model and policy boundaries

- ToolGate/policy remains in host.
- Worker handles only stage math and bounded serialization.
- No network code path is used by worker runtime.
- Preferred model loading path is host-side control; worker currently uses embedded toy fixtures.

## Operations

- One worker process per stage (LLM/WORLD/SAE/SSM/LFM).
- Host tracks spawn/kill/restart audit records.
- On timeout or IPC failure, host kills and restarts the affected stage worker.
- Host state remains source-of-truth; worker crashes do not mutate host model state.

## Caveats

- Linux memory cap is best effort via `setrlimit(RLIMIT_AS)`.
- Network namespace sandboxing is not yet applied in v1.
- Worker mode is optional and can be disabled by selecting non-worker backend packs.
