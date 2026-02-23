# Compute Sandbox v1

## Scope
`v1` adds pragmatic runtime controls for compute stages while defaulting to in-process execution.

## What it enforces
- Capability-based file IO via `IoCaps` in runtime paths.
- Bounded IPC (`<=64KB`) for stage worker messages.
- Optional per-stage worker isolation using `UCF_STAGE_ISOLATION=off|llm|ebm|all`.
- Linux worker limits: `RLIMIT_AS`, `RLIMIT_CPU`, `RLIMIT_NOFILE`, core dumps disabled.
- Best-effort network denial in worker (`unshare(CLONE_NEWNET)` on Linux).
- Crash containment: worker failures degrade to safe in-process refusal behavior.

## What it does not enforce (yet)
- Full seccomp profile.
- Containerized/user-namespace hard isolation (planned v2).
- Large tensor IPC transfer (explicitly disallowed in v1).

## Enable worker mode
```bash
export UCF_STAGE_ISOLATION=llm
export UCF_STAGE_WORKER_BIN=ucf-stage-worker
```

## Knobs
- `UCF_STAGE_ISOLATION`: stage routing toggle.
- `UCF_STAGE_WORKER_BIN`: worker executable path.

## Failure modes and audit trail
- IO checks produce deterministic denial reasons and path digests.
- Worker crash or nonzero exit degrades stage output safely and continues runtime flow.
- Existing compute budget violation records remain the primary budget audit channel.
