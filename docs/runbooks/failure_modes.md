# Failure Modes

## Compute budget exceeded
- Symptom: high risk/low confidence or degraded compute summary.
- Action: switch budget profile (`default` -> `tight` only for stress tests), keep deterministic seed fixed.

## Sandbox denied / rate-limited
- Symptom: tool calls blocked by policy/toolgate.
- Action: confirm intended capability grant path; keep `capabilities_default=deny` and explicitly scope exceptions.

## Worker crash (proc runtime)
- Symptom: sandbox runtime check fails when non-inproc runtime configured.
- Action: set `isolation_runtime=inproc` for offline recovery, then investigate worker process separately.

## ESS read failure / corruption
- Symptom: `ess_health` fails.
- Action: rerun `bringup --demo` to regenerate fixture; keep exported bugreports for forensic replay.

## Schema version mismatch
- Symptom: `verify-bugreport` fails with schema incompatibility.
- Action: regenerate bugreport from the same build tag and replay with matching binary.
