# Health Endpoint v1

`ucf-gateway` exposes a local-only, bounded health surface via protobuf request `HealthRequest` and response `HealthResponseV1`.

## Schema (`HealthResponseV1`)

- `schema_version` (`1`)
- `status` (`OK | DEGRADED | FAIL`)
- `run_id`
- `strict_mode` (bool)
- `policy_graph_digest_prefix` (hex prefix)
- `manifest_digest_prefix` (hex prefix)
- `drift_status` (`UNKNOWN | OK | DEGRADED`)
- `emergency_active` (bool)
- `last_tick_age_ms` (`u64`)
- `active_slots_summary` (bounded string)
- `recent_alarm_counts`:
  - `drift_alarms`
  - `violations`

No payload bodies, secrets, or stack traces are returned.

## Status semantics

- `OK`: no emergency and no recent drift/violation alarms.
- `DEGRADED`: no emergency but drift alarms or violations exist.
- `FAIL`: emergency is active.

## Auth and transport

- Local IPC only (Unix socket / local TCP loopback / named pipe abstraction).
- `test` / `prod`: token capability `health:read` required.
- `dev`: empty token is allowed for health probes with warning.

## Operator actions

- `OK`: continue normal operation.
- `DEGRADED`: inspect drift + violation logs and readiness gate.
- `FAIL`: enter emergency playbook and investigate runtime emergency triggers.

## `ucf-ops health check`

`ucf-ops health check` queries gateway health and writes `./out/health.json`.

Exit codes:

- `0`: OK
- `2`: DEGRADED
- `3`: FAIL
