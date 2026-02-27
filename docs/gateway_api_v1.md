# Gateway API v1

`ucf-gateway` defines a local-only interoperability boundary for UCF.

## Scope
- Submit `ControlFrame` inputs.
- Receive `DecisionEvent` outputs.
- Query ESS summaries.
- Retrieve operator reports (`explain-tick`, `readiness-gate`).

## Transport
- Default v1 bind:
  - Unix domain socket on Linux/Unix (`/tmp/ucf_gateway_v1.sock`).
  - TCP localhost fallback on non-Unix (`127.0.0.1:<port>`).
- No remote bind in v1.
- Frame codec: length-delimited protobuf (`u32 little-endian` + protobuf payload).

## Version negotiation
1. Client sends `HandshakeRequest` with:
   - `schema_version` (gateway schema),
   - `supported_versions` list,
   - local auth token.
2. Server returns highest supported intersection in `selected_version`.
3. If no overlap, gateway returns stable error code `ERR_VERSION_MISMATCH`.

## Auth model (deny-by-default)
- `UCF_GATEWAY_TOKEN` is required in `test`/`prod` mode.
- In `dev` mode, missing `UCF_GATEWAY_TOKEN` falls back to `dev-token` and emits a warning.
- Tokens are validated by hash (`hash_token`) and only token hashes are stored in config/policy data.
- Sensitive/state-changing endpoints require auth capability checks:
  - `submit`
  - `ess:read`
  - `report:read`

## Deterministic rate limits
Token bucket limits are deterministic integer arithmetic and policy-configurable per endpoint:
- `submit`: default `5/s`
- `ess_query`: default `10/s`
- `report`: default `2/s`

Rate-limit exceed returns `ERR_RATE_LIMITED` and emits a `GatewayAbuseRecord`.

## Request caps and schema checks
- Maximum framed message size: `256 KiB`.
- Submit payload max: `4096` bytes UTF-8.
- Intent summary max: `256` chars.
- List responses are capped at `64` entries.
- `schema_version` must be `1`.
- `run_id` and `policy_graph_digest_prefix` are validated against gateway config.

## Safe error surface
Gateway error responses expose only:
- `code` (stable)
- `message` (bounded, redaction-safe)
- `request_id`

Stable gateway error codes:
- `ERR_AUTH_DENIED` (`1001`)
- `ERR_RATE_LIMITED` (`1002`)
- `ERR_SCHEMA_INVALID` (`1003`)
- `ERR_POLICY_DENIED` (`1004`)
- `ERR_INTERNAL` (`1500`)
- `ERR_UNAVAILABLE` (`1501`)

No internal stack traces are returned.

## Access and abuse records
Every successful request appends a `GatewayAccessRecord` JSONL entry:
- `schema_version`
- `endpoint`
- `t_ms`
- `status`
- `client_id_digest_prefix`
- `request_id`

Every denied/abusive request appends a `GatewayAbuseRecord` JSONL entry:
- `schema_version`
- `t_ms`
- `endpoint`
- `client_id_digest_prefix`
- `reason_code` (`AuthFail`, `RateLimit`, `Malformed`, `TooLarge`, `VersionMismatch`, ...)
- `request_digest_prefix`
- `request_id`

No payload or PII is persisted.

## Endpoints (protobuf)
Defined in `proto/ucf_gateway_v1.proto`:
- `ControlFrameSubmitRequest/Response`
- `DecisionStreamSubscribeRequest/Response`
- `EssQueryRequest/Response`
- `ReportRequest/Response`
- `HandshakeRequest/Response`

## Minimal local flow
1. Handshake.
2. Submit frame.
3. Subscribe decisions.
4. Query ESS summaries.
5. Request report.
