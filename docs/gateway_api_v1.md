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
3. If no overlap, gateway returns error code `426` (`unsupported version`).

## Auth and capability model (deny-by-default)
All endpoints require a local auth token. Tokens map to capability strings:
- `submit`
- `subscribe`
- `ess:read`
- `report:read`

Sensitive endpoints (`ess_query`, `report`) are gated by capability and the configured `policy_graph_digest_prefix`.

## Data governance and redaction
- Submit payloads are accepted as bounded UTF-8 bytes only.
- Decision responses expose only `rationale_redacted` (truncated text), never raw internal structs.
- ESS endpoint returns summaries only (`experience_id`, `tick`, `kind`, `corr_id`).

## Bounded messages
- Maximum framed message size: `128 KiB`.
- Submit payload max: `4096` bytes UTF-8.
- Intent summary max: `256` chars.
- List responses are capped at `64` entries.

## Access records
Every request appends a `GatewayAccessRecord` JSONL entry:
- `schema_version`
- `endpoint`
- `t_ms`
- `status`
- `client_id_digest`

Path is configured (default test setup uses `gateway_access_records.jsonl`).

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

