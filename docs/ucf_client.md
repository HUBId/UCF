# ucf-client (Minimal Local Client v1)

`ucf-client` is a local-only CLI for operating the `ucf-gateway` API.

## Local safety constraints

- Endpoint defaults to localhost (`unix:///tmp/ucf_gateway_v1.sock` on unix, loopback tcp on non-unix).
- Non-loopback TCP endpoints are rejected.
- Output is bounded (`stream --max` and `ess query --last` are capped at 64).

## Commands

```bash
ucf-client submit --fixture fixtures/client/controlframe_min.json [--endpoint ...] [--auth ...]
ucf-client stream --max 10 [--endpoint ...] [--auth ...]
ucf-client ess query --last 32 [--endpoint ...] [--auth ...]
ucf-client report explain-tick --t 123 [--endpoint ...] [--auth ...]
ucf-client report readiness-gate --latest [--endpoint ...] [--auth ...]
```

## Auth usage

Gateway hardening expects a shared local token.

1. Start gateway with token (test/prod):
   ```bash
   export UCF_GATEWAY_TOKEN=test-token
   ```
2. Pass the same token through the client:
   ```bash
   export UCF_CLIENT_AUTH=test-token
   ```
3. Requests without valid auth return `ERR_AUTH_DENIED`.

## Deterministic fixtures

- `fixtures/client/controlframe_min.json`: minimal deterministic control frame.
- `fixtures/client/controlframe_toolintent.json`: tool-intent style fixture intended for deny-policy smoke paths.

## Operator smoke workflow

1. Ensure gateway is running locally (unix socket or loopback TCP).
2. Set auth token:
   ```bash
   export UCF_CLIENT_AUTH=test-token
   ```
3. Run smoke script:
   ```bash
   scripts/client_smoke_test.sh
   ```
4. Inspect outputs in `./out/client_smoke/`.

Windows PowerShell:

```powershell
./scripts/client_smoke_test.ps1
```

## Bringup/readiness integration (optional)

You can run the client smoke script after local bringup and before readiness review:

```bash
cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_scenario_a.json --ticks 32 --out ./out
scripts/client_smoke_test.sh ./out/client_smoke
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json
```
