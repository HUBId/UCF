# No Hidden Network v1

`offline-first` in UCF means default runtime execution must not pull hidden outbound networking paths from dependencies or transport configuration.

## What is scanned

1. **Dependency closure scan** (`ucf-ops audit net-deps`)
   - Reads `cargo metadata --format-version 1 --locked --offline`.
   - If offline metadata resolution cannot proceed due missing local registry cache, it falls back to local `Cargo.lock` graph parsing (still offline).
   - Loads `docs/network_allowlist.toml`.
   - Traverses runtime crate closures (default features only).
   - Fails if forbidden network crates appear in runtime closure.

2. **Runtime transport guard + socket audit (best effort)**
   - Gateway defaults to local IPC (Unix socket on Linux/macOS, named pipe on Windows).
   - TCP mode is allowed only with explicit `UCF_GATEWAY_BIND` loopback endpoint.
   - In strict mode (`UCF_STRICT_MODE=1`) unauthorized/non-loopback transport attempts are recorded and rejected.
   - Linux best-effort socket scan inspects `/proc/net/tcp` and records non-loopback established sockets when seen.
   - Windows support is transport-guard only; `/proc/net/tcp` scan is skipped.

## Allowlist rules

See `docs/network_allowlist.toml`.

- `runtime_crates`: roots that must remain network-neutral by default.
- `forbidden_crates`: network client/server crates blocked in runtime closure.
- `allowed_feature_notes`: documented non-runtime feature lanes where networking may exist.
- `exempt_runtime_edges`: policy-reviewed exceptions (`root_crate`, `forbidden_crate`, `reason`).

## Local usage

```bash
cargo run -p ucf-ops -- audit net-deps --out ./out/net_deps.json
```

Pass condition:
- `violations = []`.

Fail condition:
- each violation includes:
  - runtime root crate
  - forbidden crate
  - deterministic path (`A -> B -> forbidden`)
  - remediation hints

## Remediation guidance

1. Feature-gate the network dependency so it is not in default runtime features.
2. Move networking behavior into an ops-only or explicitly non-runtime crate.
3. Add an exemption only with policy intent and documented reason.

## Limitations

- Dependency scan is graph-based and conservative; false positives are acceptable and should be made actionable via allowlist/exemption updates.
- Runtime socket audit is best effort:
  - Linux: `/proc/net/tcp` visibility only.
  - Windows: no `/proc` scan; strict transport guard still enforced.
