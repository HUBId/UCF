# Portable Deployment (Hardware-/OS-agnostic)

This guide describes an offline-first deployment pattern using a portable bundle.

## 1) Build a portable bundle

```bash
python deploy/scripts/build_bundle.py --target ./bundles/releases/ucf_v1 --profile prod
```

Optional archive output:

```bash
python deploy/scripts/build_bundle.py --target ./bundles/releases/ucf_v1 --profile prod --archive zip
```

## 2) Run tools from bundle root

```bash
cd ./bundles/releases/ucf_v1
./bin/ucf-ops health check --bundle . --out ./out/health.json
./bin/ucf-ops readiness-gate --bundle . --profile test --out ./out/gate.json
./bin/ucf-ops docs lint --bundle . --strict --out ./out/docs_lint.json
./bin/ucf-ops diagnostics collect --bundle . --run <run_id> --out ./out/diag.zip
```

All artifact output is expected in `./out/<run_id>/...` or `./out/*.json` depending on command.

Health checks are backed by gateway `health` endpoint and provide strict exit codes for service managers:

- `0` = OK
- `2` = DEGRADED
- `3` = FAIL

## 3) Runtime startup validation

At startup, runtime enforces policy/manifest validation fail-fast behavior:

- policy bundle verification hash lock
- policy graph digest check (`UCF_POLICY_GRAPH_DIGEST` when set)
- promoted model manifest/digest checks from existing model governance flow

For bundle-based startup, set:

```bash
export UCF_BUNDLE_ROOT="$(pwd)"
```

## 4) Offline upgrade / rollback (bundle switching)

Bundle switch layout:

```text
./bundles/current -> ./bundles/releases/<bundle_id>/
./bundles/previous -> <previous-release>
```

POSIX:

```bash
./deploy/scripts/upgrade_bundle.sh upgrade <bundle_id>
./deploy/scripts/upgrade_bundle.sh rollback <ignored_bundle_id>
```

PowerShell:

```powershell
.\deploy\scripts\upgrade_bundle.ps1 -Action upgrade -BundleId <bundle_id>
.\deploy\scripts\upgrade_bundle.ps1 -Action rollback -BundleId <bundle_id>
```

## 5) Optional OS adapters

Templates are provided (non-blocking, optional):

- `deploy/os/linux/systemd/ucf-runtime.service`
- `deploy/os/windows/ucf-runtime-service.ps1`
- `deploy/os/macos/launchd/ucf-runtime.plist`

They wrap the same portable bundle root and do not change core runtime portability.
