# Compute Runtime Modes & Deployment Profiles (v1)

This document defines the canonical runtime configuration split for `ucf-compute`.

## Canonical runtime modes

- `production`
  - For real onboarding/production-intent compute paths.
  - Requires `UCF_COMPUTE_BACKEND=burn|worker`.
  - Rejects diagnostic-only compare/shadow settings.
- `diagnostic`
  - For compare/shadow investigations and rollout diagnostics.
  - Allows shadow/compare enablement.
- `test`
  - For local/dev/test paths (`stub`, compatibility lanes, fixtures).

Set via `UCF_RUNTIME_MODE=production|diagnostic|test`.

If not set, runtime mode defaults are deterministic by backend kind:
- `burn|worker` -> `production`
- `stub|candle` -> `test`

## Canonical deployment profiles

- `local_only`
  - Single-node/local execution path.
- `multi_worker`
  - Worker execution topology.
  - Requires `UCF_COMPUTE_BACKEND=worker`.

Set via `UCF_DEPLOYMENT_PROFILE=local_only|multi_worker`.

If not set, deployment defaults are deterministic by backend kind:
- `worker` -> `multi_worker`
- all others -> `local_only`

## Explicit misconfiguration handling

`RuntimeProfile::from_env(...)` fail-closes invalid combinations as `ComputeError::InvalidInput`:

- invalid runtime mode or deployment profile values
- unsupported combination (`multi_worker` without worker backend, or worker backend with `local_only`)
- production mode with diagnostic-only compare/shadow configuration
- production mode without production-intent backend (`burn|worker`)

## Ops visibility

`RuntimeOpsSnapshot` now includes:
- active `runtime_mode`
- active `deployment_profile`
- `diagnostic_flags` (`compare_enabled`, `shadow_enabled`, `slot_shadow_enabled`)

This keeps runtime config state visible in canonical ops/runtime snapshots without introducing a separate config platform.

## Deliberate boundaries

- No environment-management suite.
- No release/deployment orchestration layer.
- No profile matrix explosion: only the minimal modes/profiles above.
