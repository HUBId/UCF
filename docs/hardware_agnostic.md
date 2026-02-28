# Hardware Agnostic Runtime (v1)

UCF runtime configuration must stay hardware-independent. This means runtime code does not assume specific machine names, model numbers, or deploy host paths.

## Platform probe

`ucf-platform` exposes a `PlatformProbe` trait with bounded `PlatformInfo`:
- `os`: windows/linux/macos/other
- `cpu_cores`: optional
- `cpu_arch`: x86_64/aarch64/other
- `mem_total_mb`: optional
- `accel`: none/cuda/metal/other as capability only
- `monotonic_clock_ok`: best-effort check

Probe behavior is offline-only and best-effort. Missing fields are represented as `None`.

## Device profiles

Set `UCF_DEVICE_PROFILE=small|medium|large` to select resource budgets.
Profiles only tune resource controls:
- compute budget profile
- timeout defaults
- LLM max tokens
- shadow sampling/window limits
- stage isolation default

Profiles do **not** change policy semantics.

## Where hardware tuning belongs

Hardware-specific tuning is allowed only via:
- policy packs / overlays
- profile config (`configs/dev.toml`, `configs/test.toml`, `configs/prod.toml`)
- environment allowlist overrides

Do not add machine-specific names (for example workstation SKUs) in runtime defaults.

## Audit and metadata

`RunMetadataRecord` stores:
- `platform_probe_summary`
- `device_profile_name`
- `device_profile_digest`

CI also runs `cargo run -p ucf-ops -- audit hardware-scan` to block hard-coded hardware assumptions in runtime crates.
