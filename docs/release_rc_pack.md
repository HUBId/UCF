# Release Candidate Pack (v1)

The RC pack command creates one offline-transferable artifact containing:

- release binaries
- verification reports
- portable bundle
- signed RC manifest
- SHA256 list

## Build command

```bash
cargo run -p ucf-ops -- release build-rc --version v1.0-rc1 --profile prod --out ./out/rc
```

Fast CI smoke mode:

```bash
cargo run -p ucf-ops -- release build-rc --version v1.0-rc1 --profile prod --out ./out/rc --fast
```

## Included verification steps

- `cargo build --release`
- `ucf-ops docs lint --strict`
- `ucf-ops spec snapshot` with no resulting diff
- `ucf-ops readiness-gate --profile test`
- `ucf-ops adversarial-run --suite v1` (skipped in `--fast`)
- `ucf-ops goldens verify --all --os <current>` (skipped in `--fast`)
- `ucf-ops strict check`

If a step fails, `build-rc` stops and reports the failing step + report path. Partial artifacts remain in `./out/rc`.

## RC zip content

`ucf_rc_<version>_<digest>.zip` contains:

- `bundle/`
- `reports/`
- `RC_MANIFEST.json`
- `RC_MANIFEST.sig`
- `SHA256SUMS.txt`

## Offline verification

1. Verify checksums:
   - compare each file hash against `SHA256SUMS.txt`.
2. Verify signature:
   - compute digest from `RC_MANIFEST.json` canonical fields (`rc_digest` excluded),
   - verify `RC_MANIFEST.sig` with `.ucf/keys/attestation_ed25519.pub`.
3. Confirm artifact name digest suffix matches manifest-derived digest prefix.

## Manual next steps (not automated)

- create release tag
- publish bundle and RC zip through distribution channel
- hand off release checklist and verification reports
