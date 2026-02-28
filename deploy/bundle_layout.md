# Portable Runtime Bundle Layout (v1)

Canonical portable bundle root (`bundle_root/`):

```text
bundle_root/
  bin/
    ucf-runtime        # optional (if built)
    ucf-ops
    ucf-gateway        # optional
    ucf-client         # optional
  configs/
    dev.toml
    test.toml
    prod.toml
  policies/
    packs/...
    manifest.toml
  models/
    manifest.toml
    promoted/...
  data/
    ess/
  out/
    <run_id>/...
  VERSION.txt
```

## Path model

- Bundle-first and relative-by-default.
- `--bundle <path>` (for `ucf-ops`) changes working directory to bundle root.
- Runtime path roots can be pinned with `UCF_BUNDLE_ROOT=<path>`.
- External overrides remain possible through existing environment variables (`UCF_PROFILE`, `UCF_MODEL_MANIFEST`, `UCF_POLICY_OVERLAY`, ...).

## `VERSION.txt`

`deploy/scripts/build_bundle.py` writes deterministic `VERSION.txt` with:

- `code_version_tag` (`git rev-parse --short=12 HEAD` when available)
- `policy_graph_digest` (policy validate output when possible; deterministic tree digest fallback)
- `manifest_digest` (`sha256(models/manifest.toml)`)
- `profile`

## Determinism and offline behavior

- Bundle build does not download artifacts.
- Copy set and digest generation are deterministic (sorted file traversal).
- Runtime startup validates policy bundle hash and policy graph digest using existing strict checks.
