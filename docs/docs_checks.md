# Docs-as-Checks (CI blocking)

`ucf-ops docs lint` validates documentation artifacts that are part of the runtime specification.

## Command

```bash
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
```

Modes:
- `--strict`: fail on all lint failures (CI default).
- `--warn`: downgrade module-map mismatch warnings to non-blocking.

## Checks

1. **Spec snapshot up-to-date**
   - Regenerates snapshot in a temp file and compares against `docs/spec_snapshot.md`.
   - Fails when committed snapshot is stale.
   - Remediation:
     ```bash
     cargo run -p ucf-ops -- spec snapshot --policy policies/packs/base_v1 --overlay policies/packs/overlays/test --out docs/spec_snapshot.md
     git add docs/spec_snapshot.md
     ```

2. **Policy packs validate (base + overlay)**
   - Runs policy merge/validation for `policies/packs/base_v1` + `policies/packs/overlays/test`.
   - Fails on merge errors, schema issues, or unknown keys.
   - Remediation:
     ```bash
     cargo run -p ucf-ops -- policy validate --pack policies/packs/base_v1 --overlay policies/packs/overlays/test
     ```

3. **Prompt index integrity**
   - Parses prompt IDs from `docs/prompt_series_index.md`.
   - Enforces unique and strictly increasing IDs.
   - Validates `PROMPT <N> —` heading format when prompt headings are present.

4. **Module map best-effort cargo metadata consistency**
   - Parses crate-like entries in `docs/module_map.md`.
   - Compares them with local `cargo metadata --no-deps --format-version 1` package names.
   - `--strict`: mismatch fails.
   - `--warn`: mismatch warns and continues.

5. **Hardware-neutral docs guardrail**
   - Scans `docs/prompt_series_index.md`, `docs/prompt_rulebook.md`, and `docs/deploy_portable.md` for obvious hardware-specific terms.
   - Fails when forbidden terms appear in core docs outside clearly marked history sections.
   - Allows deploy/history mentions as warnings.
   - Remediation:
     ```bash
     # Replace hardware/vendor wording with DeviceProfile + explicit budgets
     cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
     ```


6. **Artifact schema snapshots up-to-date**
   - Regenerates shape snapshots for covered v3/v4/v5 governance/export artifacts and compares them with committed files in `docs/artifact_schema_snapshots/`.
   - Classifies drift conservatively as `ADDITIVE`, `BREAKING`, or `UNKNOWN`; strict lint fails on drift.
   - Remediation:
     ```bash
     cargo run -p ucf-ops -- spec artifact-schemas --out docs/artifact_schema_snapshots
     cargo run -p ucf-ops -- spec artifact-schemas-check --out ./out/artifact_schema_check.json
     git add docs/artifact_schema_snapshots
     ```

7. **Remediation registry doc up-to-date**
   - Regenerates `docs/remediation_codes_v1.md` from the canonical remediation registry and compares against the committed file.
   - Fails when generated output differs (stale registry docs are blocking).
   - Remediation:
     ```bash
     cargo run -p ucf-ops -- docs remediation-codes --out docs/remediation_codes_v1.md
     git add docs/remediation_codes_v1.md
     ```

8. **v4 docs linkage consistency**
   - Requires presence and portability/docs linkage for:
     - `docs/backend_evidence_snapshot_v4.md`
     - `docs/operator_signoff_v4.md`
     - `docs/remediation_codes_v1.md`
     - `docs/artifact_schema_snapshots.md`
   - Also requires Prompt 216 tracking in `docs/series_state_snapshot.md`.

## Report output

When `--out` is provided, lint writes deterministic JSON with per-check status and remediation hints.
