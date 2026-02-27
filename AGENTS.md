# AGENTS.md — UCF Repository Instructions

Scope: entire repository.

## Operating rules
- Always work offline-first. Do not require network access to complete standard tasks.
- No decision, no action: do not implement behavior-changing changes without explicit policy/spec intent in repo docs.
- Keep changes deterministic and reproducible.

## Canonical commands
Run these before handing off most changes:

```bash
cargo test --workspace
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json
```

Common additional checks:

```bash
cargo run -p ucf-ops -- adversarial-run --suite v1 --out ./out/adversarial_report.json
cargo run -p ucf-ops -- policy validate --pack policies/packs/base_v1 --overlay policies/packs/overlays/test
cargo run -p ucf-ops -- spec snapshot --policy policies/packs/base_v1 --overlay policies/packs/overlays/test --out docs/spec_snapshot.md
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings
```

## Key paths
- Policy packs:
  - `policies/packs/base_v1/`
  - `policies/packs/overlays/{test,dev,prod}/`
  - `policies/manifest.toml`
- Model manifest:
  - `models/manifest.toml`
- Output/artifact convention:
  - `out/<run_id>/...` for run-specific artifacts
  - `./out/docs_lint_report.json`, `./out/gate_report.json`, `./out/adversarial_report.json`
  - See `docs/artifact_convention_v0.md`

## Determinism and safety invariants
- Use fixed-point arithmetic for safety-critical scalars where applicable; avoid float drift in policy logic.
- Preserve canonical encoding rules and schema field stability.
- Avoid nondeterministic `HashMap` iteration in externally visible or hashed outputs; sort keys or use ordered maps.
- Keep determinism lock constraints in `docs/determinism_lock.md`.

## Record schema/version updates
When adding or changing record schemas:
1. Add the schema change and version bump in the relevant crate/docs.
2. Update fixtures/golden files tied to the schema.
3. Document migration/compatibility notes in `docs/` and release checklist artifacts.
4. Re-run policy/spec/docs gates and include artifacts.

## Spec snapshot workflow
1. Regenerate snapshot:
   ```bash
   cargo run -p ucf-ops -- spec snapshot --policy policies/packs/base_v1 --overlay policies/packs/overlays/test --out docs/spec_snapshot.md
   ```
2. Validate docs checks:
   ```bash
   cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
   ```
3. Commit `docs/spec_snapshot.md` with the corresponding code/policy changes.
