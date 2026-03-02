# Canonical Branch Policy

This repository uses a single **stable branch**, explicit **release-candidate branches**, and short-lived **feature branches**.

## Branch roles

- `main`
  - Always green.
  - Required gates before merge:
    - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
    - `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json`
    - `cargo run -p ucf-ops -- audit hardware-scan`
    - `cargo run -p ucf-ops -- audit path-scan`
    - `cargo run -p ucf-ops -- portability check --out ./out/portability.json`
- `vX.Y-rcN` (release candidate branches, for example `v1.0-rc1`, `v1.1-rc1`)
  - Stabilization only (bugfixes, release docs, signoff artifacts).
  - Must keep `docs/spec_snapshot.md` and release checklists aligned with changes.
- Feature branches
  - Branch from `main`.
  - Naming convention:
    - `feat/<area>-<short-description>`
    - `fix/<area>-<short-description>`
    - `docs/<area>-<short-description>`
    - `ops/<area>-<short-description>`

## Merge policy

- Preferred strategy: **Squash merge**.
- Rebase feature branches on latest `main` before merge.
- Every PR must include gate artifacts in the PR template:
  - readiness-gate report
  - adversarial report
  - docs lint report
  - spec snapshot diff (if any)
  - policy validate output
- Runtime and ops changes must not merge without a rollback note.

## Change-class requirements

- Runtime changes (`runtime/`, `core/`, `domains/`, `protocol/`):
  - Run `cargo test --workspace`.
  - Run readiness gate and adversarial run.
  - Provide rollback plan in PR.
- Ops/policy/tooling changes (`runtime/ucf-ops/`, `policies/`, `scripts/`, `release/`):
  - Run docs lint and policy validate.
  - Attach generated reports in `./out/` paths.
- Docs-only changes (`docs/`, `README.md`, templates, AGENTS.md):
  - Run docs lint at minimum.

