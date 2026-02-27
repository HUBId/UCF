## Summary
- What changed?
- Why now?

## Affected modules
- Runtime:
- Ops/Tooling:
- Policies:
- Docs:

## Gates run (paste outputs or attach artifacts)
> Use deterministic paths under `./out/` where possible.

### Readiness gate report
Command:
```bash
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json
```
Artifact / output:

### Adversarial report
Command:
```bash
cargo run -p ucf-ops -- adversarial-run --suite v1 --out ./out/adversarial_report.json
```
Artifact / output:

### Docs lint report
Command:
```bash
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
```
Artifact / output:

### Spec snapshot diff (if any)
Command:
```bash
cargo run -p ucf-ops -- spec snapshot --policy policies/packs/base_v1 --overlay policies/packs/overlays/test --out docs/spec_snapshot.md
```
Diff / note:

### Policy validate report
Command:
```bash
cargo run -p ucf-ops -- policy validate --pack policies/packs/base_v1 --overlay policies/packs/overlays/test
```
Output:

## Risk assessment
- Risk level (low/medium/high):
- Primary failure mode:
- Blast radius:

## Rollback plan
- Required for runtime, policy, weights, or toolchain changes.
- Include exact rollback commit/tag and any data/schema rollback steps.

## Checklist
- [ ] No network used for this change.
- [ ] Determinism lock respected (`docs/determinism_lock.md`).
- [ ] Tool changes respect two-phase commit model.
- [ ] Data governance constraints respected (`docs/data_governance.md`).
- [ ] Gate artifacts are attached or pasted above.
