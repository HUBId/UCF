# UCF End State v1

This document is the single source of truth for operators and reviewers on what UCF guarantees, what it does **not** guarantee, and how to run it safely in a hardware-neutral way.

## 1) Overview

UCF is a policy-governed runtime with deterministic controls around decisions, tools, model/policy binding, and audit evidence. It is designed to run offline-first and to produce verifiable artifacts for each run.

This page summarizes the operational end state and links to the normative docs:

- Strict mode semantics: [`docs/strict_mode.md`](strict_mode.md)
- Tool plan lifecycle and controls: [`docs/tool_plans_v1.md`](tool_plans_v1.md)
- Policy pack structure and overlays: [`docs/policy_packs.md`](policy_packs.md)
- Weight governance and rollback: [`docs/weights_lifecycle.md`](weights_lifecycle.md)
- Proof-carrying logs and digest checks: [`docs/proof_carrying_logs.md`](proof_carrying_logs.md)
- Attested run artifacts: [`docs/attested_runs.md`](attested_runs.md)
- Reproducibility bundle: [`docs/repro_pack.md`](repro_pack.md)
- Bug report bundle: [`docs/bug_report_kit.md`](bug_report_kit.md)
- Alerts and response signals: [`docs/alerts_v1.md`](alerts_v1.md)
- Drift budget and thresholds: [`docs/drift_budget.md`](drift_budget.md)
- Preflight checks: [`docs/preflight.md`](preflight.md)
- Portable deployment workflow: [`docs/deploy_portable.md`](deploy_portable.md)

## 2) Safety invariants

The following invariants define the intended safety envelope:

- **Offline-first operation**: standard bringup, checks, and verification use repository-local code, configs, policies, and artifacts.
- **No decision, no action**: action execution must be tied to explicit decision outputs and policy-permitted paths.
- **Tool 2PC discipline**: mutating tool actions follow a staged prepare/commit model; execution without valid commit conditions is rejected.
- **Strict mode enforcement**: strict-mode violations fail closed (do not silently downgrade to permissive behavior).
- **Policy graph digest binding**: run execution is bound to policy graph digests so that policy identity is explicit and auditable.
- **Hash-locked weights**: model weight identity is pinned by content digests for promotion, runtime use, and rollback.
- **Audit evidence chain**: logs are emitted as Merkle segments and run certificates to support post-run verification.
- **Determinism lock contract**: deterministic guarantees are exact for the toy baseline; for other backends UCF enforces a bounded envelope and explicit drift checks.

## 3) Determinism model

UCF uses a two-level determinism model:

- **Toy baseline**: exact reproducibility under the documented lock constraints.
- **Real backend envelope**: controlled, measurable variance with configured drift budgets and policy-enforced thresholds.

Operator references:

- Determinism lock constraints: [`docs/determinism_lock.md`](determinism_lock.md)
- Drift budget policy and handling: [`docs/drift_budget.md`](drift_budget.md)
- Alert semantics for drift-related incidents: [`docs/alerts_v1.md`](alerts_v1.md)

## 4) Policy and model governance

### Policy governance

- Policy packs and overlays define what is permitted.
- Policy graph digests are part of run identity and verification.
- Promotion and rollback use versioned artifacts and explicit checks.

References:

- [`docs/policy_packs.md`](policy_packs.md)
- [`docs/proof_carrying_logs.md`](proof_carrying_logs.md)

### Model/weights governance

- Runtime model slots are bound to hash-identified weight artifacts.
- Promotion requires lifecycle controls and evidence.
- Rollback restores a previously validated digest-pinned artifact set.

References:

- [`docs/weights_lifecycle.md`](weights_lifecycle.md)
- [`docs/attested_runs.md`](attested_runs.md)

## 5) Audit and verification

UCF provides verification-ready artifacts rather than implicit trust:

- Merkle-segmented audit logs.
- Run certificates that bind configuration, policy/model identities, and outputs.
- Repro/bug bundles to support deterministic replay and diagnosis.

Canonical local checks:

```bash
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json
cargo run -p ucf-ops -- preflight --profile test --out ./out/preflight_report.json
```

Operational docs:

- [`docs/proof_carrying_logs.md`](proof_carrying_logs.md)
- [`docs/attested_runs.md`](attested_runs.md)
- [`docs/repro_pack.md`](repro_pack.md)
- [`docs/bug_report_kit.md`](bug_report_kit.md)

## 6) Operating procedures (summary)

### A. Normal start (portable)

1. Prepare a portable bundle and select target profile.
2. Run preflight checks.
3. Run readiness gate.
4. Start or continue runtime operations with audit collection enabled.

References:

- [`docs/deploy_portable.md`](deploy_portable.md)
- [`docs/preflight.md`](preflight.md)
- [`docs/attested_runs.md`](attested_runs.md)

### B. Health check

- Use readiness and alert signals as primary status indicators.
- Treat strict-mode and policy-binding failures as blocking until remediated.

References:

- [`docs/readiness_gate.md`](readiness_gate.md)
- [`docs/alerts_v1.md`](alerts_v1.md)
- [`docs/strict_mode.md`](strict_mode.md)

### C. Readiness gate

- Execute the readiness gate for the active profile before release or promotion.
- Store outputs in `./out/` according to artifact conventions.

Reference:

- [`docs/readiness_gate.md`](readiness_gate.md)
- [`docs/artifact_convention_v0.md`](artifact_convention_v0.md)

### D. Preflight

- Run preflight before start/restart when policy/model/config changes are present.
- Block on any preflight hard-fail condition.

Reference:

- [`docs/preflight.md`](preflight.md)

## 7) Incident response (summary)

### A. Strict failure

1. Halt affected flow.
2. Preserve logs and run certificate artifacts.
3. Diagnose against strict-mode rules.
4. Resume only after passing preflight and readiness gate.

References:

- [`docs/strict_mode.md`](strict_mode.md)
- [`docs/proof_carrying_logs.md`](proof_carrying_logs.md)

### B. Drift alarms

1. Confirm alarm type and threshold breach.
2. Compare against configured drift budget.
3. If needed, revert to known-good policy/weights digest set.
4. Produce repro and bug report bundles.

References:

- [`docs/drift_budget.md`](drift_budget.md)
- [`docs/alerts_v1.md`](alerts_v1.md)
- [`docs/repro_pack.md`](repro_pack.md)
- [`docs/bug_report_kit.md`](bug_report_kit.md)

### C. Gateway abuse or unsafe tool path attempts

1. Quarantine the affected request/run context.
2. Verify tool-plan state transitions (prepare/commit evidence).
3. Confirm policy graph digest binding and deny unauthorized path.
4. Escalate with full evidence bundle.

References:

- [`docs/tool_plans_v1.md`](tool_plans_v1.md)
- [`docs/policy_packs.md`](policy_packs.md)
- [`docs/proof_carrying_logs.md`](proof_carrying_logs.md)

### D. Rollback (weights and/or policy)

1. Select a previously validated digest-pinned target.
2. Execute rollback per lifecycle procedure.
3. Re-run preflight and readiness gate.
4. Record attested run evidence after rollback.

References:

- [`docs/weights_lifecycle.md`](weights_lifecycle.md)
- [`docs/policy_packs.md`](policy_packs.md)
- [`docs/attested_runs.md`](attested_runs.md)

## 8) What UCF does **not** guarantee

To avoid over-claiming, UCF explicitly does **not** guarantee the following:

- **No claim of alignment perfection**: policy controls reduce risk but cannot prove universal correctness or harmlessness.
- **No universal hardware-level determinism guarantee**: exact determinism across all compute hardware/backends is not guaranteed.
- **No claim of proving consciousness metrics**: UCF does not claim to establish “true IIT/phi” or equivalent metaphysical guarantees.

## 9) Appendix: canonical commands

All commands below are repository-local and offline-capable in standard environments.

```bash
# Full workspace tests
cargo test --workspace

# Formatting and lint discipline
cargo fmt --all
cargo clippy --workspace --all-targets -- -D warnings

# Docs + readiness gates
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json

# Optional adversarial/policy/spec checks
cargo run -p ucf-ops -- adversarial-run --suite v1 --out ./out/adversarial_report.json
cargo run -p ucf-ops -- policy validate --pack policies/packs/base_v1 --overlay policies/packs/overlays/test
cargo run -p ucf-ops -- spec snapshot --policy policies/packs/base_v1 --overlay policies/packs/overlays/test --out docs/spec_snapshot.md
```
