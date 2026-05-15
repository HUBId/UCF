# UCF Readiness Gate Timeout Stability Audit

## 0. Purpose

- Audit readiness-gate timeout and readiness-spine-check drift.
- Does not weaken gate criteria.
- Does not mark timed-out gate as pass.
- Does not add UCF features, replay scheduling, real-compute activation, gateway integration, capability issuance, Evidence/Archive authority changes, or Minimal Spine v1.x changes.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `3301615e3b23ad24603c47ee0b97787d2ee84097` |
| HEAD short | `3301615e` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Consolidation closure present | yes |
| `ucf-ops` present | yes |
| CI workflow present | yes |
| Nightly workflow present | yes |

Links:

- [`docs/roadmap/full_consolidation_closure.md`](full_consolidation_closure.md)
- [`docs/roadmap/real_compute_optional_lane_closure.md`](real_compute_optional_lane_closure.md)

Report freshness note: pre-existing `out/gate_report.json` did not match the evaluated HEAD (`3301615e3b23ad24603c47ee0b97787d2ee84097`), and pre-existing `out/docs_lint_report.json` did not embed matching HEAD metadata. They were treated as stale for audit evidence.

## 2. Readiness Gate Inventory

| Check / Area | Source path | Mandatory? | Expected runtime risk | Notes |
|---|---|---:|---|---|
| CLI dispatch and final exit | `runtime/ucf-ops/src/main.rs` | mandatory | low | `readiness-gate` accepts `--profile` and `--out`, runs `readiness_gate`, prints only final `status`/`out`, and exits 2 on non-PASS. Subcommand `--help` is not implemented as help; it starts the command with default behavior. |
| Gate setup and environment | `runtime/ucf-ops/src/lib.rs` | mandatory | low | Ensures layout, creates output parent, sets `UCF_PROFILE=<profile>` and forces `UCF_SSM_KERNEL=ref`. |
| Scenario bringup runs | `runtime/ucf-ops/src/lib.rs` | mandatory | medium | Runs seven 24-tick bringups: scenario A, scenario A repeat, scenario B, EBM off, EBM shadow, EBM active, and EBM active repeat. Isolated bringup was fast in this audit. |
| Replay verify/recompute | `runtime/ucf-ops/src/lib.rs` | mandatory as checks, but can report SKIP by check semantics | low | Runs replay audit for scenario B in verify-only and recompute-stages modes; isolated audits were fast. |
| Internal workspace test check | `runtime/ucf-ops/src/lib.rs` | mandatory unless CI or `UCF_SKIP_GATE_WORKSPACE_TESTS=1` | high | Runs `cargo test --workspace --offline` inside the gate. This was the isolated 300s bottleneck. CI skips by `CI=true`; local full gate does not. |
| Offline profile check | `runtime/ucf-ops/src/lib.rs` | mandatory | low | Requires `profile == test` and `UCF_OFFLINE=1`; without the environment variable the check is FAIL if the report is reached. Offline mode did not remove the timeout because the inner workspace tests still ran. |
| Required stage profile | `runtime/ucf-ops/src/lib.rs` | mandatory when stage-profile evidence exists | low | Evaluates explain-stage profile evidence; may be SKIP when the fixture does not contain an enforceable stage profile. |
| Feature-pack disabled fast-fail | `runtime/ucf-ops/src/lib.rs` | mandatory when feature-pack evidence exists | low | Validates feature-pack disabled semantics; may be SKIP where no applicable pack evidence is present. |
| Schema versions and required records | `runtime/ucf-ops/src/lib.rs` | mandatory | low | Validates schema version presence and required records from explain output. Required records may be SKIP for missing fixture-specific records. |
| Determinism | `runtime/ucf-ops/src/lib.rs` | mandatory | medium | Compares scenario A and repeat artifacts. Passed in skip-workspace diagnostic gate. |
| Tool deny, emergency visibility, observability, plug compatibility | `runtime/ucf-ops/src/lib.rs` | mandatory or SKIP by fixture evidence | low | Policy/observability checks derived from explain and metrics artifacts; not the timeout source in isolated runs. |
| EBM/adversarial checks | `runtime/ucf-ops/src/lib.rs`; `runtime/ucf-ops/src/adversarial.rs` | mandatory or SKIP by fixture evidence | medium | EBM off/shadow/active/active-repeat bringups plus safety dominance using adversarial report path; isolated adversarial suite was fast. |
| Formal invariants | `runtime/ucf-ops/src/formal_invariants.rs` | optional/diagnostic by feature/profile | low | `formal_invariants_smt` was SKIP in the skip-workspace diagnostic gate. |
| Weights lifecycle | `runtime/ucf-ops/src/lib.rs` | optional unless lifecycle initialized | low | SKIP when no lifecycle manifest/activity exists. |
| World VLJEPA evidence | `runtime/ucf-ops/src/lib.rs` | optional unless active/recently promoted or artifact present | low | SKIP when inactive and no shadow evidence artifact is present. |
| SAE real readiness | `runtime/ucf-ops/src/lib.rs` | optional unless SAE active | low | SKIP when SAE is inactive and no probe evidence is present. |
| SSM opt drift | `runtime/ucf-ops/src/lib.rs` | optional unless optimized kernel is enabled | low | Gate forces ref kernel, so `ssm_opt` is SKIP. |
| GPU lane parity | `runtime/ucf-ops/src/lib.rs` | optional unless GPU mode enabled | low | SKIP when `UCF_GPU_MODE=off` or unset. |
| Report writing and freshness metadata | `runtime/ucf-ops/src/lib.rs` | mandatory | low | Writes only after all checks finish. Timeout before completion yields no fresh `gate_report.json`. |
| `out/` scans | `runtime/ucf-ops/src/lib.rs`; `runtime/ucf-ops/src/readiness_spine.rs` | diagnostic/optional by check | low in this audit | Root `out/` was about 3.8 MiB / 120 files and `.ucf` about 864 KiB / 108 files; no evidence that broad `out/` scanning caused the 300s timeout. |
| Workflow invocation | `.github/workflows/ci.yml`; `.github/workflows/nightly_verify.yml` | mandatory in workflows | medium/high | CI/nightly run readiness gate with explicit `--workdir` directories. CI environment skips the internal workspace test check; local command does not. |
| Progress/logging availability | `runtime/ucf-ops/src/main.rs`; `runtime/ucf-ops/src/lib.rs` | diagnostic gap | high diagnostic risk | The gate emits no per-phase progress before final status. During a timeout, the last observed output is only Cargo launching `ucf-ops`, so the active phase is hidden unless isolated externally. |

## 3. Spine Check Drift

Diagnostic command:

```bash
cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json
```

Observed result: exit 2 with status `FAIL`. This is expected diagnostic evidence for drift and is not treated as a PASS.

| Drift category | Evidence | Likely cause | Severity | Suggested follow-up |
|---|---|---|---|---|
| `REDUCTION_MISMATCH` | Report categories included `REDUCTION_MISMATCH`; generated signoff, review packet, and workflow artifacts contained `reviewability_reduction_digest_prefix = MISSING` while the freshly derived canonical spine had reduction digest prefix `6746d727b0f76066`. | Operator surfaces are not carrying the current derived reviewability reduction digest into their emitted artifacts. | high | Prompt 35C should align operator signoff/review-packet/workflow reduction digest population with the canonical readiness spine, without weakening mismatch detection. |
| `SIGNOFF_SPINE_DRIFT` | `operator_signoff_readiness_spine_check.json` emitted `reviewability_reduction_digest_prefix = MISSING` and a signoff digest that the canonical spine included. | Signoff surface is stale/incomplete relative to the canonical reduction input. | high | Update or regenerate signoff surface semantics in a follow-up so signoff embeds the canonical reduction digest when available. |
| `REVIEW_PACKET_SPINE_DRIFT` | `operator_review_packet_readiness_spine_check.json` emitted `reviewability_reduction_digest_prefix = MISSING` and a review packet digest that the canonical spine included. | Review packet surface is stale/incomplete relative to the canonical reduction input. | high | Update review-packet construction in follow-up to carry the canonical reduction digest. |
| `WORKFLOW_SPINE_DRIFT` | `operator_workflow_chain_readiness_spine_check.json` emitted `workflow_stage = WORKFLOW_BLOCKED` and `reviewability_reduction_digest_prefix = MISSING`; remediation requested `run_operator_workflow_chain`. | Workflow chain blocks because upstream operator surfaces are not reduction-aligned. | high | Close signoff/review-packet drift first, then rerun workflow chain and readiness-spine-check. |


## 3.1 Prompt 35C Drift Closure Update

Prompt 35C closed the readiness-spine drift categories without weakening the checker. The current canonical reviewability reduction digest prefix for the evaluated HEAD is `6746d727b0f76066`. The stale/mismatched surfaces were the operator signoff, operator review packet, and operator workflow chain reduction-digest fields produced during `readiness-spine-check`.

Fix summary:

- `readiness-spine-check` now materializes its freshly derived backend evidence snapshot, active-review snapshot, operator report, signoff, and review packet into the evaluated workdir `out/` surface before deriving downstream operator surfaces.
- Operator report, signoff, and review-packet discovery now reads from the command workdir `out/` instead of an unrelated process-root `./out` surface.
- Operator signoff keeps strict gate blocking semantics, but derives the reviewability reduction digest from the non-strict canonical reviewability context used by the readiness spine so strict evidence absence cannot create a digest-only drift.
- Operator review packet embeds the canonical reduction digest from the signoff surface when present; workflow then inherits the same digest from review-packet/signoff inputs.

Validation status after Prompt 35C:

| Command | Result | Notes |
|---|---|---|
| `cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json` | PASS | No drift categories; fresh report was generated for the current evaluated HEAD and intentionally not committed. |

The cold-cache readiness-gate timeout risk remains separate from spine drift. Prompt 35C does not claim that local cold-cache gate timing is resolved; it only closes the reduction/signoff/review-packet/workflow alignment failure.

## 4. Timeout Reproduction

| Command | Result | Duration | Output observed | Artifact produced? |
|---|---|---:|---|---:|
| `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json` | TIMEOUT (`EXIT=124`) | 300s | Initial audit run: Cargo finished and launched `target/debug/ucf-ops`; no further phase output before timeout. | no |
| `UCF_OFFLINE=1 timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json` | TIMEOUT (`EXIT=124`) | 300s | Initial audit run: Cargo finished and launched `target/debug/ucf-ops`; no further phase output before timeout. | no |
| `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json` after full workspace test/clippy warm-up | PASS (`EXIT=0`) | 213s | Final validation run printed `status=Pass` and report path. | yes |
| `UCF_OFFLINE=1 UCF_SKIP_GATE_WORKSPACE_TESTS=1 timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report_skip_workspace.json` | PASS diagnostic with workspace check SKIP | 3s | Final `status=Pass` and report path printed. | yes |
| `cargo run -p ucf-ops -- readiness-gate --help` | Not a help command | interrupted | Started the gate instead of printing subcommand help. | no fresh gate report relied upon |
| `cargo run -p ucf-ops -- readiness-spine-check --help` | Not a help command | seconds after build | Ran readiness-spine-check default behavior and reported drift. | yes |

## 5. Bottleneck Isolation

| Component | Isolatable command? | Result | Runtime risk | Notes |
|---|---:|---|---|---|
| Internal workspace tests | yes: `timeout 300s cargo test --workspace --offline` | TIMEOUT (`EXIT=124`) at 300s during initial audit; later non-offline `cargo test --workspace` passed after build/test cache warm-up | high | This matches the initial full gate timeout and is the dominant cold/local bottleneck. The final gate pass after warm-up confirms timeout sensitivity rather than a non-workspace phase hang. |
| Readiness gate without internal workspace tests | yes: `UCF_OFFLINE=1 UCF_SKIP_GATE_WORKSPACE_TESTS=1 timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report_skip_workspace.json` | PASS in 3s | low for non-workspace phases | This does not weaken the real gate result; it isolates the non-workspace phases as fast. |
| Scenario bringup | yes: `timeout 60s cargo run -p ucf-ops -- bringup --scenario fixtures/e2e_scenario_a.json --ticks 24 --workdir ./.ucf_audit_bringup_a --out ./out/audit_bringup_a` | PASS in <1s after build | low | Bringup itself is not the 300s blocker. |
| Replay verify | yes: `timeout 60s cargo run -p ucf-ops -- replay audit --workdir ./.ucf_audit_bringup_a --from 1 --to 24 --strict verify-only --out ./out/audit_replay_verify.json` | PASS in 1s | low | Replay verify is not the blocker. |
| Replay recompute | yes: `timeout 60s cargo run -p ucf-ops -- replay audit --workdir ./.ucf_audit_bringup_a --from 1 --to 24 --strict recompute-stages --out ./out/audit_replay_recompute.json` | PASS in <1s after build | low | Replay recompute is not the blocker. |
| Adversarial suite | yes: `timeout 60s cargo run -p ucf-ops -- adversarial-run --suite v1 --out ./out/adversarial_report.json` | PASS in 1s | low | EBM/adversarial suite is not the observed blocker. |
| Docs lint | yes: `timeout 60s cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json` | PASS in 1s | low | Docs/report writing is not the blocker. |
| Readiness spine check | yes: `cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json` | FAIL/exit 2 with drift categories | medium correctness risk, low timeout risk | Drift is real diagnostic failure, not a timeout source for readiness-gate. |
| `out/` artifact scans | partially | No timeout evidence | low | Artifact directories were small during audit; the timeout reproduces with internal workspace testing. |
| Progress logging | not currently granular | Missing | high diagnostic risk | Lack of per-phase logs hides the active phase in a timed-out gate; external isolation identified the workspace-test phase. |

## 6. Decision

| Option | Chosen? | Reason | Risk |
|---|---:|---|---|
| Option A — Audit-only | yes | The timeout source is isolated to the internal `cargo test --workspace --offline` check, and spine drift is a real semantic mismatch requiring a focused follow-up. Changing gate behavior now would risk weakening semantics or conflating audit with functional repair. | low |
| Option B — Add progress/phase logging only | no | Safe and useful, but intentionally deferred to keep this prompt audit-only and avoid touching gate code while drift remains unresolved. | low if done later |
| Option C — Timeout/per-check report metadata | no | Report metadata would require code/schema changes and cannot help when the process is externally killed before report write unless paired with incremental progress logging. | medium |
| Option D — Functional gate fix | no | A functional fix would need a policy decision about whether the gate should inline full workspace tests locally or rely on a separate mandatory lane. That decision is out of scope because gates must not be weakened. | high |

## 7. Recommended Follow-up

Recommended next prompts:

1. **UCF Prompt 35B — Readiness Gate Progress Logging and Per-Phase Timing**
   - Add stderr progress around major readiness-gate phases.
   - Add deterministic per-phase timing diagnostics if schema-compatible or separately reported.
   - Do not skip checks, do not mark missing checks as PASS, and do not hide timeouts.
2. **UCF Prompt 35C — Readiness Spine Drift Closure**
   - Align operator signoff, review packet, and workflow chain reduction digest fields with the canonical readiness spine.
   - Preserve drift failure semantics until all categories are genuinely closed.

## 8. Current Roadmap Impact

- A final warm-cache validation run produced a fresh readiness-gate PASS for the evaluated HEAD within the 300 second guard, but the initial cold/local audit runs timed out twice and demonstrate gate timeout sensitivity around the embedded workspace-test phase.
- Consolidation closure should remain explicitly gate-stability-reviewed until a follow-up decides whether the warm-cache pass is sufficient evidence or whether progress/timing diagnostics are required first.
- Replay Scheduler Roadmap should wait if the project requires cold/local gate stability before new roadmap work; if accepting the warm-cache gate pass, the remaining blocker is the readiness-spine drift follow-up.
- Readiness-spine-check drift is a separate high-severity correctness blocker and should not be reported as PASS until categories are closed.

## 9. Prompt 35B Progress Logging and Per-Phase Timing Update

Prompt 35B added readiness-gate observability only. Gate criteria, check order, skip/fail/pass semantics, replay verification semantics, required-record behavior, drift criteria, and spine-check criteria remain unchanged.

Readiness-spine drift remains closed from Prompt 35C: the canonical reviewability reduction digest prefix remains documented as `6746d727b0f76066`, and that drift closure is distinct from readiness-gate timing. Gate timing remains monitored as an operational diagnosability concern, not as a loosened acceptance criterion.

### 9.1 Observability behavior

The readiness gate now emits progress lines to stderr for major phases using this shape:

```text
[readiness-gate] start: <phase>
[readiness-gate] done: <phase> status=<PASS|FAIL|SKIP> elapsed_ms=<milliseconds>
```

The last visible `start:` line identifies the phase active if an external timeout kills the process before report writing. Completed phases are also carried in the gate report `phase_timings` metadata when the report is written successfully. A timeout before report writing is still a timeout/failure for validation purposes; the stderr phase trail is diagnostic evidence only.

### 9.2 Phase coverage

| Phase group | Purpose | Semantics changed? |
|---|---|---:|
| gate setup | Layout, output parent, profile/kernel environment setup | no |
| scenario bringups | Scenario A, A repeat, B, EBM off, EBM shadow, EBM active, EBM active repeat | no |
| replay verify/recompute | Existing verify-only and recompute-stages replay audits | no |
| explain and metrics | Existing explain/metrics extraction used by checks | no |
| internal workspace test/offline cargo check | Existing `cargo test --workspace --offline` check and existing CI/env skip semantics | no |
| required records / feature pack / stage profile checks | Existing offline profile, required stage profile, backend-disabled pack, schema version, and required-record checks | no |
| determinism / replay / policy observability checks | Existing determinism, replay-report, policy, emergency, observability, and plug-compatibility checks | no |
| adversarial / EBM checks | Existing EBM/adversarial wiring, correctness, dominance, determinism, provenance, and fallback checks | no |
| formal and optional probes | Existing formal invariants and optional weights/world/SAE/SSM/GPU probes | no |
| report assembly/write | Existing report creation and write path | no |

### 9.3 Interpretation

- If a run times out with the last visible phase `internal workspace test/offline cargo check`, the prior Prompt 35A bottleneck hypothesis is confirmed for that run.
- If the workspace-test phase completes and a later phase becomes last visible, the timing investigation should move to that later phase without changing gate policy.
- If the gate report is produced, `phase_timings` should be used to compare warm/cold runs and identify slowest completed phases.
- No timeout should be marked as PASS solely because earlier phases completed.
