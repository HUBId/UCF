# UCF Full Consolidation Closure

## 0. Purpose

- This document is the Prompt 35 closure baseline for the current bounded Micro→Meso→Macro consolidation work.
- It is not production consolidation readiness.
- It is not Replay, Sleep, Geist, or ISM readiness.
- It is not identity finalization.
- Minimal Spine v1.x remains independent and unchanged.
- Because the readiness gate timed out twice under a 300 second guard in the original closure baseline, this document keeps gate-timing risk separate from consolidation claims. Prompt 35C closed the readiness-spine drift line without changing Minimal Spine v1.x or weakening gate criteria.

## 1. Baseline

| Field | Value |
|---|---|
| Branch | `work` |
| HEAD full | `c9b76f6626f99a3c0a9bb00ff07cda2e15bf469a` |
| HEAD short | `c9b76f66` |
| Dirty state at baseline capture | clean |
| Workspace package count | 192 |
| Full consolidation roadmap present | yes |
| Schema alignment present | yes |
| Consolidation E2E present | yes |
| Macro finalization boundary present | yes |
| `ucf-replay` present | yes |
| `ucf-geist` present | yes |
| Freeze doc present | yes |

Baseline links:

- [`docs/roadmap/full_consolidation_roadmap_boundary_audit.md`](full_consolidation_roadmap_boundary_audit.md)
- [`docs/roadmap/readiness_gate_timeout_stability_audit.md`](readiness_gate_timeout_stability_audit.md)
- [`docs/roadmap/consolidation_record_authority_schema_alignment.md`](consolidation_record_authority_schema_alignment.md)
- [`docs/minimal_spine_v1_freeze.md`](../minimal_spine_v1_freeze.md)

## 2. Completed Consolidation Layers

| Layer | Status | Evidence |
|---|---|---|
| Micro candidate hook | implemented | `cargo test -p ucf-consolidation --test minimal_spine_micro_hook -- --nocapture` |
| Micro builder | implemented | `cargo test -p ucf-consolidation --test minimal_spine_micro_builder -- --nocapture` |
| Micro append/readback | implemented | `cargo test -p ucf-consolidation --test minimal_spine_micro_append -- --nocapture` |
| Meso aggregation | implemented | `cargo test -p ucf-consolidation --test minimal_spine_meso_builder -- --nocapture` |
| Meso append/readback | implemented | `cargo test -p ucf-consolidation --test minimal_spine_meso_append -- --nocapture` |
| Macro candidate | implemented | `cargo test -p ucf-consolidation --test minimal_spine_macro_candidate -- --nocapture` |
| Macro local finalization boundary | implemented | `cargo test -p ucf-consolidation --test minimal_spine_macro_finalization_boundary -- --nocapture` |
| Pipeline E2E determinism | implemented | `cargo test -p ucf-consolidation --test minimal_spine_consolidation_pipeline_e2e -- --nocapture` |
| Docs overclaim guard | implemented | `docs/roadmap/full_consolidation_roadmap_boundary_audit.md` and `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json` |

## 3. Current Allowed Claims

- A bounded deterministic Micro→Meso→Macro pipeline test exists.
- Micro and Meso explicit append/readback contracts exist.
- A Macro candidate exists.
- A local consolidation-level finalization boundary exists.
- Evidence/Archive remain the append/readback authority.
- No Replay, Sleep, Geist, ISM, Identity, Gateway, Capability, or Real Compute readiness is introduced by this line.

## 4. Forbidden Claims

- Production consolidation readiness.
- Full memory readiness.
- Replay readiness.
- Sleep readiness.
- Geist/ISM integration.
- Identity finalization.
- Identity anchor.
- `MacroMilestoneFinalized` runtime event.
- Second event log.
- Gateway-visible consolidation.

## 5. Validation Baseline

| Area | Result | Notes |
|---|---|---|
| Formatting | PASS | `cargo fmt --check` passed. |
| Docs lint | PASS | `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json` passed after the closure document and links were updated. |
| Consolidation targeted tests | PASS | All eight Prompt 27-33 consolidation tests passed. |
| Consolidation package | PASS | `cargo test -p ucf-consolidation --all-targets` passed. |
| Protocol package | PASS | `cargo test -p ucf-protocol --all-targets` passed. |
| Minimal Spine router E2E | PASS | `cargo test -p ucf-router --test minimal_spine_e2e -- --nocapture` passed. |
| Replay package | PASS | `cargo test -p ucf-replay --all-targets` passed. |
| Geist package | PASS | `cargo test -p ucf-geist --all-targets` passed. |
| Workspace tests | PASS | `cargo test --workspace` passed. |
| Clippy | PASS | `cargo clippy --workspace --all-targets -- -D warnings` passed. |
| Readiness gate | SPLIT-EVIDENCE policy available; fresh pass still required | Prompt 35D adds explicit `workspace-test-check` prerequisite evidence and lets readiness-gate validate that fresh report instead of rerunning the workspace test internally. Prompt 35E makes that evidence path observable with progress and timing diagnostics. This does not weaken PASS criteria: missing, stale, wrong-command, dirty-state-mismatched, or non-PASS workspace evidence fails the gate. |
| Readiness diagnostic | PASS | Prompt 35C `cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json` passed after reduction/signoff/review-packet/workflow digest alignment. |

## 6. Readiness Gate Status

The readiness gate remains strict, and the readiness-spine drift branch is closed by Prompt 35C. Prompt 35D adds an explicit split-evidence policy for the workspace-test bottleneck: `workspace-test-check` can be run as a mandatory prerequisite artifact, and `readiness-gate --workspace-test-report <path>` accepts it only when it is fresh for the current HEAD and dirty state. Prompt 35E confirms the evidence command can complete under a 600 second guard in this environment after cache warm-up and adds progress/timing diagnostics, but consolidation closure still requires a fresh matching workspace-test report plus a fresh readiness-gate pass before it is used as replay-readiness evidence. The follow-up audit is recorded in [`docs/roadmap/readiness_gate_timeout_stability_audit.md`](readiness_gate_timeout_stability_audit.md).

| Attempt | Command | Result | Notes |
|---|---|---|---|
| 1 | `timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json` | TIMEOUT | Timed out after Cargo launched the gate; no fresh gate report was written. |
| 2 | `UCF_OFFLINE=1 timeout 300s cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json` | TIMEOUT | Offline mode did not remove the timeout. |
| Diagnostic | `cargo run -p ucf-ops -- readiness-spine-check --out ./out/readiness_spine_check.json` | PASS | Prompt 35C closed `ReductionMismatch`, `SignoffSpineDrift`, `ReviewPacketSpineDrift`, and `WorkflowSpineDrift` by aligning operator surfaces to the canonical reduction digest. |
| Split prerequisite | `cargo run -p ucf-ops -- workspace-test-check --out ./out/workspace_test_report.json` then `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json --workspace-test-report ./out/workspace_test_report.json` | Policy added | Prompt 35D preserves mandatory workspace-test evidence while avoiding duplicate embedded workspace-test execution when fresh matching evidence is supplied. |

Static inventory shows the readiness gate runs seven bringup scenarios, two replay audits, an internal `cargo test --workspace --offline` check unless explicit split evidence, CI, or diagnostic environment behavior applies, EBM/adversarial checks, formal invariants, and optional readiness probes. The workspace-test bottleneck is now a prerequisite-evidence policy concern rather than permission to weaken gate criteria.

## 7. Remaining Gaps

- Replay Scheduler remains later work.
- Geist/ISM integration remains later work.
- Macro append/readback contract may still be needed as a separate bounded prompt if the roadmap chooses to append macro records explicitly.
- Protocol schema/provenance evolution remains later work.
- Prod-profile readiness remains later work.
- Gate timing remains tracked separately for cold/local cache conditions; readiness-spine drift is closed by Prompt 35C, and Prompt 35D provides strict split workspace-test evidence semantics.

## 8. Recommended Next Roadmap

Prompt 35C closes the gate-spine drift branch and Prompt 35D closes the workspace-test policy gap by making external workspace-test evidence explicit, fresh, and mandatory in split mode. If validation produces a fresh split-evidence readiness-gate pass, the next roadmap prompt can proceed to Replay Scheduler boundary work. If the gate still times out outside the workspace-test phase, run a focused follow-up on the remaining phase.

**Recommended next prompt when split-evidence validation is fresh and passing: UCF Prompt 36 — Replay Scheduler Roadmap and Boundary Audit**. The resulting Replay planning audit is [`docs/roadmap/replay_scheduler_roadmap_boundary_audit.md`](replay_scheduler_roadmap_boundary_audit.md), caveated as roadmap/boundary-only and not as Replay, Sleep, Geist/ISM, identity, Gateway, or Evidence/Archive authority readiness.

**Fallback if more diagnostics are needed: UCF Prompt 35E — Readiness Gate Workspace Evidence Integration Follow-up**.

After the gate-stability follow-up produces a stable fresh readiness baseline, the next large roadmap block should be either Replay Scheduler Roadmap and Boundary Audit or Prod-profile Readiness, without changing the forbidden claims above.
