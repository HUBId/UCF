# Serie P: First Domain Rollout Candidate on Final Compute Line v1

Status: targeted, repo-based domain rollout candidate consolidation on top of the finalized compute core.

Boundary: this is **not** a new core buildout, product platform, governance rollout, or tenant/auth/billing program.
It remains pinned to the existing final compute integration language only.

Canonical contract line (unchanged):
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- outward status/evidence surface: `status_evidence_export_surface`
- outward hook boundary: `integration_hook_view` (`read_only_integration_safe` / `caveated_conditional`)
- explicit non-rollout boundary: `build_backend(kind=stub|candle|worker)` and `domains/ai*` compatibility lanes

Code source of truth:
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_FIRST_DOMAIN_ROLLOUT_CANDIDATE_MAP`
  - `DomainRolloutCandidateClass`
  - `CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
  - `CanonicalComputeEntryPoint::{submit,status,status_evidence_export_surface,integration_hook_view}`

## 1) Real domain candidate check (repo-treu)

Checked candidates from Serie M and Serie N plus load-bearing repo surfaces:

- `ops_compute_probe`
- `runtime_orchestrator_env_bootstrap`
- `replay_diff_backend_recompute`
- `bench_compute_subcommand`
- `domains_ai_compat_lane`

Result:
- **first real rollout candidate now:** `ops_compute_probe`
- **near candidate with narrow residual edge:** `runtime_orchestrator_env_bootstrap`
- **not rollout baseline now:** replay/bench/legacy compat lanes

## 2) Minimal rollout-candidate map (no platform layer)

Exactly four classes are used:
- `rollout-ready candidate`
- `rollout-plausible with caveats`
- `mixed/transitional candidate`
- `not a real rollout candidate now`

Classification:

| candidate | class | short reason |
|---|---|---|
| `ops_compute_probe` | `rollout-ready candidate` | already on canonical outward submit + status/evidence contracts |
| `runtime_orchestrator_env_bootstrap` | `rollout-plausible with caveats` | load-bearing but still mixed env/compat intake |
| `replay_diff_backend_recompute` | `mixed/transitional candidate` | replay/compat comparison lane, not outward service rollout contract |
| `bench_compute_subcommand` | `not a real rollout candidate now` | internal benchmark harness |
| `domains_ai_compat_lane` | `not a real rollout candidate now` | explicit legacy/compat boundary |

## 3) Explicit bind-back of selected candidate to final compute line

Selected candidate: `ops_compute_probe`.

Technical bind-back (explicit):
1. outward execution contract:
   - `CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})`
2. outward status/evidence exports:
   - `CanonicalComputeEntryPoint::status`
   - `CanonicalComputeEntryPoint::status_evidence_export_surface`
3. hooks posture:
   - `integration_hook_view` remains read-only/caveated observer boundary
4. explicitly excluded from rollout basis:
   - compatibility/internal constructors (`build_backend(kind=stub|candle|worker)`)
   - `domains/ai*` legacy compatibility lane

## 4) Minimal rollout-critical hardening in this step

Only narrow consolidation/hardening is included:
- one canonical map for first domain rollout candidate classification,
- tests that enforce canonical outward contract usage for rollout-ready,
- tests that prevent mixed/legacy/internal paths from appearing as rollout-ready,
- no second integration language and no core architecture expansion.

## 5) Rollout status/evidence semantics (Prompt 2 hardening)

For the selected candidate (`ops_compute_probe`) rollout semantics now stay explicitly anchored on
the same outward-facing compute surface as the finalized core line:

- status semantic source:
  - `status_evidence_export_surface.status` (`service_trust`, `snapshot_consistency`, top-level caveats)
- evidence semantic source:
  - `status_evidence_export_surface.evidence` (`bundle_refs`, evidence caveat refs)
- canonical outward aggregation:
  - `status_evidence_export_surface.canonical_consumer_view()`

`ucf-ops::run_compute_probe` reports these as outward rollout keys (`outward_*`) and keeps any
internal-only signal clearly segregated as diagnostic-only (`internal_diag_pipeline_state`), so
rollout semantics do not drift into expert/internal details.

This keeps one semantic line for status/trust/caveat/evidence-reference usage and avoids adding a
parallel domain-specific status model.

## 6) Explicit rollout caveats

Stable now:
- canonical execution/status/evidence contract line,
- `ops_compute_probe` as first outward domain rollout-ready anchor.

Constrained but acceptable:
- `runtime_orchestrator_env_bootstrap` remains plausible with caveats pending narrow canonicalization.

Deliberately out of this rollout:
- replay diff uplift to outward runtime service contract,
- benchmark harness path,
- legacy `domains/ai*` compatibility lanes,
- any new platform/control/governance layer.

## 7) Integration doc continuity

This document extends existing M/N/O lines without replacing them:
- Serie M: first post-core consumer alignment map.
- Serie N: broader system review/prioritization.
- Serie O: maintenance boundary for compute core.

Serie P adds one precise thing only: a real first domain rollout candidate anchored on canonical outward contracts.

## 8) Targeted check intent

Checks in `reference_map` enforce at minimum:
- rollout-ready candidate is actually canonical outward (`submit`, `status_evidence_export_surface`),
- mixed/legacy/internal lanes are not accidentally promoted to rollout-ready,
- status/evidence semantics stay tied to the same final contract line,
- no second integration language.
