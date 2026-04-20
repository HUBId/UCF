# Serie Q: Post-Rollout Adoption Map v1 (repo-basiert, schmal, technisch)

Status: hard technical prioritization after the first stabilized real domain rollout line.  
Boundary: no new compute-core program, no new platform wave, no governance/product portfolio layer.

## 1) Repo-basierte Re-Prüfung nach erstem echten Domain-Rollout

Re-checked surfaces:
- Serie M consumer map (`CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP`)
- Serie N broader system map (`broader_system_integration_map_serie_n_v1.md`)
- Serie P first real rollout line (`CANONICAL_FIRST_DOMAIN_ROLLOUT_COMPLETION_MAP`, `ops_compute_probe`)
- adjacent UCF surfaces now technically less ambiguous due to the first rollout anchor

Result (honest, narrow):
- **high-readiness:** `runtime_orchestrator_env_bootstrap` (load-bearing and closest to the proven outward execution/status/evidence line).
- **plausible but caveated:** `replay_diff_backend_recompute` (technically close, but still replay/comparison-first instead of outward service-first).
- **legacy-entangled:** `domains_ai_compat_lane` (historical coupling + compatibility adapters).
- **not worth adopting now:** `bench_compute_subcommand` (internal benchmark harness), `ops_compute_probe` (already anchor baseline, not next target).

## 2) Minimal post-rollout adoption classes

This step introduces only one narrow class set in code/doc:

1. `high-readiness adoption candidate`
2. `plausible but caveated candidate`
3. `legacy-entangled candidate`
4. `not worth adopting now`

No large adoption matrix, no parallel adoption framework.

## 3) Single evaluation anchor: first real rollout line

All surfaces are evaluated against the same proven line (`ops_compute_probe`) and only with technical criteria:
- outward-facing execution contract (`CanonicalComputeEntryPoint::submit`)
- outward-facing status/evidence semantics (`status_evidence_export_surface`)
- integration-safe hooks (`integration_hook_view`, read-only/caveated posture)
- low reliance on legacy/internal rollout authority (`build_backend(kind=stub|candle|worker)` and `domains/ai*` are not rollout authority)
- low demand for new compute-core/platform work

Canonical anchor line remains:

`submit -> compute_canonical -> result/fault/status -> execution_snapshot`

This keeps **no second adoption language** and no alternate rollout semantics.

## 4) Explicit downgrade of legacy / superficially-near surfaces

Downgraded on purpose (hidden coupling explicit):
- `domains_ai_compat_lane`: compatibility seam; attractive only via host ABI adapter history and legacy-coupled semantics.
- `bench_compute_subcommand`: internal benchmark harness; no outward-facing status/evidence contract.
- `ops_compute_probe` as “next candidate”: rejected because it is already the established rollout anchor baseline, not new adoption scope.

Also constrained:
- `replay_diff_backend_recompute` is technically useful and now less ambiguous, but still a replay/comparison lane with indirect hook usage (not primary outward rollout contract).

## 5) 1–3 echte nächste Adoptionsrichtungen (strict)

**Technically prioritized next directions (no wishlist):**
1. `runtime_orchestrator_env_bootstrap`
   - why high-readiness: load-bearing runtime surface and nearest practical continuation of the already proven outward line.
   - why still narrow: only acceptable via progressive tightening to canonical submit + outward status/evidence semantics.
2. `replay_diff_backend_recompute`
   - why plausible but caveated: can reuse core semantics/evidence references with low core churn.
   - why not high-readiness: still comparison-first and not an outward execution/status service contract.

No additional surfaces are promoted beyond these directions in this step.  
**Keine Wunschliste.**

## 6) Consolidated post-rollout adoption map (single truth)

Pinned in code as:
- `CANONICAL_POST_ROLLOUT_ADOPTION_MAP`

Map summary:

| surface | class | short technical rationale |
|---|---|---|
| `runtime_orchestrator_env_bootstrap` | `high-readiness adoption candidate` | closest load-bearing continuation; can be narrowed to canonical outward contract line with small integration work |
| `replay_diff_backend_recompute` | `plausible but caveated candidate` | demystified by rollout anchor but still replay/comparison-first lane |
| `domains_ai_compat_lane` | `legacy-entangled candidate` | explicit legacy/compat boundary, no canonical outward authority |
| `bench_compute_subcommand` | `not worth adopting now` | internal harness only |
| `ops_compute_probe` | `not worth adopting now` | already established first rollout anchor baseline |

## 7) Minimal consistency checks added

Added only narrow checks:
- class coverage for the new post-rollout adoption map
- exactly one `high-readiness adoption candidate` (`runtime_orchestrator_env_bootstrap`)
- `replay_diff_backend_recompute` remains explicitly caveated (not auto-promoted to high-readiness)
- explicit downgrade enforcement for legacy/internal surfaces
- doc/code anchor sync (same outward contract language, same exclusions)

No new test wave beyond map/doc consistency.
