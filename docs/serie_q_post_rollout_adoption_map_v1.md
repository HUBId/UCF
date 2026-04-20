# Serie Q: Post-Rollout Adoption Map v1 (repo-basiert, schmal, technisch)

Status: narrow adoption re-evaluation after the first stabilized real domain rollout line.  
Boundary: no new compute-core program, no new platform wave, no governance/product portfolio layer.

## 1) Repo-basierte Re-Prüfung nach erstem echten Domain-Rollout

Re-checked surfaces:
- Serie M consumer map (`CANONICAL_DOMAIN_FACING_COMPUTE_CONSUMER_MAP`)
- Serie N broader system map (`broader_system_integration_map_serie_n_v1.md`)
- Serie P first real rollout line (`CANONICAL_FIRST_DOMAIN_ROLLOUT_COMPLETION_MAP`, `ops_compute_probe`)
- adjacent UCF surfaces now technically less ambiguous due to the first rollout anchor

Result (honest, narrow):
- **more realistic now:** `runtime_orchestrator_env_bootstrap` (still caveated, but technically closest next outward adoption lane).
- **still indirect/legacy/internal:** `domains_ai_compat_lane`, `bench_compute_subcommand`.
- **demystified but not yet true outward candidate:** `replay_diff_backend_recompute`.
- **anchor, not next target:** `ops_compute_probe`.

## 2) Minimal post-rollout adoption classes

This step introduces only one narrow class set in code/doc:

1. `genuine next adoption candidate`
2. `plausible after first rollout but caveated`
3. `still indirect / compatibility-touching`
4. `not meaningful for compute adoption now`

No large adoption matrix, no parallel adoption framework.

## 3) Single evaluation anchor: first real rollout line

All surfaces are evaluated against the same proven line (`ops_compute_probe`):
- outward-facing execution contract (`CanonicalComputeEntryPoint::submit`)
- outward-facing status/evidence semantics (`status_evidence_export_surface`)
- integration-safe hooks (`integration_hook_view`, read-only/caveated posture)
- absence of legacy/internal rollout authority (`build_backend(kind=stub|candle|worker)` and `domains/ai*` are not rollout authority)

Canonical anchor line remains:

`submit -> compute_canonical -> result/fault/status -> execution_snapshot`

This keeps **no second adoption language** and no alternate rollout semantics.

## 4) Explicit downgrade of legacy / superficially-near surfaces

Downgraded on purpose:
- `domains_ai_compat_lane`: compatibility seam, legacy adjacency only.
- `bench_compute_subcommand`: internal benchmark harness, not domain-facing adoption.
- `ops_compute_probe` as “next candidate”: rejected because it is already the established rollout anchor baseline.

Also constrained:
- `replay_diff_backend_recompute` is technically useful and now less ambiguous, but still a replay/comparison lane (not primary outward rollout contract).

## 5) 1–3 genuine next adoption candidates (strict)

**Kept as genuine next candidate(s):**
1. `runtime_orchestrator_env_bootstrap`
   - why now genuine: load-bearing runtime surface and nearest practical continuation of the already proven outward line.
   - why still narrow: only acceptable via progressive tightening to canonical submit + outward status/evidence semantics.

No additional surfaces are promoted to “genuine next” in this step.  
**Keine Wunschliste.**

## 6) Consolidated post-rollout adoption map (single truth)

Pinned in code as:
- `CANONICAL_POST_ROLLOUT_ADOPTION_MAP`

Map summary:

| surface | class | short technical rationale |
|---|---|---|
| `runtime_orchestrator_env_bootstrap` | `genuine next adoption candidate` | closest load-bearing continuation; can be narrowed to canonical outward contract line |
| `replay_diff_backend_recompute` | `plausible after first rollout but caveated` | demystified by rollout anchor but still replay/comparison-first lane |
| `domains_ai_compat_lane` | `still indirect / compatibility-touching` | explicit legacy/compat boundary, no canonical outward authority |
| `bench_compute_subcommand` | `not meaningful for compute adoption now` | internal harness only |
| `ops_compute_probe` | `not meaningful for compute adoption now` | already established first rollout anchor baseline |

## 7) Minimal consistency checks added

Added only narrow checks:
- class coverage for the new post-rollout adoption map
- exactly one `genuine next adoption candidate` (`runtime_orchestrator_env_bootstrap`)
- explicit downgrade enforcement for legacy/internal surfaces
- doc/code anchor sync (same outward contract language, same exclusions)

No new test wave beyond map/doc consistency.
