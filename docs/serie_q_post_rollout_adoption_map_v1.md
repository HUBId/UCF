# Serie Q: Reviewable Follow-up Option for Broader Adoption v1

Status: documentation-only follow-up option after integration stabilization.  
Boundary: no new rollout wave, no compute-core expansion, no roadmap/governance/meta-planning package.

## 1) Side-by-side baseline (clear separation)

- **Final compute line (already aligned):**
  `submit -> compute_canonical -> result/fault/status -> execution_snapshot`
- **First real domain rollout line (established):**
  `ops_compute_probe` on canonical outward contracts (`submit`, `status_evidence_export_surface`, `integration_hook_view`).
- **Further prioritized candidates (review-only):**
  `runtime_orchestrator_env_bootstrap`, `replay_diff_backend_recompute`.

This is a **reviewable follow-up option**, not an implementation step.

## 2) Narrow follow-up option classes

Only these classes are used:

1. `already aligned`
2. `first real rollout established`
3. `broader adoption review candidate`
4. `not pursued now`

No additional adoption matrix and no second rollout language.

- Explicitly excluded as rollout authority in this step: `build_backend(kind=stub|candle|worker)` and `domains/ai*` compatibility lanes.

## 3) Minimal review statements for prioritized further candidates

### `runtime_orchestrator_env_bootstrap` (`broader adoption review candidate`)
- **Why connectable:** load-bearing runtime surface close to the already proven outward line.
- **Outward-facing contracts that would be sufficient:** canonical `CanonicalComputeEntryPoint::submit` + `status_evidence_export_surface` (+ non-mutating `integration_hook_view` posture).
- **Why no rollout now:** env/compat intake is still mixed and requires narrowing before any rollout claim; therefore review-only.

### `replay_diff_backend_recompute` (`broader adoption review candidate`)
- **Why connectable:** shares core semantics/evidence references and is technically less ambiguous after first rollout establishment.
- **Outward-facing contracts that would be sufficient:** would need primary outward execution + status/evidence service semantics equivalent to canonical outward contracts.
- **Why no rollout now:** remains replay/comparison-first and lacks outward service-first contract posture; therefore review-only.

## 4) Explicit non-rollout boundaries

- `domains_ai_compat_lane`: `not pursued now` (legacy/compat seam; no canonical outward authority).
- `bench_compute_subcommand`: `not pursued now` (internal benchmark harness).
- `ops_compute_probe`: `first real rollout established` baseline only, not a new adoption target.

## 5) Consolidated reviewable map (single truth)

Pinned in code as `CANONICAL_POST_ROLLOUT_ADOPTION_MAP`.

| surface | class | concise boundary statement |
|---|---|---|
| `final_compute_reference_line` | `already aligned` | final technical production line is complete baseline, not expansion scope |
| `ops_compute_probe` | `first real rollout established` | established first rollout line; keep stable as reference baseline |
| `runtime_orchestrator_env_bootstrap` | `broader adoption review candidate` | technically connectable if narrowed to canonical outward contracts |
| `replay_diff_backend_recompute` | `broader adoption review candidate` | technically useful but still replay-first; outward service posture not yet met |
| `domains_ai_compat_lane` | `not pursued now` | explicit legacy/internal compatibility boundary |
| `bench_compute_subcommand` | `not pursued now` | internal-only benchmark harness |

## 6) Consistency guardrails (small, explicit)

- class coverage check for all four review-option classes,
- first rollout baseline remains explicit and separate from review candidates,
- prioritized broader candidates stay review-only,
- docs/code keep the same canonical outward contract language,
- **no unplanned rollout** is asserted in this step.
