# Serie P Prompt 3: First Domain Rollout Completion Proof v1

Status: narrow technical completion proof for the first real domain rollout case on the finalized compute line.

Boundary: no audit/governance/certification platform; no second truth source; no new integration wave.

Canonical line (unchanged):
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 1) Selected rollout case and contract checks

Selected first real domain rollout case: `ops_compute_probe`.

Technical checks against the final compute core:
1. **canonical execution contract**
   - `CanonicalComputeEntryPoint::submit(ComputeSubmitRequest{ExecuteInline})`
2. **outward-facing status/evidence exports**
   - `CanonicalComputeEntryPoint::status`
   - `CanonicalComputeEntryPoint::status_evidence_export_surface`
   - outward semantics from `status_evidence_export_surface.canonical_consumer_view()`
3. **integration-safe hooks**
   - `integration_hook_view` remains read-only/caveated for outward use
4. **absence of hidden legacy/internal dependency**
   - no rollout authority via `build_backend(kind=stub|candle|worker)`
   - no rollout authority via `domains/ai*`

## 2) Completion-status map (narrow, explicit)

Allowed statuses:
- `aligned`
- `aligned with caveats`
- `mixed/transitional`
- `not yet true rollout completion`

Current map for this proof step:

| rollout case | completion status | note |
|---|---|---|
| `ops_compute_probe` | `aligned` | canonical outward submit + status/evidence exports + integration-safe hook posture + no hidden legacy dependency in rollout authority |

## 3) Minimal hardening applied in Prompt 3

- code-pinned completion map for the first real rollout case in `reference_map`
- outward probe details include `rollout_completion_status` as a single canonical key
- small consistency checks ensure this proof remains bound to the final compute line and legacy/internal boundaries

## 4) Distinction from transitional/internal lanes

This prompt does **not** reclassify other lanes as rollout completion. It only provides a technical completion proof anchor for `ops_compute_probe` and keeps:
- mixed/transitional consumers outside aligned rollout completion,
- internal/legacy lanes outside true rollout completion.

## 5) Immediate Serie P follow-up

Use this proof case as the baseline for next domain rollouts:
- require the same four checks (execution contract, outward status/evidence export, integration-safe hook, hidden dependency exclusion),
- classify only with the same four statuses,
- avoid introducing parallel rollout semantics or platform layers.

## 6) Continuity to Prompt 4 closure

Prompt 3 provides the single-case completion proof anchor.
Final Serie-P closure matrix, explicit first rollout line statement, and next-direction prioritization are recorded in `docs/serie_p_readiness_sweep_prompt4_v1.md`.
