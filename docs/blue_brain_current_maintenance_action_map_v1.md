# Blue-Brain current maintenance action map v1

Status: narrow maintenance-facing cleanup map for the current post-BR6/IR1/MD2/MD3/SC1 Blue-Brain state. This file is a **supporting current reference** only. It does not supersede `docs/blue_brain_authority_chain_status_map.md`, does not create a second operative authority, and does not add a region, model-deepening candidate, global model/relation platform, planner/agent layer, policy/governance layer, retry/orchestration layer, retrieval/consolidation/reasoning layer, memory persistence path, allowed-actions expansion, HH production integration, or compute-core work.

Code anchor: `CANONICAL_BLUE_BRAIN_CURRENT_MAINTENANCE_ACTION_MAP` and `CANONICAL_BLUE_BRAIN_CURRENT_MAINTENANCE_ACTION_CLASS_MAP` in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`.

## 1) Action classes

| Action class | Maintenance target | Cleanup decision |
| --- | --- | --- |
| `authority/discoverability cleanup` | README entrypoint, authority chain, and supporting-reference routing. | Keep `docs/blue_brain_authority_chain_status_map.md` as the classifier; supporting references may clarify but must not become parallel authority. |
| `relation wording cleanup` | Implemented, mediated, deferred, blocked, caveated, and diagnostic-only relation wording. | Preserve the canonical relation status and mediation path; do not imply platform coupling, direct execution, retry, or promotion authority. |
| `model-boundary wording cleanup` | Abstract current mode, bounded Kuramoto-like current mode, HH simulation-only/diagnostic-only, and later-HH/deferred wording. | Keep current model wording bounded and relation-local; HH remains non-productive and deferred/diagnostic-only. |
| `guard wording/visibility cleanup` | no-direct action, execution, retry, memory, compute, and safety override guard language. | Keep all no-direct-* denials visible together in current docs and tests. |
| `evidence/reference cleanup` | Baseline, report, and check references under `out/` plus tracked evidence docs. | Treat reports as evidence for their recorded commit/run; do not promote report files into operational authority. |
| `no-change-needed finding` | Six-region inventory, IR1 bounded architecture, and exactly two selective model-deepening lines. | Record that no new region, third deepening, platform, policy, retry, memory, HH production, or compute-core work is justified. |

## 2) Current maintenance caveat locations

| Caveat | Current reading | Primary reference path |
| --- | --- | --- |
| Current authority vs supporting references | Current operational authority is classified only by `docs/blue_brain_authority_chain_status_map.md`; compact maps are supporting references. | `docs/README.md`, `docs/blue_brain_authority_chain_status_map.md` |
| Implemented vs mediated relation wording | Implemented relations are active only in bounded advisory/read-only diagnostics; mediated relations remain on their named reference-, selection-, execution-interface-, or direct-bounded-advisory path. | `docs/blue_brain_canonical_inter_region_relation_map_v1.md`, `docs/blue_brain_relation_wording_findings_map_v1.md` |
| Deferred vs blocked relation wording | Deferred means architecture-known but inactive/not-yet-implemented; blocked means fail-closed/unavailable and is not failed execution or retry state. | `docs/blue_brain_relation_wording_findings_map_v1.md` |
| Diagnostic-only relation wording | Diagnostic-only states are observable/readable only and do not steer runtime transitions, selection, retries, memory commits, or compute invocation. | `docs/blue_brain_relation_wording_findings_map_v1.md`, `docs/blue_brain_guard_semantic_drift_map_v1.md` |
| Model boundary wording | Abstract current mode remains the default region-facing mode; only `Amygdala ↔ Thalamus` and `Amygdala ↔ Basal Ganglia` use bounded Kuramoto-like current mode; HH is simulation-only/diagnostic-only or later/deferred. | `docs/blue_brain_canonical_model_boundary_map_v1.md` |
| Guard visibility | The guard set must stay complete: no direct action trigger, no direct execution trigger, no direct retry trigger, no direct memory commit, no direct compute invocation, and no safety override. | `docs/blue_brain_guard_semantic_drift_map_v1.md`, `docs/blue_brain_sc1_prompt4_final_system_consolidation_sweep_v1.md` |
| Evidence baseline readability | `out/` reports and audit baseline folders are reproducibility evidence for named runs/commits; authority remains document/code classified, not report-derived. | `docs/blue_brain_audit_baseline_map_v1.md`, `out/docs_lint_report.json`, `out/gate_report.json` |

## 3) No-change-needed findings retained

- Real Compute Stack remains maintenance-only.
- The bounded anatomical region set remains exactly Hippocampus, Amygdala, Thalamus, Basal Ganglia, Cerebellum, and Hypothalamus.
- IR1 remains the bounded inter-region architecture; it is not a global relation platform.
- The only current selective model-deepening lines remain `Amygdala ↔ Thalamus` and `Amygdala ↔ Basal Ganglia`.
- No direct action, execution, retry, memory commit, compute invocation, or safety override authority is introduced.
- No new expansion block is justified by this maintenance pass.

## 4) Completion note for this pass

This map consolidates the current maintenance action surface so future readers can find the remaining caveats without rereading historical BB25/BB27/BB29 handoffs as current authority. The maturity state remains **maintenance-ready with caveats**: the caveats are wording/discoverability/evidence-facing, not missing compute functionality.
