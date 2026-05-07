# Blue-Brain Maintenance Findings Map (Serie MAINT Prompt 1)

Status: narrow maintenance/bugfix/cleanup pass for the current UCF Blue-Brain state. This file is a maintenance-facing findings map, not a roadmap for new functionality.

## Scope lock

The active maintenance scope remains bounded to the already-integrated anatomical region set:

1. `hippocampus_like_region`
2. `amygdala_like_region`
3. `thalamus_like_region`
4. `basal_ganglia_like_region`
5. `cerebellum_like_region`

No new anatomical region is introduced by this pass. The current model mode remains `abstract functional current mode`; the only first minimal model-deepening line remains maintenance-stabilized and diagnostic/reference-bounded. There is no second deepening candidate in this pass.

## Maintenance findings map

| Finding class | Maintenance meaning | Required handling in this pass |
| --- | --- | --- |
| `real_bug` | Code behavior contradicts canonical guard semantics or deterministic classification intent. | Fix narrowly, add/adjust tests, avoid new capability. |
| `semantic_inconsistency` | Naming/order/contract semantics blur states that docs require to remain distinct. | Align code/tests/docs so advisory-only, caveated, deferred, blocked, insufficient, diagnostic-only, reference-only, and non-canonical/internal-only stay separated. |
| `guard_weakness` | A boundary condition can be misclassified or consumed too strongly. | Fail closed, keep no-direct-action/retry/memory/compute/policy/planner/agent guards intact. |
| `doc_test_drift` | Existing docs/tests describe behavior more strictly than the tested implementation. | Add focused assertions or trim wording; do not broaden scope. |
| `non_canonical_residual_path` | Legacy/internal/test-only wording or paths look externally promotable. | Mark as non-canonical/internal-only or keep reference-only; never promote to runtime/selection authority. |
| `no_change_needed` | Audited area already matches the maintenance boundary. | Leave unchanged and record the reason briefly. |

## Findings resolved in this pass

| Finding class | Finding | Resolution |
| --- | --- | --- |
| `real_bug` | Reference validity marker checks were case-sensitive even though reference kind and execution outcome classification are case-insensitive. Uppercase `BLOCKED`, `INSUFFICIENT`, `STALE`, `INVALIDATED`, or caveat markers could be missed. | Normalize validity marker checks to lowercase before matching. |
| `semantic_inconsistency` | Generic diagnostic/reference-only classification ran before explicit blocked/insufficient/stale/invalidated/caveated markers. A path such as `diag:runtime:insufficient_basis` was swallowed as generic `ReferenceOnly`. | Explicit guard/lifecycle markers now override generic diagnostic/reference-only fallback while plain diagnostics remain `ReferenceOnly`. |
| `guard_weakness` | Missing tests allowed diagnostic-only/reference-only paths with explicit guard markers to blur into weak reference-only semantics. | Added tests proving explicit blocked/insufficient/stale markers stay distinct while plain diagnostic paths remain reference-only. |
| `doc_test_drift` | Maintenance docs require insufficient, blocked, diagnostic-only, and reference-only to remain distinguishable; the previous unit coverage did not pin that edge. | Added focused canonical-reference unit coverage for those distinctions. |
| `non_canonical_residual_path` | Non-canonical/internal-only paths were reviewed at reference consumption points. | Existing fail-closed consumption tests remain valid; no code expansion needed. |
| `no_change_needed` | Runtime/selection inter-region advisory coupling, no-direct guard caveats, and HH diagnostic-only boundary were audited and already stayed bounded. | No behavior change. |

## Non-expansion checks

- No new anatomical region.
- No implicit second model-deepening candidate.
- No new allowed action, execution, retry, queue, memory persistence, retrieval, planner, agent, policy, governance, or compute-core authority.
- Advisory-only remains advisory-only; diagnostic-only/reference-only remain non-authoritative.
- Bounded inter-region relations remain bounded contract/diagnostic surfaces only.

## Follow-up maintenance notes

Further maintenance can keep reducing legacy wording around older generic brain-port test fixtures, but that should remain cleanup-only and must not become a region expansion or a productive HH integration.
