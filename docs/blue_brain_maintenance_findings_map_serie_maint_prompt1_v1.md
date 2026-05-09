# Blue-Brain Maintenance Findings Map (Serie MAINT Prompt 1)

Status: narrow maintenance/bugfix/cleanup pass for the current UCF Blue-Brain state. This file is a maintenance-facing findings map, not a roadmap for new functionality.

## Scope lock

The active maintenance scope remains bounded to the already-integrated anatomical region set:

1. `hippocampus_like_region`
2. `amygdala_like_region`
3. `thalamus_like_region`
4. `basal_ganglia_like_region`
5. `cerebellum_like_region`
6. `hypothalamus_like_region`

No new anatomical region is introduced by this pass. Prefrontal Cortex, Anterior Cingulate Cortex and Insula references remain non-canonical/internal-only or historical candidate residues unless a future explicit re-scope says otherwise. The current model mode remains `abstract functional current mode` for non-deepened regions. The two existing selective model-deepening lines remain bounded to `Amygdala ↔ Thalamus` (first/MD1-MD2) and `Amygdala ↔ Basal Ganglia` (second/MD3). There is no third or additional model-deepening candidate in this pass.

## Maintenance findings map

The canonical code-side finding classes are pinned in `CANONICAL_BLUE_BRAIN_MAINTENANCE_FINDINGS_CLASS_MAP`.

| Finding class | Maintenance meaning | Required handling in this pass |
| --- | --- | --- |
| `real_bug` | Code behavior contradicts canonical guard semantics, current region scope, or deterministic classification intent. | Fix narrowly, add/adjust tests, avoid new capability. |
| `semantic_inconsistency` | Naming/order/contract semantics blur states that docs require to remain distinct. | Align code/tests/docs so advisory-only, caveated, deferred, blocked, insufficient, diagnostic-only, reference-only, current model mode and non-canonical/internal-only stay separated. |
| `guard_weakness` | A boundary condition can be misclassified, consumed too strongly, or read as active scope. | Fail closed, keep no-direct-action/retry/memory/compute/policy/planner/agent guards intact. |
| `doc_test_drift` | Existing docs/tests describe behavior more strictly than the tested implementation. | Add focused assertions or trim wording; do not broaden scope. |
| `non_canonical_residual_path` | Legacy/internal/test-only wording or paths look externally promotable. | Mark as non-canonical/internal-only or keep reference-only; never promote to runtime/selection authority. |
| `no_change_needed` | Audited area already matches the maintenance boundary. | Leave unchanged and record the reason briefly. |

## Findings resolved in this pass

| Finding class | Finding | Resolution |
| --- | --- | --- |
| `real_bug` | The MD1 region-deepening decision map still covered only five bounded regions after BR6. `Hypothalamus` therefore fell through to the non-canonical/internal-only fallback even though it is now a current bounded region. | Added a bounded Hypothalamus MD1 decision as `NoDeepeningNeededNow` / `KeepAbstractOrDeferred`, with no coupling leverage, no HH/productive model work, no third/additional candidate and no direct authority. |
| `semantic_inconsistency` | The maintenance findings scope lock still listed five regions while SC1 Prompt 4 and the authority map define six current bounded regions. | Updated this map to six regions and explicitly separated historical/non-canonical region residues from the current bounded set. |
| `guard_weakness` | The current bounded region set was implicit; generic anatomical candidate residues could be misread as currently integrated. | Added a code-side `CURRENT_BOUNDED_BLUE_BRAIN_ANATOMICAL_REGION_MAP` and `is_current_bounded_blue_brain_anatomical_region` guard helper, with tests excluding Prefrontal Cortex, Anterior Cingulate Cortex and Insula from current scope. |
| `doc_test_drift` | No focused test proved that Hypothalamus is canonical for maintenance but not a new model-deepening candidate. | Added tests pinning the six-region scope, maintenance finding classes, Hypothalamus MD1 no-deepening status and unchanged single bounded Kuramoto-like region candidate count. |
| `non_canonical_residual_path` | Historical candidate regions remained visible through the broader anatomical candidate map. | Left the historical/candidate map intact for compatibility, but added the explicit current bounded map so consumers have a non-promoting current-scope read. |
| `no_change_needed` | Inter-region relation guards, MD2 no-second-candidate decision, no-direct action/execution/retry/memory/compute/safety flags and Hypothalamus relation boundaries were audited. | No behavior expansion needed; existing advisory/diagnostic/reference semantics remain bounded. |

## Non-expansion checks

- No new anatomical region.
- No implicit third or additional model-deepening candidate.
- No new allowed action, execution, retry, queue, memory persistence, retrieval, planner, agent, policy, governance, or compute-core authority.
- Advisory-only remains advisory-only; diagnostic-only/reference-only remain non-authoritative.
- Caveated, deferred, blocked, insufficient, diagnostic-only, reference-only, current model mode and non-canonical/internal-only remain distinct.
- Bounded inter-region relations remain bounded contract/diagnostic surfaces only.

## Follow-up maintenance notes

Further maintenance can keep reducing legacy wording around older generic brain-port test fixtures and broad anatomical candidate maps, but that should remain cleanup-only and must not become a region expansion, a third/additional model-deepening line or a productive HH integration.
