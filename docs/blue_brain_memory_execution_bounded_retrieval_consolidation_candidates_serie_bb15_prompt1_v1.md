# Serie BB15 Prompt 1: Memory retrieval expansion / bounded consolidation candidates

Status: BB15 erweitert die kanonische Retrieval-Basis **nur bounded** über bereits vorhandene BB8-Memory-Records und BB14-Execution-Result-Referenzen. Es wird **keine** Consolidation-Engine, **kein** Ranking, **keine** Semantic Search und **kein** Reasoning-/Action-Autoritäts-Pfad eingeführt.

Hinweis: Prompt 2 härtet diese Linie über eine kanonische Combined-Retrieval-Diagnostics-Map sowie explizite stale/invalidated/failed/cancelled/blocked/insufficient-Trennungen weiter (siehe `docs/blue_brain_combined_retrieval_diagnostics_stale_invalidated_failed_feedback_serie_bb15_prompt2_v1.md`).

## Kanonische Combined-Retrieval-Linie

`runtime/ucf-compute/src/blue_brain_combined_retrieval.rs` führt eine minimale gemeinsame Basis ein:

- Kandidatenklassen:
  - `memory_retrieval_candidate`
  - `execution_result_retrieval_candidate`
  - `combined_reference_candidate`
  - `retrieval_supporting_context_candidate`
  - `consolidation_candidate_only`
  - `insufficient_retrieval_basis`
  - `non_canonical_internal_only_retrieval_path`
- Combined-Reference-Status:
  - `combined_reference_available`
  - `combined_reference_caveated`
  - `combined_reference_insufficient`
  - `consolidation_candidate_only`
  - `no_consolidation_performed`

## Referenztypen und Trennung der Basen

Die strukturierte Basis trägt explizit und getrennt:

- `memory_record_reference` (aus BB8 persisted memory read/reference)
- `execution_result_reference` (aus BB14 canonical execution result/failure/cancelled/blocked reference)
- optionaler Bezug auf `candidate_reference` / `proposal_reference` / `context_reference`
- `caveats`, `freshness_or_staleness`, `maintenance_or_failure_state`

Damit bleiben unterscheidbar:

1. retrieval basis (Memory/Execution-Referenzbeobachtung),
2. reference basis (kanonische Pfade),
3. execution basis (Outcome/Failure-Grenzen),
4. consolidation-candidate only (ohne automatische Consolidation).

## Advisory-only Boundaries (Runtime/Selection/Context)

Die Combined-Retrieval-Basis bleibt explizit advisory-only:

- `reference_basis_supports_selection_or_proposal_only = true`
- `automatic_compute_invoked = false`
- `automatic_action_executed = false`
- `automatic_memory_persisted = false`

Stale/invalidated memory sowie failed/cancelled/blocked execution references schwächen die Basis explizit (`stale_invalidated_or_failed_references_weaken_basis`) und werden nicht geglättet.

## Non-canonical Pfade

Internal-only/non-canonical Memory- oder Execution-Pfade führen zu `non_canonical_internal_only_retrieval_path` und werden nicht als kanonische Combined-Reference-Basis normalisiert.
