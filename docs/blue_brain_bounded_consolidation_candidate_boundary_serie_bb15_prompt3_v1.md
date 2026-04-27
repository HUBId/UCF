# Serie BB15 Prompt 3: Bounded consolidation candidate boundary / no-merge & no-ranking guards

Status: Die BB15-Linie bleibt strikt bei **consolidation-candidate only** auf kombinierter Memory-/Execution-Referenzbasis. Es gibt weiterhin **keine** Consolidation-Engine, **kein** Record-Merge, **kein** Ranking, **keine** Semantic Search, **kein** Reasoning-Output und keine automatische Runtime-/Action-/Persistenzwirkung.

## Kanonische Consolidation-Candidate-Map

`runtime/ucf-compute/src/blue_brain_combined_retrieval.rs` führt `CANONICAL_BLUE_BRAIN_CONSOLIDATION_CANDIDATE_MAP` mit genau diesen Zuständen:

- `consolidation_candidate_only`
- `caveated_consolidation_candidate`
- `insufficient_consolidation_candidate`
- `blocked_consolidation_candidate`
- `not_a_consolidation_candidate`
- `non_canonical_internal_only_consolidation_path`

Diese Map ist die harte Boundary zwischen Combined-Retrieval-Basis und echter Consolidation.

## Candidate-only bleibt strikt getrennt von echter Consolidation

Die Combined-Basis markiert nur Kandidatenstatus. Sie ist explizit:

- kein merged record,
- kein ranked output,
- kein semantic-retrieval Produkt,
- kein reasoning output,
- keine direkte Änderung bestehender Records.

## Explizite no-merge / no-ranking / no-semantic-search Guards

Die Retrieval-Basis trägt feste Guard-Felder:

- `merge_or_record_mutation_permitted = false`
- `ranking_permitted = false`
- `semantic_search_permitted = false`
- `reasoning_output_permitted = false`

Zusätzlich bleiben advisory-only Guards erhalten:

- `reference_basis_supports_selection_or_proposal_only = true`
- `automatic_compute_invoked = false`
- `automatic_action_executed = false`
- `automatic_memory_persisted = false`

## Maintenance-/Staleness-/Failure-Grenze

Stale/invalidated memory sowie failed/cancelled/blocked execution outcomes schwächen Candidate-Status deterministisch. Der Status kann dadurch nur caveated/insufficient/blocked werden; es erfolgt keine implizite Reparatur oder Glättung durch Consolidation.
