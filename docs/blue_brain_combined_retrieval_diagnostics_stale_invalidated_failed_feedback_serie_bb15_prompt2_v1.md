# Serie BB15 Prompt 2: Combined retrieval diagnostics / stale-invalidated-failed reference feedback hardening

Status: Die BB15 Combined-Retrieval-Linie bleibt strikt **bounded advisory-only**. Diese Härtung ergänzt eine kanonische Diagnostics-Map und trennt Memory-/Execution-Zustände für Combined-Reference-Feedback ohne Consolidation, Ranking, Semantic Search oder Reasoning.

Hinweis BB15 Prompt 3: Die harte Candidate-Boundary wird zusätzlich über eine kanonische Consolidation-Candidate-Map abgesichert (siehe `docs/blue_brain_bounded_consolidation_candidate_boundary_serie_bb15_prompt3_v1.md`).

## Kanonische Combined-Retrieval-Diagnostics-Map

`runtime/ucf-compute/src/blue_brain_combined_retrieval.rs` führt `CANONICAL_BLUE_BRAIN_COMBINED_RETRIEVAL_DIAGNOSTICS_MAP` mit genau diesen Klassen:

- `combined_reference_available_diagnostic`
- `combined_reference_caveated_diagnostic`
- `combined_reference_stale_diagnostic`
- `combined_reference_invalidated_diagnostic`
- `combined_reference_failed_diagnostic`
- `combined_reference_cancelled_diagnostic`
- `combined_reference_blocked_diagnostic`
- `combined_reference_insufficient_diagnostic`
- `non_canonical_internal_only_combined_reference_diagnostic`

Damit bleibt die Combined-Linie als Diagnostics-/Feedback-Map eindeutig und konkurriert nicht mit einer zweiten Retrieval-Sprache.

## Explizite Trennung von Memory- und Execution-Basis

Die Combined-Basis trägt kompakt:

- `memory_basis_state`: `current | caveated | stale | invalidated | missing | blocked | unavailable`
- `execution_basis_state`: `completed | failed | cancelled | blocked | unavailable | unsupported | placeholder_only | non_canonical_internal_only_path | not_observed`

Diese Basiszustände bleiben rein diagnostisch/advisory und erzeugen keine automatische Compute-/Action-/Persistenzwirkung.

## Kanonische Combined-Reference-Status-Semantik

Für Combined-Kandidaten bleiben differenzierbar:

- `combined_reference_available`
- `combined_reference_caveated`
- `combined_reference_stale`
- `combined_reference_invalidated`
- `combined_reference_failed`
- `combined_reference_cancelled`
- `combined_reference_blocked`
- `combined_reference_insufficient`

Die Trennlinien sind explizit:

- stale memory != invalidated memory,
- failed execution != cancelled execution,
- blocked/unavailable execution != successful basis,
- caveated combined basis != strong available basis.

## Advisory-only Rückbindung in Runtime/Selection/Context

Die Combined-Retrieval-Basis bleibt bewusst read-only/advisory:

- `reference_basis_supports_selection_or_proposal_only = true`
- `automatic_compute_invoked = false`
- `automatic_action_executed = false`
- `automatic_memory_persisted = false`

Dadurch informiert Combined-Retrieval nur bounded Context/Selection/Candidate-/Proposal-Linien; es entstehen keine direkten Runtime-Mutationen, keine automatische Proposal-Erzeugung und keine automatische Execution.

## Guards: no-consolidation / no-ranking / no-semantic-search

Die Härtung hält explizit:

- keine automatische Consolidation,
- kein Ranking,
- keine Semantic-Search-Semantik,
- kein Reasoning-Output aus Diagnostics.

Combined-Retrieval bleibt eine deterministische Feedback-Schicht auf BB8/BB14-Referenzpfaden.
