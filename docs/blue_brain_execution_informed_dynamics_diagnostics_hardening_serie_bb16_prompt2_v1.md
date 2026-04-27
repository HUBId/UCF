# Serie BB16 Prompt 2: execution-informed dynamics diagnostics hardening (advisory-only)

Diese Härtung konsolidiert die diagnostische Rückkanal-Linie zwischen BB12 dynamics, BB14 execution/result-integrity,
BB15 bounded references und BB2/BB4 Runtime-/Selection-Semantik.

## Kanonische execution-informed dynamics feedback Zustände

`CANONICAL_BLUE_BRAIN_DYNAMICS_EXECUTION_FEEDBACK_MAP` bleibt die einzige kanonische Map und trennt kompakt:

1. `execution_informed_dynamics_input`
2. `reference_informed_dynamics_input`
3. `caveated_execution_informed_dynamics_input`
4. `insufficient_dynamics_feedback_basis`
5. `blocked_dynamics_feedback_basis`
6. `unavailable_dynamics_feedback_basis`
7. `diagnostic_only_dynamics_feedback`
8. `non_canonical_internal_only_feedback_path`

## Erfolgreiche vs. nicht erfolgreiche Execution-Basis

Die Linie bleibt strikt getrennt:

- `...:result:completed` → execution-informed basis,
- `...:result:failed` / `...:result:cancelled` → caveated basis,
- `...:result:ExecutionBlocked` → blocked basis,
- `...:result:ExecutionUnavailable` / `...:result:ExecutionUnsupported` → unavailable basis,
- fehlende belastbare Basis → insufficient,
- reine diagnostic refs (`diag:*`, placeholders) → diagnostic-only.

Es gibt keine implizite Promotion von failed/cancelled/blocked/unavailable zu „erfolgreich“.

## Runtime-/Selection-Rückbindung bleibt advisory-only

Kuramoto Runtime-Hints führen weiterhin nur Diagnose-/Caveat-Tokens zurück (`KURAMOTO_EXEC_FEEDBACK=...`).
Diese Tokens informieren Runtime/Selection über die beobachtete Basis, erzeugen aber keine direkte Entscheidungsschicht.

## Caveat / insufficient / blocked / unavailable / diagnostic-only Gründe

Die Rückkanal-Semantik bleibt deterministisch und kompakt:

- caveated = schwache oder nur partielle Execution-/Reference-Basis,
- insufficient = fehlende bounded Feedback-Basis,
- blocked = Guard-/Boundary-bedingte Blockade,
- unavailable = operative Voraussetzungen fehlen,
- diagnostic-only = keine advisory Modulation begründbar.

## No-direct-* Grenzen bleiben hart sichtbar

Guard-nahe Zustände bleiben diagnostisch sichtbar durch explizite Caveats, ohne Autoritätsaufwertung:

- `no_direct_action_allowed`
- `no_direct_memory_allowed`
- `no_direct_compute_allowed`
- `no_direct_reexecute_allowed`
- `no_direct_retry_orchestration_allowed`
- `no_safety_override_allowed`

Keine direkte Re-Execution, keine Retry-Orchestrierung, keine Policy-Entscheidungsübernahme,
keine automatische Compute-Invocation, keine automatische Memory-Persistenz.

## Referenzlinie bleibt bounded

Canonical execution result references sind rein informativ für bounded dynamics diagnostics.
Sie werden nicht implizit zu Re-Execution-Triggern, Retry-Entscheidungen, Policy-Entscheidungen oder Memory-Writes.
