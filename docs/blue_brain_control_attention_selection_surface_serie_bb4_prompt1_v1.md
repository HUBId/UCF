# Serie BB4 Prompt 1: Kanonische Blue-Brain Control/Attention/Selection Surface (Runtime + Context)

Status: BB4 Prompt 1 zieht die kanonische Auswahl-/Kontroll-/Aufmerksamkeitsfläche über Runtime,
Context, Evidence/Reference und Memory-Candidates fest. Es wird **keine** neue Planning- oder
Reasoning-Engine gebaut und **kein** Memory-Commit-System eingeführt.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP`
  - `CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP`
  - `CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_UPDATE_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
- `runtime/ucf-runtime/src/orchestrator.rs`

Finale Referenzlinie bleibt unverändert:
- `submit -> compute_canonical -> result/fault/status -> execution_snapshot`

## 1) Bedeutung von Selection in BB4

BB4 meint hier explizit:
- attention target setzen,
- context/evidence/reference/memory-candidate strukturiert auswählen,
- Compute-Trigger optional aus expliziter Selection-Lage ableiten,
- Selection-Zustände sichtbar machen: selected, deferred, ignored, blocked, insufficient, caveated, rejected.
- selected context / selected evidence/reference / selected memory candidate explizit trennen.

Nicht enthalten:
- keine Planning- oder Reasoning-Engine,
- keine Policy-/Governance-Plattform,
- keine RL-Schicht,
- keine Memory-Consolidation- oder Commit-Engine,
- keine neurodynamische Spezialmodell-Integration.

## 2) Kanonische Selection-Klassen

`CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP` definiert die kanonischen Klassen:
- attention target,
- context selection,
- evidence/reference selection,
- memory-candidate selection,
- compute-trigger selection,
- non-canonical/internal-only selection paths (explizit ausgeschlossen).

Damit werden implizite Vermischungen zwischen Runtime-Control, Memory-Persistenz, Policy und
Reasoning vermieden.

## 3) Context Selection

Die Surface macht mindestens explizit unterscheidbar:
- context selected for current transition,
- context selected for compute trigger (optional caveated),
- context deferred,
- context ignored/irrelevant,
- context blocked due to stale or insufficient basis,
- context selected with caveat.

Leitplanke:
- Context Selection bleibt runtime-scoped und impliziert keinen Memory-Commit.

## 4) Evidence/Reference Selection

Evidence-/Reference-Basis kann Selection stützen, ohne Audit- oder Reasoning-Plattform zu werden.
Explizit modelliert:
- evidence/reference selected,
- evidence/reference ignored,
- evidence/reference deferred,
- evidence/reference insufficient,
- evidence/reference caveated.

Leitplanke:
- no memory commit implied.

## 5) Memory-Candidate Selection (ohne Persistenz)

Memory-Candidates bleiben proposal-/lifecycle-nah und werden nicht persistiert.
Explizit modelliert:
- candidate selected for future memory handling,
- candidate deferred,
- candidate rejected,
- candidate ignored,
- candidate blocked due to weak reference/context,
- candidate not persisted.

Leitplanke:
- Candidate Selection ≠ Memory Commit.

## 6) Compute-Trigger Selection auf BB2-Handoffs

Compute Trigger bleiben auf BB2-kanonischen Handoffs:
- compute trigger selected from context,
- compute trigger selected from evidence/reference need,
- compute trigger blocked due to insufficient selection basis,
- compute trigger deferred,
- no internal/expert-only trigger used.

Leitplanke:
- Trigger-Semantik bleibt an `CanonicalComputeEntryPoint::submit` gebunden und öffnet keine neue
  Compute-Trigger-Engine.

## 7) Selection-Basisqualität

BB4 konsolidiert die Basisqualität ohne numerische Ranking-Plattform:
- sufficient selection basis,
- partial selection basis,
- stale selection basis,
- caveated selection,
- insufficient selection basis.

Diese Qualität informiert Runtime-Übergänge, Trigger-Posture und Candidate-Lifecycle,
bleibt aber explizit nicht als Score-/Policy-Optimierer ausgelegt.

## 8) Non-canonical Selection/Control Paths

Internal/expert/compat Pfade bleiben ausgeschlossen:
- `run_operation_with_entry` / `replay_with_entry`,
- `build_backend(kind=stub|candle|worker)`,
- `domains/ai*` compatibility lanes.

Diese Pfade sind non-canonical/internal-only und dürfen nicht als kanonische Blue-Brain
Selection-/Control-Autorität erscheinen.

no internal/expert-only trigger used als kanonische Selection-Autorität.

## 9) Ergebnis für BB4-Anschluss

BB4 Prompt 1 liefert eine belastbare Auswahlfläche über Runtime und Context:
- Selection-Zustände sind strukturiert und testbar unterscheidbar.
- Context/Evidence/Candidate/Trigger Selection sind explizit gekoppelt, ohne semantische
  Vermischung mit Memory-Commit, Policy oder Reasoning.
- Non-canonical Pfade sind explizit markiert und als kanonische Autorität ausgeschlossen.
