# Serie BB4 Prompt 2: Compute-trigger Arbitration und Selection-gated Invocation (Runtime + Context + Evidence)

Status: BB4 Prompt 2 präzisiert die kanonische Trigger-Arbitration-Schicht über der BB4 Prompt-1 Selection-Surface,
bleibt aber strikt auf derselben finalen Compute-Linie (`CanonicalComputeEntryPoint::submit`) ohne neue
Scheduler-/Planning-/Policy-/Reasoning-Plattform.

Repo-Anker:
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_COMPUTE_TRIGGER_ARBITRATION_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP`
  - `CANONICAL_BLUE_BRAIN_TRANSITION_TRIGGER_MAP`
  - `CANONICAL_BLUE_BRAIN_REFERENCE_CONTEXT_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_CANDIDATE_LIFECYCLE_MAP`
  - `CANONICAL_BLUE_BRAIN_RUNTIME_FEEDBACK_MAP`
- `runtime/ucf-compute/src/service_surface.rs`
- `runtime/ucf-compute/src/contracts.rs`
- `runtime/ucf-compute/src/reference_map.rs` (Tests + Doc-Pinning)

## 1) Bedeutung von Trigger-Arbitration in BB4

BB4 Prompt 2 bedeutet hier explizit:
- mehrere Triggerquellen als **trigger candidate** sichtbar halten,
- Selection/Attention als Gate nutzen,
- Trigger als **selected / deferred / suppressed / blocked / insufficient / caveated** unterscheidbar halten,
- Invocation nur über outward-facing Compute-Contract (`submit`) auslösen,
- keine implizite Compute-Ausführung über interne Helper-/Orchestration-Pfade.

Trigger-Arbitration ist hier **keine** Scheduler- oder Planungsschicht und ersetzt weder Policy noch Reasoning.

## 2) Kanonische Trigger-Arbitration-Map

Die Semantik ist in `CANONICAL_BLUE_BRAIN_COMPUTE_TRIGGER_ARBITRATION_MAP` codiert.
Die minimalen Klassen sind:
- trigger candidate,
- selected trigger,
- deferred trigger,
- suppressed trigger,
- blocked trigger,
- insufficient trigger basis,
- caveated trigger,
- non-canonical/internal-only trigger,
- invocation result feedback.

Damit bleiben Triggerkandidaten, Triggerauswahl und Invocation-Resultat auf einer gemeinsamen,
aber schmalen und expliziten Surface.

## 3) Kanonische Trigger-Quellen

Die Trigger-Basis wird in der Map explizit getrennt:
- context-derived trigger,
- evidence/reference-derived trigger,
- runtime-state-derived trigger,
- memory-candidate-derived trigger,
- feedback-derived trigger,
- manually/internal-only trigger (non-canonical).

Kein Quellentyp impliziert automatisch Compute-Invocation oder Memory-Commit.

## 4) Selection-gated Invocation

Selection-gated Invocation bleibt strikt sichtbar:
- trigger selected and invocation requested,
- trigger deferred and no invocation,
- trigger blocked and no invocation,
- trigger caveated but allowed,
- trigger insufficient and requires more context/evidence.

Nur selektierte, kanonische Trigger führen in den outward-facing Contract (`submit`).
Deferred/blocked/insufficient/suppressed Trigger erzeugen keinen Compute-Aufruf.

## 5) Qualitätsbindung an Context / Evidence

Die Trigger-Arbitration nutzt BB3/BB4-Qualitäten als Gate:
- sufficient,
- partial,
- stale,
- caveated,
- insufficient.

Diese Qualitäten steuern Triggerposture direkt (z. B. deferred/blocked/caveated) ohne numerische
Scoring- oder Ranking-Plattform.

## 6) Rückbindung an Memory Candidates

Memory Candidates können Trigger-Arbitration beeinflussen, bleiben aber non-commit:
- candidate selected as trigger basis,
- candidate deferred,
- candidate insufficient for trigger,
- candidate rejected and no trigger,
- no memory commit implied.

Wichtig: candidate-basierte Trigger sind Basis-/Arbitrationssignale, keine Auto-Invocation und keine Persistenz.

## 7) Invocation-Resultat zurück in Runtime/Selection

Nach selection-gated Invocation bleibt Ergebnisposture explizit:
- invocation completed,
- invocation failed,
- invocation blocked by Compute contract,
- invocation caveated/degraded,
- invocation result updates runtime context but not memory automatically.

Damit bleibt eine einzige, kanonische Compute-Result-Semantik erhalten.

## 8) Non-canonical Pfade sind explizit ausgegrenzt

Nicht kanonisch für BB4-Triggerautorität:
- compute-interne/expert-only Hooks,
- legacy/compat lanes (`build_backend(kind=stub|candle|worker)`, `domains/ai*`),
- unstabile internal/test surfaces als direkte Triggerquelle.

Diese Pfade sind entweder:
- explizit non-canonical markiert,
- oder müssen erst auf outward-facing status/evidence/submit down-gemappt werden.

## 9) Grenzen (nicht Teil von BB4 Prompt 2)

Bewusst nicht eingeführt:
- neue globale Scheduler-/Planning-Engine,
- neue Policy-/Governance-/RL-/Reasoning-Plattform,
- neue Compute-Core-Semantik,
- Memory-Consolidation- oder Commit-Engine,
- neurodynamische Spezialmodellierung (z. B. Hodgkin-Huxley/Kuramoto) in der Trigger-Arbitration.

## 10) Anschluss

BB4 Prompt 2 liefert eine belastbare Trigger-Arbitration-Linie:
- Quellen sind explizit,
- Zustände sind explizit,
- Invocation bleibt selection-gated und outward-contract-gebunden,
- Runtime-/Feedback-Rückspiegelung bleibt einheitlich,
- Memory bleibt weiterhin klar non-automatic und commit-separiert.
