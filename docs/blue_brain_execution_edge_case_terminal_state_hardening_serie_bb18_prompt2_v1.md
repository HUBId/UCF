# Serie BB18 Prompt 2: Execution edge-case semantics & terminal-state integrity hardening

## Zielbild (narrow pass, keine Scope-Ausweitung)

Diese Härtung zieht die minimale echte Execution-Linie enger an den Rändern:

- terminale Zustände bleiben eindeutig (`completed`, `failed`, `cancelled`, `blocked`, `unavailable`, `unsupported`),
- doppelte oder widersprüchliche Terminalisierung wird explizit als Integritätsverletzung erkannt,
- partielle/unvollständige Pfade erzeugen keine irreführende Result-/Reference-Repräsentation,
- no-direct-\* Grenzen und maintenance-only Scope bleiben unverändert.

Keine Workflow-Engine, keine Retry-Orchestrierung, keine neue Agenten- oder Governance-Plattform.

## Kanonische Edge-Case-Map

Die minimale Execution-Linie führt eine kanonische Edge-Case-Klassifikation:

- `ConflictingTerminalStateAttempt`
- `DuplicateTerminalizationAttempt`
- `PartialExecutionPath`
- `IncompleteResultPath`
- `BlockedBeforeStartEdgeCase`
- `CancelledAfterStartEdgeCase`
- `FailureAfterStartEdgeCase`
- `NonCanonicalInternalOnlyEdgePath`

Damit werden widersprüchliche Vorzustände und inkonsistente Übergangsdeutungen explizit, deterministisch und testbar.

## Terminal-State-Integrität

Integritätsprüfung bewertet nicht nur Status/Boundary/Lifecycle-Flags, sondern zusätzlich die Edge-Case-Map.
Insbesondere führen jetzt zu `IntegrityMismatch`:

- widersprüchliche Terminal-Referenzkombinationen,
- doppelte terminale Referenzen,
- partielle Übergänge mit inkonsistenten Lifecycle-Flags,
- unvollständige Ergebnis-/Referenzpfade.

So gibt es keine stille Umdeutung oder Überschreibung terminaler Zustände.

## Result-/Reference-Konsistenz an Randfällen

Die Edge-Case-Prüfung erzwingt:

- `blocked`/`unavailable`/`unsupported` ohne completed/failure/cancelled-Result-Mischung,
- `cancelled` getrennt von `failed`,
- `completed` nur mit vollständiger completed-Result-Repräsentation,
- keine zweite Result-Sprache neben den kanonischen Referenzklassen.

## Eligibility-/Safety-Bindung & no-direct-\* Grenzen

Die bestehende Bindung bleibt bestehen:

- blocked-before-start bleibt Safety-/Boundary-seitig,
- failure-after-start bleibt Execution-seitig,
- unavailable bleibt vom Failure-Pfad getrennt,
- cancelled bleibt getrennt von blocked/failed.

Es wurden keine automatischen Folge-Executions, keine impliziten Re-Openings terminaler Pfade,
keine Retry-Orchestrierung und keine Memory-Autocommits eingeführt.
