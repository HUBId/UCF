# Serie BB20 Prompt 2: Cross-line terminology/state harmonization pass

Status: **Schmaler, repo-weiter Semantikabgleich über die operativen BlueBrain-Linien**.
Kein neuer Plattform-Scope, keine neue Zustandsmaschine, keine Funktionsausweitung.

## 1) Kanonische Kernterminologie (repo-weit bindend)

Die folgenden Begriffe sind die gemeinsame operative Sprache über Runtime, Selection,
Memory/Reference, Execution, Retrieval, Dynamics und Runtime/Selection-Contracts:

- **state**: technischer Zustandsvektor einer Linie (z. B. Runtime-, Selection-, Execution-, Contract-State).
- **contract**: bindende Übergangs- oder Austauschgrenze zwischen Linien; kein Ergebnisobjekt.
- **result**: tatsächliches Ausführungs-Outcome (completed/failed/cancelled).
- **reference**: Verweis-/Ankerobjekt auf Kontext/Evidence/Result/Memory; kein eigenes Result.
- **diagnostic**: beobachtende Klassifikation; keine Ausführungs- oder Autoritätsentscheidung.
- **feedback**: rückgebundene Diagnose-/Caveat-/Hinweisinformation; keine implizite Entscheidung.

## 2) Kanonische Zustandsfamilien (cross-line)

- `blocked`: harte Guard-/Safety-/Integritätsgrenze.
- `deferred`: expliziter, bounded Aufschub innerhalb eines gültigen Flows.
- `caveated`: nutzbar, aber mit expliziten Caveats/Unsicherheiten.
- `insufficient`: Basis fehlt/ist zu schwach; kein impliziter blocked- oder unavailable-Ersatz.
- `unavailable`: Pfad/Resource aktuell nicht verfügbar; kein failed-Ersatz.
- `advisory-only`: liefert Hinweise/Diagnostik ohne operative Autorität.
- `candidate-only`: markiert Eignung/Bereitschaft, ohne Ausführung/Konsolidierung.
- `non-canonical`: intern/expert/test/legacy Pfad ohne kanonische operative Authority.

## 3) Verbindliche Abgrenzungen

- `blocked` **ist nicht** `deferred`.
- `caveated` **ist nicht** `insufficient`.
- `unavailable` **ist nicht** `failed`.
- `advisory-only` **ist keine** direkte Runtime-/Selection-/Execution-Autorität.
- `candidate-only` **ist keine** implizite Consolidation- oder Execution-Realisation.
- `result`/`reference`/`diagnostic`/`feedback` bleiben getrennte Rollen.
- `contract` und `state` bleiben getrennt (`contract` steuert Austauschgrenzen, `state` beschreibt Laufzustand).

## 4) BB20 Prompt-2 gezielte Harmonisierung (dieser Sweep)

1. **Kuramoto feedback basis** trennt `insufficient` jetzt explizit von `unavailable` auf
   der Runtime→Dynamics-Evidence-Klassifikation.
2. **Cross-line Doku-Kern** für Terminologie/Zustandsfamilien als zentrale BB20-Referenz ergänzt.
3. Keine Änderung der no-direct-* Guard-Rails, keine neue Execution- oder Retrieval-Autorität.

## 5) Legacy/non-canonical Sprachreste

- Nicht-kanonische Pfade bleiben explizit markiert (`non-canonical/internal-only`) und werden
  nicht sprachlich als operative Runtime-/Selection-/Execution-Linie dargestellt.
- Interne Diagnosepfade bleiben Diagnosepfade und werden nicht als Result-/Contract-Authority benannt.

## 6) Ergebnis

BB20 Prompt 2 liefert eine einheitliche, belastbare Kernsemantik für die bereits operativen Linien.
Damit kann die verbleibende BB20-Readiness-Härtung auf konsistenter Terminologie statt auf
impliziten, linien-spezifischen Bedeutungen aufsetzen.
