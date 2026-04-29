# Blue Brain — Serie BB22 Prompt 3: Remaining Internal-only Transition/Doc Cleanup Pass

Status: Schmaler, repo-treuer Cleanup-Pass über verbleibende **internal-only / test-only / legacy / non-canonical** Ränder. Keine neue Plattform, keine neue Capability, keine Autoritätsausweitung.

## 1) Canonical cleanup map (finaler Restkatalog)

| Kategorie | Geltung | Cleanup-Entscheid |
|---|---|---|
| `canonical_operational_transition` | operative BB-Linie | bleibt kanonisch, ohne implizite Nebenpfade |
| `internal_only_transition` | intern/expert-only | bleibt explizit non-canonical und fail-closed |
| `test_only_transition` | test/deferred/helper | bleibt test-only, nicht operativ promotable |
| `deprecated_legacy_transition` | legacy/compat/historical | bleibt deprecated/legacy, keine operative Primärquelle |
| `non_canonical_internal_only_doc_claim` | Doku-/Kommentarrest | als non-canonical/internal-only Claim markieren oder entfernen |

## 2) Remaining cleanup decisions

- Internal-only Übergänge bleiben explizit ausgeschlossen (`NonCanonicalInternalOnlyPath`, `allowed=false`).
- Test-only/deferred Übergänge bleiben sichtbar als **test-only boundary**, ohne operative Geltung.
- Deprecated/legacy Übergänge bleiben **historisch/kompatibel**, aber nicht kanonische Autorität.
- Implizite Shortcut-Lesarten werden nicht als zweite operative Wirklichkeit geführt.

## 3) Guard-/Boundary-Sichtbarkeit (bewusst unverändert)

Die folgenden Leitplanken bleiben explizit sichtbar und bindend:

- no-direct-action
- no-direct-retry
- no-direct-memory
- no-direct-compute
- no-direct-policy/planner/agent
- advisory-only bleibt advisory-only
- candidate-only/test-only/deferred/non-canonical bleiben getrennt

## 4) Operative Klarheit nach dem Cleanup

Klarer operativ bleiben nur:

1. Canonical Runtime/Selection/Execution/Reference Übergänge.
2. Bounded advisory-only Dynamics-Signale ohne direkte Autoritätswirkung.
3. Fail-closed Behandlung von non-canonical/internal-only/test-only/legacy Rändern.

## 5) BB22 Abschlusslinie

Dieser Prompt-3 Pass entfernt keine Guardrails und erweitert keine Fähigkeiten. Er schließt nur Restunschärfen bei Übergangsdeutung und Dokuclaims, damit nach BB22 kein relevanter interner Schattenpfad wie ein operativer BlueBrain-Pfad wirkt.
