# Serie BB25 Prompt 4: Nach-BB25 Roadmap-Entscheidung (festgezogen)

Status: **Entscheidung festgezogen: Maintenance/Bugfix/Cleanup ist der Default nach BB25; kein automatischer Start einer neuen Serie.**

Diese Referenz dokumentiert ausschließlich die Nach-BB25-Entscheidung auf Basis des aktuellen Repo-Stands. Sie führt **keine** neue Implementierungsarbeit ein und öffnet **keine** zusätzliche Regionenklasse.

## 1) Repo-basierte Stabilitätsprüfung für Region 1

Region 1 ist im aktuellen Stand technisch als maintenance-hardened abgeschlossen:
- BB24 Prompt 10 fixiert die erste kontrollierte Regionenexpansion als bewusst einzelne geöffnete Regionenklasse.
- BB25 Prompt 1–2 stabilisieren und schärfen die Region-1-Referenzfläche.
- BB25 Prompt 3 bestätigt final den Stabilitätsstatus inklusive unveränderter Guard Rails.

Daraus folgt: Für den Normalbetrieb ist **Maintenance/Bugfix/Cleanup** ausreichend und der technisch saubere Default.

## 2) Prüfung auf Region-2-Hebel

Es gibt aktuell **keinen** im Repo klar lokalisierbaren technischen Zwang, der eine unmittelbare Region-2-Öffnung erfordert.

Region 2 bleibt daher:
- nicht geöffnet,
- hinter `NotOpenedYetExplicitRescopeRequired`,
- nur über einen späteren, explizit begründeten Re-Scope denkbar.

Ein Region-2-Schritt wäre erst dann technisch gerechtfertigt, wenn neue Anforderungen dokumentiert sind, die außerhalb der stabilisierten Region-1-/BB23-Grenzen liegen und nicht maintenance-seitig lösbar sind.

## 3) Festgezogene Entscheidung nach BB25

**Verbindliche Entscheidung:**
- Standardmodus nach BB25 ist **Maintenance/Bugfix/Cleanup ohne neue Serie**.
- Ein späterer **expliziter Region-2-Re-Scope** bleibt als Option erhalten, ist aber **nicht** automatisch aktiviert und **nicht jetzt** Teil des aktiven Pfads.

## 4) Trennlinie: Maintenance vs. Re-Scope

Maintenance (zulässig im Default):
- Bugfixes innerhalb bestehender Guard Rails,
- Dokumentationskonsistenz und Referenzpflege,
- Cleanup ohne Capability-Ausweitung.

Nicht-Maintenance (nur per explizitem Re-Scope):
- Öffnung einer zweiten Regionenklasse,
- neue operative Autoritätskanäle,
- Scope-/Capability-Erweiterung über BB23-Envelope hinaus.

## 5) Kanonische Post-BB25 Decision-Map

Die explizite Decision-Map für den Maintenance-Default nach BB25 steht in:
- `docs/blue_brain_post_bb25_maintenance_default_decision_map_serie_bb25_prompt5_v1.md`

Sie hält Region 1 als einzige aktive Expansion fest, markiert Region 2 als bewusst nicht aktiv (nur expliziter Re-Scope) und verhindert implizite Serienfortsetzung.

## 5) Konsequenz für die Roadmap-Führung

Zur Vermeidung einer impliziten neuen Serie gilt:
- Kein automatischer Mehrfachausbau nach BB25.
- Keine Umdeutung von Maintenance-Arbeit als neue Expansionsserie.
- Re-Scope nur bei expliziter technischer Begründung und separater Priorisierungsentscheidung.
