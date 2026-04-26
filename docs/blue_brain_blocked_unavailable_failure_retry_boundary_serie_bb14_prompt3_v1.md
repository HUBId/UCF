# Serie BB14 Prompt 3: Blocked/Unavailable/Failure-Semantik und Retry-Boundary (minimal)

Status: Diese Ergänzung schärft die minimale echte Execution-Linie um eine explizite blocked/unavailable/failed/cancelled-Trennung plus minimale retry-/nonretry-Boundary. Kein Retry-Manager, keine Queue, keine Orchestrierungsplattform.

## Kanonische Outcome-Klassen

Die BB14-Minimallinie führt eine explizite Outcome-Klasse je Report:

- `execution completed`
- `execution blocked`
- `execution unavailable`
- `execution failed`
- `execution cancelled`
- `execution unsupported`
- `execution placeholder-only`
- `non-canonical/internal-only path`

Damit bleibt klar:

- `blocked`: Boundary-/Safety-/Eligibility-Stop vor echter Ausführung,
- `unavailable`: Ausführungspfad/Subsystem ist operativ nicht verfügbar,
- `failed`: echte Ausführung wurde gestartet und endete negativ,
- `cancelled`: bewusster Abbruch, kein failed-Ergebnis.

## Minimale Retry-/Nonretry-Grenze

Für jeden Report wird eine minimale Retry-Disposition kodiert:

- `retryable failure`
- `nonretryable failure`
- `retry not applicable`

Regeln:

- Nur `execution failed` kann `retryable` oder `nonretryable` sein.
- `blocked`, `unavailable`, `cancelled`, `unsupported`, `placeholder-only`, `non-canonical` sind nicht automatisch retrybar und werden als `retry not applicable` geführt.
- Es wird **keine** automatische Wiederholung ausgelöst.

## Failure-Path-Klasse

Zusätzlich bleibt die Failure-Path-Klasse explizit:

- `canonical failure path`
- `non-canonical/internal-only failure path`
- `not-a-failure path`

So bleiben non-canonical/internal-only Pfade explizit getrennt und werden nicht implizit in kanonische Failure-/Retry-Semantik hochgestuft.

## Rückbindung in Runtime/Selection/Memory

Die Backbind-Schicht trägt die Retry-Disposition in Runtime-Feedback mit, während bestehende Guardrails unverändert bleiben:

- keine automatische Folge-Execution,
- keine automatische Proposal-Generierung,
- keine automatische Memory-Persistenz,
- keine automatische Compute-Invocation.

## Bewusste Grenzen

Unverändert out-of-scope:

- Retry-Engine,
- Job-Queue,
- Orchestration-Plattform,
- Agenten-/Policy-Governance-Logik,
- zweite Fehler- oder Referenzsprache neben der kanonischen Linie.
