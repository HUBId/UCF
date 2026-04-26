# Serie BB13 Prompt 3: minimale Tool-Capability-Surface und kanonischer Allowed-Actions-Scope

Status: BB13 Prompt 3 zieht die reale Execution-Handlungsfläche bewusst eng und trennt allowed/blocked/unsupported/unavailable/non-canonical deterministisch.

## Kanonischer Minimal-Scope

Die operative Linie bleibt absichtlich klein:

- **Allowed canonical action**
  - `emit_canonical_signal`
  - nur bei `FutureActionReady` + `ExecutionEligibleHandoff` + Safety `Passed|Caveated`
- **Allowed canonical tool call**
  - aktuell **keiner**; keine zusätzliche Tool-Palette in BB13 Prompt 3

Damit ist die einzige reale Ausführung in dieser Linie:
`execute_blue_brain_minimal_action` mit `EmitCanonicalSignal`.

## Capability-Klassen (kanonisch)

`blue_brain_minimal_capability_scope` unterscheidet explizit:

1. `allowed canonical action`
2. `allowed canonical tool call` (reserviert, derzeit nicht aktiv genutzt)
3. `blocked action`
4. `unsupported action`
5. `unavailable action`
6. `non-canonical/internal-only action path`

## Trennung der Nicht-Ausführungsfälle

- **unsupported**: requested action liegt außerhalb der kanonischen Minimalfläche
  (`ExecutionUnsupported`, `UnsupportedNoResult`).
- **blocked**: prinzipiell kanonischer Scope, aber aktuelle Handoff-/Eligibility-/Safety-Bedingung verhindert Ausführung.
- **unavailable**: Subsystempfad nicht verfügbar (`safety_precheck == Unavailable`).
- **non-canonical/internal-only**: interner Pfad bleibt nie operativ ausführbar.

Diese Zustände werden nicht ineinander umgeschrieben.

## Eligibility/Safety/Execution/Result auf derselben Scope-Sprache

- Scope-Klassifikation läuft vor der eigentlichen Ausführung.
- Nur `allowed canonical action` kann real in Execution laufen.
- blocked/unsupported/unavailable/non-canonical erzeugen keinen echten Action-/Tool-Result-Output.
- Runtime-/Selection-/Memory-Feedback bleibt daran gebunden:
  - `execution-unsupported feedback` ist separat von `blocked`/`unavailable`.
  - kein Auto-Follow-up, kein Auto-Memory-Commit.

## No-direct-* Grenzen bleiben erhalten

Unverändert gilt:

- keine Agentenplattform / keine autonome Tool-Wahl
- keine Multi-Step-Orchestrierung
- keine Safety-Overrides
- keine implizite Memory-Persistenz
- keine Compute-Core-Mutation

