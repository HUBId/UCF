# Serie BB19 Prompt 1: Runtime/Selection Contract Hardening Line

Status: **stable bounded contract hardening** auf der bestehenden BB2/BB4/BB12/BB13–BB18-Linie (keine neue Planner-/Agent-/Policy-/Orchestration-Plattform).

## 1) Kanonische Runtime/Selection-Contract-Signale

Die operative Kopplung bleibt auf sieben explizite Signalklassen begrenzt:

- `runtime_to_selection_advisory_signal`
- `runtime_to_selection_blocked_or_deferral_signal`
- `selection_to_runtime_advisory_state`
- `selection_to_runtime_deferred_state`
- `caveated_contract_signal`
- `insufficient_contract_basis`
- `non_canonical_internal_only_contract_path`

Diese Signale transportieren nur bounded Runtime/Selection-Contract-Zustände und erweitern keine Entscheidungsautorität.

## 2) Richtungs- und Zustandsgrenzen

- Runtime → Selection bleibt ein advisory/blocked/deferral/caveat/insufficient Signalpfad.
- Selection → Runtime bleibt ein advisory/deferred/caveat/insufficient Signalpfad.
- `deferred ist nicht blocked`.
- `blocked ist nicht failed execution`.
- `caveated ist nicht strong signal`.
- `insufficient ist nicht blocked`.
- `advisory-only bleibt advisory-only`.

## 3) Bounded Einbindung von Execution/Dynamics/References

- Execution-Feedback geht nur als bounded Contract-Basis in die Runtime/Selection-Line ein.
- Kuramoto-Dynamics bleibt advisory-only; keine direkte Action-/Compute-/Memory-Autorität.
- Canonical References bleiben Basis-Signalquelle; non-canonical/internal-only Pfade bleiben explizit ausgeschlossen.

## 4) No-direct-* Grenzen (unverändert verpflichtend)

- keine direkte Action-Execution
- keine direkte Retry-Orchestrierung
- keine automatische Compute-Invocation
- keine automatische Memory-Persistenz
- keine Policy-/Agenten-Autoritätserweiterung
- kein Safety-Override durch Runtime/Selection-Contract-Signale

## 5) Operative Wirkung in BB19

BB19 Prompt 1 konsolidiert die bestehende Runtime/Selection-Kopplung in eine klar benennbare Contract-Map. Dadurch werden spätere produktionsnahe Kopplungsschritte auf einer gehärteten, reproduzierbaren und auditierbaren Signalgrenze aufgesetzt.
