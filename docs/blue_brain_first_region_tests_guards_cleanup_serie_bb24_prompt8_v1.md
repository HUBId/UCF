# Serie BB24 Prompt 8: First-Region Tests/Guard-Härtungen und Non-Canonical Cleanup

Status: **erste Regionenexpansion ist testseitig und guard-seitig gehärtet; non-canonical/internal-only Restpfade sind klar operativ ausgeschlossen**.

Diese Linie erweitert BB24 Prompt 5–7 nur in einem schmalen Hardening-Scope: keine neue Regionenklasse, keine neue Runtime-Autorität, keine Planner-/Policy-/Retry-/Compute-/Memory-Ausweitung.

## 1) Canonical first-region hardening map

Kanonisch geführt in `runtime/ucf-compute/src/blue_brain_region_first_integration.rs` via
`CANONICAL_BLUE_BRAIN_FIRST_REGION_HARDENING_MAP`:

- `GuardedCanonicalRegionSurface`
- `GuardedDiagnosticsPath`
- `BlockedForbiddenAuthorityPath`
- `NonCanonicalInternalOnlyRegionPath`
- `TestOnlyHelperNonOperationalPath`

Damit bleibt die operative first-region surface ausdrücklich auf bounded advisory-only beschränkt.

## 2) Input/State/Output/Reference guards

Die first-region Input-Guards unterscheiden jetzt explizit zwischen kanonischen und verbotenen Quellen:

- kanonisch: runtime selection/deferral/context reference input,
- explizit rejected: tool action control, compute internal state, safety override, implicit memory mutation.

State/Output/Reference bleiben davon getrennt und ausschließlich advisory/diagnostic gebunden.

## 3) No-direct-* hardening (unverändert bindend)

`BlueBrainFirstRegionOutputSurface` bleibt hart auf:

- `direct_action_selection = false`
- `direct_execution_trigger = false`
- `direct_retry_trigger = false`
- `direct_memory_commit = false`
- `direct_compute_invocation = false`
- `safety_override = false`

Keine implizite zweite Regionsausweitung und keine implizite Autoritätsanhebung.

## 4) Diagnostics-/Deferred-/Blocked-/Insufficient-Grenzen

Die Trennlinien bleiben testgebunden stabil:

- advisory-only bleibt getrennt von caveated,
- caveated bleibt getrennt von deferred,
- deferred bleibt getrennt von blocked,
- blocked bleibt getrennt von insufficient,
- diagnostic-only bleibt nicht-operativ.

## 5) Runtime-/Selection-/Reference-Konsistenz

Runtime/Selection/Reference lesen dieselbe first-region Contract-Semantik durch dedizierte, identische Konsumpunkte (`*_contract_signal(...)`) auf derselben Output-Surface.

Damit gibt es keine Schicht-spezifischen Sonderdeutungen für dieselbe Regionssurface.

## 6) Non-canonical/internal-only cleanup

Die verbleibenden internen Restpfade sind klar als nicht-operativ markiert:

- `NonCanonicalInternalOnlyRegionPath`
- `TestOnlyHelperNonOperationalPath`

Diese Klassen sind sichtbar für Guard-/Testzwecke, aber nicht als operative Autorität nutzbar.

## 7) Bewusst unveränderte Grenzen

- keine HH-Produktivkopplung,
- keine neue Compute-Core-Arbeit,
- keine neue allowed-actions-Erweiterung,
- keine Retry-/Queue-/Orchestrierung,
- keine automatische Memory-Persistenz,
- keine Mehrfachregionen-Plattform.
