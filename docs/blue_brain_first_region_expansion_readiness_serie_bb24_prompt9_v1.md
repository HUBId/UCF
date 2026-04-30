# Serie BB24 Prompt 9: First-Region Expansion Readiness Sweep und Expansion-Grenze

Status: **kanonische BB24-Abschlusslinie für die erste kontrollierte Regionenexpansion**.

Diese Referenz konsolidiert BB24 Prompt 5–8 als harte Abschlussprüfung für genau **eine** reale Regionsexpansion und zieht die Expansion-Grenze explizit. Kein Mehrfachausbau, keine Compute-Core-Neuöffnung.

## 1) Repo-basierte First-Region-Expansion (operativ)

**Operativ zuerst expandierte Regionenklasse:**
- `AttentionSelectionRelated`.

Kanonische technische Trägerfläche:
- `runtime/ucf-compute/src/blue_brain_region_first_integration.rs`
  - `BlueBrainFirstRegionClass::AttentionSelectionRelated`
  - `evaluate_blue_brain_first_region_attention_selection(...)`
  - dedizierte Input/State/Output/Reference/Diagnostics/Contract-Semantik.

## 2) BB24 Expansion Readiness Map (kanonisch)

### A. Stable first-region operational surface
- `RegionInputSurface` (bounded Runtime/Deferral/Reference Inputs).
- `RegionStateSurface` (`ActiveBoundedAdvisoryOnly`, `BlockedDeferred`, `NonCanonicalInternalOnly`).
- `RegionOutputAdvisorySurface` (advisory-only, keine direkte Autorität).
- `RegionReferenceSurface` (reference-bounded, read-only semantics).
- Contract-Read-Punkte für Runtime/Selection/Reference (`*_contract_signal(...)`) mit identischer Signal-Lesung.

### B. Usable with caveats
- `Caveated` bei caveated/stale reference validity.
- `Deferred` bei deferred-deferral-Lifecycle.
- `Insufficient` bei insufficient references/context.
- `Blocked` bei rejected/stale/insufficient candidate lifecycle.

### C. Advisory-only
- `runtime_advisory_only = true`
- `selection_advisory_only = true`
- `reference_bounded_only = true`
- `direct_action_selection = false`
- `direct_execution_trigger = false`
- `direct_retry_trigger = false`
- `direct_memory_commit = false`
- `direct_compute_invocation = false`
- `safety_override = false`

### D. Deferred / blocked / insufficient / diagnostic-only
- Deferred/Blocked/Insufficient bleiben eigene Contract- und Diagnostic-States.
- `ReferenceOnly`-Lagen werden als `DiagnosticOnly` geführt (`reference_only = true`) und nicht in operative Autorität promotet.

### E. Non-canonical / internal-only
- `NonCanonicalInternalOnlyRegionPath`
- `TestOnlyHelperNonOperationalPath`
- `NonCanonicalInternalOnly` Contract/Diagnostic-Pfade

Diese Pfade bleiben explizit nicht-operativ.

## 3) First-region expansion line (explizit)

Für `AttentionSelectionRelated` sind kanonisch:
- **Input:** `BlueBrainFirstRegionInputSurface` + `classify_blue_brain_first_region_input_guard(...)`.
- **State:** `BlueBrainFirstRegionStateSurface`.
- **Output/Advisory:** `BlueBrainFirstRegionOutputSurface` + `BlueBrainFirstRegionAdvisoryOutputClass`.
- **Reference:** reference-bounded Signale ohne Persistenz-/Mutationsautorität.
- **Diagnostics:** `BlueBrainFirstRegionDiagnosticState`.
- **Contract:** `BlueBrainFirstRegionContractSignal` + einheitliche Runtime/Selection/Reference-Konsumpunkte.

Bounded Information Flow:
- Runtime/Selection/Reference lesen dieselbe Contract-Surface, keine Sonderkanäle pro Linie.

Explizit **nicht** operativ:
- zweite Regionenklasse,
- direkte Action-Steuerung,
- Retry-Orchestrierung,
- automatische Memory-Mutation/Persistenz,
- Safety-Override,
- Compute-Core-Ausweitung.

## 4) Finale Surface-/Diagnostics-/Contract-Grenzen

Hard-Grenzen bleiben:
- Input/state/output/reference/diagnostics/contract sind getrennte, benannte Sichten.
- advisory-only bleibt advisory-only (keine versteckte Autorität).
- caveated/deferred/blocked/insufficient/diagnostic-only bleiben getrennte Zustände.
- reference-bounded/read-only wird nicht als direkte operative Autorität behandelt.

## 5) No-direct-* und Out-of-scope final

BB24-first-region bleibt strikt ohne:
- direct execution,
- retry orchestration,
- planner/agent/policy governance semantics,
- implicit memory persistence,
- safety override control,
- HH-Produktivintegration,
- Multi-Region-Plattformisierung.

## 6) Konsistenz mit bestehenden BlueBrain-Linien

Die First-Region-Expansion bleibt in den bestehenden Leitplanken:
- BB2 runtime/transition/feedback: advisory-bounded, keine Execution-Autorität.
- BB4 selection/priority/deferral: Contract-Signale bleiben selektionsinformierend.
- BB8 + BB17 context/memory/reference hardening: reference-bounded, keine implizite Memory-Autorität.
- BB12 bounded dynamics: nur advisory-only, sofern gekoppelt.
- BB19 runtime/selection contract line: einheitliche Contract-Signal-Lesung.
- BB21 execution/reference interaction: reference interaction ohne direkte execution promotion.

## 7) Compute-Core-Abschlusslinie

Unverändert gültig:
- Compute-Core bleibt finale Exit-Linie,
- outward-facing contracts bleiben stabil,
- Core bleibt maintenance-only,
- BB24 öffnet keine neue Compute-Core-Arbeit.

## 8) Entscheidung: zweite Regionenklasse vs. Stabilisierung

Repo-basierte Entscheidung nach Prompt 9:
- **Priorität: Stabilisierung der ersten Regionsexpansion (`AttentionSelectionRelated`)**.

Technischer Grund:
- Die erste Expansion ist jetzt klar kanonisiert, aber ihr Hebel liegt kurzfristig in stabiler Diagnostik-/Contract-Telemetrie und langfristiger Drift-Kontrolle entlang der bestehenden Linien.
- Ein direkter Sprung in eine zweite Regionenklasse erhöht Cross-line- und Scope-Drift-Risiko stärker als den aktuellen operativen Nutzen.

Damit gilt:
- keine reflexartige zweite Expansion in BB24 Prompt 9,
- falls später erweitert wird, dann genau eine zweite Klasse erst nach zusätzlicher Stabilitätsbeobachtung der ersten Expansion.
