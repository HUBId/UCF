# Serie BB24 Prompt 5: First-Region Integration Line (minimal, bounded)

Status: **erste reale Regionenexpansion minimal und bounded integriert**.

Auf Basis von BB24 Prompt 4 wird genau eine Regionenklasse angebunden:
`Attention/Selection-related`.

## 1) Kanonische First-Region-Integration-Map

Die kanonische Integrationsmap ist in
`runtime/ucf-compute/src/blue_brain_region_first_integration.rs`
als `CANONICAL_BLUE_BRAIN_FIRST_REGION_INTEGRATION_MAP` geführt und trennt strikt:

1. `RegionInputSurface`
2. `RegionStateSurface`
3. `RegionOutputAdvisorySurface`
4. `RegionReferenceSurface`
5. `BlockedDeferredRegionPath`
6. `NonCanonicalInternalOnlyRegionPath`

Damit bleibt die erste Regionsexpansion explizit und als Vorlage für spätere kontrollierte
Einhängungen nutzbar, ohne Meta-Plattform-Aufbau.

## 2) Minimale Input-Surface (allowed vs. blocked)

Erlaubte minimale Inputs (`BlueBrainFirstRegionInputSurface`):

- `attention_class` aus BB4 Selection-Semantik,
- `deferral_class` aus BB4/BB19 Deferral-Linie,
- `reference_validity` aus BB17/BB21 Reference-Linie,
- `context_priority` aus BB2/BB8 Context-/Runtime-Signalen.

Input-Guard (`BlueBrainFirstRegionInputGuard`) hält explizit fest, dass **nicht zulässig** bleibt:

- direkte Tool-/Action-Steuersignale,
- compute-interne Rohzustände,
- direkte Safety-Override-Eingänge,
- implizite Memory-Mutationsinputs.

## 3) Minimale State-Surface

Die Region trägt nur drei Zustände (`BlueBrainFirstRegionStateSurface`):

- `ActiveBoundedAdvisoryOnly`
- `BlockedDeferred`
- `NonCanonicalInternalOnly`

Nicht enthalten (bewusst out-of-scope):

- Execution-State-Mutationen,
- Retry-Orchestrierungszustände,
- Memory-Commit-/Persistenzzustände,
- Compute-Invocation-Zustände.

## 4) Minimale Output-/Advisory-Surface

`BlueBrainFirstRegionOutputSurface` erlaubt nur bounded advisory Signale:

- `CaveatHint`
- `PriorityHint`
- `DeferralHint`
- `ReferenceBoundedSignal`
- plus `BlockedDeferred` / `NonCanonicalInternalOnly` als boundary states.

Die Struktur enthält harte Verbotsfelder, die immer `false` bleiben:

- `direct_action_selection`
- `direct_execution_trigger`
- `direct_retry_trigger`
- `direct_memory_commit`
- `direct_compute_invocation`
- `safety_override`

Damit bleibt die Region explizit ohne neue Autoritätskanäle.

## 5) Runtime/Selection/Reference-Anbindung (bounded)

Die Region wird nur über bestehende Linien gespiegelt:

- **Runtime/Selection:** via vorhandene BB4 attention/deferral Klassen
  (`BlueBrainControlAttentionSelectionClass`, `BlueBrainCandidateDeferralLifecycleClass`).
- **Reference/Context:** via `BlueBrainReferenceValidity` und
  `BlueBrainContextEvidencePriorityClass`.

Die Auswertung erfolgt über
`evaluate_blue_brain_first_region_attention_selection(...)`
und liefert ausschließlich advisory-only Output.

## 6) Dynamics-Entscheid

Für diese erste Regionenklasse bleibt die Integration **abstract-functional**.
Es wird keine zusätzliche Kuramoto- oder HH-Produktivintegration aktiviert.

## 7) Guard-/Scope-/Safety-Linie

Verbindlich erhalten:

- keine direkte Action-/Retry-/Memory-/Compute-Autorität,
- keine Safety-Override-Semantik,
- keine implizite Reaktivierung deferred/non-canonical Pfade,
- keine Mehrfachregionen-Expansion in BB24 Prompt 5.

## 8) Nächste Schritte (BB24, 3-5 schmale Folgepunkte)

1. First-region surface in runtime diagnostics snapshots sichtbar machen (read-only).
2. Optionalen, rein diagnostischen router-pass-through für die advisory class ergänzen.
3. Kontrakt-/Doc-Alignment mit BB19/BB21 für reference-bounded signaling schärfen.
4. Zweite Regionenklasse vorbereiten, aber weiter `viable_but_not_first` halten,
   bis die first-region Telemetrie stabil ist.
