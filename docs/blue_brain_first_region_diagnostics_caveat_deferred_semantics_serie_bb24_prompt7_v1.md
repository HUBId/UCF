# Serie BB24 Prompt 7: First-Region Diagnostics-/Caveat-/Deferred-Semantik (gehärtet)

Status: **erste regionsspezifische Diagnostics-Linie ist kanonisch und runtime-/selection-/reference-konsistent geschärft**.

Diese Linie baut direkt auf BB24 Prompt 6 auf und erweitert die bestehende first-region Contract-Surface **nur diagnostisch**, ohne neue Autoritätskanäle oder Mehrfach-Regionen-Ausbau.

## 1) Kanonische First-Region Diagnostics Map

Kanonisch geführt in
`runtime/ucf-compute/src/blue_brain_region_first_integration.rs` über
`CANONICAL_BLUE_BRAIN_FIRST_REGION_DIAGNOSTIC_MAP`:

- `RegionAdvisoryOnlyDiagnostic`
- `RegionCaveatedDiagnostic`
- `RegionDeferredDiagnostic`
- `RegionBlockedDiagnostic`
- `RegionInsufficientDiagnostic`
- `RegionDiagnosticOnlyState`
- `NonCanonicalInternalOnlyRegionDiagnosticPath`

Damit sind advisory-only/caveated/deferred/blocked/insufficient/diagnostic-only plus non-canonical/internal-only explizit getrennt.

## 2) Advisory-only vs Caveated

- `RegionAdvisoryOnlyDiagnostic` bleibt ein bounded positives Hinweis-Signal, ohne direkte Action/Execution/Retry/Memory/Compute-Autorität.
- `RegionCaveatedDiagnostic` bleibt ein abgeschwächter Zustand bei caveated/stale Referenzbasis.
- Caveated wird nicht stillschweigend zu starkem advisory-only Signal promoted.

## 3) Deferred vs Blocked

- `RegionDeferredDiagnostic` = bounded Zurückstellung (z. B. `CandidateDeferred*`).
- `RegionBlockedDiagnostic` = begrenzender Contract-/Safety-/Reference-Zustand (z. B. `CandidateRejected`, `CandidateInsufficient`, `CandidateStale`).
- `deferred != blocked`, `blocked != failed execution` bleiben explizit getrennt.

## 4) Insufficient und Diagnostic-only

- `RegionInsufficientDiagnostic` signalisiert fehlende tragfähige bounded Basis (`BlueBrainReferenceValidity::Insufficient`).
- `RegionDiagnosticOnlyState` signalisiert reine Sichtbarkeit (u. a. reference-only), aber keine operative advisory support basis.
- Beide bleiben strikt ohne direkte Execution-/Retry-/Memory-/Compute-Side-Effects.

## 5) Runtime-/Selection-/Reference-Konsistenz

`BlueBrainFirstRegionOutputSurface` trägt nun zusätzlich ein einheitliches `diagnostic_state` Feld.
Damit lesen Runtime, Selection und Reference dieselbe kanonische regionsspezifische Diagnostics-Sprache statt getrennten Ad-hoc-Interpretationen.

## 6) Bounded dynamics und Scope-Grenzen

Für BB24 Prompt 7 unverändert:

- keine zusätzliche produktive Dynamics-Steuerkopplung,
- keine HH-Produktivintegration,
- keine neue Regionsklasse,
- keine Planner-/Policy-/Retry-/Queue-Plattform.

No-direct-* bleibt unverändert hart:

- kein direct action trigger,
- kein direct execution trigger,
- kein direct retry trigger,
- kein direct memory commit,
- kein direct compute invocation,
- kein safety override.
