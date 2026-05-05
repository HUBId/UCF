# Serie BR1 Prompt 2: Hippocampus minimal und bounded in UCF einhängen

Status: Die erste echte Hippocampus-Integrationslinie ist jetzt als **minimal bounded region surface** explizit festgezogen und bleibt advisory-only ohne direkte Autorität.

## 1) Kanonische hippocampus integration map

Die kanonische Map enthält genau sechs Klassen:

1. `hippocampus input surface`
2. `hippocampus state surface`
3. `hippocampus output/advisory surface`
4. `hippocampus reference surface`
5. `blocked/deferred hippocampus path`
6. `non-canonical/internal-only hippocampus path`

Keine neue Meta-Plattform und keine neue Compute-Core-Linie wird eingeführt.

## 2) Minimaler Hippocampus-Input (bounded)

Zulässige Inputs bleiben strikt bounded:

- runtime/selection/context signal (bounded)
- advisory reference signal

Explizit nicht zulässig:

- direct tool/action control signal
- compute-internal raw state
- direct safety override signal
- implicit memory mutation signal

## 3) Minimaler Hippocampus-State

Der Hippocampus-State ist auf **context/reference-advisory shaping** begrenzt:

- keine Proposal-/Action-Autorität
- keine Execution-Autorität
- keine Retry-Orchestrierung
- keine Memory-Commit-Autorität
- keine Compute-Invocation-Autorität

## 4) Minimaler Hippocampus-Output (advisory-only)

Zulässige bounded Outputs:

- context-binding hint
- reference-binding hint
- retrieval-support hint
- caveat signal

Diese werden nur als advisory gelesen:

- runtime advisory read
- selection advisory read
- reference/context bounded read

Explizit nicht zulässig:

- no direct action trigger
- no direct execution trigger
- no direct retry trigger
- no direct memory commit
- no direct compute invocation
- no safety override

## 5) Context/Memory/Reference-Anbindung (bounded)

Hippocampus-bezogene Referenzen bleiben kanonisch innerhalb bestehender Reference-Semantik:

- current reference: bounded advisory support
- caveated reference: quality-limited advisory support
- stale/deferred reference: deferred or diagnostic-only read
- reference-only lane: kein Autoritätseskalationspfad

Wichtig: Es entsteht keine zweite Referenzwirklichkeit und keine implizite Memory-Persistenz.

## 6) Runtime/Selection-Rückbindung (bounded)

Runtime und Selection konsumieren denselben Hippocampus-Contract nur als bounded advisory Signal:

- keine direkte Proposal-Promotion
- keine direkte Action-Selektion
- keine direkte Execution-Freigabe
- keine implizite Planner-Logik

## 7) Modellgrenze (unverändert)

Der Hippocampus bleibt in BR1 Prompt 2 im:

- `abstract functional current mode`

Nicht automatisch geöffnet:

- Kuramoto-Aufweitung
- HH-Produktivintegration
- globale Neurodynamikplattform

Modellvertiefung bleibt eine gesonderte spätere Entscheidung.

## 8) Guard-/Scope-Grenzen

Die No-direct-* Grenze bleibt explizit:

- no direct action trigger
- no direct execution trigger
- no direct retry trigger
- no direct memory commit
- no direct compute invocation
- no safety override

Zusätzlich bleibt Scope begrenzt auf Hippocampus-only in BR1:

- no parallel opening of additional anatomical regions
- no policy-governance platform expansion
- no retry/queue/orchestration platform expansion
- no planner/agent platform expansion

## 9) Nächste Härtungsstufe (BR1)

1. Contract-level caveat/deferred/reference-only fixtures für Hippocampus ergänzen.
2. Runtime/Selection diagnostics snapshots um hippocampus-spezifische advisory Marker schärfen.
3. Reference freshness/staleness handling für hippocampus hints gezielt regressionssichern.
4. Doc-index/roadmap Referenzpfade für BR1 konsolidieren.
5. Erst danach über selektive Modellvertiefung entscheiden (expliziter Re-scope).
