# Serie BR4 Prompt 2: Basal Ganglia minimal und bounded integrieren

Status: `basal_ganglia_like_region` ist als vierte echte anatomische UCF-/Blue-Brain-Region nach BR1 Hippocampus, BR2 Amygdala und BR3 Thalamus minimal eingehängt. Die Integration bleibt eine kontrollierte, bounded, advisory-only Action-Gating-Linie und erzeugt keine direkte Action-, Retry-, Memory-, Compute-, Execution- oder Safety-Autorität.

Diese Datei konsolidiert die kanonische BR4-Prompt-2-Integrationsfläche. Sie ergänzt die BR4-Prompt-1-Rollenkarte und hängt an bestehende Selection-/Priority-/Deferral-/Contract-, Runtime-/Execution-interface- und Reference-/Context-Linien an, ohne eine zweite Wahrheitsquelle oder neue Meta-Plattform zu öffnen.

## 1) Harte Prüfung gegen bestehende UCF-Linien

Die kleinste echte Basal-Ganglia-Integrationsfläche liegt an den bestehenden Selection-/Action-gating- und Execution-readiness-Konsumpunkten:

- **Selection/Priority/Deferral/Contracts:** Basal Ganglia darf selection-, priority- und deferral-nahe Signale lesen und daraus nur bounded advisory hints ableiten.
- **Action-gating/channel-selection/suppression:** Basal Ganglia darf action-gating posture, suppression/inhibition posture und channel-selection arbitration nur als Hinweisform tragen.
- **Runtime/Execution-interface:** Runtime und execution-interface sehen Basal Ganglia nur als advisory/caveated readiness signal, nie als Auslöser.
- **Reference/Context:** Reference/Context dürfen eine bounded reference basis liefern; stale, caveated, blocked, insufficient und reference-only bleiben diagnostisch begrenzt.
- **Regionenabgrenzung:** `hippocampus_like_region` bleibt context/reference/memory-association-lastig, `amygdala_like_region` bleibt salience/valence/caveat-lastig, `thalamus_like_region` bleibt relay/gating/routing-lastig, `basal_ganglia_like_region` bleibt action-gating/suppression/channel-selection-lastig.

Basal Ganglia darf nur region-lokale advisory state tragen. Nicht berührt werden dürfen Tool-/Action-Control, compute-interne Rohzustände, Safety-Override-Zustände, Retry-Orchestrierung, Memory-Mutation/Persistenz und neue allowed-actions-Flächen.

## 2) Kanonische Basal-Ganglia integration map

Die kanonische Integration map enthält genau diese Klassen:

| Klasse | Status | Bedeutung |
| --- | --- | --- |
| `basal-ganglia input surface` | canonical bounded | Bestehende Runtime-readiness-, Selection-priority-, Selection-deferral-, Action-gating-posture-, Context-reference- und Reference-validity-Signale als bounded/advisory Lesebasis. |
| `basal-ganglia state surface` | canonical bounded | Region-lokale Zustände für action-gating advisory, suppression/inhibition advisory, channel-selection arbitration, execution-readiness caveat, reference-only, deferred, blocked, insufficient und non-canonical. |
| `basal-ganglia output/advisory surface` | advisory-only | `gating-hint`, `suppression-hint`, `channel-selection hint`, `execution-readiness caveat`, `reference-bounded signal`, blocked/deferred und insufficient diagnostics. |
| `basal-ganglia reference surface` | reference-bounded only | Reference/Context-Kopplung ohne Memory-Persistenz, Retrieval-Ausbau oder zweite Referenzwirklichkeit. |
| `blocked/deferred basal-ganglia path` | diagnostic-only | Blocked/deferred/insufficient/caveated Signale bleiben diagnostisch und erzeugen keine Autorität. |
| `non-canonical/internal-only basal-ganglia path` | excluded | Interne oder nicht-kanonische Pfade sind nicht Teil der operationalen Surface. |

## 3) Minimale Input-Surface

Zulässige Input-Signale sind:

- `RuntimeReadinessSignal` als advisory-only Runtime-readiness Lesebasis,
- `SelectionPrioritySignal`, `SelectionDeferralSignal` und `ActionGatingPostureSignal` als advisory-only Selection-/Action-gating-Lesebasis,
- `ContextReferenceSignal` und `ReferenceValiditySignal` als reference-only bounded Lesebasis.

Explizit unzulässig sind:

- direkte Tool-/Action-Steuersignale,
- compute-interne Rohzustände,
- direkte Safety-Override-Eingänge,
- implizite Memory-Mutationsinputs.

## 4) Minimale Output-/Advisory-Surface

Zulässige bounded Outputs sind ausschließlich:

- `gating-hint` für bestehende Selection-/Action-gating-Sichten,
- `suppression-hint` für bounded inhibition/defer/block Lesarten,
- `channel-selection hint` für bounded channel arbitration,
- `execution-readiness caveat` für caveated execution-interface Lesarten,
- `reference-bounded signal` für bestehende Reference/Context-Semantik,
- blocked/deferred/insufficient diagnostics,
- non-canonical/internal-only diagnostic exclusion.

Nicht zulässig bleiben:

- direct action selection,
- direct action trigger,
- direct execution trigger,
- direct retry trigger,
- direct memory commit,
- direct compute invocation,
- safety override.

## 5) Bounded Anschluss an Selection und Action-gating

Selection sieht Basal Ganglia ausschließlich als advisory action-gating/suppression/channel-selection Unterstützer. Gating, suppression und channel-selection dürfen bestehende Selection-Lesarten nur caveated, deferred, blocked, insufficient oder advisory markieren. Es entsteht keine Proposal-, Planner-, Action- oder Execution-Autorität.

## 6) Bounded Anschluss an Runtime und Execution-interface

Runtime sieht Basal Ganglia ausschließlich als bounded diagnostic/advisory Contract-Signal. Das Execution-interface darf daraus nur eine `execution-readiness caveat` Lesart konsumieren. Basal Ganglia kann keine Execution starten, keine Compute-Invocation auslösen, keine Retry-Queue bedienen und keine Safety-Entscheidung überschreiben.

## 7) Bounded Anschluss an Reference und Context

Kanonische basal-ganglia-bezogene Referenzen sind ausschließlich reference-bounded Context-/Reference-validity-Hinweise. Stale, caveated, reference-only, blocked und insufficient Fälle bleiben diagnostisch sichtbar und erzeugen keine zweite Referenzwirklichkeit, keine Retrieval-/Consolidation-Linie und keine implizite Memory-Persistenz.

## 8) Modellgrenze

Der aktuelle Modus bleibt der in BR4 Prompt 1 festgelegte `abstract functional current mode`. Diese Integration erzeugt keine Kuramoto-Produktivaufweitung, keine Hodgkin-Huxley-Produktivintegration, keine Subnukleus-/Dopamin-/Spiking-Architektur und keine globale Neurodynamikplattform. Spätere Modellvertiefung erfordert einen gesonderten, expliziten Re-Scope.

## 9) No-direct-* und Scope-Grenzen

Die Guard-Linie bleibt unverändert:

- kein direct action trigger,
- kein direct action selection,
- kein direct execution trigger,
- kein direct retry trigger,
- kein direct memory commit,
- kein direct compute invocation,
- kein safety override,
- keine neue allowed-actions-Erweiterung,
- keine Planner-/Agenten-/Policy-/Governance-/Retry-/Queue-/Orchestration-Plattform,
- keine Retrieval-/Consolidation-/Reasoning-Plattform,
- keine parallele Öffnung weiterer anatomischer Regionen.

## 10) BR4 nächste Schritte

1. Diagnostics/contract hardening für Basal-Ganglia-Surfaces gegen Dokumentation und Tests schärfen.
2. Readiness-sweep für advisory-only und no-direct-* Evidenz ergänzen.
3. Cross-region contract matrix für Hippocampus/Amygdala/Thalamus/Basal Ganglia ohne neue Autorität konsolidieren.
4. Stale/caveated/reference-only Basal-Ganglia-Fälle in Guard-/Readiness-Doku sichtbar machen.
5. Erst danach über eine gesonderte bounded Modellvertiefung entscheiden.
