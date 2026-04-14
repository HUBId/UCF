# Serie F Abschluss: Expert Runtime Surface / API Hardening

## 1) Serie-F-Kernprüfung (hart, repo-basiert)

| Bereich | Ist-Zustand | Kurzbefund |
| --- | --- | --- |
| Standard vs Expert Entry Contracts | **stable expert surface core** | `RuntimeEntryClass` trennt `standard_canonical`, `expert_high_trust`, `internal_dev_test` mit zentraler Shape-/Safety-Mapping-Quelle; Standard blockiert Expert-Replay/Ops deterministisch. |
| Expert Ops Actions / Interventions | **production-usable but constrained** | `run_operation_with_entry` trägt class/scope/result/mutation-Semantik; `drain_scheduler`/`rehydrate_history` sind high-trust und rail-gebunden, `refresh_runtime` bleibt explizit unsupported im in-memory Runtime-Kontext. |
| Canonical Runtime Snapshots / Expert Diagnostics | **production-usable but constrained** | Snapshot-/Diagnostics-Core sind auf gemeinsame Kernzustände normalisiert (`current|partial|stale|unavailable` und `available|partial|unavailable|blocked`), mit bewusst separatem `internal_only` Extension-Seam. |
| Expert Replay-/Rollout-/Ops Workflows | **partial / diagnostic** | `workflow_view` liefert kanonische High-Trust-Workflowklassen und Transition-States; es bleibt ein Diagnose-/Contract-Layer ohne Workflow-Engine oder zusätzliche Orchestrierungsplattform. |
| Mutation Boundaries / Safety Rails | **stable expert surface core** | Runtime-Ops reporten `mutation_boundary`, `mutation_result`, optionale `blocked_by` und intended/resulting-state-change; Core-Semantik zwischen Outcome-Code und Mutation-Result wird explizit validiert. |
| Shared Surface Core / Contract Drift Control | **stable expert surface core** | Shared Mapping/Compatibility (`RuntimeEntryClass`-Mapping + code↔mutation consistency rule) reduziert Drift zwischen Standard-, Expert- und Internal-Pfaden; Abdeckung durch gezielte Konsistenztests. |
| Bewusst außerhalb Serie F | **intentionally deferred** | Keine Auth-/RBAC-/Tenant-Plattform, keine Admin-/Governance-Control-Plane, keine Workflow-Orchestrierungs-Engine. |

## 2) Serie-F-Abschlussmatrix

| Status | Bereiche |
| --- | --- |
| **stable expert surface core** | Entry-Contract-Kern (Standard/Expert/Internal), Mutation-Boundary-/Safety-Rail-Semantik, Shared Contract-Drift-Control |
| **production-usable but constrained** | Expert runtime interventions (history/scheduler/runtime ops), canonical snapshot + diagnostics core |
| **partial / diagnostic** | Workflow view für replay/rollout/ops als technische Diagnose- und Übergangsabbildung |
| **intentionally deferred** | Auth/Governance/Control-Plane- und Workflow-Engine-Themen außerhalb von `runtime/ucf-compute` |

## 3) Explizite Abschlusslinie für Serie F

Serie F gilt im aktuellen Repo-Stand als **abgeschlossen** für Expert Runtime Surface / API Hardening.

Als gebaut und tragfähig gelten jetzt:
1. Explizite Expert-vs-Standard-vs-Internal Entry-Verträge mit zentraler Shape-/Safety-Mapping-Quelle.
2. Kontrollierte High-Trust Runtime-Ops inkl. Outcome-/Mutation-Semantik und Safety-Rail-Blockern.
3. Canonical Snapshot-/Diagnostics-Core mit klarer Trennung zwischen produktivem Kern und `internal_only`-Extension.
4. Replay-/Ops-/Rollout-nahe Workflow-Contracts als einheitliche, im Snapshot sichtbare technische Übergangsfläche.

Offene Punkte, die **nicht mehr load-bearing** für Serie F sind:
- fehlende Auth/RBAC/Tenant- oder Governance-Plattform,
- fehlende externe Workflow-Orchestrierung,
- weitergehende Reliability-/Control-Plane-Automation jenseits des bestehenden Runtime-Surface-Kerns.

Weitere Arbeit an diesen Punkten ist ab jetzt **neue Vertiefungsserie** und nicht mehr Teil von Serie F.

## 4) Nächste Serien (1–3) mit höchstem Hebel

1. **Serie G (priorisiert): Long-run operational resilience / service hardening**
   - Höchster Hebel: Nach stabilisiertem Expert-Surface ist der nächste Engpass die Langlaufstabilität der bestehenden Runtime-/Ops-Pfade (Recovery-Verhalten, Betriebsrobustheit unter Dauerlast, degradationsfeste Service-Resilienz).
2. **Serie H (nachrangig): Advanced evidence / trace / reasoning integration**
   - Hebel: vertieft technische Nachvollziehbarkeit auf dem jetzt stabileren Surface, ist aber weniger load-bearing als direkte Resilience-Härtung.
3. **Serie I (nachrangig): Narrow final cleanup / canonical reference consolidation**
   - Hebel: reduziert Referenz-/Dokudrift, aber primär Konsolidierung statt unmittelbarer Runtime-Härtung.

## 5) Priorisierung: exakt nächste Serie

**Start als Nächstes: Serie G (Long-run operational resilience / service hardening).**

Warum zuerst:
- Höchster unmittelbarer technischer Hebel auf reale Betriebsfestigkeit der bereits gehärteten Expert-Surface-Verträge.
- Stabilisiert den bestehenden Runtime-Kern unter längeren/fehleranfälligen Betriebsbedingungen statt neue Oberflächen zu eröffnen.

Warum die anderen nachrangig sind:
- Serie H profitiert von einem robusteren Langlauf-Betriebskern und liefert danach verlässlichere Evidence/Trace-Mehrwerte.
- Serie I ist wichtig für Pflege/Kanonisierung, erhöht aber kurzfristig weniger die operative Belastbarkeit.

## 6) Referenzen (Repo-Basis)

- Entry-/Contract-Kern: `runtime/ucf-compute/src/contracts.rs`, `runtime/ucf-compute/src/service_surface.rs`.
- Exportierte Runtime-Surface-Verträge: `runtime/ucf-compute/src/lib.rs`.
- Workflow-/Mutation-Referenz: `docs/expert_runtime_workflows_v1.md`.
- Runtime-README-Serie-F-Abschnitte: `runtime/ucf-compute/README.md`.
