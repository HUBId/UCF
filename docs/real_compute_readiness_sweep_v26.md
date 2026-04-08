# Real Compute Stack Abschlussmatrix (Prompt 40)

Stand: Repo-Zustand am 2026-04-08.

Ziel: harte technische Abschlusslinie für die aktuelle Real-Compute-Ausbaustufe und Priorisierung der **nächsten** Vertiefungsreihe.

## 1) Abschlussmatrix (repo-basiert, kurz)

| Bereich | Einstufung | Harte Repo-Basis |
|---|---|---|
| Kanonischer produktiver Pfad (`build_onboarding_reference_backend` → `compute_canonical`) | **stable core** | Burn-pinned Canonical-Onboarding (`CANONICAL_ONBOARDING_BACKEND/PACK`) + fester Canonical-Stage-Contract in `pipeline`. |
| Compute Entry / Service Surface (`CanonicalComputeEntryPoint`) | **stable core** | Einheitlicher Entry für submit/status/lifecycle/ops/replay/baseline compare inkl. typisierter Outcomes. |
| Pipeline/Contracts/Failure-Taxonomie | **stable core** | `CanonicalPipelineResult` / `CanonicalPipelineFailure` + Failure-Kind/Fault-Domain/Isolation-Dispositions als tragende Vertragsfläche. |
| Bounded Service + Scheduling (Admission/Queue/Lifecycle/Timeout/Accounting) | **stable core** | `InMemoryComputeService` + deterministische Datenstrukturen (`BTreeMap/BTreeSet/VecDeque`) + strukturierte Lifecycle-Events. |
| Multi-Worker/Placement/Capacity (lokal + Worker IPC) | **production-usable but constrained** | Suitability-/Placement-/Capacity-Pfade und Worker-States sind vorhanden; Device-Semantik bleibt bewusst grob (`cpu`, `worker`). |
| Promotion/Rollout/Gates/Baselines | **production-usable but constrained** | Slot-Pfade (`active/candidate/compare/shadow`), Readiness-Gates inkl. `required_stage_profile`, Baseline-Compare vorhanden; bewusst schmaler Automationsgrad. |
| Ops/History/Replay/Recovery | **production-usable but constrained** | History-store, Replay, Recovery-Dispositionen und Runtime-Operationen vorhanden; Failures bei fehlender Persistenz bleiben explizit sichtbar. |
| Config/Modes/Readiness/Isolation | **production-usable but constrained** | `configs/prod.toml` pinnt Burn; Readiness-Gate erzwingt profilgebundene Stage-Checks; Isolation/Fault-Domain-Semantik ist typisiert. |
| Tiefe heterogene Device-/Accelerator-Spezialisierung | **intentionally deferred** | Nicht Teil des aktuellen Placement-Designs; keine feingranulare Accelerator-Klassifikation im Kernpfad. |
| Vollautomatische verteilte Orchestrierung/Fleet-Scheduling | **intentionally deferred** | Kein Cluster-Orchestrator in dieser Ausbaustufe; bounded local scheduler + optional worker lane bleiben die Grenze. |

## 2) Explizite Abschlusslinie dieser Ausbaustufe

Diese Reihe ist technisch **abgeschlossen** für den Grundaufbau des Real Compute Stacks:

1. Der kanonische produktive Pfad ist gebaut, eindeutig und belastbar.
2. Der bounded compute service inkl. Admission/Lifecycle/Placement-/Failure-/Provenance-Semantik ist tragfähig.
3. Promotion-/Rollout-/Readiness-/Replay-/Recovery-Flächen sind produktiv nutzbar, aber bewusst mit engen Grenzen.

Nicht mehr in diese Reihe zurückziehen:

- Neubau eines verteilten Orchestrierungs- oder Governance-Kontrollplanes,
- breite Plattform-/Dashboard-/Meta-Programm-Pakete,
- „zweiter“ konkurrierender Compute-Grundpfad.

Ab hier ist weitere Arbeit **gezielte Vertiefung** auf dem bestehenden Kern, kein weiterer Grundaufbau.

## 3) Nächste Vertiefungsrichtungen (Top-Hebel, 1–3)

1. **Priorität 1: Replay/Reproducibility-Härtung auf kritischen Pfaden**
   - Prod-Profil fail-closed auf History-Verfügbarkeit,
   - engere Replay-Vollständigkeit (canonical request/history) für Compare/Baseline,
   - klare, testbare Repro-Garantien für Audit-relevante Jobs.
2. **Priorität 2: Rollout-/Promotion-Automation robuster machen**
   - stärker deterministische Candidate→Baseline→Promotion-Übergänge,
   - klarere Blockgründe/Auto-Abbruchkanten bei Gate-Mismatch.
3. **Priorität 3: Worker-/Placement-Robustheit vertiefen**
   - additive Capability-Tags und robustere Degradation bei heterogener Worker-Flotte,
   - ohne neuen Orchestrator-Unterbau.

## 4) Genau eine nächste Prompt-Reihe

**Nächste Reihe: Replay/Reproducibility-Härtung (Priorität 1).**

Warum jetzt höchster Hebel:

- Sie erhöht direkt die Verlässlichkeit der bereits gebauten Promotion-/Baseline-Entscheidungen,
- sie schließt die größte operative Restlücke zwischen „läuft“ und „auditierbar reproduzierbar“,
- sie nutzt bestehende History/Replay-Strukturen statt neue Architekturflächen zu öffnen.

Warum die anderen nachrangig:

- Rollout-/Promotion-Automation bringt erst dann maximalen Nutzen, wenn Repro-Pfade hart belastbar sind.
- Worker-/Placement-Vertiefung ist wichtig, aber aktuell weniger kritisch als reproduzierbare Entscheidungsgrundlagen auf dem kanonischen Pfad.

## 5) Minimale Konsistenzchecks für die Abschlussaussage

- Kanonischer Referenzpfad weiterhin vorhanden (Onboarding-Builder + canonical compute contract).
- Produktiver Entry-Point weiterhin eindeutig (`CanonicalComputeEntryPoint`).
- `prod` bleibt auf Burn gepinnt und Readiness-Stage-Profile bleiben profilgebunden fail-closed.
