# Real Compute Stack Readiness Sweep v26

Stand: Repo-Zustand am 2026-04-07, ohne Zusatzannahmen.

Ziel: harte Abschlussprüfung der realen Compute-Surface (nicht Governance/Release-Prozesse).

## 1) Technische Readiness-Matrix (kurz, repo-basiert)

| Bereich | Status | Repo-Evidenz |
|---|---|---|
| Kanonische Modellpipeline (World→SAE→SSM→LFM) | **minimally production-usable** | Feste Stufenreihenfolge + strukturierte Result/Failure-Typen in `pipeline.rs`; kanonischer Onboarding-Builder auf Burn gepinnt. |
| Artifact-/Manifest-/Compatibility-Nutzung | **minimally production-usable** | `ModelStore` erzwingt Manifest, Hash-Verifikation, Allowlist-Pfade; Slot-Provenance + Compatibility-Gate in `backend_pack.rs`. |
| Burn als primärer Runtime-Pfad | **minimally production-usable** | `CANONICAL_ONBOARDING_BACKEND=Burn`, `CANONICAL_ONBOARDING_PACK=BurnToyV1`; Auswahl priorisiert Burn-Lane. |
| Candle als sekundärer Seam | **partial / constrained** | Candle ist optional feature-gated und nur als Compatibility-Seam dokumentiert/verdrahtet. |
| JEPA / NSR / LFM Readiness | **partial / constrained** | JEPA im Hauptpfad; LFM/NSR haben explizite Readiness- und Failure-States, können aber blockiert/disabled sein. |
| Kanonischer E2E-Referenzpfad | **real productive** | Eindeutig dokumentiert: `build_onboarding_reference_backend(seed) -> compute_canonical(request)`; keine stille Promotion auf Candle/Stub. |
| Validation / structured failures / Provenance | **real productive** | Kanonische Failure-Klassen, Validation-Summary, Stage-/Slot-/Backend-Provenance sind feste Top-Level-Contracts. |
| Bounded compute service | **real productive** | In-Memory-Service mit Admission, Queue, Lifecycle, Timeout/Failure-Klassen und deterministischen Datenstrukturen (BTree/VecDeque). |
| Scheduling / Worker Execution / Multi-Worker | **minimally production-usable** | Multi-Worker-Placement inkl. Suitability, Device-Suitability, Burn-Prio und Candle-Fallback. |
| Model Pack / Promotion / Active Slot Lifecycle | **minimally production-usable** | Active/Pin-Auflösung über `models/promoted/<slot>/<hash>`; Activation-Fehler und Slot-Target-State sind typisiert. |
| Compute API / Service Surface | **real productive** | `CanonicalComputeEntryPoint` bietet Submit/Status/Lifecycle/Operations/Replay/Baseline-Compare mit strukturierten Outcomes. |
| Capability / Placement / Device-Semantik | **minimally production-usable** | Separate Backend- vs. Device-Suitability und explizite Failure-Klassen vorhanden; Device-Klassen bewusst schmal (`cpu`, `worker`). |
| Operations Surface | **minimally production-usable** | Runtime-Operationen (`Snapshot`, `DrainScheduler`, `RefreshRuntime`) und degradierter Zustand sind explizit im API-Modell. |
| Job History / Replay | **minimally production-usable** | JSONL-Store + Replay-Report mit Determinism-Klassifizierung und Konfig-Diff. |
| Compatibility Gates | **minimally production-usable** | `ProductionCompatibilityGate` + Blockgründe (Contract/Slot/Backend/Placement/Activation) sind im Slot-Provenance-Modell. |
| Baseline Comparison | **minimally production-usable** | Candidate-vs-Baseline-Compare mit Konfig-Gleichheit, Completion/Failure-Delta und Work-Delta vorhanden. |
| Legacy Cleanup | **partial / constrained** | Legacy-Backend-Aliase sind parserseitig blockiert; Default-Backend bleibt aber `stub` (bewusste Compat-Lane). |

## 2) Kanonischer produktiver Pfad (explizit festgezogen)

1. **Kanonischer Compute-Entry-Point:** `CanonicalComputeEntryPoint` (`runtime/ucf-compute/src/service_surface.rs`).
2. **Kanonischer E2E-Referenzpfad:** `build_onboarding_reference_backend(seed)` + `ComputePipelineBackend::compute_canonical(request)`.
3. **Primärer Runtime-Pfad:** Burn (`BurnToyV1`) als kanonische Onboarding-Lane.
4. **Nicht-primäre Pfade:**
   - Candle = Compatibility-/Compare-Seam,
   - Stub = Dev/Compat-Default,
   - Worker = Ausführungs-/Isolation-Lane.

## 3) Load-bearing Restbrüche / Widersprüche

### A) Canonical vs Runtime-Default
- **Modul/Pfad:** `runtime/ucf-compute/src/backends.rs`, `runtime/ucf-compute/README.md`.
- **Problem:** Kanonischer Pfad ist Burn, aber `ComputeBackendConfig::default()` bleibt `Stub`.
- **Produktivrelevanz:** Ohne explizite Env-/Builder-Setzung kann produktionsnahe Laufzeit ungewollt in Compat-Lane landen.
- **Minimaler Fix:** In produktiven Startpfaden explizit `build_canonical_production_backend` (oder `UCF_COMPUTE_BACKEND=burn`) erzwingen und das in Ops-Runbooks als Pflicht markieren.

### B) LFM/NSR readiness is explicit but not uniformly hard-gated
- **Modul/Pfad:** `runtime/ucf-compute/src/pipeline.rs`.
- **Problem:** LFM/NSR sind ehrlich typisiert (disabled/unavailable/blocked), aber je nach Feature/Mode optional statt global erzwungen.
- **Produktivrelevanz:** Unterschiedliche Deployments können formal „grün“ erscheinen, obwohl die gleiche Stage-Abdeckung fehlt.
- **Minimaler Fix:** Einen kleinen „required stage profile“-Check (z. B. burn-prod) im Readiness-Gate fest verdrahten.

### C) Multi-Worker bleibt device-semantisch minimal
- **Modul/Pfad:** `runtime/ucf-compute/src/compute_service.rs`.
- **Problem:** Device-Klassen sind bewusst nur `Cpu`/`Worker`; Accelerator-Spezifika sind nicht Teil dieses Placement-Layers.
- **Produktivrelevanz:** Für echte heterogene Hardware-Flotten ist die aktuelle Platzierungssemantik nur begrenzt aussagekräftig.
- **Minimaler Fix:** Optionalen, strikt additiven Capability-Tag pro Worker einführen (ohne neue Orchestrierungsarchitektur).

### D) Replay/Baseline hängen an verfügbarer canonical_request/history
- **Modul/Pfad:** `runtime/ucf-compute/src/service_surface.rs`.
- **Problem:** Replay/Compare sind robust modelliert, aber bewusst blockiert bei fehlender Persistenz/inkompletter Replay-Konfiguration.
- **Produktivrelevanz:** Ohne konsequente History-Persistenz sinkt die Reproduzierbarkeit auf Live-Status statt belastbarer Audit-Tiefe.
- **Minimaler Fix:** History-Store in allen produktiven Launch-Profilen als required-on konfigurieren.

## 4) „Nicht mehr bauen“ im Kern (Fokusdisziplin)

Diese Ausbauten sind **nicht** Teil des Compute-Kernabschlusses:
- Governance-/Approval-Framework-Erweiterungen,
- Dashboard-/Monitoring-Plattform-Neubau,
- breite MLOps-/Experiment-Suiten,
- Cluster-/Orchestrierungs-Overengineering.

Kern bleibt: kanonischer Compute-Pfad, belastbare Failure-/Provenance-Semantik, reproduzierbare Ausführung.

## 5) Nächste 3–7 lohnende technische Schritte

1. **Burn als Runtime-Default in produktiven Profilen hart setzen** (keine Stub-Drift in Prod-Bootstraps).
2. **Kleiner Required-Stage-Readiness-Check** für burn-prod (LFM/NSR-Anforderungen explizit profilgebunden).
3. **History required-on für Prod-Launchpfade** inkl. Fail-fast bei nicht öffnendem History-Store.
4. **1 gezielter Guard-Test für „no silent fallback to stub in canonical launch“** im Runtime/Ops-Pfad.
5. **Placement-Capability-Tag (minimal)** pro Worker (z. B. `burn_capable`, `candle_capable`) für ehrlichere Scheduling-Entscheide.

## 6) Abschlussurteil

Der Real-Compute-Stack ist für den **kanonischen Burn-basierten Referenzpfad** technisch belastbar. Für breitere Produktionsnutzung sind vor allem Default-/Profile-Härtung und verpflichtende Replay/History-Disziplin die verbleibenden High-Leverage-Lücken.
