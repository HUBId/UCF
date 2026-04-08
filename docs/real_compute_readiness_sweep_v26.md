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
- **Status (2026-04-07, Prompt 27):** **gezielt entschärft im produktiven Profilpfad**. `configs/prod.toml` pinnt jetzt `compute_backend = "burn"` und `validate_config_ladder` erzwingt für `prod` hart `compute_backend=burn` (kein stiller Fallback auf Stub).

### B) LFM/NSR readiness is explicit but not uniformly hard-gated
- **Modul/Pfad:** `runtime/ucf-compute/src/pipeline.rs`.
- **Problem:** LFM/NSR sind ehrlich typisiert (disabled/unavailable/blocked), aber je nach Feature/Mode optional statt global erzwungen.
- **Produktivrelevanz:** Unterschiedliche Deployments können formal „grün“ erscheinen, obwohl die gleiche Stage-Abdeckung fehlt.
- **Minimaler Fix:** Einen kleinen „required stage profile“-Check (z. B. burn-prod) im Readiness-Gate fest verdrahten.
- **Status (2026-04-07, Prompt 28):** **gezielt entschärft im produktiven Readiness-Gate**. `readiness-gate` enthält jetzt `required_stage_profile`: im `prod`-Profil wird fail-closed erzwungen, dass NSR im Explain-Outcome als `used` vorliegt und LFM-Stagesichtbarkeit vorhanden ist; nicht-`prod` bleibt explizit `SKIP`.

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

## 7) Readiness-Delta nach Prompt 27 (gezielter Single-Blocker-Fix)

- **Gewählter Blocker:** A) Canonical vs Runtime-Default (`prod` konnte auf Stub stehen bleiben).
- **Behoben:** Produktionsprofil (`configs/prod.toml`) ist auf Burn gepinnt; Config-Ladder verweigert `prod` ohne Burn.
- **Readiness-Effekt:** Der kanonische produktive Pfad ist jetzt im realen Ops-Startpfad hart gegen Stub-Drift abgesichert.
- **Weiterhin offen:** B) Required-Stage-Profilhärtung (LFM/NSR), C) feinere Worker-Capabilities, D) History required-on in allen Prod-Launchpfaden.

## 8) Readiness-Delta nach Prompt 28 (gezielter Single-Blocker-Fix)

- **Gewählter Blocker:** B) LFM/NSR readiness war explizit, aber nicht profilgebunden hart-gated.
- **Vorheriger Fehlerzustand:** `prod` konnte im Gate ohne harte Required-Stage-Prüfung formal passieren, obwohl NSR nicht als tatsächlich genutzte Stage (`used`) abgesichert war.
- **Behoben:** `readiness-gate` führt jetzt den Check `required_stage_profile` aus:
  - `prod`: `FAIL`, wenn NSR nicht `used` oder LFM-Sichtbarkeit fehlt.
  - nicht-`prod`: bewusst `SKIP` (keine stille Ausweitung auf Dev/Test).
- **Readiness-Effekt:** Der kanonische produktive Pfad hat jetzt eine explizite, diagnostisch sichtbare Stage-Coverage-Härtung direkt im Gate-Report statt impliziter Hoffnung.
- **Weiterhin offen:** C) feinere Worker-Capabilities, D) History required-on in allen Prod-Launchpfaden.

## 9) Entscheidung nach Prompt 29: Restblocker-Phase sauber beendet

### Repo-basierte Neubewertung der verbleibenden Punkte

- **C) Multi-Worker Device-Semantik minimal (`Cpu`/`Worker`)**
  - **Einordnung:** technisch echt offen, aber **kein Load-Bearing-Blocker** für den kanonischen Burn-Produktivpfad.
  - **Grund:** Der aktuelle kanonische Pfad benötigt keine weiter ausdifferenzierte Accelerator-Klassifikation, um deterministisch/fail-closed/produktiv nutzbar zu sein.
- **D) History required-on in allen Prod-Launchpfaden**
  - **Einordnung:** hohes Audit-/Repro-Verbesserungspotenzial, aber **kein verbleibender Kernblocker** für den bereits gehärteten kanonischen Compute-Startpfad.
  - **Grund:** Replay/Compare-Failures sind bereits explizit typisiert und diagnostisch sichtbar; fehlende History ist damit ein klarer, nachvollziehbarer Betriebszustand statt stiller Inkonsistenz.

### Harte Entscheidung

Es bleibt **kein weiterer load-bearing Restblocker** übrig, der die Reihenfolge des Ausbaupfads weiterhin zwingend auf „Restblockerabbau“ festnagelt.

Damit ist die **Restblocker-Phase abgeschlossen**. Ab diesem Stand ist der Wechsel auf den nächsten echten Ausbaupfad legitim.

### Erreichte Mindeststabilität (technische Beendigungsgrenze)

- Kanonischer Prod-Backendpfad ist fail-closed auf Burn gehärtet (`prod` ohne Stub-Drift).
- Required-Stage-Profile sind im `prod`-Readiness-Gate hart und sichtbar verdrahtet (NSR/LFM-Sichtbarkeit).
- Failure-/Provenance-/Readiness-Semantik bleibt explizit, strukturiert und diagnostisch auswertbar.

### Nachrangige offene Punkte (nicht blockerhaft, nächster Ausbaupfad)

1. Additive Worker-Capability-Tags für ehrlichere Heterogenitäts-Placement-Entscheide.
2. `history required-on` für produktive Launchprofile als Repro/Audit-Härtung.
3. Zusätzlicher Guard-Test für historische Persistenzpflicht im Prod-Profil.
4. Optional feinere Replay/Baseline-Betriebsreports für Ops-Ergonomie.
