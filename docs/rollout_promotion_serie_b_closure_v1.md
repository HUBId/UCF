# Rollout / Promotion Hardening — Serie B Abschlussmatrix v1

Stand: Repo-Zustand am 2026-04-10.

Scope dieser Abschlussprüfung: nur der tatsächlich implementierte Rollout-/Promotion-Kern in `runtime/ucf-compute` (Promotion-Entscheidung, Compare/Shadow-Signalpfad, Activation/Fallback/Rollback, Rollout-Diagnostics), ohne Governance/Release/Incident-Management.

## 1) Serie-B-Kernprüfung (hart, repo-basiert)

- **Promotion Decisions:** belastbar als signalgebundene technische Entscheidung (`SlotPromotionDecision`) mit expliziten Blockern, Disposition und compare/shadow Kontext; Promotion-State wird aus verifizierten Pfaden + Warmup + Gate-/Runtime-Signalen abgeleitet.
- **Compare-/Shadow-Evaluierung:** produktionsnutzbar, aber constrained: Compare/Shadow werden deterministisch aggregiert, erfassen Drift/Envelope-Probleme und liefern technische Outcomes/Context; sie sind Diagnose-/Promotion-Signale, kein autonomes Scoring-/Ranking-System.
- **Activation / Fallback / Rollback:** belastbar mit fail-closed Activation-Planung, Guardrail-Gründen, Fallback auf prior active und verifizierbarer Rollback-Assessments.
- **Rollout Provenance / History / Diagnostics:** belastbar als kompakte `SlotRolloutDiagnostics`-Sicht inkl. Events, ProblemKind, RecoveryOutcome und konsolidierter Rollout-Lage.
- **Guardrails / Blast-Radius-Begrenzung:** belastbar durch GuardedActive/Blocked/Reverted-Scope, deduplizierte Guardrail-Reasons sowie Events, die wider activation explizit verhindern bzw. revertieren.
- **Rollout signal consolidation:** belastbar als feste Konsolidierungsfunktion über Gate-/Compare-/Guardrail-/Runtime-Gruppen und kanonische `RolloutSignalSituation`.
- **Recovery nach bad activation / unstable candidate:** belastbar als explizite Recovery-Klassifikation (`FallbackToPriorActive`, `RollbackCompleted`, `IncompleteOrBlocked`) plus Ereignisfolge für Stabilisierung oder unvollständige Erholung.

Bewusst außerhalb Serie B:
- Keine externe Audit-Plattform oder Incident-Governance.
- Keine dauerhafte Orchestrierungs-/Release-Historie außerhalb des Runtime-Kerns.
- Keine adaptive Scoring- oder Policy-Ranking-Engine für Promotion.

## 2) Serie-B-Abschlussmatrix

| Bereich | Status | Harte Repo-Basis |
|---|---|---|
| Promotion Decisions auf technische Signale | **stable rollout core** | `slot_promotion_decision` koppelt State/Blocker/Disposition an verifizierte Active/Candidate/Compare/Shadow-Pfade, Warmup und technische Signale (`baseline_comparison_ready`, `runtime_path_production_usable`, `compare_or_shadow_diagnostic_ready`). |
| Activation / Fallback / Rollback Semantik | **stable rollout core** | `assess_slot_activation` + `assess_slot_rollback` erzwingen fail-closed Aktivierung, Guardrails, Fallback-Status und verifizierbaren Rollback-Pfad. |
| Rollout Provenance / History / Diagnostics | **stable rollout core** | `slot_rollout_diagnostics` erzeugt typisierte Fortschritt-/Problem-/Recovery-Klassifikation und kanonische Rollout-Events in einer kompakten Diagnosefläche. |
| Guardrails / Blast-Radius Begrenzung | **stable rollout core** | Guardrail-Gründe und Scope-Transitionen (`GuardedActive`, `Blocked`, `Reverted`) verhindern ungeprüfte Weitung und erfassen erzwungene Reverts. |
| Compare-/Shadow-Evaluierung im Rollout | **production-usable but constrained** | `EnablementComputeBackend` sammelt deterministische Compare-Window-Signale inkl. Digest-/Envelope-Abweichungen, bleibt aber bewusst ein technischer Vergleichs- und Diagnosepfad. |
| Rollout signal consolidation | **production-usable but constrained** | `consolidate_rollout_signals` konsolidiert belastbar, aber als regelbasierte, kompakte Lagebestimmung (kein probabilistisches Ranking). |
| Diagnose-Feingranularität für Grenzfälle | **partial / diagnostic** | Detailereignisse und `RolloutProblemKind` sind vorhanden, aber weiterhin auf runtime-nahe technische Diagnostik begrenzt. |
| Externe Governance/Approval/Incident-Prozesse | **intentionally deferred** | Absichtlich außerhalb des Implementierungsscopes von `runtime/ucf-compute` und der Serie-B-Härtung. |

## 3) Explizite Abschlusslinie für Serie B

Serie B gilt als **technisch abgeschlossen** für Rollout / Promotion Hardening im aktuellen Repo-Kern:

1. Promotion-Entscheidungen, Activation/Fallback/Rollback und Guardrail-gesteuerte Begrenzung sind gebaut und load-bearing tragfähig.
2. Compare/Shadow sowie Rollout-Signal-Konsolidierung liefern produktionsnutzbare, reproduzierbare technische Evidenz für Promotion-/Aktivierungsentscheidungen.
3. Recovery nach schlechter Aktivierung bzw. instabilem Kandidaten ist mit Fallback-/Rollback-Ausgang und Diagnoseereignissen explizit modelliert.

Offene Punkte, die **nicht mehr load-bearing für Serie B** sind:
- Externe Governance-/Approval- und Incident-Prozesswelten.
- Ausbau zu einer eigenständigen Scoring/Ranking-Plattform für Promotion.
- Breitere kosten-/kapazitätsorientierte Runtime-Optimierung.

Konsequenz: weitere Arbeit ist eine **neue Vertiefungsserie**, nicht „Serie B weiterführen“.

## 4) Nächste Vertiefungsserien nach Serie B (Top-Hebel)

1. **Priorität 1 — Serie C: Replay / Reproducibility Hardening für Rollout-Pfade**
   - Hebel: macht Promotion-/Recovery-Entscheidungen revisionssicherer und reproduzierbar über längere Zeiträume.
2. **Priorität 2 — Serie D: Capacity / Cost / Runtime Optimization**
   - Hebel: verbessert Effizienz auf einem nun belastbaren Rollout-Kern.
3. **Priorität 3 — Serie E: Device / Backend Specialization Hardening**
   - Hebel: vertieft hardware/backend-spezifische Aktivierungspfade, nachdem Repro- und Effizienzgrundlagen stehen.

### Genau eine priorisierte nächste Serie

**Start als nächstes: Serie C (Replay / Reproducibility Hardening für Rollout-Pfade).**

Kurzbegründung:
- Höchster unmittelbarer Hebel auf Verlässlichkeit der bereits load-bearing Rollout-/Promotion-Entscheidungskette.
- Serie D optimiert primär Kosten/Latenz, erhöht aber ohne zusätzliche Repro-Härtung nicht die Beweisfestigkeit von Aktivierungsentscheidungen.
- Serie E bleibt nachrangig, da Spezialisierung auf Geräte/Backends erst nach robuster Repro-Linie maximalen Nutzen bringt.

## 5) Minimale Konsistenzchecks für diese Abschlusslinie

- `cargo test -p ucf-compute rollout_diagnostics_capture_blocked_and_inconclusive_signals`
- `cargo test -p ucf-compute rollout_diagnostics_capture_progressed_active_path_when_no_candidate`
- `cargo test -p ucf-compute rollout_diagnostics_distinguish_compare_shadow_only_scope`
- `cargo test -p ucf-compute activation_assessment_distinguishes_degraded_from_rollback_semantics`
- `cargo test -p ucf-compute promotion_decision_marks_runtime_context_mismatch_as_not_comparable`
- `cargo test -p ucf-compute shadow_failure_never_breaks_primary`

Diese Checks sichern gezielt die Abschlussaussage (Promotion/Compare/Activation/Recovery/Guardrails), ohne eine neue Testwelle zu starten.
