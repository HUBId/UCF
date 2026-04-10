# Runtime Optimization View (Serie C Prompt 6)

Die Runtime-Optimization-View konsolidiert bestehende Queue-/Capacity-/Warmup-/Work-/Hotspot-Signale in eine **schmale operative Sicht** ohne Dashboard-/Analytics-Plattform.

## Konsolidierte Signale

- Queue-/Deferral-Druck aus `capacity_queue_disposition`, Queue-Wartezeit und Placement-Outcomes (`deferred`, `queued awaiting better placement`).
- Capacity-Druck aus `CapacityPressure` (`constrained`, `saturated`, `backpressured`, `temporarily_unschedulable`).
- Warmup-/Readiness-Kontext aus Warmup-State (`cold`, `stale`, `blocked`) inkl. Cold-Start-Hinweisen.
- Work-/Cost-/Hotspot-Kontext aus `degraded_stage_count` und dominantem Stage-Anteil.
- Historisches Feedback aus `PlacementOptimizationFeedbackView` (strong/weak/stale/contradicted/insufficient).

## Kanonische Runtime-Lagen

Die abgeleitete Optimization-View unterscheidet jetzt explizit:

- `healthy_and_efficient`
- `constrained_by_capacity`
- `constrained_by_cold_or_warmup`
- `constrained_by_dominant_stage_hotspot`
- `degraded_but_serviceable`
- `mixed_optimization_picture`
- `inconclusive`
- `failure_unrelated_to_optimization` (nur wenn Fehlerklasse nicht auf typische Runtime-Optimierungsengpässe zeigt)

## Nutzung in Scheduling/Placement/Warmup/History/Ops

- Multi-Worker-Placement ergänzt Entscheidungen um `optimization_state=...` in `decisive_signals`, damit Deferral-/Constrained-Entscheidungen im konsolidierten Lagebild erklärbar bleiben.
- Job-History persistiert `optimization_view` pro Record als kompakte Lage (state, bottleneck, Druckdimensionen, caveats).
- Runtime-Ops-Snapshot führt `optimization_view` als aktuellen operativen Überblick mit Hauptengpass, Mixed-Flag und Caveats.
- Runtime-Optimization-Snapshot im Multi-Worker-Service verbindet aktuelle Lage mit historischem Feedback (`historical_feedback_alignment`, `repeated_pattern_confirmed`).

## Bewusste Grenzen

- Keine Scoring-/Ranking-Engine.
- Keine adaptive globale Optimierungsplattform.
- Keine neue Monitoring-/BI-/Dashboard-Datenplattform.
- Ableitung bleibt deterministisch und stützt sich nur auf bereits vorhandene Runtime-Diagnostik plus minimale Konsolidierungslogik.
