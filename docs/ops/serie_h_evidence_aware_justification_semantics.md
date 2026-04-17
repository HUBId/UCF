# Serie H: Evidence-aware Expert Diagnostics / Action Justification (Prompt 3)

## Kanonische Begriffe (schmal)

- `DecisionJustificationView` ist die gemeinsame schmale Begründungssicht für Expert-Diagnostics und High-Trust-Ops.
- `decision_target` benennt den technischen Entscheidungs-/Aktionskontext (z. B. Placement, Replay, Recovery, konkrete Runtime-Operation).
- `primary_evidence_refs` verweist auf primäre Evidence-Bundles.
- `primary_trace_slice_refs` verweist auf primäre Trace-Slices.
- `primary_reasons` enthält die kompakten tragenden technischen Gründe.
- `outcome_or_next_step` drückt Ergebnis oder direkte empfohlene nächste Aktion aus.

## Semantik für Evidence-Posture

- `evidence_backed`: belastbar evidenzgestützt.
- `partial_evidence`: teilweise evidenzgestützt.
- `stale_or_caveated`: Evidenz ist vorhanden, aber caveated/stale.
- `insufficient_for_high_trust_mutation`: für mutierende High-Trust-Intervention nicht ausreichend.
- `no_meaningful_justification_available`: aktuell keine tragfähige Begründung verfügbar.

Diese Semantik bleibt **diagnostics-nah**; sie ersetzt keine Policy- oder Explainability-Engine.

## Wie die Schichten zusammenhängen

- `CanonicalRuntimeSnapshot.evidence_bundle_refs` bleibt der kanonische Evidence-Entry.
- `CanonicalRuntimeSnapshot.justification_anchors` ergänzt dazu load-bearing Begründungsanker für:
  - placement capacity decision
  - rollout path decision
  - replay/repro decision
  - recovery readiness decision
- `RuntimeOperationOutcome.action_justification` macht mutierende/high-trust Aktionen explizit nachvollziehbar (allowed/blocked/caveated + Evidence/Trace + next step).

## Allow / Block / Caveat / Recommendation

- `disposition=allowed`: Aktion/Entscheidung ist technisch tragfähig.
- `disposition=blocked`: Aktion/Entscheidung ist technisch blockiert.
- `disposition=caveated`: Aktion/Entscheidung ist möglich, aber mit caveat (z. B. stale basis).
- Empfehlungen bleiben bewusst kurz (`outcome_or_next_step`) und operational (refresh/resync/recheck), ohne Workflow-Orchestrierung.

## Bewusste Grenzen

- Keine Explainability-/Reasoning-Plattform.
- Keine Governance-/Approval-Automation.
- Keine zweite Snapshot- oder Diagnostics-Welt.
- Keine neue Meta-Sprache über vorhandene Runtime-/Ops-/Replay-/Rollout-/Recovery-/Placement-Contracts hinaus.
