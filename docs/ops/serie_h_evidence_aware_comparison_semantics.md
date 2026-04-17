# Serie H: Evidence-aware Comparison Semantics (Prompt 4)

## Kanonische, schmale Vergleichssicht

`EvidenceAwareComparisonView` ist die gemeinsame minimale Vergleichssicht über Replay-, Rollout- und Recovery-Kontexte.

Sie trägt nur:
- `compared_entities` (z. B. original/replay, baseline/candidate, before/after trust state)
- `shared_evidence_refs`
- `contrasting_evidence_refs`
- `primary_differences`
- `primary_caveats`
- `class` (`meaningful`, `caveated`, `inconclusive`, `not_meaningful`, `blocked_missing_prerequisites`)

Keine allgemeine Diff-, Audit- oder Analytics-Plattform.

## Zusammenführung über Replay / Rollout / Recovery

- Replay-Mismatch-Sichten (`ReplayMismatchView`) tragen jetzt dieselbe evidence-aware Vergleichssemantik.
- Rollout-nahe Baseline-Vergleiche (`BaselineComparisonSummary`) enthalten die gleiche Vergleichssicht.
- Recovery-/Mutations-Operationen (`RuntimeOperationOutcome`) liefern eine `recovery_comparison` mit before/after Trust-Hinweisen.

Damit werden vorher subsystem-separierte Vergleichsformen auf eine gemeinsame evidenzseitige Minimal-Sprache ausgerichtet.

## Primary differences / caveats (vereinheitlicht)

Primäre Unterschiede und Caveats werden jetzt als kompakte, wiederverwendbare Felder geführt, u. a. für:
- changed execution context
- changed rollout context
- improved trust after recovery
- still inconclusive / blocked due to missing prerequisites

## Expert diagnostics Anbindung

Action-Justifications referenzieren die load-bearing Vergleichssicht über stabile `comparison_ref`-Gründe und können die Vergleichsklasse aus `RuntimeOperationOutcome.recovery_comparison` nutzen.

## Bewusste Grenzen

- Keine neue Reasoning-Engine.
- Keine zweite Diagnose- oder Snapshot-Welt.
- Keine autonome Governance-/Approval-Logik.
- Vergleichssemantik bleibt technisch-operativ und evidenznah.
