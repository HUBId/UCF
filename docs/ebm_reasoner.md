# EBM Reasoner v0

Der EBM-Reasoner ist eine additive Re-Ranking-Schicht zwischen CandidateSet-Erzeugung und finaler Auswahl.

## Rolle in der Pipeline

1. CandidateSet wird wie bisher durch `DefaultCandidateGeneratorV0` erzeugt.
2. Der EBM bewertet Kandidaten deterministisch über quantisierte Signale.
3. In `active` wird die Auswahl nach minimaler Energie gerankt, aber **Governor/Policy/ToolGate bleiben final**.
4. In `shadow`/`compare` werden nur ESS-Audits erzeugt; die bestehende Auswahl bleibt unverändert.

## v0 Energie-Funktion

`CpuEbmStubV0` nutzt feste Fixed-Point-Gewichte (`UQ0_16`) über:

- risk
- uncertainty
- pressure
- surprise

Zusatzregeln:

- `ToolIntent`: hoher Energie-Penalty
- `Json`: kleiner Penalty
- `NoOp`: kleiner Bonus
- `emergency_active` oder Tier ≥ 3: massive Tool-Penalty, NoOp-Bias

Alle Energien werden auf `[0, 1]` geklemmt (`UQ0_16`). Tie-Break ist stabil über `candidate_id`.

## Safety-Properties

- Keine Tool-Ausführung durch EBM (nur Re-Ranking).
- Boundaries: max Kandidaten und Top-N limitiert.
- Budget-Überschreitung führt zu `DegradedFallback` und blockiert den Basispfad nicht.
- Deterministischer Digest über quantisierte Inputs + Energien.

## Enablement Modes

- `off`: EBM deaktiviert.
- `shadow`: EBM läuft, Entscheidung unverändert.
- `compare`: EBM läuft wie shadow (v0).
- `active`: EBM-Re-Ranking wird für die Kandidatenauswahl genutzt.

Empfohlen: erst shadow, dann active.

## Konfiguration

- Shadow: `UCF_SLOT_EBM_MODE=shadow`
- Active: `UCF_SLOT_EBM_MODE=active`

## Future

`ModelSlot::EbmReasoner` ist vorbereitet für spätere Candle/Burn Gewichtsladung (`WeightSpec`/`ModelStore`) ohne Architekturbruch.
