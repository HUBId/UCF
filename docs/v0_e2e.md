# v0 E2E: ControlFrame → CandidateSet → Decision → ESS Append

Dieser Test deckt den deterministischen v0-End-to-End-Flow ab:

1. `ControlFrame` Intake.
2. minimale Candidate-Erzeugung (SafeText + NoOp).
3. Policy-gated Decision Selection.
4. ESS-Append der relevanten Kette pro Tick:
   - `WorldSummary`
   - `SaeSummary`
   - `SsmSummary`
   - `LfmSummary`
   - `SignalBundle`
   - `DecisionInputs`
   - `DecisionOut` (`DecisionFrame`)
   - `Output` (Experience-Audit)
5. Optional-Hooks (`consolidation` / `geist`) werden aufgerufen, dürfen Core-Outputs aber nicht ändern.

## Fixture

Fixture-Datei: `fixtures/e2e/v0_flow_a.json`

- fixed ticks: `8`
- intent: `produce safe text response`
- overlay: `test`
- determinism: `strict`
- stub backends: `true`
- keine Tool-Request-Payload

## Test ausführen

```bash
cargo test -p ucf-runtime --test v0_flow_e2e
```

## PASS bedeutet

- Lauf ist deterministisch (2 identische Runs → gleiche letzte Digest-Prefixes für Signal/Decision/Experience).
- CandidateSet enthält mindestens `SafeText` und `NoOp` und nutzt stabile Candidate-Digests.
- Keine Tool-Ausführung (`MockAdapter.mem_writes == 0`, deny-by-default).
- Record-Kette pro Tick ist vollständig und verlinkt (inkl. Audit-Digest-Chain).
- Optional-Hooks erzeugen nur Hook-seitige Aktivität, aber verändern nicht:
  - ausgewählte Kandidaten
  - Signal-Digest
  - Decision-Digest
- Replay-Audit im `verify-only` Modus ist `Ok`.

## Inspektion / Explain-Tick

Falls `ucf-ops explain-tick` im Setup aktiv ist, kann derselbe Lauf zusätzlich über die ESS-Artefakte inspiziert werden, um Tick-weise Digests und Entscheidungskette nachzuverfolgen.
