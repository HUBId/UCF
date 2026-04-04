# Supported Scope Decision v21

## Zweck (Prompt 381)

`supported-scope-decision` beweist deterministisch eine **binäre** Entscheidung direkt nach `GovernanceLockSweepV1`:

- `SCOPE_EXPANSION_APPLIED`
- `SCOPE_FREEZE_REINFORCED`

Es gibt keinen dritten Zustand. Governance-Lock alleine erweitert den Supported Scope nicht automatisch.

## Warum nach Governance-Lock nicht auto-expand?

Der Lock-Sweep beweist die kanonische Governance-Kette, aber nicht automatisch, dass Scope-Ausführung, Semantik, Export und Kontinuität ohne Überdehnung konsistent erweitert werden können. Deshalb bleibt der Default konservativ/fail-closed.

## Binäres Entscheidungsmodell

Die Entscheidung akzeptiert Expansion nur, wenn exakt ein Slot die vollständige Evidenzkette trägt:

1. Governance erlaubt den Slot.
2. Current Scope Execution repräsentiert ihn kanonisch.
3. Keine neue Runtime-/Backend-Aktivierung.
4. Readiness widerspricht nicht.
5. Export-/Bundle-Semantik bleibt präzise.
6. Primary-Semantik bleibt exakt (kein Inflation-Risiko).
7. Continuity bleibt single-chain-kompatibel.

Wenn eine Bedingung fehlt, wird Freeze verstärkt (`SCOPE_FREEZE_REINFORCED`).

## Command

```bash
cargo run -p ucf-ops -- supported-scope-decision --out ./out/supported_scope_decision.json
```

## Bedeutung der Statuswerte

- `SCOPE_EXPANSION_APPLIED`: genau ein Slot wurde als vollständig evidenzgetragen bestätigt.
- `SCOPE_FREEZE_REINFORCED`: kein Slot erfüllt die komplette Kette; bestehender Scope bleibt unverändert und fail-closed.
