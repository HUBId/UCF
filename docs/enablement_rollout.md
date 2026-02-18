# Real Compute Enablement Rollout

Dieses Dokument beschreibt das sichere, deterministische Rollout von Real-Compute pro Slot.

## Modi

- `UCF_REAL_ENABLEMENT_MODE=off|shadow|compare|active`
  - `off`: nur Toy/Stubs.
  - `shadow`: primär Toy; Real läuft parallel, beeinflusst keine Entscheidungen.
  - `compare`: wie shadow, zusätzlich Envelope-Verletzungszähler.
  - `active`: Real kann pro Slot aktiv geschaltet werden.

## Per-Slot Steuerung

- `UCF_SLOT_LLM_MODE=toy|shadow|active`
- `UCF_SLOT_LFM_MODE=toy|shadow|active`
- `UCF_SLOT_WORLD_JEPA_MODE=toy|shadow|active`
- `UCF_SLOT_SAE_MODE=toy|shadow|active`
- `UCF_SLOT_SSM_MODE=toy|shadow|active`

Zusätzlich kann der globale Shadow-Frequenzteiler gesetzt werden:

- `UCF_SHADOW_EVERY_N_TICKS=4`

## Stage-Ladder Rollout Beispiel

Für deterministische Stufen kann eine Ladder geparst werden:

- `phase1@t0;phase2@t64;phase3@t128`

Semantik:

- `phase1`: LLM in `shadow`.
- `phase2`: LLM `active`, LFM `shadow`.
- `phase3`: LLM+LFM `active`, JEPA `shadow`.

## Envelope Checks (Compare)

In Compare/Shadow werden keine Digest-Gleichheiten erzwungen, sondern Invarianten:

- `surprise`, `pressure`, `risk` sind finite und in `[0,1]`.
- `lfm_uncertainty` (falls vorhanden) ist finite und in `[0,1]`.

Bei Verstoß wird `ucf_compare_envelope_violation_total{slot="compute"}` erhöht.

## Fallback- und Timeout-Signale

Shadow-Läufe sind begrenzt (`1/n` Ticks) und verwenden reduziertes Budget (`max_micros/2`).

- Erfolg: `ucf_shadow_runs_total{slot="compute"}`
- Fehler/Timeout: `ucf_shadow_timeouts_total{slot="compute"}`

Die Primärentscheidung bleibt immer Toy-first in Shadow/Compare.

## Audit-Output

`ShadowComparisonRecord` enthält:

- Tick `t`
- Digest-Präfix Toy/Real
- `elapsed_ms`
- Ergebnisstatus

Damit bleibt der Vergleich reproduzierbar und offline auswertbar.

## LLM v1 (Candle CPU) Rollout Notes

- LLM bleibt in `shadow`/`compare` strikt nicht-entscheidend: Toy/Stub ist primär, Candle läuft nur beobachtend.
- `active` ist nur zulässig, wenn ModelSlot (`llm`) + Tokenizer-Hash verifiziert sind.
- Timeout/Fehler führen deterministisch zu SafeText-Fallback (`System busy; try again.`) statt Tool-Ausführung.
- OutputClass bleibt harter finaler Choke-Point (invalid output => refusal).
