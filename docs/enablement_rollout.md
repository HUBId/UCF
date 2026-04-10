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

## Rollout Provenance / History / Diagnostics (Serie B)

Zusätzlich zur Slot-Provenance wird jetzt pro Slot eine kompakte
`SlotRolloutDiagnostics`-Sicht geführt (im Runtime-Provenance-Objekt):

- Referenzen: `prior_active_hash`, `active_hash`, `candidate_hash`, `compare_hash`, `shadow_hash`,
  `resulting_active_hash`
- Entscheidungen: `promotion_state`, `promotion_disposition`, `activation_outcome`,
  `fallback`, `rollback`
- Konsolidierte Signal-View (`consolidated`), die load-bearing Quellen gemeinsam auswertet:
  - `compatibility_gate`
  - `baseline_compare_shadow`
  - `activation_guardrails`
  - `runtime_outcomes`
- Kanonische konsolidierte Rollout-Lage (`rollout_signal_situation`):
  - `insufficient_signal`
  - `technically_blocked`
  - `comparable_but_inconclusive`
  - `promotable_under_guardrails`
  - `fully_promotable`
  - `active_but_unstable`
- Pro Lage sichtbar:
  - primäre stützende Signalgruppen (`supporting_groups`)
  - primäre blockierende Signalgruppen (`blocking_groups`)
  - explizite Spannungslagen (`tensions`), u.a.:
    - `gates_passed_but_runtime_unstable`
    - `comparison_positive_but_guardrail_required`
    - `activation_possible_but_signal_insufficient`
    - `active_but_degraded_questions_stability`
- Diagnose-Klassifikation:
  - `progressed`
  - `blocked`
  - `inconclusive`
  - `degraded_but_active`
  - `fallback_or_rollback_after_issue`
- Blocker-Semantik:
  - `gate_blocked`
  - `missing_comparison_signal`
  - `activation_issue`
  - `compare_shadow_inconclusive`
  - `fallback_or_rollback_required`

Die Events sind absichtlich schmal und technisch:

- `candidate_introduced`
- `candidate_compared_or_shadowed`
- `candidate_promotable|candidate_blocked|candidate_inconclusive`
- `activation_attempted`
- `activation_succeeded|activation_degraded|activation_blocked|activation_failed_technically`
- `fallback_occurred`
- `rollback_occurred`

Grenzen (bewusst):

- Keine Audit-Plattform oder Release-Management-Historie.
- Keine Governance-/Approval-/Incident-Welt.
- Keine separate Evidenz-Parallelwelt; Compare/Baseline/Gate-Signale bleiben in der
  bestehenden Promotion-/Activation-Semantik.
- Keine Scoring-/Ranking-Engine; die Sicht bleibt eine schmale technische Konsolidierung.

## LLM v1 (Candle CPU) Rollout Notes

- LLM bleibt in `shadow`/`compare` strikt nicht-entscheidend: Toy/Stub ist primär, Candle läuft nur beobachtend.
- `active` ist nur zulässig, wenn ModelSlot (`llm`) + Tokenizer-Hash verifiziert sind.
- Timeout/Fehler führen deterministisch zu SafeText-Fallback (`System busy; try again.`) statt Tool-Ausführung.
- OutputClass bleibt harter finaler Choke-Point (invalid output => refusal).
