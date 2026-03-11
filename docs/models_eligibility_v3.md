# Unified Models Eligibility v3

`ucf-ops models eligibility` erzeugt einen einzigen deterministischen Operator-Report über die aktuell unterstützten Real-Slots (maximal 2 in v3-Baseline):

- Probe-Ready
- Shadow-Ready
- Active-Eligible

Der Report ist rein informativ (read-only), offline-fähig und führt **keine** Aktivierung/Promotion/Rollback-Aktion aus.

## Begriffe

## Probe-Ready
Ein Slot ist `probe_ready=true`, wenn der letzte per-slot `ProbeReportV1`:

1. `status=PASS` hat,
2. zur Slot-ID passt,
3. und der Probe-Hash-Präfix zum aktuellen Target-Hash passt.

## Shadow-Ready
Ein Slot ist `shadow_ready=true`, wenn die bestehende `ShadowReadyEvidenceV1`-Logik erfüllt ist:

- Probe vorhanden/PASS,
- Compare-Window vorhanden,
- No-Impact-Nachweis vorhanden,
- Drift nicht `SEVERE`.

## Active-Eligible
Ein Slot ist `active_eligible=true`, wenn die bestehende Active-Evidence-Logik (`ActiveEnablementEvidenceV1`) PASS liefert.

Wichtig: Diese Sicht fügt **keine neue Semantik** hinzu, sie konsolidiert nur bestehende Evidence-Entscheidungen.

## Kommando

```bash
cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json
```

Optional für einen unterstützten Slot:

```bash
cargo run -p ucf-ops -- models eligibility --slot world --out ./out/models_eligibility_world.json
```

## Reportstruktur

`AggregatedEligibilityReportV1` enthält:

- `overall_status` (`NONE_READY|PROBE_ONLY|SHADOW_READY_PARTIAL|SHADOW_READY_ALL|ACTIVE_ELIGIBLE_PARTIAL|ACTIVE_ELIGIBLE_ALL`)
- `slots[]` mit `UnifiedEligibilityStatusV1`
- `generated_from` (Probe-/Shadow-/Active-Digest-Bezüge)
- `policy_graph_digest_prefix`
- `report_digest`

Pro Slot (`UnifiedEligibilityStatusV1`) sind u.a. enthalten:

- `probe_ready`, `shadow_ready`, `active_eligible`
- `denial_reason_probe|shadow|active`
- `remediation_codes` (stabil sortiert, bounded)
- Evidence- und Status-Digest-Präfixe

## Partial Readiness interpretieren

- `ACTIVE_ELIGIBLE_PARTIAL`: mindestens ein Slot active-eligible, mindestens ein anderer nicht.
- `SHADOW_READY_PARTIAL`: kein Slot active-eligible, aber mindestens ein Slot shadow-ready.
- `PROBE_ONLY`: weder shadow-ready noch active-eligible, aber mindestens ein Slot probe-ready.
- `NONE_READY`: kein Slot probe-ready.

Damit müssen Operatoren Probe-/Shadow-/Active-Reports nicht mehr manuell korrelieren.


## Additional v3 evidence feed
- Unified eligibility also includes second-slot parity report digest evidence (`out/<slot>_parity_report.json`) for shadow-readiness interpretation.


## Strict mode coupling

Strict mode v3 consumes the same unified evidence primitives (probe, shadow-ready, active-eligible, normalized compare freshness + drift/hash denial reasons) instead of re-deriving an independent decision tree.

This keeps semantics aligned between:

- `ucf-ops models eligibility`
- strict startup/runtime checks
- `ucf-ops strict check`


## Operator first-stop

For operations, start with the consolidated report first:

```bash
cargo run -p ucf-ops -- operator report --out ./out/operator_report.json
```

Then inspect `eligibility_section` in that report and drill into `models eligibility` only when needed.


## v4 consistency note
Eligibility now consumes the shared `SupportedRealSlotSetV1` and `SlotEvidenceSnapshotV1` evidence layer used across strict/operator/gate surfaces.
