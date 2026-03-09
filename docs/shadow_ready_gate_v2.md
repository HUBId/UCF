# Shadow-Ready Gate v2

## Purpose

`Shadow-Ready` is a deterministic, offline evidence gate for the currently supported real-backend slots in v2:

- `world_jepa`
- and exactly one additional slot declared in `docs/series_state_snapshot.md` (`sae` in current repo state)

The gate evaluates whether a slot has enough recent shadow/probe evidence to be considered safe for **shadow operation scaffolding**.

## What Shadow-Ready means

A slot is `shadow_ready=true` iff all required conditions hold:

1. Latest probe exists and is `PASS`.
2. Latest compare window evidence is present.
3. `shadow_no_decision_impact` verification is present and `PASS`.
4. Latest drift status is not `SEVERE`.

Output is represented as `ShadowReadyEvidenceV1` per slot and aggregated into `AggregatedEvidenceReportV1`.

## What Shadow-Ready does **not** mean

- It does **not** grant Active mode.
- It does **not** bypass active-evidence checks.
- It is purely a shadow-scaffolding readiness signal.

`Shadow-Ready != Active-Eligible`.

## Command

Generate report for supported slots (max two):

```bash
cargo run -p ucf-ops -- models shadow-ready --out ./out/shadow_ready_report.json
```

Generate for one supported slot:

```bash
cargo run -p ucf-ops -- models shadow-ready --slot world_jepa --out ./out/shadow_ready_world.json
```

## Evidence sources

- Probe reports: `./out/probe_<slot>.json`
- Compare window / drift alarms: `.ucf/ess/ess_fixture.json`
- No-decision-impact check: `./out/gate_report.json` (`shadow_no_decision_impact`)
- Policy graph digest prefix for persisted check records: from `./out/gate_report.json` when present

## Persisted records

`ShadowReadyCheckRecordV1` entries are appended to:

- `.ucf/out/shadow_ready_checks.json`

Each record contains:

- slot id
- target hash
- PASS/FAIL
- evidence digest prefix
- denial reason code (if any)
- policy graph digest prefix
