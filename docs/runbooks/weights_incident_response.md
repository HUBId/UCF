# Weights Incident Response Runbook

This runbook handles model-weight operational incidents for v1.1 workflows (promotion, rollback, and emergency pinning).

## Scope

Applies to incidents involving:

- Promotions from `models/staging/` to `models/promoted/`.
- Regressions after promotion.
- Invalid/mismatched artifacts that pass into staging.
- Emergency model hash pinning and rollback workflows.

## Prerequisites

- Access to `ucf-ops`.
- Access to readiness/probe reports.
- Known slot identifier and suspect model hash.

## Incident Triage

1. Identify affected slot/hash and blast radius.
2. Confirm whether policy-facing behavior is impacted.
3. Determine severity:
   - Sev-1: production-impacting policy regression.
   - Sev-2: degraded quality/latency with fallback available.
   - Sev-3: staging-only issue.

## Immediate Containment

1. Freeze promotions for the affected slot.
2. Pin to last known-good promoted hash:
   ```bash
   cargo run -p ucf-ops -- models pin --slot <slot> --hash <known_good_hash>
   ```
3. If needed, roll back promoted pointer:
   ```bash
   cargo run -p ucf-ops -- models rollback --slot <slot> --to <known_good_hash>
   ```
4. Re-run readiness gate for verification:
   ```bash
   cargo run -p ucf-ops -- readiness-gate --profile incident --out ./out/readiness_incident.json
   ```

## Diagnosis Workflow

1. Verify `WeightSpec` conformance in probe outputs.
2. Compare shadow metrics versus promoted metrics:
   - latency envelopes
   - prediction/pressure error envelopes
   - quantized output stability
3. Check manifest history and signature trail.
4. Confirm no unauthorized pin/manifest modification.

## Recovery

1. Keep affected slot pinned until stable evidence exists.
2. Validate known-good hash across required gate profiles.
3. Resume promotion only with explicit signoff and updated mitigation notes.

## Evidence to Attach to Incident

- Probe report for suspect and known-good hash.
- Shadow/readiness gate reports.
- Manifest history diff (before/after).
- Pin/rollback command logs with operator and timestamp.
- Root cause analysis + preventive actions.

## Post-Incident Actions

1. Add a regression test reproducing the failure class.
2. Update promotion checklist/readiness thresholds if needed.
3. Document learning in release notes and ops handoff.
4. Remove emergency pin only after explicit approval:
   ```bash
   cargo run -p ucf-ops -- models unpin --slot <slot>
   ```
