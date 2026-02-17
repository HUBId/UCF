# Adversarial Harness (v1)

Run:

```bash
ucf-ops adversarial-run --suite v1 --out ./out/adversarial_report.json
```

## What it covers
- Prompt-injection fixtures (`fixtures/adversarial/prompts/*.txt`).
- Tool misuse attempts (NetHttp/FileRead deny paths).
- Sandbox path traversal and symlink escape checks.
- Governor stress over fixed 32 ticks.
- Emergency-mode case reporting (deny-all safe-only expected).

## Determinism and bounds
- Fixed suite `v1`.
- Sorted fixture loading.
- Capped case count and bounded stress tick count.
- JSON report schema stable for CI ingestion.

## Report interpretation
`AdversarialReport` includes:
- suite version, code version tag, policy hash prefix.
- per-case PASS/FAIL, observed governor tier, denial reasons,
  emergency flag, output class, evidence digest prefixes.

When a case fails, use `failure_reason` + `hint` to triage policy scope,
fixture expectation, or policy bundle configuration.

## Add a new case
1. Add deterministic fixture text or JSON under `fixtures/adversarial/*`.
2. Extend `runtime/ucf-ops/src/adversarial.rs` case execution.
3. Keep runtime bounded and offline-compatible.
4. Verify with `cargo test -p ucf-ops` and `ucf-ops adversarial-run --suite v1`.

## Threat model mapping
See `docs/threat_model_v1.md` for attack-surface-to-mitigation mapping used by this harness.
