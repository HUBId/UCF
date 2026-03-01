# Hardware-Neutral Migration Note

## Why this migration exists

The prompt series briefly introduced machine-specific wording (for example, NUC/cluster-centric assumptions). UCF runtime and policy behavior are intended to be hardware-neutral and reproducible across supported environments.

To preserve determinism and portability in planning docs, hardware-specific language is being removed from core prompt guidance.

## What changed

- Prompt entries tied to hardware assumptions are now kept only as historical records and marked deprecated/obsolete in the prompt index.
- Successor prompts (147/148) define hardware-neutral guidance and compliance scanning.
- `ucf-ops docs lint` includes a hardware-neutral docs guardrail that flags obvious hardware terms in core docs.

## Replacement guidance

Use **DeviceProfile** classes and explicit budget envelopes:

- `small`
- `medium`
- `large`

Describe constraints in measurable terms instead of machine names:

- latency budget (p95 / p99)
- memory budget (MiB/GiB)
- compute budget (CPU-seconds, accelerator quota)
- storage and artifact budget

## Historical compatibility

Historical prompt IDs remain in index documentation and are not renumbered or deleted.

## Validation

Run:

```bash
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
```

The hardware-neutral docs check will fail for forbidden terms in core docs and only warn for allowed history/deploy scopes.
