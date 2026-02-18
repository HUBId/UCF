# EBM Training (Deferred, Offline, Safe Skeleton)

## Safety Statement

`ucf-runtime` does **not** perform online learning. Model weights are never updated from runtime code paths.
Training is an operator workflow using a separate binary (`ucf-ebm-train`) and is disabled by default.

## Compile/Feature Gates

- Training runner binary: `runtime/ucf-ebm-train`, compiled only with feature `ebm-train`.
- Runtime training mode guard: if `UCF_EBM_TRAINING_MODE=1`, runtime initialization fails fast.

## Offline Dataset Export Workflow

Export dataset samples from ESS as bounded JSONL with redaction-aware metadata only:

```bash
ucf-ops ebm export-dataset \
  --workdir .ucf \
  --run run_2026_02_ebm \
  --from 0 \
  --to 999999 \
  --out ./out/ebm_dataset_v1.jsonl
```

Notes:
- Retention/redaction policy is applied (`policies/bundle_v1/retention_v1.json` by default).
- Export contains digests, quantized signals, candidate metadata, labels, and constraint term IDs.
- Raw output text is never exported.

## Offline Training Runner Workflow

Run bounded deterministic training in a sandbox-style operator flow:

```bash
cargo run -p ucf-ebm-train --features ebm-train -- \
  --enable-training-runner \
  --dataset ./out/ebm_dataset_v1.jsonl \
  --initial-weights ./models/ebm_seed.safetensors \
  --steps 10 --lr 0.0005 --batch 16 --seed 7 \
  --out-dir ./out
```

Outputs:
- New staged weights: `./models/staging/ebm_<sha256>.safetensors`
- Report: `./out/ebm_training_report.json`
  - dataset digest
  - input/output weight hashes
  - config digest
  - bounded loss summary
  - nondeterminism risk flag

## Promotion Workflow (No Runtime Auto-Update)

1. Verify staged artifact hash from file name and report.
2. Update `models/manifest.toml` for `slots.ebm_reasoner` with:
   - `enabled = true`
   - `path = "staging/ebm_<sha256>.safetensors"`
   - `expected_sha256 = "<sha256>"`
3. Execute normal release/sign-off pipeline.

Runtime loads only manifest-hash-locked weights. If an enabled slot has no hash, load is refused.

## Verify After Promotion

```bash
ucf-ops models verify --manifest models/manifest.toml
```

If hash/path are valid and in allowlist root, EBM slot verifies successfully.
Unpromoted or hash-mismatched staging weights are refused.
