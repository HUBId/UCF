# OptionalRealRuntime Fixture (Synthetic, Metadata-Only)

This directory contains a tiny local fixture set for OptionalRealRuntime **manifest and metadata validation only**.

## Purpose
- Validate fixture/manifest structure and pinned SHA-256 digests.
- Validate metadata compatibility with `OptionalRealRuntimeCandidateContract`.

## Non-goals
- No runtime inference execution.
- No OptionalRealRuntime activation.
- No Real-Compute runtime activation.
- No production-readiness claim.

## Constraints
- Local/offline only.
- No network access required.
- No external services required.
- Artifact is synthetic and deterministic.
- Toy/backend mappings are not promoted.

## Files
- `artifact.fixture.bin`: synthetic tiny binary payload (not a trained model).
- `input.fixture.json`: deterministic JSON input fixture.
- `expected_output.fixture.bytes`: deterministic planned-output bytes fixture for pinned digest.
- `fixture_manifest.json`: pinned metadata + SHA-256 digests.
