# UCF Spine

This repository follows a simple spine layout at the top level:

- `core/`: canonical shared types, IDs, ports, and foundational crates.
- `domains/`: domain-specific crates and features built on core contracts.
- `runtime/`: executable/runtime wiring and orchestration crates.
- `assets/`: static assets and generated data artifacts.
- `vendor/`: vendored third-party code or snapshots.
- `docs/`: architecture and operational documentation.

## Rule for domain crates

Domain crates must depend on canonical core types and must not duplicate canonical IDs or shared type definitions.
