# UCF Chip 2 Regulation Engine Scaffold

## Workspace Overview
The workspace is organized as a Rust workspace rooted at the repository root. All crates forbid unsafe code and expose placeholder APIs only.

- **wire**: Frame IO primitives with placeholders for signing and verification. Intended to wrap transport and cryptographic concerns for regulator messaging.
- **rsv**: Regulator State Vector types and storage abstraction. Provides a simple `RegulatorState` structure and `StateStore` trait for persistence backends.
- **profiles**: Profile and overlay composition primitives. Offers serializable definitions and a `ProfileComposer` trait with a simple `StaticProfileComposer` placeholder.
- **engine**: Update engine surface. Defines the `UpdateEngine` trait, input/output structures, and helper utilities for staging profile resolutions.
- **hpa**: Placeholder for the DBM-HPA layer, with a no-op client and configuration skeleton for future memristor-backed implementations.
- **pvgs_client**: Client placeholder for CBV/HBV retrieval. Includes configuration and snapshot scaffolding with minimal validation.
- **app**: Binary crate that wires together config paths and placeholder components. It validates config path presence, triggers a dummy profile resolution, and announces successful boot.

## Configuration Directory
The `config/` directory contains placeholder YAML files for profiles, overlays, update tables, windowing, class thresholds, and HPA settings. They are referenced by the `app` crate but not parsed yet.

## CI Workflow
The GitHub Actions workflow at `.github/workflows/ci.yml` runs `cargo fmt`, `cargo clippy`, and `cargo test` across the workspace to ensure the scaffold remains buildable and warning-free.
