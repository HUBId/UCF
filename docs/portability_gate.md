# Portability Gate v1 (Linux + Windows)

`Portability Gate` blocks merges when core runtime/ops checks are not cross-platform safe.

## What is checked

1. **Cross-platform CI matrix (blocking)**
   - Linux lane:
     - `cargo test --workspace --all-targets`
     - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
     - `cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json`
     - `cargo run -p ucf-ops -- audit hardware-scan`
     - `cargo run -p ucf-ops -- audit path-scan`
     - `cargo run -p ucf-ops -- audit net-deps --out ./out/net_deps.json`
     - `cargo run -p ucf-ops -- portability check --out ./out/portability.json`
   - Windows lane:
     - `cargo test --workspace --all-targets`
     - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
     - `cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json`
     - `cargo run -p ucf-ops -- audit hardware-scan`
     - `cargo run -p ucf-ops -- audit path-scan`
     - `cargo run -p ucf-ops -- audit net-deps --out ./out/net_deps.json`
     - `cargo run -p ucf-ops -- portability check --out ./out/portability.json`

2. **v1 minimal checks (blocking where present)**
   - `models verify` runs only if `models/MANIFEST.toml` or `models/manifest.toml` exists; otherwise CI emits deterministic skip output.
   - `models probe` runs deterministic offline probes for `llm`, `sae`, and `world_jepa` slots.
   - `v1 smoke` runs a minimal smoke report:
     - Linux: `ucf-ops v1 smoke --shadow` (single-slot shadow observational check)
     - Windows: `ucf-ops v1 smoke` (probe-only smoke)

3. **Path hygiene scan (`audit path-scan`)**
   - Scans runtime crate source files (`runtime/*/src/*.rs`) for hard-coded OS-specific assumptions:
     - `"/etc/"`
     - `"/var/"`
     - `"systemd"`
     - `"systemctl"`
   - Allowlist/exclusions:
     - `deploy/` templates
     - vendor/target/fuzz scopes
     - `runtime/ucf-ops/src/` (scanner implementation)

4. **Hardware-neutral guardrails (`audit hardware-scan`)**
   - Scans runtime crates plus core portability docs for forbidden vendor/machine terms.
   - Fails portability lane on violations.

## Determinism across OS

- The gate enforces deterministic behavior **within each OS lane**.
- Reports are produced in stable ordering for scan outputs.
- Cross-OS exact digest parity can differ in edge cases; fixed-point envelope fields + schema stability are the portable contract.

## Local run instructions

### Linux/macOS shell

```bash
cargo test --workspace --all-targets
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json
if [ -f models/MANIFEST.toml ]; then cargo run -p ucf-ops -- models verify --manifest models/MANIFEST.toml --out ./out/models_verify_report.json; elif [ -f models/manifest.toml ]; then cargo run -p ucf-ops -- models verify --manifest models/manifest.toml --out ./out/models_verify_report.json; else echo "models_verify=skip reason=no_manifest"; fi
cargo run -p ucf-ops -- models probe --slot llm --out ./out/probe_llm.json
cargo run -p ucf-ops -- models probe --slot sae --out ./out/probe_sae.json
cargo run -p ucf-ops -- models probe --slot world_jepa --out ./out/probe_world_jepa.json
cargo run -p ucf-ops -- v1 smoke --shadow --out ./out/v1_smoke_report.json
cargo run -p ucf-ops -- audit hardware-scan
cargo run -p ucf-ops -- audit path-scan
cargo run -p ucf-ops -- audit net-deps --out ./out/net_deps.json
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json --workdir ./.ucf_gate
cargo run -p ucf-ops -- portability check --out ./out/portability.json
```

### Windows PowerShell

```powershell
cargo test --workspace --all-targets
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json
if (Test-Path "models/MANIFEST.toml") { cargo run -p ucf-ops -- models verify --manifest models/MANIFEST.toml --out ./out/models_verify_report.json } elseif (Test-Path "models/manifest.toml") { cargo run -p ucf-ops -- models verify --manifest models/manifest.toml --out ./out/models_verify_report.json } else { Write-Host "models_verify=skip reason=no_manifest" }
cargo run -p ucf-ops -- models probe --slot llm --out ./out/probe_llm.json
cargo run -p ucf-ops -- models probe --slot sae --out ./out/probe_sae.json
cargo run -p ucf-ops -- models probe --slot world_jepa --out ./out/probe_world_jepa.json
cargo run -p ucf-ops -- v1 smoke --out ./out/v1_smoke_report.json
cargo run -p ucf-ops -- audit hardware-scan
cargo run -p ucf-ops -- audit path-scan
cargo run -p ucf-ops -- audit net-deps --out ./out/net_deps.json
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report_windows.json --workdir ./.ucf_gate_windows
cargo run -p ucf-ops -- portability check --out ./out/portability.json
```

## Common failures and remediation

- **`audit path-scan` failed**
  - Remove hard-coded `/etc`/`/var` paths from runtime code.
  - Move OS/service-specific defaults to deploy templates or config.

- **`audit hardware-scan` failed**
  - Replace vendor/machine assumptions with neutral `DeviceProfile` + budget/config controls.

- **`models verify` skipped in CI**
  - Add `models/MANIFEST.toml` (preferred) or `models/manifest.toml` if lifecycle validation is required.

- **`v1 smoke` failed**
  - Run `ucf-ops models probe --slot llm|sae|world_jepa` locally and inspect probe reports.
  - If Linux shadow check fails, confirm `shadow` mode remains observational and does not alter decision selection in the smoke scenario.
