# Portability Gate v3 Refresh (Linux + Windows)

`Portability Gate` blocks merges when core runtime/ops checks are not cross-platform safe.

## What is checked

1. **Cross-platform CI matrix (blocking)**
   - Linux lane:
     - `cargo test --workspace --all-targets`
     - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
     - `cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json`
     - `cargo run -p ucf-ops -- v1 gate --out ./out/v1_gate_report.json`
     - `cargo run -p ucf-ops -- v2 gate --out ./out/v2_gate_report.json`
     - `cargo run -p ucf-ops -- audit hardware-scan`
     - `cargo run -p ucf-ops -- audit path-scan`
     - `cargo run -p ucf-ops -- audit net-deps --out ./out/net_deps.json`
     - `cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json`
     - `cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json`
     - `cargo run -p ucf-ops -- operator report --out ./out/operator_report.json`
     - `cargo run -p ucf-ops -- portability check --out ./out/portability.json`
     - `cargo run -p ucf-ops -- portability report --out ./out/portability_report.json`
   - Windows lane:
     - `cargo test --workspace --all-targets`
     - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
     - `cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json`
     - `cargo run -p ucf-ops -- v1 gate --out ./out/v1_gate_report.json`
     - `cargo run -p ucf-ops -- v2 gate --out ./out/v2_gate_report.json`
     - `cargo run -p ucf-ops -- audit hardware-scan`
     - `cargo run -p ucf-ops -- audit path-scan`
     - `cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json`
     - `cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json`
     - `cargo run -p ucf-ops -- operator report --out ./out/operator_report.json`
     - `cargo run -p ucf-ops -- portability check --out ./out/portability.json`
     - `cargo run -p ucf-ops -- portability report --out ./out/portability_report.json`

2. **v3 report generation smoke checks (blocking)**
   - `models eligibility` must run in bounded offline mode and write deterministic schema output.
   - `strict check --strict` must emit v3 check family output and fail with actionable denial codes.
   - `operator report` must produce consolidated output and never panic when source reports are missing.

3. **v1/v2 compatibility checks (blocking where present)**
   - `models verify` runs only if `models/manifest.toml` (preferred) or `models/MANIFEST.toml` exists; otherwise CI emits deterministic skip output.
   - `models probe` runs deterministic offline probes for `llm`, `sae`, and `world_jepa` slots.

4. **Path hygiene scan (`audit path-scan`)**
   - Scans runtime crate source files (`runtime/*/src/*.rs`) for hard-coded OS-specific assumptions:
     - `"/etc/"`
     - `"/var/"`
     - `"systemd"`
     - `"systemctl"`
   - Allowlist/exclusions:
     - `deploy/` templates
     - vendor/target/fuzz scopes
     - `runtime/ucf-ops/src/` (scanner implementation)

5. **Hardware-neutral guardrails (`audit hardware-scan`)**
   - Scans runtime crates plus core portability docs for forbidden vendor/machine terms.
   - Includes v3 docs:
     - `docs/models_eligibility_v3.md`
     - `docs/strict_mode_v3.md`
     - `docs/operator_report_v3.md`
   - Fails portability lane on violations.

6. **Docs consistency (via `docs lint --strict`)**
   - Ensures v3 docs exist and are linked from `docs/portability_gate.md` and `docs/strict_mode.md`.
   - Ensures prompt/state indexes stay consistent for Prompt 207 tracking.

7. **Consolidated portability summary (`portability report`)**
   - Orchestrates existing checks and writes `./out/portability_report.json`.
   - Emits explicit PASS/FAIL (and SKIP where applicable in future optional paths) per check.

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
cargo run -p ucf-ops -- v1 gate --out ./out/v1_gate_report.json
cargo run -p ucf-ops -- v2 gate --out ./out/v2_gate_report.json
if [ -f models/manifest.toml ]; then cargo run -p ucf-ops -- models verify --manifest models/manifest.toml --out ./out/models_verify_report.json; elif [ -f models/MANIFEST.toml ]; then cargo run -p ucf-ops -- models verify --manifest models/MANIFEST.toml --out ./out/models_verify_report.json; else echo "models_verify=skip reason=no_manifest"; fi
cargo run -p ucf-ops -- models probe --slot llm --out ./out/probe_llm.json
cargo run -p ucf-ops -- models probe --slot sae --out ./out/probe_sae.json
cargo run -p ucf-ops -- models probe --slot world_jepa --out ./out/probe_world_jepa.json
cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json
cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json
cargo run -p ucf-ops -- operator report --out ./out/operator_report.json
cargo run -p ucf-ops -- audit hardware-scan
cargo run -p ucf-ops -- audit path-scan
cargo run -p ucf-ops -- audit net-deps --out ./out/net_deps.json
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json --workdir ./.ucf_gate
cargo run -p ucf-ops -- portability check --out ./out/portability.json
cargo run -p ucf-ops -- portability report --out ./out/portability_report.json
```

### Windows PowerShell

```powershell
cargo test --workspace --all-targets
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- v0 gate --scenario fixtures/e2e/v0_flow_a.json --out ./out/v0_gate_report.json
cargo run -p ucf-ops -- v1 gate --out ./out/v1_gate_report.json
cargo run -p ucf-ops -- v2 gate --out ./out/v2_gate_report.json
if (Test-Path "models/manifest.toml") { cargo run -p ucf-ops -- models verify --manifest models/manifest.toml --out ./out/models_verify_report.json } elseif (Test-Path "models/MANIFEST.toml") { cargo run -p ucf-ops -- models verify --manifest models/MANIFEST.toml --out ./out/models_verify_report.json } else { Write-Host "models_verify=skip reason=no_manifest" }
cargo run -p ucf-ops -- models probe --slot llm --out ./out/probe_llm.json
cargo run -p ucf-ops -- models probe --slot sae --out ./out/probe_sae.json
cargo run -p ucf-ops -- models probe --slot world_jepa --out ./out/probe_world_jepa.json
cargo run -p ucf-ops -- models eligibility --out ./out/models_eligibility_report.json
cargo run -p ucf-ops -- strict check --strict --out ./out/strict_check.json
cargo run -p ucf-ops -- operator report --out ./out/operator_report.json
cargo run -p ucf-ops -- audit hardware-scan
cargo run -p ucf-ops -- audit path-scan
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report_windows.json --workdir ./.ucf_gate_windows
cargo run -p ucf-ops -- portability check --out ./out/portability.json
cargo run -p ucf-ops -- portability report --out ./out/portability_report.json
```

## Common failures and remediation

- **`audit path-scan` failed**
  - Remove hard-coded `/etc`/`/var` paths from runtime code.
  - Move OS/service-specific defaults to deploy templates or config.

- **`audit hardware-scan` failed**
  - Replace vendor/machine assumptions with neutral `DeviceProfile` + budget/config controls.

- **`models verify` skipped in CI**
  - Add `models/manifest.toml` (preferred for current portability lane) or `models/MANIFEST.toml` if lifecycle validation is required.

- **`models eligibility` failed**
  - Validate probe/shadow/active evidence artifacts under `./out` and re-run eligibility.

- **`strict check` failed**
  - Use denial codes in `./out/strict_check.json` to remediate stale/missing compare/drift/hash evidence.

- **`operator report` failed**
  - Re-generate input artifacts (`v0/v1/v2 gate`, `eligibility`, `strict`, `drift`, `alerts`) and re-run.

## FAIL vs SKIP semantics

- **FAIL**: blocking regression; command executed but returned non-pass status or errored.
- **SKIP**: command is not applicable for the selected profile/fixture and reports explicit skip reason (non-blocking for optional paths only).
- In this v3 refresh, required checks above are expected to PASS on supported Linux/Windows setups.
