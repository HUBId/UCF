# Portability Gate v1 (Linux + Windows)

`Portability Gate` blocks merges when core runtime/ops checks are not cross-platform safe.

## What is checked

1. **Cross-platform CI matrix (blocking)**
   - Linux lane:
     - `cargo test --workspace --all-targets`
     - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
     - `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json --workdir ./.ucf_gate`
     - `cargo run -p ucf-ops -- audit hardware-scan`
     - `cargo run -p ucf-ops -- audit path-scan`
     - `cargo run -p ucf-ops -- portability check --out ./out/portability.json`
   - Windows lane:
     - `cargo test --workspace --all-targets`
     - `cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json`
     - `cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report_windows.json --workdir ./.ucf_gate_windows`
     - `cargo run -p ucf-ops -- audit hardware-scan`
     - `cargo run -p ucf-ops -- audit path-scan`
     - `cargo run -p ucf-ops -- portability check --out ./out/portability.json`

2. **Path hygiene scan (`audit path-scan`)**
   - Scans runtime crate source files (`runtime/*/src/*.rs`) for hard-coded OS-specific path/system assumptions:
     - `"/etc/"`
     - `"/var/"`
     - `"systemd"`
     - `"systemctl"`
   - Allowlist/exclusions:
     - `deploy/` templates
     - vendor/target/fuzz scopes
     - `runtime/ucf-ops/src/` (scanner implementation)

3. **Portability check (`portability check`)**
   - Runs a short deterministic toy/stub scenario.
   - Emits `out/portability.json` with:
     - schema version + OS/arch
     - digest prefixes
     - fixed-point scalar summary (`risk_q`, `pressure_q`, `surprise_q`, `uncertainty_q`)
     - deterministic-within-OS status
     - remediation guidance

4. **Hardware-neutral guardrails in CI**
   - `docs lint` hardware-neutral checks
   - `audit hardware-scan` runtime scan
   - `audit path-scan` runtime path hygiene

## Determinism across OS

- For toy/stub paths we target deterministic behavior and canonical ordering.
- The gate enforces deterministic behavior **within each OS lane** (`deterministic_within_os=true`).
- Cross-OS exact digest parity can still differ in edge cases; use fixed-point envelope fields + schema stability as the portable contract.

## Local run instructions

### Linux/macOS shell

```bash
cargo test --workspace --all-targets
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- audit hardware-scan
cargo run -p ucf-ops -- audit path-scan
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report.json --workdir ./.ucf_gate
cargo run -p ucf-ops -- portability check --out ./out/portability.json
```

### Windows PowerShell

```powershell
cargo test --workspace --all-targets
cargo run -p ucf-ops -- docs lint --strict --out ./out/docs_lint_report.json
cargo run -p ucf-ops -- audit hardware-scan
cargo run -p ucf-ops -- audit path-scan
cargo run -p ucf-ops -- readiness-gate --profile test --out ./out/gate_report_windows.json --workdir ./.ucf_gate_windows
cargo run -p ucf-ops -- portability check --out ./out/portability.json
```

## Common failures and remediation

- **`audit path-scan` failed**
  - Remove hard-coded `/etc`/`/var` paths from runtime code.
  - Move OS/service-specific defaults to deploy templates or config.

- **`portability check` deterministic-within-OS failed**
  - Check ordering stability (sort keys, avoid random iteration).
  - Ensure canonical serialization for externally visible digests.
  - Keep fixed-point paths canonical and avoid float-only gate logic.

- **hardware scan failed**
  - Replace vendor/machine assumptions with neutral `DeviceProfile` + budget/config controls.
