# UCF Linux-Only Target Boundary and Windows Portability Audit

## 0. Purpose
- Planning/audit only.
- No broad deletion in this prompt.
- No prod-readiness claim.
- No gate weakening.

## 1. Baseline
- HEAD: `60f3bcc14b9f3ca0816a226be79dc262fb0e16fc`
- Branch: `work`
- Dirty state at start: clean (`git status --short` had no entries)
- Required source links:
  - `.github/workflows/ci.yml`
  - `.github/workflows/nightly_verify.yml`
  - `docs/current_state_architecture_index.md`
  - `docs/roadmap/workspace_evidence_stability_roadmap_boundary_audit.md`

## 2. Windows / Portability Inventory

| Concern | Path | Current behavior | Linux relevance | Windows-only? | Risk |
|---|---|---|---|---:|---|
| CI OS matrix in main CI | `.github/workflows/ci.yml` | Main smoke/readiness workflow uses `matrix.os: [ubuntu-latest, windows-latest]`; many steps branch with `if: runner.os == 'Windows'` and `shell: pwsh`. | Linux lane is directly relevant; Windows lane is not required for Linux deployment. | yes (for Windows branch steps) | High CI overhead and maintenance churn. |
| Nightly Windows job | `.github/workflows/nightly_verify.yml` | Contains a dedicated `runs-on: windows-latest` job in addition to Linux nightly job. | Linux nightly remains relevant; Windows nightly appears additive. | yes | Extra runtime and false-negative surface unrelated to target deployment. |
| Linux-only jobs already present | `.github/workflows/ci.yml`, `.github/workflows/nightly_verify.yml` | Jobs/lanes such as `runs-on: ubuntu-latest` and feature-matrix Linux runs are already Linux-only. | Required baseline should keep these. | no | Low risk; these are the core target lanes. |
| Docs claiming Linux+Windows portability gate | `README.md`, `docs/continuous_verification.md`, `docs/portability_gate.md` | Existing language references portability and cross-OS smoke/reporting as first-class checks. | Portability documentation still partly relevant where it protects deterministic Linux artifacts. | mixed | Documentation ambiguity about officially supported OS targets. |
| Readiness docs with platform references | `docs/readiness_gate.md` | Describes deterministic/offline readiness behavior and CI comparability; interacts with workflows that currently include Windows lanes. | Yes, readiness semantics are Linux-relevant and should stay strict. | no (semantics), yes (Windows lane references) | Risk of weakening readiness if platform pruning is done incorrectly. |
| Portability reports/checks | `runtime/ucf-ops/src/*gate*.rs`, `README.md`, CI commands | Commands like `ucf-ops portability check/report` and `repro_portability.zip` are used in gates and CI smoke. | Keep if they validate deterministic behavior within Linux; remove cross-OS expectations later. | no | Removing too much could drop Linux determinism protections. |
| Windows/PowerShell-specific CI scripting | `.github/workflows/ci.yml` | Multiple `pwsh` blocks, Windows path copies, and Windows-only smoke/readiness invocations (`workspace_test_report_windows.json`, `gate_report_windows.json`). | Not required for Linux deployment target. | yes | Pure overhead once Windows support is out of scope. |
| Path normalization and newline normalization | `runtime/ucf-ops/src/docs_lint.rs` and related docs checks | `normalize_newlines` style logic (`\r\n` to `\n`) and portability docs link checks guard reproducibility/document consistency. | Yes, these checks prevent Linux doc drift from CRLF or inconsistent inputs. | no | Must not be removed blindly; could regress Linux reproducibility. |
| Tests mentioning portability/Windows | `runtime/ucf-ops` gates/tests and CI smoke step names | Some checks encode portability as a concept; most are not strictly Windows runtime tests but docs/report consistency checks. | Often yes when scoped to deterministic artifact contracts. | mixed | Misclassification can remove useful Linux safety coverage. |
| Historical portability/readiness docs | `docs/*sweep*`, historical sections in `docs/current_state_architecture_index.md` | Many portability/readiness sweep docs are historical and not live support commitments. | Useful as audit history only. | no | Overclaim risk if historical docs are read as current support policy. |

### Required answers from the audit
- Workflows on `windows-latest`: `ci.yml` (matrix lane and many Windows-only steps) and `nightly_verify.yml` (dedicated job).
- Workflows on `macos-latest`: none found in audited workflows.
- Jobs already Linux-only: `verify` job in nightly (`ubuntu-latest`), initial Linux CI job(s), and `feature-matrix` in `ci.yml` (`ubuntu-latest`).
- Docs claiming Windows support: `README.md` portability section and continuous-verification/portability docs references imply Linux+Windows parity.
- Tests mentioning portability/Windows: mainly `ucf-ops` portability/gate/report checks and CI smoke step coverage.
- Reports mentioning portability: `portability_report*.json`, `portability_check_*.json`, and repro portability bundle references.
- Path normalization utilities required for Linux: newline and deterministic docs normalization checks remain relevant for Linux reproducibility.
- Windows-specific CI overhead: `pwsh` steps, Windows copy/path handling, Windows workspace-test and readiness smoke outputs.
- Cross-platform checks to keep: deterministic portability checks that validate docs/report consistency and within-OS determinism for Linux outputs.

## 3. Target Support Decision

| Option | Chosen? | Reason | Risk |
|---|---:|---|---|
| A. Linux-only deployment + Linux-only required CI | yes | Best scope cut for stated UCF target, removes non-target Windows readiness overhead while preserving strict Linux gates. | Must avoid deleting Linux-relevant determinism checks during pruning. |
| B. Linux required, macOS best-effort dev, Windows removed | no | Possible fallback, but not needed for this boundary decision. | Could retain unnecessary breadth. |
| C. Keep cross-platform support | no | Misaligned with declared deployment target. | Ongoing CI/docs complexity and ambiguous support contract. |

Recommended wording:
- Supported deployment target: Linux x86_64.
- Required CI target: Linux.
- macOS: optional/best-effort developer environment, not release target.
- Windows: unsupported; Windows-specific CI/readiness is removed or historical only.
- WSL2: acceptable developer workaround, but not a formal target.

## 4. Removal / Declassification Plan

| Item | Action | Reason | Safe now? | Follow-up prompt |
|---|---|---|---:|---|
| `.github/workflows/ci.yml` Windows matrix lane and `runner.os == 'Windows'` smoke/readiness steps | remove workflow job | Non-target CI overhead for unsupported OS. | no | Prompt 79W2 |
| `.github/workflows/nightly_verify.yml` Windows job | remove workflow job | Nightly verification should match required Linux target. | no | Prompt 79W2 |
| `docs/continuous_verification.md` Windows wording | mark historical / update support language | Align docs with Linux-only support contract. | no | Prompt 79W2 |
| `docs/readiness_gate.md` platform scope text | declassify from required readiness on Windows | Keep readiness semantics strict but Linux-targeted. | no | Prompt 79W2 |
| `docs/current_state_architecture_index.md` platform support note | keep as current planning pointer + add Linux-only boundary audit link | Prevent support ambiguity; central index should point to boundary decision. | yes | this prompt (link only) |
| `docs/module_implementation_depth_registry.md` platform/support mentions | review before removal | Mostly status taxonomy; may not need hard removals, only wording alignment. | no | Prompt 79W2 |
| Portability report docs/tests | keep because Linux-relevant | Some checks protect deterministic Linux outputs; classify per-check before pruning cross-OS clauses. | no | Prompt 79W2 |
| Path normalization code/tests | keep because Linux-relevant | Prevent CRLF/path drift and deterministic output regressions. | yes | Prompt 79W2 review checklist |

## 5. Risk Matrix

| Risk | Severity | Mitigation |
|---|---|---|
| Accidentally removing Linux-relevant portability checks | High | Keep determinism/newline/path normalization checks unless proven Windows-only; prune only OS lanes first. |
| Docs drift after CI scope change | Medium | Update index + verification/readiness docs in same change set as workflow pruning prompt. |
| CI not testing intended target | High | Require Linux lanes as mandatory checks in CI/nightly after matrix reduction. |
| Hidden Windows references still claiming support | Medium | Run targeted `rg` sweeps for `windows|pwsh|windows-latest|PowerShell` in docs/workflows in Prompt 79W2. |
| Path handling regression | High | Preserve and explicitly retain Linux-relevant normalization tests/checks during declassification. |
| Operator confusion around WSL/macOS | Medium | Add explicit guardrail wording: WSL2 convenience only; macOS best-effort only; Linux required target. |
| Release target ambiguity | High | Canonical support statement in current-state index and readiness/continuous verification docs. |

## 6. Guardrails
- Linux x86_64 is the supported deployment target.
- Required CI should be Linux-only unless a future prompt explicitly reauthorizes extra target lanes.
- Windows support is unsupported/deferred.
- macOS is best-effort only (not release target).
- Removing Windows does not imply prod readiness.
- Do not remove Linux-relevant path/determinism checks.
- Do not weaken readiness gates; only remove unsupported platform lanes.

## 7. Recommended Next Prompt
- UCF Prompt 79W2 — Windows CI/Readiness Declassification and Linux-Only Docs Alignment.
