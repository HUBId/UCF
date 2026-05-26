# UCF Post-Metabolic Roadmap Selection

## 0. Purpose
- Selection only.
- No runtime/authority implementation.

## 1. Baseline
- HEAD: `4aa80452004aaf18b4e722be8be5125412588296`.
- Closure references:
  - `docs/roadmap/metabolic_hormone_control_layer_closure.md`
  - `docs/roadmap/metabolic_hormone_control_layer_roadmap_boundary_audit.md`
  - `docs/current_state_architecture_index.md`
  - `docs/module_implementation_depth_registry.md`

## 2. Current Closure Inventory

| Line | Current status | Deferred risks | Candidate next relevance |
|---|---|---|---|
| OptionalRealRuntime | Optional-real compile-only taxonomy and artifact planning documented; runtime invocation still deferred/absent. | Overclaiming compile support as runtime proof; fragile artifact/runtime environment. | Medium: useful later, but still blocked by runtime evidence prerequisites. |
| Consolidation | Bounded Micro/Meso/Macro candidate/audit/readback and local boundary documented. | Runtime replay/sleep/geist/identity integration remains deferred. | Medium: relevant as downstream consumer, not immediate next authority line. |
| Replay | Bounded Token→Schedule→Audit→AppliedBoundary + deterministic E2E + bounded append/readback documented. | Runtime replay scheduler/queue/worker and apply execution remain deferred. | High for handoff planning; avoid execution scope creep. |
| Sleep | Bounded SleepPlan candidate/audit/applied-boundary + deterministic E2E documented. | Runtime sleep coordinator, SleepCompleted semantics, and execution authority deferred. | High for handoff planning; keep candidate-only boundary. |
| Geist/ISM | Bounded projection/audit/local ISM candidate boundary + deterministic E2E documented. | ISM write/upsert, identity anchor/finalization, runtime authority deferred. | Medium: depends on replay/sleep maturity and identity decisions. |
| Evidence/Archive | Bounded audit/provenance append/readback and cross-layer readback E2E documented. | Runtime/identity/Gateway authority semantics intentionally deferred. | Medium: stable support line; avoid authority expansion. |
| Platform/Linux-only | Linux-only required target selected; Windows required readiness declassified. | CI/runtime portability constraints if runtime-heavy lines start too early. | High as operational guardrail for all next lines. |
| Metabolic/Hormone | M1–M8 bounded closure complete: state/update/modulation/replay-sleep candidates/verify-only audit present. | Runtime scheduler and direct replay/sleep/geist wiring deferred. | Very high: immediate upstream for next roadmap branch selection. |
| Prod/Workspace stability | Readiness/doc lint workflows exist; no new prod-readiness claim allowed. | Environment drift can invalidate runtime-heavy roadmap steps. | High as parallel operational line before major runtime promotion. |

## 3. Candidate Ranking

Scoring method: higher is better for Strategic value, Dependency readiness, Codex-Web suitability, Architecture unlock value; lower risk is rewarded by scoring Authority risk and Runtime/testing risk as **risk-adjusted safety** (5 = low risk, 1 = high risk).

| Candidate | Strategic value | Dependency readiness | Codex-Web suitability | Authority risk | Runtime/testing risk | Architecture unlock value | Total | Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A. Replay/Sleep Handoff Boundary from Metabolic Candidates | 5 | 5 | 5 | 3 | 3 | 5 | 26 | 2 |
| B. Runtime Scheduler / Queue Roadmap | 4 | 2 | 2 | 1 | 1 | 5 | 15 | 7 |
| C. OptionalRealRuntime continuation | 3 | 2 | 2 | 3 | 1 | 4 | 15 | 8 |
| D. Gateway Read API Expansion | 3 | 4 | 4 | 3 | 4 | 3 | 21 | 4 |
| E. Policy Ecology Read-Only Constraint Layer | 5 | 4 | 5 | 4 | 4 | 5 | 27 | 1 |
| F. Identity Anchor Authority Roadmap | 4 | 1 | 2 | 1 | 2 | 5 | 15 | 6 |
| G. Prod/Workspace Execution Environment | 4 | 4 | 4 | 5 | 3 | 4 | 24 | 3 |
| H. Metabolic → Replay/Sleep Docs/Interface only | 4 | 5 | 5 | 5 | 5 | 3 | 27 | 1 (tie) |

Tie-break: E is selected primary because it reduces cross-line governance ambiguity before any new cross-domain handoff semantics; H remains close and is selected as secondary line.

## 4. Roadmap Decision

| Line | Decision | Reason |
|---|---|---|
| E. Policy Ecology Read-Only Constraint Layer | Primary next line | Highest governance value with low authority/runtime risk and strong Codex-Web suitability; constrains future handoff semantics safely. |
| H. Metabolic → Replay/Sleep Docs/Interface only | Secondary line | Immediate metabolic continuity with low risk; can proceed once read-only policy constraints are explicit. |
| G. Prod/Workspace Execution Environment | Parallel operational line | Keeps CI/runner/hardware stability planning active without forcing runtime authority expansion. |
| A, B, C, D, F | Deferred | A deferred until read-only policy layer clarifies constraints; B/C/F high runtime/authority risk; D useful but less strategic than policy-first governance. |

## 5. Selected Prompt Series

| Prompt | Title | Goal | Guardrails |
|---|---|---|---|
| P1 | Policy Ecology Roadmap and Boundary Audit | Define policy-ecology bounded scope, authority boundaries, and dependencies. | No policy mutation implementation; docs/roadmap only. |
| P2 | Policy Record Authority and Schema Alignment | Align policy record ownership/schema boundaries across docs/specs. | No record runtime migration behavior; no gateway authority changes. |
| P3 | Read-Only Policy Field Contract | Specify immutable top-down read-only policy field contract. | No write path, no self-grant, no capability issuance activation. |
| P4 | Policy Constraint Evaluation Candidate | Define deterministic candidate-level constraint evaluation shape. | Candidate/verify-only; no runtime decision loop activation. |
| P5 | Policy Verify-Only Audit Contract | Add verify-only policy audit contract and bounded invariants. | No archive authority expansion; no execution side effects. |
| P6 | Policy Docs Overclaim Guard | Harden docs to prevent policy/governance overclaims. | No behavior changes, documentation guard only. |
| P7 | Policy Readiness Refresh | Run targeted formatting/docs validation for the policy lane outputs. | No full workspace/clippy/test expansion beyond approved scope. |

## 6. Guardrails
- no runtime scheduler without dedicated prompt.
- no replay/sleep execution.
- no policy mutation.
- no identity anchor/finalization.
- no Gateway/action authority.
- no prod-readiness claim.
- full workspace/clippy remains deferred in Codex-Web.

## 7. Recommended Next Prompt
- **P1 — Policy Ecology Roadmap and Boundary Audit**.
