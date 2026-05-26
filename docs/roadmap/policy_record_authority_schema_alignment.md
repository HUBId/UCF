# UCF Policy Record Authority and Schema Alignment

## 0. Purpose
- Authority alignment only.
- No policy mutation.
- No action authority.
- No runtime enforcement.

## 1. Baseline
- HEAD: `88f41c758c19b67efe8caa0009a919b0f096da8c`.
- Policy roadmap present: yes (`docs/roadmap/policy_ecology_roadmap_boundary_audit.md`).
- Crate/path baseline:
  - `core/crates/ucf-policy-ecology`: present.
  - `protocol`: present.
  - `runtime/ucf-ops`: present.
  - `domains/ucf-policy-ecology`: not present (scope uses `core/crates/ucf-policy-ecology`).

## 2. Policy Record Inventory

| Record / API | Path | Current role | Maturity | Authority risk |
|---|---|---|---|---|
| `v1::spec::PolicyDecision` (`kind`, `action`, `constraint_ids`) | `protocol/crates/ucf-protocol/src/lib.rs`, `protocol/crates/ucf-protocol/spec/messages_v1.md` | Protocol decision payload primitive in ControlFrame schema. | protocol primitive | `action` naming can be misread as runtime action authority if not bounded to record semantics. |
| `policy_id` (ControlFrame) | `protocol/crates/ucf-protocol/src/lib.rs`, `protocol/crates/ucf-protocol/spec/messages_v1.md` | Stable policy version identifier in protocol envelope. | protocol primitive | Could be misread as live mutable policy store handle. |
| `policy_decision_digest` / `policy_status` fields | `protocol/crates/ucf-protocol/src/lib.rs`, `protocol/crates/ucf-protocol/spec/messages_v1.md` | Deterministic linkage/status fields for candidate/output records. | protocol primitive | Could be overread as enforcement completion marker. |
| `PolicyRule` enum | `core/crates/ucf-policy-ecology/src/lib.rs` | Local typed rule set used by gate checks. | local prototype | Includes allow/deny semantics; must not be treated as global governance authority. |
| `PolicyEcology` (`version`, `rules`) | `core/crates/ucf-policy-ecology/src/lib.rs` | In-memory policy container instantiated locally. | local prototype | Constructor-time rule injection could be confused with approved governance update path. |
| `ReplayGate` / `GeistGate` / `SleepPhaseGate` traits | `core/crates/ucf-policy-ecology/src/lib.rs` | Read/check interfaces for replay/sleep/geist boundaries. | local prototype | Gate naming can imply authority beyond bounded local checks. |
| `DefaultPolicyEcology` | `core/crates/ucf-policy-ecology/src/lib.rs` | Deterministic default local rule profile. | local prototype | Could be mistaken for canonical policy authority profile. |
| Policy gateway evaluator + no-op surface (from P1 inventory) | `domains/policy-gateway/crates/ucf-policy-gateway/src/lib.rs` | Prototype evaluator skeleton/no-op behavior. | local prototype | High naming collision with gateway/action authority. |
| Governance sweep/report artifacts | `runtime/ucf-ops/src/*governance*`, docs referenced by docs-lint | Audit/reporting and consistency checks for governance surfaces. | docs-only | Broad “governance authority” terms can be conflated with policy mutation authority. |
| Policy ecology roadmap boundary decisions | `docs/roadmap/policy_ecology_roadmap_boundary_audit.md` | Planning constraints and forbidden authority surfaces. | docs-only | Drift if not treated as hard guardrail for P3 scope. |

Phase-2 required answers:
- Policy types in protocol: `PolicyDecision`, `policy_id`, `constraint_ids`, `policy_decision_digest`, `policy_status` exist as schema-level record fields.
- Policy types in `ucf-policy-ecology`: `PolicyRule`, `PolicyEcology`, `DefaultPolicyEcology`, gate traits (`ReplayGate`, `GeistGate`, `SleepPhaseGate`) exist.
- Gateway/evaluator surfaces: policy-gateway evaluator/no-op skeleton exists (prototype/supporting only).
- Mutable stores present: no canonical persistent mutable policy store identified in scoped sources.
- Update pathway present: no explicit human-authorized policy update contract/API implemented.
- Policy decisions read-only vs action-like: encoded as record payloads; must be treated read-only/reporting primitives, not action grants.
- Constraints typed vs string/docs-only: both exist in fragments (`PolicyRule` typed enums + protocol `constraint_ids` references); no single bounded typed PolicyField contract yet.
- Governance update semantics defined: not for policy mutation authority; governance sweeps are audit/report surfaces.

## 3. Authority Classification

| Record/API | Authority decision | Reason |
|---|---|---|
| Protocol `PolicyDecision` | primitive/supporting only | Protocol payload primitive; no runtime enforcement or gateway execution path is granted by schema presence. |
| Protocol `policy_id`, `constraint_ids`, policy digests/status | primitive/supporting only | Referential/canonical metadata only; no mutable policy storage semantics attached. |
| `PolicyRule` + `PolicyEcology` in `ucf-policy-ecology` | prototype only | Local bounded logic exists but no canonical cross-layer ownership/governance update contract. |
| `ReplayGate`/`SleepPhaseGate`/`GeistGate` | explicitly not authority | Read/check boundary hooks only; do not authorize writes, execution, or gateway action. |
| Policy-gateway evaluator/no-op surface | explicitly not authority | Prototype/no-op and naming-risky; must not be treated as final authority plane. |
| Governance sweep surfaces in `ucf-ops` | audit/report only | Report/consistency artifacts, not policy mutation semantics. |
| Future `PolicyFieldV1` (or equivalent) in `ucf-policy-ecology` | future bounded authority candidate | Intended read-only bounded authority object for P3 with explicit non-mutation contract. |
| Runtime policy mutation/update APIs | forbidden for P3 | Out of scope by roadmap boundary and Minimal Spine freeze constraints. |

## 4. Naming / Semantics Boundary

| Term | Allowed meaning | Forbidden meaning |
|---|---|---|
| PolicyField | Immutable read-only snapshot of bounded policy state. | Mutable live policy control plane or auto-updating authority. |
| PolicyConstraint | Typed deterministic constraint descriptor evaluated read-only. | Dynamic imperative action trigger or lower-layer override instruction. |
| PolicyContext | Candidate evaluation input context package. | Runtime execution context that can write/apply actions. |
| PolicyEvaluationCandidate | Candidate-only verify/evaluate result with reasons. | Final action approval, execution grant, or gateway decision token. |
| PolicyDecision | Protocol/report decision record and rationale metadata. | Direct runtime enforcement command or gateway action authority. |
| PolicyVerifyAudit | Verify-only boundary/determinism audit record. | Enforcement engine output or mutation authorizer. |
| PolicyGovernanceUpdatePlan | Human-authorized docs/process plan for future updates. | Automatic mutation pathway or self-authorizing governance write. |
| Constitution | High-level normative framing documentation. | Runtime mutable rule source bypassing bounded policy schema/versioning. |
| Enforcement | Deferred future runtime capability only when explicitly authorized. | Any current P2/P3 behavior claim. |
| Approval | Human/manual governance review concept only. | Machine-issued action approval for gateway/lower-layer execution. |
| Override | Explicitly disallowed for lower-layer authority boundaries. | Policy layer overriding replay/sleep/geist/runtime writes at execution time. |

## 5. Schema Placement Decision

| Option | Chosen? | Reason | Risk |
|---|---:|---|---|
| A. Local bounded schema in `ucf-policy-ecology` | yes | Matches P1 boundary: targeted, deterministic, read-only contract can be added without promoting protocol/runtime authority too early. | Potential later migration work when protocol promotion is approved. |
| B. Protocol schema first | no | Premature before bounded read-only contract semantics are proven locally and naming risks are settled. | Early protocol lock-in could encode overbroad authority semantics. |
| C. Docs-only | no | Insufficient for P3 contract acceptance and testability. | Ambiguity persists; no enforceable local schema boundaries. |

## 6. P3 Acceptance Criteria

| Criterion | Required? | Notes |
|---|---:|---|
| Implement `PolicyFieldV1` (or equivalent) in `ucf-policy-ecology` | yes | Must be local bounded schema authority candidate only. |
| Immutable/read-only snapshot semantics | yes | No mutable fields exposed for update-in-place behavior. |
| `PolicyConstraintV1` typed constraints | yes | Avoid string-only ambiguity for bounded deterministic checks. |
| No mutation/update method | yes | No `set`, `update`, `upsert`, or governance auto-write path. |
| Deterministic canonical bytes/digest (if feasible) | yes | Prefer canonical encoding order and stable digest derivation. |
| No action approval semantics | yes | Candidate/verify-only outputs only. |
| No gateway authority | yes | No gateway execute/approve/write link introduced. |
| No identity authority | yes | No IdentityAnchor/IdentityFinalization semantics. |
| No Evidence/Archive append authority | yes | No append mutation side effects added by policy records. |
| Tests prove read-only/no mutation/no action authority | yes | Targeted tests in `ucf-policy-ecology`; no full workspace expansion required. |

## 7. Current Status
- Policy Ecology bounded `PolicyFieldV1` implementation is now present in `core/crates/ucf-policy-ecology/src/policy_field_v1.rs` with immutable/read-only validation and deterministic digest bytes.
- Existing policy-related surfaces are protocol primitives, local prototypes, and audit/supporting docs.
- P3 (Read-Only Policy Field Contract) is implemented with targeted tests in `core/crates/ucf-policy-ecology/tests/policy_field_v1.rs`.
- P4 (Policy Constraint Evaluation Candidate) is implemented with `PolicyContextV1`, `PolicyEvaluationCandidateV1`, and `evaluate_policy_constraints_v1(...)` in `core/crates/ucf-policy-ecology/src/policy_evaluation_candidate_v1.rs`, covered by `core/crates/ucf-policy-ecology/tests/policy_evaluation_candidate_v1.rs`.
- P5 (Policy Verify-Only Audit Contract) is implemented with `PolicyVerifyAuditStatusV1`, `PolicyVerifyAuditFailureV1`, `PolicyVerifyAuditV1`, and `verify_policy_evaluation_v1(...)` in `core/crates/ucf-policy-ecology/src/policy_verify_audit_v1.rs`, covered by `core/crates/ucf-policy-ecology/tests/policy_verify_audit_v1.rs`.

## 8. Open Questions
- Should protocol promote local policy records later after P3/P4 stabilization?
- What exact typed constraint expression model should `PolicyConstraintV1` use?
- How should `PolicyField` versioning be represented (`u32` schema version + digest domain)?
- How should human-authorized governance updates be represented without enabling runtime mutation?
- Which concrete interfaces guarantee lower-layer read-only consumption across replay/sleep/geist?
- What checks prevent gateway/action authority from being inferred by naming or API shape?

## 9. Recommended Next Prompt
UCF Prompt P6 — Policy Docs Overclaim Guard
