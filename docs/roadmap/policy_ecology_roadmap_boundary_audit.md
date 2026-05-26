# UCF Policy Ecology Roadmap and Boundary Audit

## 0. Purpose
- Inventory/roadmap only.
- No policy mutation.
- No action authority.
- No runtime enforcement engine.

## 1. Baseline
- HEAD: `8b51bc0e4143be79b1fb3949461dcfbb51099909`.
- Required references inspected:
  - `docs/roadmap/post_metabolic_roadmap_selection.md`
  - `docs/current_state_architecture_index.md`
  - `docs/module_implementation_depth_registry.md`
  - `docs/minimal_spine_v1_freeze.md`
  - `README.md`
- Relevant crate/doc surfaces inspected via targeted grep/inventory:
  - `core/crates/ucf-policy-ecology`
  - `domains/policy-gateway/crates/ucf-policy-gateway`
  - `runtime/ucf-ops`
  - `runtime/ucf-replay`
  - `core/crates/ucf-sleep-coordinator`
  - `domains/geist/crates/ucf-geist`
  - `domains/ucf-neuromod`
  - policy/governance mentions in `docs/**/*.md`

## 2. Policy Surface Inventory

| Concern | Path | Current behavior | Maturity | Risk |
|---|---|---|---|---|
| Post-M policy selection | `docs/roadmap/post_metabolic_roadmap_selection.md` | P1-P7 policy-ecology series selected as docs-first, no runtime authority implementation. | docs-only | Overclaim if interpreted as implementation completion. |
| Current architecture authority framing | `docs/current_state_architecture_index.md` | Current-state authority order already warns against overclaim and deferred/runtime confusion. | docs-only | Drift risk if policy roadmap is not indexed as a current planning artifact. |
| Minimal spine freeze boundaries | `docs/minimal_spine_v1_freeze.md` | Explicitly forbids policy mutation/runtime authority expansion in v1.x claims. | bounded/tested | Scope creep could violate freeze if policy lane is interpreted as active authority. |
| Policy ecology crate exists | `core/crates/ucf-policy-ecology/src/lib.rs` | In-memory rule set (`PolicyEcology`, `PolicyRule`) and gate traits (`ReplayGate`, `GeistGate`, `SleepPhaseGate`) support deterministic allow/deny checks. No persistence/governance update protocol. | functional-prototype | Can be mistaken as full governance authority although currently local/bounded. |
| Gateway policy evaluator surface | `domains/policy-gateway/crates/ucf-policy-gateway/src/lib.rs` | `PolicyEvaluator` trait with `NoOpPolicyEvaluator`; emits no-op `PolicyDecision` envelope; no external authority wiring shown here. | skeleton | Naming overlap with “gateway” can be misread as action-approval authority. |
| Protocol policy records/types | `protocol/crates/ucf-protocol/src/lib.rs` + `protocol/crates/ucf-protocol/spec/*.md` | Contains `PolicyDecision`, `policy_id`, `constraint_ids`, policy digest/status fields as canonical message types. | bounded/tested | Message presence may be misread as enforcement/runtime policy engine activation. |
| Replay boundary | `runtime/ucf-replay/src/lib.rs` | Replay line is verify/recompute/audit oriented; no new policy mutation path identified in this audit. | bounded/tested | Terminology collision (“governance signals”) could imply policy authority if undocumented. |
| Sleep boundary | `core/crates/ucf-sleep-coordinator/src/lib.rs` | Candidate/audit/boundary records include explicit no-side-effect flags; `SleepPhaseGate` read path present. | bounded/tested | Could be overread as authorization for runtime sleep apply. |
| Geist/ISM boundary | `domains/geist/crates/ucf-geist/src/lib.rs` | Candidate/audit boundary fields explicitly deny identity/policy mutation/runtime authority; can read policy gate trait. | bounded/tested | ISM naming can be conflated with identity authority if boundaries are not restated. |
| Governance/docs sweeps | `runtime/ucf-ops/src/*governance*` and governance docs | Broad governance/reporting/sweep surfaces exist as checks and artifacts; policy-ecology write semantics not defined there. | partial | Governance vocabulary may be conflated with policy-ecology mutation authority. |

### Phase-2 required answers
- **Is there a policy crate?** Yes: `core/crates/ucf-policy-ecology`.
- **Are policy records/types present?** Yes, protocol-level message types exist (`PolicyDecision`, `policy_id`, `constraint_ids`, policy-status/digest fields).
- **Is Policy Ecology only docs?** No. There is docs planning plus a bounded crate-level prototype; still no approved top-down immutable PolicyField contract.
- **Do any crates mutate policy?** No canonical governance update/mutation pipeline was identified in scoped sources; current crate API exposes constructor-time rule provisioning, not documented lower-layer runtime mutation authority.
- **Are there gateway/action authority overlaps?** Overlap risk exists by naming (`policy-gateway`), but current audited implementation is no-op/skeleton and does not establish action-approval authority.
- **Are there identity/ISM overlaps?** Yes as risk surface only; Geist/Sleep docs and types explicitly keep identity anchor/finalization and ISM write authority deferred/forbidden in bounded lines.
- **Are constraints machine-readable?** Partially. There are typed rule enums and protocol fields, but no canonical cross-layer Policy Ecology schema/authority package selected yet.
- **Are governance update semantics defined?** Not for this policy-ecology line; governance sweeps exist, but no approved policy-governance update contract for mutable policy state.
- **Are there tests?** Yes for existing bounded crates/types; no dedicated P1 policy-ecology roadmap implementation tests are introduced here.

## 3. Boundary Decisions

| Boundary | Decision | Reason |
|---|---|---|
| Lower-layer writes | forbidden | Policy ecology must remain top-down and read-only to lower layers; mutation would violate bounded-line assumptions. |
| Gateway/action | forbidden | Prevents policy ecology from becoming direct action approval/execution authority. |
| Runtime enforcement | deferred | P1 is roadmap/audit only; no runtime enforcement engine in current scope. |
| Candidate evaluation | future read-only only | Candidate evaluation may consume policy constraints but must not execute actions or grant authority. |
| Identity/ISM | forbidden | Identity anchor/finalization and ISM write/upsert are outside this line and remain deferred. |
| Evidence/Archive | no append | Policy ecology line must not add append side effects or new authority paths. |
| Governance updates | deferred/human-authorized only | Any future policy updates require explicit human-authorized governance contract, not automatic runtime mutation. |
| Policy format | future explicit schema | Current surfaces are partial; a canonical typed schema/ownership decision is required in P2+. |

## 4. Risk Matrix

| Risk | Severity | Guardrail |
|---|---|---|
| policy becomes hidden global mutable state | high | Enforce docs boundary: no lower-layer writes; require explicit immutable snapshot contract before implementation. |
| policy evaluation becomes action approval | high | Keep evaluation candidate-only/verify-only; ban execute/approve-action semantics in policy component scope. |
| policy ecology becomes gateway authority | high | Forbid gateway write/action authority coupling; keep policy lane non-authoritative. |
| policy constraints overrule human governance | high | Governance updates deferred and human-authorized-only; no automatic override path. |
| lower layers modify policy | high | Declare lower-layer mutation forbidden; require read-only interfaces only. |
| identity anchoring through policy | high | Explicitly forbid identity anchor/finalization in policy lane. |
| policy audit mistaken for enforcement | medium | Label audit outputs verify-only and non-executing in schema/docs naming. |
| safety docs mistaken for runtime safety guarantee | high | Preserve overclaim guard text and “planning-only” status in roadmap/index docs. |
| non-machine-readable policy causing ambiguity | medium | P2/P3 must define canonical schema and ownership. |
| policy drift without governance/versioning | high | Require versioned snapshot semantics and human-authorized update plan before any mutation capability. |

## 5. Proposed Architecture Shape

| Proposed component | Purpose | Inputs | Outputs | Non-goals |
|---|---|---|---|---|
| `PolicyFieldV1` | Immutable/read-only policy snapshot as top-down normative field. | Versioned policy snapshot artifact + declared source authority. | Read-only policy field object/digest. | No mutation, no action execution, no gateway authority. |
| `PolicyConstraintV1` | Typed constraint unit for deterministic evaluation. | `PolicyFieldV1` entries + typed context keys. | Deterministic constraint records/reasons. | No runtime side effects, no override channel. |
| `PolicyContextV1` | Candidate context packaging only. | Candidate metadata, bounded replay/sleep/geist context digests. | Typed context value for evaluation. | No scheduler/worker execution, no identity finalization. |
| `PolicyEvaluationCandidateV1` | Candidate-level policy evaluation result. | `PolicyFieldV1` + `PolicyContextV1` + constraints. | Allow/deny/escalate candidate with traceable reasons (non-authoritative). | Not an action approval grant; no execution trigger. |
| `PolicyVerifyAuditV1` | Verify-only audit for determinism and boundary compliance. | Candidate evaluation outputs + snapshot digests. | Audit status/reasons/provenance digest. | Not enforcement engine; no append/write authority. |
| `PolicyGovernanceUpdatePlanV1` | Human-authorized docs-first update plan surface. | Proposed policy change package + governance justification metadata. | Planned/approved/rejected update record (manual workflow). | No automatic mutation, no lower-layer self-update. |

## 6. Prompt Series Plan

| Prompt | Title | Goal | Acceptance criteria | Guardrails |
|---|---|---|---|---|
| P2 | Policy Record Authority and Schema Alignment | Decide ownership + canonical schema boundaries for policy records. | Clear crate ownership, canonical record map, and non-authority wording aligned across docs/spec. | No runtime mutation/enforcement/gateway authority changes. |
| P3 | Read-Only Policy Field Contract | Define immutable `PolicyFieldV1` interface and snapshot semantics. | Typed read-only contract + versioning/provenance expectations documented. | No write/upsert path; no identity authority. |
| P4 | Policy Constraint Evaluation Candidate | Specify deterministic candidate evaluation shape only. | Candidate input/output schema, reason taxonomy, and forbidden terms list accepted. | No action execution/approval or runtime loop activation. |
| P5 | Policy Verify-Only Audit Contract | Specify verify-only policy audit structure and checks. | Deterministic audit status/failure reason set and provenance fields documented. | No enforcement semantics, no archive authority expansion. |
| P6 | Policy Docs Overclaim Guard | Harden docs against policy/governance overclaim language. | Explicit “not enforcement/not authority” guard text added where needed. | Docs-only; no behavior changes. |
| P7 | Policy Readiness Refresh | Run targeted formatting/docs checks for policy docs line. | fmt/docs lint clean for touched artifacts and links. | No full-workspace expansion claims. |

## 7. Current Status
- Policy Ecology is **not yet implemented** as a bounded typed top-down read-only layer (`PolicyFieldV1` etc.).
- Current work here is planning-only and boundary-audit only.
- P2 alignment is now available at `docs/roadmap/policy_record_authority_schema_alignment.md`.
- Recommended next step: **P3 — Read-Only Policy Field Contract**.

## 8. Open Questions
- Which crate owns `PolicyField`?
- Should policy be protocol/type layer or separate domain?
- What is the canonical format?
- How are human-authorized updates represented?
- How to avoid Gateway/action authority?
- How to keep lower layers read-only?
- How to version policy snapshots?
- How to test no mutation?

## 9. Recommended Next Prompt
UCF Prompt P3 — Read-Only Policy Field Contract
