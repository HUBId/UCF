# UCF Post-Policy Roadmap Selection

## 0. Purpose
- Selection only.
- No runtime/authority implementation.

## 1. Baseline
- HEAD: `3f256300b9e9e2c899ea710a99c880e700e8eb6a`.
- Closure references confirmed:
  - `docs/roadmap/policy_ecology_closure.md`
  - `docs/roadmap/metabolic_hormone_control_layer_closure.md`
  - `docs/current_state_architecture_index.md`
  - `docs/module_implementation_depth_registry.md`

## 2. Current Closure Inventory

| Line | Current status | Deferred risks | Candidate next relevance |
|---|---|---|---|
| OptionalRealRuntime | Metadata/fixture/contract planning exists; active runtime remains absent. | Runtime activation overclaim; backend promotion confusion. | Medium: useful boundary hardening, but weaker near-term architecture unlock than read APIs. |
| Consolidation | Bounded Micro/Meso/Macro closure line documented. | Overreading as full memory/runtime closure. | Medium: dependency context for query/handoff readbacks. |
| Replay | Bounded token/schedule/audit/applied-boundary closure documented. | Misread as runtime scheduler/worker readiness. | High as readback/query source; low for runtime scheduler rollout. |
| Sleep | Bounded candidate/audit/applied-boundary closure documented. | Misread as Sleep runtime activation. | High as readback/query source and policy-constrained context input. |
| Geist/ISM | Bounded projection/audit/ISM candidate closure documented. | Identity/ISM write authority overread risk. | High as readback/query source; authority-sensitive for direct activation lines. |
| Evidence/Archive | Bounded append/readback + cross-layer readback closure documented. | Could be overread as broad authority if query wording is loose. | Very high: natural base for read-only query layer before Gateway read expansion. |
| Metabolic/Hormone | Bounded state/update/modulation/replay-sleep candidate/audit closure documented. | Could be misread as action authority when combined with policy. | High: strong secondary line when constrained to candidate-only handoff semantics. |
| Policy Ecology | Bounded read-only/candidate/audit closure documented (P1-P7 closed). | Governance/mutation/approval semantic drift risk. | High: informs bounded policy-aware roadmap sequencing. |
| Platform/Linux-only | Linux-only required target; Windows required CI/readiness declassified/deferred. | Cross-platform overclaim and CI scope confusion. | Medium: keeps planning realistic for Codex-Web lane. |
| Prod/Workspace stability | Prod/runtime still blocked by absent OptionalRealRuntime and unstable full workspace/clippy conditions in this lane. | False prod-readiness claim risk. | High operational relevance; supports keeping runtime/scheduler deferred. |

## 3. Candidate Ranking

Scoring notes: 1–5 per criterion. Higher authority/runtime risk reduces score contribution.

| Candidate | Strategic value | Dependency readiness | Codex-Web suitability | Authority risk | Runtime/testing risk | Architecture unlock value | Total | Rank |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| F. Evidence/Archive Read API / Query Layer | 5 | 5 | 5 | 3 | 4 | 5 | 27 | 1 |
| A. Metabolic + Policy Handoff Boundary | 4 | 4 | 5 | 2 | 4 | 4 | 23 | 2 |
| G. Execution Environment / CI Runner Plan | 4 | 4 | 4 | 5 | 3 | 5 | 25 | 3 |
| B. Gateway Read API Expansion | 4 | 3 | 4 | 2 | 3 | 4 | 20 | 4 |
| E. OptionalRealRuntime continuation | 3 | 2 | 3 | 4 | 1 | 4 | 17 | 5 |
| C. Governance Update Model | 3 | 2 | 3 | 1 | 3 | 3 | 15 | 6 |
| D. Runtime Scheduler / Queue Roadmap | 3 | 1 | 2 | 1 | 1 | 4 | 12 | 7 |
| H. Identity Anchor Authority Roadmap | 2 | 1 | 2 | 1 | 2 | 3 | 11 | 8 |

## 4. Roadmap Decision

| Line | Decision | Reason |
|---|---|---|
| F. Evidence/Archive Read API / Query Layer | **Primary next line** | Highest immediate value with strongest dependency readiness and bounded read-only semantics before Gateway expansion. |
| A. Metabolic + Policy Handoff Boundary | **Secondary line** | High value after query-layer boundary work; must stay candidate-only/verify-only to avoid authority drift. |
| G. Execution Environment / CI Runner Plan | **Parallel operational line** | Needed to improve workspace/clippy/readiness confidence without coupling to authority semantics. |
| B, C, D, E, H | **Deferred** | Either higher authority semantics or higher runtime/environment dependence than current bounded planning lane allows. |

## 5. Selected Prompt Series

| Prompt | Title | Goal | Guardrails |
|---|---|---|---|
| EAQ1 | Evidence/Archive Query Layer Roadmap and Boundary Audit | Define bounded query-layer scope and non-goals. | Read-only only; no append mutation; no gateway/action authority. |
| EAQ2 | Query Record Authority and Read-Only Semantics Alignment | Map query record ownership and semantics across layers. | No policy mutation; no identity authority; no runtime execution semantics. |
| EAQ3 | Replay/Sleep/Geist/ISM Readback Query Candidate | Specify candidate query surfaces over bounded readback records. | Candidate/readback only; no scheduler/worker activation; no ISM write/upsert. |
| EAQ4 | Cross-Layer Query Verify-Only Audit Contract | Define deterministic verify-only query audit schema. | Verify-only; no enforcement/action approval. |
| EAQ5 | Query Docs Overclaim Guard | Add explicit docs guardrails against authority overclaim. | Docs-only language hardening; no behavior change. |
| EAQ6 | Query Readiness Refresh | Run targeted docs/format checks for query line artifacts. | No full workspace claims; no prod-readiness claim. |
| EAQ7 | Post-Query Roadmap Selection | Re-rank next strategic line after EAQ closure. | Planning-only; no runtime/authority rollout. |

## 6. Guardrails
- no runtime scheduler without dedicated prompt.
- no gateway/action authority without dedicated prompt.
- no policy mutation.
- no identity anchor/finalization.
- no prod-readiness claim.
- full workspace/clippy remains deferred in Codex-Web.

## 7. Recommended Next Prompt
- **EAQ1 — Evidence/Archive Query Layer Roadmap and Boundary Audit**.
