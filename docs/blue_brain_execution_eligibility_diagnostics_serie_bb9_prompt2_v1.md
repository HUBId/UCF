# Serie BB9 Prompt 2: Execution eligibility diagnostics und blocked/safety feedback bind-back

Status: BB9 Prompt 2 konsolidiert die kanonische **Execution-Eligibility-Diagnostics-Schicht** für Blue-Brain-Handoffs.
Der Scope bleibt strikt diagnostisch: **keine Action Execution**, **keine Tool Invocation**, **keine Compute Invocation**,
**kein Memory Commit**, keine Planner-/Policy-/Governance-/RL-Engine.

Repo-Anker (code-pinned):
- `runtime/ucf-compute/src/reference_map.rs`
  - `CANONICAL_BLUE_BRAIN_ACTION_EXECUTION_ELIGIBILITY_MAP`
  - `CANONICAL_BLUE_BRAIN_SAFETY_PRECHECK_MAP`
  - `CANONICAL_BLUE_BRAIN_EXECUTION_ELIGIBILITY_DIAGNOSTICS_MAP`
  - `CANONICAL_BLUE_BRAIN_FUTURE_ACTION_HANDOFF_MAP`
  - `CANONICAL_BLUE_BRAIN_CANDIDATE_COMPARISON_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTROL_ATTENTION_SELECTION_MAP`
  - `CANONICAL_BLUE_BRAIN_CONTEXT_EVIDENCE_PRIORITY_MAP`
  - `CANONICAL_BLUE_BRAIN_MEMORY_DIAGNOSTICS_MAP`

## 1) Kanonische Eligibility-Diagnostics-Klassen

- `execution-eligible diagnostic`
- `execution-ineligible diagnostic`
- `execution-blocked diagnostic`
- `execution-caveated diagnostic`
- `execution-insufficient diagnostic`
- `safety-precheck-passed diagnostic`
- `safety-precheck-failed diagnostic`
- `safety-precheck-blocked diagnostic`
- `safety-precheck-caveated diagnostic`
- `safety-precheck-unavailable diagnostic`
- `non-canonical/internal-only execution diagnostic`

Diese Klassen werden ausschließlich über `CANONICAL_BLUE_BRAIN_EXECUTION_ELIGIBILITY_DIAGNOSTICS_MAP` geführt.

## 2) Kompakte kanonische Gründe

Die Diagnostics tragen kompakte, kanonische reason-Klassen und reason-compacts:

- eligible wegen ausreichender proposal/context/evidence/memory Basis
- ineligible wegen unzureichender proposal basis
- blocked wegen stale/invalidated memory
- blocked wegen missing context/evidence basis
- blocked wegen failed safety precheck
- caveated wegen partial evidence oder caveated memory
- unavailable weil precheck/execution subsystem nicht verfügbar
- blocked wegen internal-only dependency (non-canonical)

Kein freies Tool- oder Policy-Result-Narrativ wird als kanonische Ursache eingeführt.

## 3) Blocked-/Safety-Feedback bedeutet explizit nur Boundary-Status

`blocked`/`failed`/`unavailable`/`caveated` Feedback bedeutet:

1. Eligibility oder Precheck hat die Ausführungsfähigkeit nicht freigegeben, **oder**
2. die Ausführungsgrenze / das Subsystem ist nicht verfügbar, **oder**
3. die Basis ist unzureichend.

Es bedeutet ausdrücklich **nicht**:

- Tool wurde ausgeführt,
- Action wurde ausgeführt oder ist fehlgeschlagen,
- Policy/Governance hat entschieden,
- Planner hat abgelehnt,
- Compute wurde invokiert.

## 4) Bind-back in Handoff, Proposal, Selection, Context, Memory, Runtime

### 4.1 Future-Action-Handoff + Proposal
- `future-action-ready` kann als rein diagnostischer Zustand erhalten bleiben.
- `future-action-ready` kann in `execution-eligible` übergehen, wenn Basis + Precheck passen.
- `action-ready proposal` kann `execution-ineligible` bleiben.
- Handoff kann `blocked`/`caveated`/`insufficient` bleiben.

### 4.2 Selection / Priority / Deferral
- `execution-eligible` kann als auswählbarer Future-Action-Handoff markiert sein.
- `blocked` bleibt deferred/blocked.
- `caveated` bleibt caveated.
- `insufficient` bleibt nicht execution-eligible.
- `safety-precheck-failed` schließt aktuelle execution-eligibility aus.

### 4.3 Context / Evidence / Memory
- Diagnostics binden weiter auf BB3/BB8-Basis:
  - context/evidence/reference sufficiency,
  - memory current/stale/invalidated/caveated/missing,
  - candidate/proposal diagnostics.
- Kein Memory Commit wird impliziert.

### 4.4 Runtime
Runtime kann explizit sehen:
- execution eligibility observed,
- execution eligible/ineligible/blocked/caveated/insufficient,
- safety precheck passed/failed/blocked/caveated/unavailable,
- und dabei weiterhin: no action execution, no tool invocation, no compute invocation, no memory commit.

## 5) Non-canonical Diagnostics Abgrenzung

Internal/expert/legacy/test-dev-only Pfade sind nicht kanonisch, solange kein explizites Down-Mapping auf
Handoff-/Eligibility-/Precheck-/Context-/Memory-/Selection-Referenzen erfolgt.

`non-canonical/internal-only execution diagnostic` bleibt daher explizit `canonical=false` und ist
keine Autorität für kanonische Eligibility-/Safety-Aussagen.

## 6) Harte Grenzen (BB9 Prompt 2)

Prompt 2 baut eine Diagnostik- und Feedbackschicht — **keine** Execution-/Tool-/Policy-/Planner-/Monitoring-Plattform.

- no action execution
- no tool invocation
- no compute invocation
- no memory commit

Compute-Kern bleibt maintenance-only. Hodgkin-Huxley/Kuramoto bleiben außerhalb dieses Schritts.
