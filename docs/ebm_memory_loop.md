# EBM Memory Loop v1

This document describes the v1 memory loop wiring for ESS/consolidation without any EBM training.

## Experience energy tagging

- ESS records can carry a bounded `ExperienceEbmTagRecord` on `ExperienceRecord.ebm_tag`.
- Tag payload is audit-safe and bounded:
  - `ebm_energy_min_q`
  - `ebm_energy_mean_topk_q`
  - `ebm_constraints_digest_prefix`
  - `ebm_top_terms` (max 4 entries)
  - `ebm_reasoning_digest_prefix`
- `make_ebm_tag_from_reasoning` derives tags deterministically from `EbmReasoningRecord` and the associated `evidence_chain_digest`.

## Retrieval roles and safety constraints

Retrieval uses deterministic energy-bias policy (`apply_ebm_bias`) and assigns one role:

- `PrecedentSafe`
- `Template`
- `AvoidExample`

Safety dominance rule:

- High-energy experiences (`energy_q >= high_energy_threshold_q`) are *never* returned as `Template`.
- High-energy items can only be emitted as bounded avoid reminders (`max_avoid <= 2` by policy).

## Retrieval audit trail

`RetrievalDecisionRecord` persists bounded evidence:

- query digest prefix (no raw query)
- selected experience ids/digest prefixes and role assignments
- thresholds used
- policy hash prefix
- evidence-chain digest prefix
- reason codes:
  - `EbmBiasApplied`
  - `AvoidExamplesIncluded`
  - `HighRiskContext`

## Consolidation energy-aware milestone selection

Consolidation cycle uses an energy-biased prefilter before milestone construction:

- retain low-energy stable records
- retain a bounded number of high-energy anomalies (`MAX_ANOMALIES = 2`) as warning exemplars
- publish `MilestoneEbmSummary` with selected counts + thresholds

This keeps behavior deterministic and bounded while preserving anomaly signal for later replay analysis.

## Operational notes

- No EBM training is required.
- All added records are digest-based and bounded.
- Retrieval and consolidation decisions are deterministic for fixed inputs and policy values.
