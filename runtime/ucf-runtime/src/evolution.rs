use blake3::Hasher;

pub const STRUCTURAL_DELTA_SCHEMA_VERSION: u16 = 1;
pub const MAX_DELTA_OPS: usize = 8;
pub const MAX_DELTA_CANDIDATES: usize = 8;
pub const MAX_SCORE_AUDIT_ITEMS: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeltaTarget {
    FepWeights,
    PolicyThresholds,
    ComputeBudgetHints,
    CoherenceGating,
    BiophysGating,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SmallKey {
    BetaPolicyRisk,
    BetaCoherenceLock,
    StructureDeltaCap,
    CoherenceMinClosedLoopGain,
    CoherenceMaxUncheckedDrift,
    CoherenceMaxMemoryPressure,
    CoherenceRiskInhibitMin,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum DeltaOp {
    Set { key: SmallKey, value: f32 },
    Add { key: SmallKey, delta: f32 },
    Clamp { key: SmallKey, min: f32, max: f32 },
}

pub type DeltaId = [u8; 32];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeltaProvenance {
    pub evidence_chain_digest: [u8; 32],
    pub source_window: (u64, u64),
    pub engine_id: &'static str,
    pub seed: u64,
    pub score_digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct StructuralDelta {
    pub schema_version: u16,
    pub delta_id: DeltaId,
    pub t: u64,
    pub target: DeltaTarget,
    pub ops: Vec<DeltaOp>,
    pub provenance: DeltaProvenance,
    pub digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TunableSnapshot {
    pub beta_policy_risk: f32,
    pub beta_coherence_lock: f32,
    pub structure_delta_cap: f32,
    pub coherence_min_closed_loop_gain: f32,
    pub coherence_max_unchecked_drift: f32,
    pub coherence_max_memory_pressure: f32,
    pub coherence_risk_inhibit_min: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EvolutionContext {
    pub t: u64,
    pub source_window: (u64, u64),
    pub evidence_chain_digest: [u8; 32],
    pub risk_mean: f32,
    pub confidence_mean: f32,
    pub coherence_mean: f32,
    pub instability_mean: f32,
    pub budget_exceeded_rate: f32,
    pub denied_tool_rate: f32,
    pub stress_index: Option<f32>,
    pub neuro_arousal: Option<f32>,
    pub params: TunableSnapshot,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EvolutionBudget {
    pub work_units: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum ReasonCode {
    ImprovesStability,
    ReducesBudgetExceed,
    DegradesConfidence,
    ViolatesClamp,
    TooAggressiveChange,
    IncreasesRisk,
    WeakSafetyMargin,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScoreAudit {
    pub metrics_used: Vec<SmallKey>,
    pub reasons: Vec<ReasonCode>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct DeltaScore {
    pub fitness: f32,
    pub risk_penalty: f32,
    pub stability_penalty: f32,
    pub budget_penalty: f32,
    pub audit: ScoreAudit,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, PartialEq)]
pub struct SelectionResult {
    pub accepted: Option<(StructuralDelta, DeltaScore)>,
    pub rejected: Vec<(DeltaId, Vec<ReasonCode>)>,
}

pub trait EvolutionEngine {
    fn name(&self) -> &'static str;
    fn propose(&mut self, ctx: EvolutionContext, budget: EvolutionBudget) -> Vec<StructuralDelta>;
    fn evaluate(&self, delta: &StructuralDelta, ctx: &EvolutionContext) -> DeltaScore;
    fn select(&self, candidates: &[(StructuralDelta, DeltaScore)]) -> SelectionResult;
}

#[derive(Clone, Debug)]
pub struct MockEvolutionEngineV0 {
    seed: u64,
}

impl MockEvolutionEngineV0 {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn key_bounds(key: SmallKey) -> (f32, f32) {
        match key {
            SmallKey::BetaPolicyRisk => (0.5, 3.0),
            SmallKey::BetaCoherenceLock => (0.5, 3.0),
            SmallKey::StructureDeltaCap => (0.05, 0.6),
            SmallKey::CoherenceMinClosedLoopGain => (0.2, 0.8),
            SmallKey::CoherenceMaxUncheckedDrift => (0.4, 0.95),
            SmallKey::CoherenceMaxMemoryPressure => (0.5, 0.95),
            SmallKey::CoherenceRiskInhibitMin => (0.4, 0.9),
        }
    }

    fn value_of(params: TunableSnapshot, key: SmallKey) -> f32 {
        match key {
            SmallKey::BetaPolicyRisk => params.beta_policy_risk,
            SmallKey::BetaCoherenceLock => params.beta_coherence_lock,
            SmallKey::StructureDeltaCap => params.structure_delta_cap,
            SmallKey::CoherenceMinClosedLoopGain => params.coherence_min_closed_loop_gain,
            SmallKey::CoherenceMaxUncheckedDrift => params.coherence_max_unchecked_drift,
            SmallKey::CoherenceMaxMemoryPressure => params.coherence_max_memory_pressure,
            SmallKey::CoherenceRiskInhibitMin => params.coherence_risk_inhibit_min,
        }
    }

    fn apply_add(params: TunableSnapshot, key: SmallKey, delta: f32) -> f32 {
        let (min, max) = Self::key_bounds(key);
        (Self::value_of(params, key) + delta).clamp(min, max)
    }

    fn build_delta(
        &self,
        ctx: &EvolutionContext,
        target: DeltaTarget,
        ops: Vec<DeltaOp>,
    ) -> StructuralDelta {
        let ops = ops.into_iter().take(MAX_DELTA_OPS).collect::<Vec<_>>();
        let mut hasher = Hasher::new();
        hasher.update(b"ucf.evolution.structural_delta.v1");
        hasher.update(&ctx.t.to_le_bytes());
        hasher.update(&(target as u8).to_le_bytes());
        for op in &ops {
            match op {
                DeltaOp::Set { key, value } => {
                    hasher.update(&[0]);
                    hasher.update(&[*key as u8]);
                    hasher.update(&value.to_le_bytes());
                }
                DeltaOp::Add { key, delta } => {
                    hasher.update(&[1]);
                    hasher.update(&[*key as u8]);
                    hasher.update(&delta.to_le_bytes());
                }
                DeltaOp::Clamp { key, min, max } => {
                    hasher.update(&[2]);
                    hasher.update(&[*key as u8]);
                    hasher.update(&min.to_le_bytes());
                    hasher.update(&max.to_le_bytes());
                }
            }
        }
        let digest = *hasher.finalize().as_bytes();
        StructuralDelta {
            schema_version: STRUCTURAL_DELTA_SCHEMA_VERSION,
            delta_id: digest,
            t: ctx.t,
            target,
            ops,
            provenance: DeltaProvenance {
                evidence_chain_digest: ctx.evidence_chain_digest,
                source_window: ctx.source_window,
                engine_id: self.name(),
                seed: self.seed,
                score_digest: [0; 32],
            },
            digest,
        }
    }
}

impl EvolutionEngine for MockEvolutionEngineV0 {
    fn name(&self) -> &'static str {
        "openevolve_mock_v0"
    }

    fn propose(&mut self, ctx: EvolutionContext, budget: EvolutionBudget) -> Vec<StructuralDelta> {
        if budget.work_units == 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        if ctx.budget_exceeded_rate > 0.25 {
            out.push(self.build_delta(
                &ctx,
                DeltaTarget::ComputeBudgetHints,
                vec![
                    DeltaOp::Add {
                        key: SmallKey::BetaPolicyRisk,
                        delta: 0.1,
                    },
                    DeltaOp::Add {
                        key: SmallKey::StructureDeltaCap,
                        delta: -0.05,
                    },
                ],
            ));
        }
        if ctx.coherence_mean < 0.35 || ctx.instability_mean > 0.7 {
            out.push(self.build_delta(
                &ctx,
                DeltaTarget::CoherenceGating,
                vec![
                    DeltaOp::Add {
                        key: SmallKey::CoherenceMinClosedLoopGain,
                        delta: 0.05,
                    },
                    DeltaOp::Add {
                        key: SmallKey::CoherenceRiskInhibitMin,
                        delta: 0.05,
                    },
                ],
            ));
        }
        if ctx.risk_mean > 0.65 && ctx.confidence_mean < 0.45 {
            out.push(self.build_delta(
                &ctx,
                DeltaTarget::FepWeights,
                vec![
                    DeltaOp::Add {
                        key: SmallKey::BetaPolicyRisk,
                        delta: 0.1,
                    },
                    DeltaOp::Add {
                        key: SmallKey::BetaCoherenceLock,
                        delta: 0.05,
                    },
                ],
            ));
        }
        if out.is_empty() {
            let tightened = Self::apply_add(ctx.params, SmallKey::StructureDeltaCap, -0.02);
            out.push(self.build_delta(
                &ctx,
                DeltaTarget::FepWeights,
                vec![DeltaOp::Set {
                    key: SmallKey::StructureDeltaCap,
                    value: tightened,
                }],
            ));
        }
        out.sort_by(|a, b| a.digest.cmp(&b.digest));
        out.truncate(MAX_DELTA_CANDIDATES);
        out
    }

    fn evaluate(&self, delta: &StructuralDelta, ctx: &EvolutionContext) -> DeltaScore {
        let mut reasons = Vec::new();
        let mut metrics_used = Vec::new();
        let mut risk_penalty: f32 = 0.0;
        let mut stability_penalty: f32 = 0.0;
        let mut budget_penalty: f32 = 0.0;

        for op in &delta.ops {
            if let DeltaOp::Add { key, delta } = op {
                metrics_used.push(*key);
                if delta.abs() > 0.2 {
                    reasons.push(ReasonCode::TooAggressiveChange);
                }
                let next = Self::apply_add(ctx.params, *key, *delta);
                let (min, max) = Self::key_bounds(*key);
                if next < min || next > max {
                    reasons.push(ReasonCode::ViolatesClamp);
                }
            }
        }

        if ctx.risk_mean > 0.7 {
            risk_penalty += 0.4;
            reasons.push(ReasonCode::IncreasesRisk);
        }
        if ctx.confidence_mean < 0.4 {
            risk_penalty += 0.2;
            reasons.push(ReasonCode::DegradesConfidence);
        }
        if ctx.instability_mean > 0.7 {
            stability_penalty += 0.2;
            reasons.push(ReasonCode::ImprovesStability);
        }
        if ctx.budget_exceeded_rate > 0.3 {
            budget_penalty += 0.2;
            reasons.push(ReasonCode::ReducesBudgetExceed);
        }
        if ctx.coherence_mean < 0.3 {
            stability_penalty += 0.2;
            reasons.push(ReasonCode::WeakSafetyMargin);
        }

        reasons.sort();
        reasons.dedup();
        metrics_used.sort();
        metrics_used.dedup();
        reasons.truncate(MAX_SCORE_AUDIT_ITEMS);
        metrics_used.truncate(MAX_SCORE_AUDIT_ITEMS);

        let fitness = (1.0 - (risk_penalty + stability_penalty + budget_penalty)).clamp(0.0, 1.0);
        let mut hasher = Hasher::new();
        hasher.update(b"ucf.evolution.score.v1");
        hasher.update(&delta.digest);
        hasher.update(&fitness.to_le_bytes());
        hasher.update(&risk_penalty.to_le_bytes());
        hasher.update(&stability_penalty.to_le_bytes());
        hasher.update(&budget_penalty.to_le_bytes());
        for r in &reasons {
            hasher.update(&[*r as u8]);
        }
        for m in &metrics_used {
            hasher.update(&[*m as u8]);
        }
        let digest = *hasher.finalize().as_bytes();

        DeltaScore {
            fitness,
            risk_penalty,
            stability_penalty,
            budget_penalty,
            audit: ScoreAudit {
                metrics_used,
                reasons,
            },
            digest,
        }
    }

    fn select(&self, candidates: &[(StructuralDelta, DeltaScore)]) -> SelectionResult {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|(da, sa), (db, sb)| {
            sb.fitness
                .total_cmp(&sa.fitness)
                .then_with(|| da.delta_id.cmp(&db.delta_id))
        });
        if let Some((delta, score)) = sorted.into_iter().next() {
            let blocked = score.audit.reasons.iter().any(|r| {
                matches!(
                    r,
                    ReasonCode::ViolatesClamp
                        | ReasonCode::TooAggressiveChange
                        | ReasonCode::IncreasesRisk
                        | ReasonCode::WeakSafetyMargin
                )
            });
            if !blocked && score.fitness >= 0.55 {
                return SelectionResult {
                    accepted: Some((delta, score)),
                    rejected: Vec::new(),
                };
            }
            return SelectionResult {
                accepted: None,
                rejected: vec![(delta.delta_id, score.audit.reasons)],
            };
        }

        SelectionResult {
            accepted: None,
            rejected: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_ctx() -> EvolutionContext {
        EvolutionContext {
            t: 64,
            source_window: (1, 64),
            evidence_chain_digest: [7; 32],
            risk_mean: 0.8,
            confidence_mean: 0.2,
            coherence_mean: 0.25,
            instability_mean: 0.8,
            budget_exceeded_rate: 0.5,
            denied_tool_rate: 1.0,
            stress_index: Some(0.8),
            neuro_arousal: Some(0.9),
            params: TunableSnapshot {
                beta_policy_risk: 1.4,
                beta_coherence_lock: 1.1,
                structure_delta_cap: 0.3,
                coherence_min_closed_loop_gain: 0.35,
                coherence_max_unchecked_drift: 0.8,
                coherence_max_memory_pressure: 0.85,
                coherence_risk_inhibit_min: 0.65,
            },
        }
    }

    #[test]
    fn delta_digest_is_stable() {
        let ctx = sample_ctx();
        let engine = MockEvolutionEngineV0::new(42);
        let delta = engine.build_delta(
            &ctx,
            DeltaTarget::FepWeights,
            vec![DeltaOp::Add {
                key: SmallKey::BetaPolicyRisk,
                delta: 0.1,
            }],
        );
        let delta2 = engine.build_delta(
            &ctx,
            DeltaTarget::FepWeights,
            vec![DeltaOp::Add {
                key: SmallKey::BetaPolicyRisk,
                delta: 0.1,
            }],
        );
        assert_eq!(delta.digest, delta2.digest);
    }

    #[test]
    fn candidate_generation_is_deterministic() {
        let mut engine = MockEvolutionEngineV0::new(9);
        let ctx = sample_ctx();
        let budget = EvolutionBudget { work_units: 16 };
        let first = engine.propose(ctx, budget);
        let second = engine.propose(ctx, budget);
        assert_eq!(first, second);
    }

    #[test]
    fn evaluation_rejects_too_aggressive_delta() {
        let ctx = sample_ctx();
        let engine = MockEvolutionEngineV0::new(1);
        let delta = engine.build_delta(
            &ctx,
            DeltaTarget::FepWeights,
            vec![DeltaOp::Add {
                key: SmallKey::BetaPolicyRisk,
                delta: 0.25,
            }],
        );
        let score = engine.evaluate(&delta, &ctx);
        assert!(score
            .audit
            .reasons
            .contains(&ReasonCode::TooAggressiveChange));
    }

    #[test]
    fn selection_tie_break_is_deterministic() {
        let ctx = sample_ctx();
        let engine = MockEvolutionEngineV0::new(11);
        let a = engine.build_delta(
            &ctx,
            DeltaTarget::FepWeights,
            vec![DeltaOp::Add {
                key: SmallKey::BetaPolicyRisk,
                delta: 0.01,
            }],
        );
        let b = engine.build_delta(
            &ctx,
            DeltaTarget::FepWeights,
            vec![DeltaOp::Add {
                key: SmallKey::BetaCoherenceLock,
                delta: 0.01,
            }],
        );
        let score = DeltaScore {
            fitness: 0.6,
            risk_penalty: 0.0,
            stability_penalty: 0.0,
            budget_penalty: 0.0,
            audit: ScoreAudit {
                metrics_used: Vec::new(),
                reasons: Vec::new(),
            },
            digest: [0; 32],
        };
        let result = engine.select(&[(a.clone(), score.clone()), (b.clone(), score)]);
        let expected = if a.delta_id < b.delta_id {
            a.delta_id
        } else {
            b.delta_id
        };
        assert_eq!(
            result.accepted.as_ref().map(|(d, _)| d.delta_id),
            Some(expected)
        );
    }
}
