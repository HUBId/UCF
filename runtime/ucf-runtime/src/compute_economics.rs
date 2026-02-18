use std::collections::BTreeMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeTokens(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct TokenBalance {
    pub available: u64,
    pub spent: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetPool {
    Primary,
    Shadow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ComputeStage {
    Governor,
    Llm,
    Jepa,
    Sae,
    Ssm,
    Lfm,
    Tool,
}

impl ComputeStage {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Governor => "governor",
            Self::Llm => "llm",
            Self::Jepa => "jepa",
            Self::Sae => "sae",
            Self::Ssm => "ssm",
            Self::Lfm => "lfm",
            Self::Tool => "tool",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CostSchedule {
    pub llm_base: u64,
    pub llm_per_in_token: u64,
    pub llm_per_out_token: u64,
    pub jepa_base: u64,
    pub jepa_per_dim: u64,
    pub sae_base: u64,
    pub sae_per_feature: u64,
    pub ssm_base: u64,
    pub ssm_per_dim: u64,
    pub lfm_base: u64,
    pub lfm_per_dim: u64,
    pub governor_base: u64,
    pub tool_base: u64,
}

impl Default for CostSchedule {
    fn default() -> Self {
        Self {
            llm_base: 40,
            llm_per_in_token: 2,
            llm_per_out_token: 5,
            jepa_base: 50,
            jepa_per_dim: 1,
            sae_base: 40,
            sae_per_feature: 1,
            ssm_base: 45,
            ssm_per_dim: 1,
            lfm_base: 45,
            lfm_per_dim: 1,
            governor_base: 12,
            tool_base: 20,
        }
    }
}

impl CostSchedule {
    pub fn llm_cost(&self, in_tokens: u32, out_tokens: u32) -> ComputeTokens {
        ComputeTokens(
            self.llm_base
                .saturating_add(self.llm_per_in_token.saturating_mul(u64::from(in_tokens)))
                .saturating_add(self.llm_per_out_token.saturating_mul(u64::from(out_tokens))),
        )
    }

    pub fn stage_cost(&self, stage: ComputeStage, dim: u32) -> ComputeTokens {
        let dim = u64::from(dim);
        let cost = match stage {
            ComputeStage::Governor => self.governor_base,
            ComputeStage::Jepa => self
                .jepa_base
                .saturating_add(self.jepa_per_dim.saturating_mul(dim)),
            ComputeStage::Sae => self
                .sae_base
                .saturating_add(self.sae_per_feature.saturating_mul(dim)),
            ComputeStage::Ssm => self
                .ssm_base
                .saturating_add(self.ssm_per_dim.saturating_mul(dim)),
            ComputeStage::Lfm => self
                .lfm_base
                .saturating_add(self.lfm_per_dim.saturating_mul(dim)),
            ComputeStage::Tool => self.tool_base,
            ComputeStage::Llm => self.llm_base,
        };
        ComputeTokens(cost)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeEconomicsProfile {
    pub primary_window_budget: u64,
    pub shadow_window_budget: u64,
    pub llm_window_budget: u64,
    pub others_window_budget: u64,
    pub primary_per_tick_budget: u64,
    pub shadow_per_tick_budget: u64,
    pub llm_per_tick_budget: u64,
}

impl ComputeEconomicsProfile {
    pub fn from_env() -> Self {
        let mut out = Self::default();
        if let Ok(v) = std::env::var("UCF_PRIMARY_COMPUTE_BUDGET") {
            if let Ok(n) = v.parse::<u64>() {
                out.primary_window_budget = n;
            }
        }
        if let Ok(v) = std::env::var("UCF_SHADOW_COMPUTE_BUDGET") {
            if let Ok(n) = v.parse::<u64>() {
                out.shadow_window_budget = n;
            }
        }
        if let Ok(v) = std::env::var("UCF_LLM_COMPUTE_BUDGET") {
            if let Ok(n) = v.parse::<u64>() {
                out.llm_window_budget = n;
            }
        }
        out
    }
}

impl Default for ComputeEconomicsProfile {
    fn default() -> Self {
        Self {
            primary_window_budget: 24_000,
            shadow_window_budget: 4_000,
            llm_window_budget: 10_000,
            others_window_budget: 14_000,
            primary_per_tick_budget: 1_200,
            shadow_per_tick_budget: 200,
            llm_per_tick_budget: 650,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeBudgetWindowRecord {
    pub t0: u64,
    pub t1: u64,
    pub window: u64,
    pub primary_start: TokenBalance,
    pub primary_end: TokenBalance,
    pub shadow_start: TokenBalance,
    pub shadow_end: TokenBalance,
    pub spent_per_stage: BTreeMap<ComputeStage, u64>,
    pub governor_tier_mean_q: u16,
    pub governor_tier_max: u8,
    pub policy_hash_prefix: [u8; 8],
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ComputeBudgetViolationRecord {
    pub t: u64,
    pub stage: ComputeStage,
    pub pool: BudgetPool,
    pub reason: &'static str,
    pub attempted_cost: u64,
    pub available: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LlmReservation {
    pub estimated: u64,
    pub pool: BudgetPool,
}

#[derive(Debug, Clone)]
pub struct ComputeEconomy {
    pub profile: ComputeEconomicsProfile,
    pub schedule: CostSchedule,
    primary_window: TokenBalance,
    shadow_window: TokenBalance,
    llm_window_remaining: u64,
    others_window_remaining: u64,
    primary_tick_remaining: u64,
    shadow_tick_remaining: u64,
    llm_tick_remaining: u64,
    spent_per_stage: BTreeMap<ComputeStage, u64>,
}

impl ComputeEconomy {
    pub fn new(profile: ComputeEconomicsProfile, schedule: CostSchedule) -> Self {
        Self {
            profile,
            schedule,
            primary_window: TokenBalance {
                available: profile.primary_window_budget,
                spent: 0,
            },
            shadow_window: TokenBalance {
                available: profile.shadow_window_budget,
                spent: 0,
            },
            llm_window_remaining: profile.llm_window_budget,
            others_window_remaining: profile.others_window_budget,
            primary_tick_remaining: profile.primary_per_tick_budget,
            shadow_tick_remaining: profile.shadow_per_tick_budget,
            llm_tick_remaining: profile.llm_per_tick_budget,
            spent_per_stage: BTreeMap::new(),
        }
    }

    pub fn begin_tick(&mut self, tier: u8, emergency: bool) {
        let scale = if emergency {
            10
        } else {
            match tier.min(3) {
                0 => 100,
                1 => 70,
                2 => 45,
                _ => 20,
            }
        };
        self.primary_tick_remaining = self
            .profile
            .primary_per_tick_budget
            .saturating_mul(scale)
            .saturating_div(100);
        self.shadow_tick_remaining = self
            .profile
            .shadow_per_tick_budget
            .saturating_mul(scale)
            .saturating_div(100);
        self.llm_tick_remaining = self
            .profile
            .llm_per_tick_budget
            .saturating_mul(scale)
            .saturating_div(100);
    }

    pub fn reset_window(&mut self) {
        self.primary_window = TokenBalance {
            available: self.profile.primary_window_budget,
            spent: 0,
        };
        self.shadow_window = TokenBalance {
            available: self.profile.shadow_window_budget,
            spent: 0,
        };
        self.llm_window_remaining = self.profile.llm_window_budget;
        self.others_window_remaining = self.profile.others_window_budget;
        self.spent_per_stage.clear();
    }

    pub fn try_charge(
        &mut self,
        pool: BudgetPool,
        stage: ComputeStage,
        tokens: ComputeTokens,
        t: u64,
    ) -> Result<(), ComputeBudgetViolationRecord> {
        self.try_charge_raw(pool, stage, tokens.0, t)
    }

    fn try_charge_raw(
        &mut self,
        pool: BudgetPool,
        stage: ComputeStage,
        tokens: u64,
        t: u64,
    ) -> Result<(), ComputeBudgetViolationRecord> {
        let (window, tick) = match pool {
            BudgetPool::Primary => (&mut self.primary_window, &mut self.primary_tick_remaining),
            BudgetPool::Shadow => (&mut self.shadow_window, &mut self.shadow_tick_remaining),
        };
        let mut available = window.available.min(*tick);
        if stage == ComputeStage::Llm {
            available = available
                .min(self.llm_window_remaining)
                .min(self.llm_tick_remaining);
        } else {
            available = available.min(self.others_window_remaining);
        }
        if tokens > available {
            return Err(ComputeBudgetViolationRecord {
                t,
                stage,
                pool,
                reason: "insufficient_budget",
                attempted_cost: tokens,
                available,
            });
        }
        window.available = window.available.saturating_sub(tokens);
        window.spent = window.spent.saturating_add(tokens);
        *tick = tick.saturating_sub(tokens);
        if stage == ComputeStage::Llm {
            self.llm_window_remaining = self.llm_window_remaining.saturating_sub(tokens);
            self.llm_tick_remaining = self.llm_tick_remaining.saturating_sub(tokens);
        } else {
            self.others_window_remaining = self.others_window_remaining.saturating_sub(tokens);
        }
        *self.spent_per_stage.entry(stage).or_insert(0) = self
            .spent_per_stage
            .get(&stage)
            .copied()
            .unwrap_or(0)
            .saturating_add(tokens);
        metrics::counter!("ucf_compute_tokens_spent_total", "stage" => stage.as_str().to_string())
            .increment(tokens);
        Ok(())
    }

    pub fn reserve_llm(
        &mut self,
        pool: BudgetPool,
        in_tokens: u32,
        max_out_tokens: u32,
        t: u64,
    ) -> Result<LlmReservation, ComputeBudgetViolationRecord> {
        let est = self.schedule.llm_cost(in_tokens, max_out_tokens).0;
        self.try_charge_raw(pool, ComputeStage::Llm, est, t)?;
        Ok(LlmReservation {
            estimated: est,
            pool,
        })
    }

    pub fn settle_llm(
        &mut self,
        reservation: LlmReservation,
        in_tokens: u32,
        out_tokens: u32,
        t: u64,
    ) -> Option<ComputeBudgetViolationRecord> {
        let actual = self.schedule.llm_cost(in_tokens, out_tokens).0;
        if actual <= reservation.estimated {
            let refund = reservation.estimated - actual;
            self.refund(reservation.pool, ComputeStage::Llm, refund);
            None
        } else {
            Some(ComputeBudgetViolationRecord {
                t,
                stage: ComputeStage::Llm,
                pool: reservation.pool,
                reason: "actual_gt_estimate_clamped",
                attempted_cost: actual,
                available: reservation.estimated,
            })
        }
    }

    fn refund(&mut self, pool: BudgetPool, stage: ComputeStage, refund: u64) {
        if refund == 0 {
            return;
        }
        let window = match pool {
            BudgetPool::Primary => &mut self.primary_window,
            BudgetPool::Shadow => &mut self.shadow_window,
        };
        window.available = window.available.saturating_add(refund);
        window.spent = window.spent.saturating_sub(refund);
        if stage == ComputeStage::Llm {
            self.llm_window_remaining = self.llm_window_remaining.saturating_add(refund);
            self.llm_tick_remaining = self.llm_tick_remaining.saturating_add(refund);
        } else {
            self.others_window_remaining = self.others_window_remaining.saturating_add(refund);
        }
        self.spent_per_stage
            .entry(stage)
            .and_modify(|s| *s = s.saturating_sub(refund));
    }

    pub fn remaining(&self, pool: BudgetPool) -> u64 {
        match pool {
            BudgetPool::Primary => self.primary_window.available,
            BudgetPool::Shadow => self.shadow_window.available,
        }
    }

    pub fn snapshot_window(
        &self,
        t0: u64,
        t1: u64,
        window: u64,
        tier_mean_q: u16,
        tier_max: u8,
        policy_hash_prefix: [u8; 8],
    ) -> ComputeBudgetWindowRecord {
        ComputeBudgetWindowRecord {
            t0,
            t1,
            window,
            primary_start: TokenBalance {
                available: self.profile.primary_window_budget,
                spent: 0,
            },
            primary_end: self.primary_window,
            shadow_start: TokenBalance {
                available: self.profile.shadow_window_budget,
                spent: 0,
            },
            shadow_end: self.shadow_window,
            spent_per_stage: self.spent_per_stage.clone(),
            governor_tier_mean_q: tier_mean_q,
            governor_tier_max: tier_max,
            policy_hash_prefix,
        }
    }
}

pub fn estimate_input_tokens(prompt: &str) -> u32 {
    prompt.split_whitespace().count().min(u32::MAX as usize) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llm_cost_is_deterministic() {
        let s = CostSchedule::default();
        assert_eq!(s.llm_cost(10, 20), s.llm_cost(10, 20));
    }

    #[test]
    fn charging_saturates_and_denies() {
        let profile = ComputeEconomicsProfile {
            primary_window_budget: 10,
            primary_per_tick_budget: 10,
            llm_window_budget: 10,
            llm_per_tick_budget: 10,
            others_window_budget: 10,
            shadow_window_budget: 1,
            shadow_per_tick_budget: 1,
        };
        let mut e = ComputeEconomy::new(profile, CostSchedule::default());
        e.begin_tick(0, false);
        assert!(e
            .try_charge(
                BudgetPool::Primary,
                ComputeStage::Governor,
                ComputeTokens(8),
                1
            )
            .is_ok());
        assert!(e
            .try_charge(
                BudgetPool::Primary,
                ComputeStage::Governor,
                ComputeTokens(8),
                1
            )
            .is_err());
    }

    #[test]
    fn shadow_pool_is_separate() {
        let profile = ComputeEconomicsProfile {
            primary_window_budget: 100,
            shadow_window_budget: 5,
            llm_window_budget: 100,
            others_window_budget: 100,
            primary_per_tick_budget: 100,
            shadow_per_tick_budget: 5,
            llm_per_tick_budget: 100,
        };
        let mut e = ComputeEconomy::new(profile, CostSchedule::default());
        e.begin_tick(0, false);
        assert!(e
            .try_charge(BudgetPool::Shadow, ComputeStage::Sae, ComputeTokens(4), 1)
            .is_ok());
        assert_eq!(e.remaining(BudgetPool::Primary), 100);
        assert_eq!(e.remaining(BudgetPool::Shadow), 1);
    }
}
