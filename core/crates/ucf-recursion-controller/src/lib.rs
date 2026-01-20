#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

const BUDGET_DOMAIN: &[u8] = b"ucf.rdc.budget.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecursionInputs {
    pub phi: u16,
    pub drift_score: u16,
    pub surprise: u16,
    pub risk: u16,
    pub attn_gain: u16,
    pub focus: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecursionBudget {
    pub max_depth: u8,
    pub per_cycle_steps: u16,
    pub level_decay: u16,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecursionController {
    base_max_depth: u8,
    base_steps: u16,
    base_decay: u16,
}

impl Default for RecursionController {
    fn default() -> Self {
        Self {
            base_max_depth: 3,
            base_steps: 36,
            base_decay: 8600,
        }
    }
}

impl RecursionController {
    pub fn compute(&self, inp: &RecursionInputs) -> RecursionBudget {
        let mut max_depth = self.base_max_depth;
        let mut per_cycle_steps = self.base_steps;
        let mut level_decay = self.base_decay;

        if inp.drift_score >= 7000 {
            max_depth = max_depth.saturating_sub(1);
            per_cycle_steps = per_cycle_steps.saturating_sub(6);
            level_decay = level_decay.saturating_sub(400);
        }
        if inp.drift_score >= 8500 {
            max_depth = max_depth.saturating_sub(1);
            per_cycle_steps = per_cycle_steps.saturating_sub(6);
            level_decay = level_decay.saturating_sub(400);
        }
        if inp.risk >= 7000 {
            max_depth = max_depth.saturating_sub(1);
            per_cycle_steps = per_cycle_steps.saturating_sub(6);
            level_decay = level_decay.saturating_sub(300);
        }
        if inp.risk >= 9000 {
            max_depth = max_depth.saturating_sub(1);
            per_cycle_steps = per_cycle_steps.saturating_sub(6);
            level_decay = level_decay.saturating_sub(300);
        }

        if inp.phi >= 7000 && inp.risk <= 3000 {
            max_depth = max_depth.saturating_add(2);
            per_cycle_steps = per_cycle_steps.saturating_add(8);
            level_decay = level_decay.saturating_add(150);
        }
        if inp.phi >= 8500 && inp.risk <= 2000 {
            max_depth = max_depth.saturating_add(1);
            per_cycle_steps = per_cycle_steps.saturating_add(6);
            level_decay = level_decay.saturating_add(100);
        }

        if inp.surprise >= 8000 {
            max_depth = max_depth.saturating_sub(1);
            per_cycle_steps = per_cycle_steps.saturating_add(10);
            level_decay = level_decay.saturating_sub(200);
        } else if inp.surprise >= 5000 {
            max_depth = max_depth.saturating_sub(1);
            per_cycle_steps = per_cycle_steps.saturating_sub(2);
            level_decay = level_decay.saturating_sub(100);
        }

        if inp.focus >= 8000 {
            level_decay = level_decay.saturating_sub(200);
        }
        level_decay = level_decay.saturating_sub(inp.attn_gain / 25);

        max_depth = max_depth.min(8);
        per_cycle_steps = per_cycle_steps.clamp(4, 200);
        level_decay = level_decay.clamp(5000, 10000);

        let commit = commit_budget(inp, max_depth, per_cycle_steps, level_decay);
        RecursionBudget {
            max_depth,
            per_cycle_steps,
            level_decay,
            commit,
        }
    }
}

fn commit_budget(
    inp: &RecursionInputs,
    max_depth: u8,
    per_cycle_steps: u16,
    level_decay: u16,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(BUDGET_DOMAIN);
    hasher.update(&inp.phi.to_be_bytes());
    hasher.update(&inp.drift_score.to_be_bytes());
    hasher.update(&inp.surprise.to_be_bytes());
    hasher.update(&inp.risk.to_be_bytes());
    hasher.update(&inp.attn_gain.to_be_bytes());
    hasher.update(&inp.focus.to_be_bytes());
    hasher.update(&[max_depth]);
    hasher.update(&per_cycle_steps.to_be_bytes());
    hasher.update(&level_decay.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base_inputs() -> RecursionInputs {
        RecursionInputs {
            phi: 4000,
            drift_score: 2000,
            surprise: 1000,
            risk: 1500,
            attn_gain: 1200,
            focus: 2000,
        }
    }

    #[test]
    fn determinism_holds_for_identical_inputs() {
        let controller = RecursionController::default();
        let inputs = base_inputs();
        let first = controller.compute(&inputs);
        let second = controller.compute(&inputs);

        assert_eq!(first, second);
        assert_eq!(first.commit, second.commit);
    }

    #[test]
    fn high_drift_reduces_depth() {
        let controller = RecursionController::default();
        let mut inputs = base_inputs();
        let base = controller.compute(&inputs);
        inputs.drift_score = 9000;
        let adjusted = controller.compute(&inputs);

        assert!(adjusted.max_depth < base.max_depth);
        assert!(adjusted.per_cycle_steps <= base.per_cycle_steps);
    }

    #[test]
    fn high_phi_low_risk_increases_depth() {
        let controller = RecursionController::default();
        let base = controller.compute(&base_inputs());
        let inputs = RecursionInputs {
            phi: 9000,
            risk: 1200,
            ..base_inputs()
        };
        let adjusted = controller.compute(&inputs);

        assert!(adjusted.max_depth > base.max_depth);
        assert!(adjusted.per_cycle_steps > base.per_cycle_steps);
    }
}
