use std::collections::HashMap;

use crate::v0::CausalGraph;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifyVerdict {
    Verified,
    Rejected,
    Unknown,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RuleCfg {
    pub min_confidence: f32,
    pub max_out_degree: u32,
}

impl Default for RuleCfg {
    fn default() -> Self {
        Self {
            min_confidence: 0.6,
            max_out_degree: 8,
        }
    }
}

pub fn verify_graph(g: &CausalGraph, cfg: RuleCfg) -> (VerifyVerdict, f32) {
    if !g.is_acyclic() {
        return (VerifyVerdict::Rejected, 0.0);
    }

    let mut out_degree: HashMap<u32, u32> = HashMap::new();
    let mut verified = 0_u32;

    for h in &g.hyps {
        if h.confidence >= cfg.min_confidence {
            verified = verified.saturating_add(1);
            *out_degree.entry(h.edge.from).or_insert(0) += 1;
        }
    }

    if out_degree.values().any(|&d| d > cfg.max_out_degree) {
        return (VerifyVerdict::Unknown, 0.3);
    }

    let total = g.hyps.len().max(1) as f32;
    let verified_ratio = verified as f32 / total;

    if verified_ratio >= 0.7 {
        (VerifyVerdict::Verified, verified_ratio)
    } else {
        (VerifyVerdict::Unknown, verified_ratio)
    }
}
