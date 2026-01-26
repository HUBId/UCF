#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_geist::SelfState;
use ucf_ism::IsmAnchor;
use ucf_predictive_coding::SurpriseBand;
use ucf_types::Digest32;

const REPORT_DOMAIN: &[u8] = b"ucf.consistency.report.v1";
const ACTION_DOMAIN: &[u8] = b"ucf.consistency.action.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DriftBand {
    Low,
    Medium,
    High,
    Critical,
}

impl DriftBand {
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Low => 1,
            Self::Medium => 2,
            Self::High => 3,
            Self::Critical => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsistencyReport {
    pub drift_score: u16,
    pub band: DriftBand,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConsistencyActionKind {
    DampNoise,
    ReduceRecursion,
    IncreaseReplay,
    ThrottleOutput,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsistencyAction {
    pub kind: ConsistencyActionKind,
    pub intensity: u16,
    pub commit: Digest32,
}

#[derive(Clone, Debug)]
pub struct ConsistencyInputs<'a> {
    pub self_state: &'a SelfState,
    pub self_symbol: Digest32,
    pub ism_root: Digest32,
    pub anchors: &'a [IsmAnchor],
    pub suppression_count: u16,
    pub policy_class: u16,
    pub policy_stable: bool,
    pub risk_score: u16,
    pub surprise_band: SurpriseBand,
    pub phi: u16,
}

#[derive(Clone, Debug, Default)]
pub struct ConsistencyEngine;

impl ConsistencyEngine {
    pub fn evaluate(
        &self,
        inputs: &ConsistencyInputs<'_>,
    ) -> (ConsistencyReport, Vec<ConsistencyAction>) {
        let mut drift = bytewise_diff(inputs.self_symbol, inputs.ism_root);
        if inputs.suppression_count >= 2 {
            drift = drift.saturating_add(1200);
        }
        drift = drift.saturating_add(surprise_drift(inputs.surprise_band));
        if inputs.policy_stable && inputs.phi >= 7000 {
            drift = drift.saturating_sub(1000);
        }
        drift = drift.min(10_000);

        let band = band_for_drift(drift);
        let report_commit = commit_report(inputs, drift, band);
        let report = ConsistencyReport {
            drift_score: drift,
            band,
            commit: report_commit,
        };
        let actions = select_actions(&report);
        (report, actions)
    }
}

fn bytewise_diff(a: Digest32, b: Digest32) -> u16 {
    let mut total: u32 = 0;
    for (left, right) in a.as_bytes().iter().zip(b.as_bytes().iter()) {
        total = total.saturating_add(u32::from(left.abs_diff(*right)));
    }
    total.min(u16::MAX as u32) as u16
}

fn surprise_drift(band: SurpriseBand) -> u16 {
    match band {
        SurpriseBand::Low => 0,
        SurpriseBand::Medium => 600,
        SurpriseBand::High => 1500,
        SurpriseBand::Critical => 2500,
    }
}

fn band_for_drift(score: u16) -> DriftBand {
    match score {
        0..=2499 => DriftBand::Low,
        2500..=4999 => DriftBand::Medium,
        5000..=7499 => DriftBand::High,
        _ => DriftBand::Critical,
    }
}

fn commit_report(inputs: &ConsistencyInputs<'_>, drift: u16, band: DriftBand) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(REPORT_DOMAIN);
    hasher.update(inputs.self_state.commit.as_bytes());
    hasher.update(inputs.self_symbol.as_bytes());
    hasher.update(inputs.ism_root.as_bytes());
    for anchor in inputs.anchors {
        hasher.update(anchor.commit.as_bytes());
    }
    hasher.update(&inputs.suppression_count.to_be_bytes());
    hasher.update(&inputs.policy_class.to_be_bytes());
    hasher.update(&[inputs.policy_stable as u8]);
    hasher.update(&inputs.risk_score.to_be_bytes());
    hasher.update(&inputs.phi.to_be_bytes());
    hasher.update(&[inputs.surprise_band as u8]);
    hasher.update(&drift.to_be_bytes());
    hasher.update(&[band.as_u8()]);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn select_actions(report: &ConsistencyReport) -> Vec<ConsistencyAction> {
    let intensity = report.drift_score.min(10_000);
    match report.band {
        DriftBand::Low => Vec::new(),
        DriftBand::Medium => vec![action(
            ConsistencyActionKind::DampNoise,
            intensity,
            report.commit,
        )],
        DriftBand::High | DriftBand::Critical => vec![
            action(
                ConsistencyActionKind::ReduceRecursion,
                intensity,
                report.commit,
            ),
            action(
                ConsistencyActionKind::IncreaseReplay,
                intensity,
                report.commit,
            ),
            action(
                ConsistencyActionKind::ThrottleOutput,
                intensity,
                report.commit,
            ),
        ],
    }
}

fn action(
    kind: ConsistencyActionKind,
    intensity: u16,
    report_commit: Digest32,
) -> ConsistencyAction {
    let mut hasher = Hasher::new();
    hasher.update(ACTION_DOMAIN);
    hasher.update(&[kind as u8]);
    hasher.update(&intensity.to_be_bytes());
    hasher.update(report_commit.as_bytes());
    let commit = Digest32::new(*hasher.finalize().as_bytes());
    ConsistencyAction {
        kind,
        intensity,
        commit,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drift_increases_when_self_symbol_diverges() {
        let anchor = IsmAnchor::new(Digest32::new([1u8; 32]), Digest32::new([2u8; 32]), 1, 1);
        let self_state = SelfState {
            cycle_id: 1,
            ssm_commit: Digest32::new([1u8; 32]),
            workspace_commit: Digest32::new([2u8; 32]),
            risk_commit: Digest32::new([3u8; 32]),
            attn_commit: Digest32::new([4u8; 32]),
            ncde_commit: Digest32::new([5u8; 32]),
            consistency: 0,
            commit: Digest32::new([9u8; 32]),
        };
        let base_inputs = ConsistencyInputs {
            self_state: &self_state,
            self_symbol: Digest32::new([0u8; 32]),
            ism_root: Digest32::new([0u8; 32]),
            anchors: &[anchor],
            suppression_count: 0,
            policy_class: 1,
            policy_stable: false,
            risk_score: 0,
            surprise_band: SurpriseBand::Low,
            phi: 2000,
        };
        let engine = ConsistencyEngine;
        let (base_report, _) = engine.evaluate(&base_inputs);
        let mut shifted = base_inputs.clone();
        shifted.self_symbol = Digest32::new([10u8; 32]);
        let (shifted_report, _) = engine.evaluate(&shifted);

        assert!(shifted_report.drift_score > base_report.drift_score);
    }
}
