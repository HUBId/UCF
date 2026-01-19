#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldStateVec {
    pub dims: u16,
    pub data: Vec<i16>,
}

impl WorldStateVec {
    pub fn new(dims: u16, data: Vec<i16>) -> Self {
        Self { dims, data }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prediction {
    pub state: WorldStateVec,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Observation {
    pub state: WorldStateVec,
    pub commit: Digest32,
}

impl Observation {
    pub fn new(state: WorldStateVec) -> Self {
        let commit = commit_state(b"ucf.predictive.observation.v1", &state);
        Self { state, commit }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PredictionError {
    pub l1: u32,
    pub l2: u32,
    pub commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurpriseSignal {
    pub score: u16,
    pub band: SurpriseBand,
    pub commit: Digest32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurpriseBand {
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SurpriseUpdated {
    pub score: u16,
    pub band: SurpriseBand,
    pub commit: Digest32,
}

impl From<&SurpriseSignal> for SurpriseUpdated {
    fn from(signal: &SurpriseSignal) -> Self {
        Self {
            score: signal.score,
            band: signal.band,
            commit: signal.commit,
        }
    }
}

pub const SURPRISE_SCORE_MAX: u16 = 10_000;
pub const SURPRISE_LOW_MAX: u16 = 2_500;
pub const SURPRISE_MEDIUM_MAX: u16 = 5_000;
pub const SURPRISE_HIGH_MAX: u16 = 8_000;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WorldModel {
    coefficients: [i16; 4],
    bias: i16,
}

impl Default for WorldModel {
    fn default() -> Self {
        Self {
            coefficients: [3, -2, 5, 1],
            bias: 7,
        }
    }
}

impl WorldModel {
    pub fn predict(&self, current: &WorldStateVec) -> Prediction {
        let mut data = Vec::with_capacity(current.data.len());
        for (idx, value) in current.data.iter().enumerate() {
            let coeff = self.coefficients[idx % self.coefficients.len()];
            let raw = i32::from(*value)
                .saturating_mul(i32::from(coeff))
                .saturating_add(i32::from(self.bias))
                .saturating_add(i32::from(idx as i16));
            data.push(clamp_i16(raw));
        }
        let state = WorldStateVec {
            dims: current.dims,
            data,
        };
        let commit = commit_state(b"ucf.predictive.prediction.v1", &state);
        Prediction { state, commit }
    }
}

pub fn error(pred: &Prediction, obs: &Observation) -> PredictionError {
    let pred_len = pred.state.data.len();
    let obs_len = obs.state.data.len();
    let len = pred_len.min(obs_len);
    let mut l1 = 0u32;
    let mut l2 = 0u32;
    for idx in 0..len {
        let delta = i32::from(pred.state.data[idx]) - i32::from(obs.state.data[idx]);
        let abs = delta.unsigned_abs();
        l1 = l1.saturating_add(abs);
        let sq = abs.saturating_mul(abs);
        l2 = l2.saturating_add(sq);
    }
    let mismatch = pred_len.abs_diff(obs_len) as u32;
    if mismatch > 0 {
        let penalty = mismatch.saturating_mul(500);
        l1 = l1.saturating_add(penalty);
        l2 = l2.saturating_add(penalty);
    }
    let commit = commit_error(l1, l2);
    PredictionError { l1, l2, commit }
}

pub fn surprise(err: &PredictionError) -> SurpriseSignal {
    let combined = err.l1.saturating_add(err.l2 / 16);
    let scaled = combined / 4;
    let score =
        u16::try_from(scaled.min(u32::from(SURPRISE_SCORE_MAX))).unwrap_or(SURPRISE_SCORE_MAX);
    let band = band_for_score(score);
    let commit = commit_surprise(score, band);
    SurpriseSignal {
        score,
        band,
        commit,
    }
}

pub fn band_for_score(score: u16) -> SurpriseBand {
    if score <= SURPRISE_LOW_MAX {
        SurpriseBand::Low
    } else if score <= SURPRISE_MEDIUM_MAX {
        SurpriseBand::Medium
    } else if score <= SURPRISE_HIGH_MAX {
        SurpriseBand::High
    } else {
        SurpriseBand::Critical
    }
}

fn clamp_i16(value: i32) -> i16 {
    value.clamp(i32::from(i16::MIN), i32::from(i16::MAX)) as i16
}

fn commit_state(domain: &[u8], state: &WorldStateVec) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    hasher.update(&state.dims.to_be_bytes());
    for value in &state.data {
        hasher.update(&value.to_be_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_error(l1: u32, l2: u32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.predictive.error.v1");
    hasher.update(&l1.to_be_bytes());
    hasher.update(&l2.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_surprise(score: u16, band: SurpriseBand) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.predictive.surprise.v1");
    hasher.update(&score.to_be_bytes());
    hasher.update(&[band as u8]);
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prediction_is_deterministic() {
        let model = WorldModel::default();
        let state = WorldStateVec::new(4, vec![10, -3, 7, 2]);

        let first = model.predict(&state);
        let second = model.predict(&state);

        assert_eq!(first, second);
        assert_eq!(first.commit, second.commit);
    }

    #[test]
    fn surprise_band_thresholds_are_correct() {
        assert_eq!(band_for_score(0), SurpriseBand::Low);
        assert_eq!(band_for_score(SURPRISE_LOW_MAX), SurpriseBand::Low);
        assert_eq!(band_for_score(SURPRISE_LOW_MAX + 1), SurpriseBand::Medium);
        assert_eq!(band_for_score(SURPRISE_MEDIUM_MAX), SurpriseBand::Medium);
        assert_eq!(band_for_score(SURPRISE_MEDIUM_MAX + 1), SurpriseBand::High);
        assert_eq!(band_for_score(SURPRISE_HIGH_MAX), SurpriseBand::High);
        assert_eq!(
            band_for_score(SURPRISE_HIGH_MAX + 1),
            SurpriseBand::Critical
        );
        assert_eq!(band_for_score(SURPRISE_SCORE_MAX), SurpriseBand::Critical);
    }
}
