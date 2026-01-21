#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_attn_controller::{AttentionWeights, FocusChannel};
use ucf_types::Digest32;

const ACT_VIEW_DOMAIN: &[u8] = b"ucf.feature_translator.activation_view.v1";
const FEATURE_DOMAIN: &[u8] = b"ucf.feature_translator.feature.v1";
const FEATURE_SET_DOMAIN: &[u8] = b"ucf.feature_translator.feature_set.v1";
const LENS_SELECTION_DOMAIN: &[u8] = b"ucf.feature_translator.lens_selection.v1";
const MOCK_SAE_WEIGHT_DOMAIN: &[u8] = b"ucf.feature_translator.mock_sae.weight.v1";
const MOCK_LENS_DOMAIN: &[u8] = b"ucf.feature_translator.mock_lens.seed.v1";
const MAX_WEIGHT_ABS: i16 = 10_000;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivationView {
    pub layer_id: u16,
    pub act_digest: Digest32,
    pub energy: u16,
    pub commit: Digest32,
}

impl ActivationView {
    pub fn new(layer_id: u16, act_digest: Digest32, energy: u16) -> Self {
        let commit = commit_activation_view(layer_id, act_digest, energy);
        Self {
            layer_id,
            act_digest,
            energy,
            commit,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseFeature {
    pub id: u32,
    pub weight: i16,
    pub commit: Digest32,
}

impl SparseFeature {
    pub fn new(id: u32, weight: i16, seed: Digest32) -> Self {
        let commit = commit_feature(seed, id, weight);
        Self { id, weight, commit }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SparseFeatureSet {
    pub features: Vec<SparseFeature>,
    pub commit: Digest32,
}

impl SparseFeatureSet {
    pub fn new(mut features: Vec<SparseFeature>, seed: Digest32) -> Self {
        features.sort_by_key(|feature| feature.id);
        let commit = commit_feature_set(seed, &features);
        Self { features, commit }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LensSelection {
    pub topk: Vec<SparseFeature>,
    pub commit: Digest32,
}

impl LensSelection {
    pub fn new(topk: Vec<SparseFeature>, seed: Digest32) -> Self {
        let commit = commit_lens_selection(seed, &topk);
        Self { topk, commit }
    }
}

pub trait SaePort {
    fn encode(&self, acts: &ActivationView) -> SparseFeatureSet;
}

pub trait LensPort {
    fn select(&self, set: &SparseFeatureSet, attn: &AttentionWeights) -> LensSelection;
}

#[derive(Clone, Debug)]
pub struct MockSaePort {
    feature_count: usize,
}

impl Default for MockSaePort {
    fn default() -> Self {
        Self { feature_count: 6 }
    }
}

impl MockSaePort {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_feature_count(feature_count: usize) -> Self {
        Self { feature_count }
    }

    fn weight_from_activation(acts: &ActivationView, id: u32) -> i16 {
        let mut hasher = Hasher::new();
        hasher.update(MOCK_SAE_WEIGHT_DOMAIN);
        hasher.update(acts.act_digest.as_bytes());
        hasher.update(&id.to_be_bytes());
        let digest = hasher.finalize();
        let bytes = digest.as_bytes();
        let base = i16::from_be_bytes([bytes[0], bytes[1]]);
        let bias = (acts.energy as i32) / 2;
        let signed_bias = if base >= 0 { bias } else { -bias };
        clamp_weight(base as i32 + signed_bias)
    }
}

impl SaePort for MockSaePort {
    fn encode(&self, acts: &ActivationView) -> SparseFeatureSet {
        let bytes = acts.act_digest.as_bytes();
        let mut features = Vec::new();
        let count = self.feature_count.min(bytes.len() / 4).max(1);
        for i in 0..count {
            let idx = i * 4;
            let id =
                u32::from_be_bytes([bytes[idx], bytes[idx + 1], bytes[idx + 2], bytes[idx + 3]]);
            let weight = Self::weight_from_activation(acts, id);
            features.push(SparseFeature::new(id, weight, acts.commit));
        }
        SparseFeatureSet::new(features, acts.commit)
    }
}

#[derive(Clone, Debug)]
pub struct MockLensPort {
    top_k: usize,
}

impl Default for MockLensPort {
    fn default() -> Self {
        Self { top_k: 4 }
    }
}

impl MockLensPort {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_top_k(top_k: usize) -> Self {
        Self { top_k }
    }

    fn adjust_weight(weight: i16, attn: &AttentionWeights) -> i16 {
        let scaled = (weight as i32).saturating_mul(attn.gain as i32) / 1000;
        let bias = match attn.channel {
            FocusChannel::Threat => 260,
            FocusChannel::Task => 200,
            FocusChannel::Social => 140,
            FocusChannel::Memory => 180,
            FocusChannel::Exploration => 160,
            FocusChannel::Idle => 0,
        };
        let adjusted = if scaled >= 0 {
            scaled.saturating_add(bias)
        } else {
            scaled.saturating_sub(bias)
        };
        clamp_weight(adjusted)
    }
}

impl LensPort for MockLensPort {
    fn select(&self, set: &SparseFeatureSet, attn: &AttentionWeights) -> LensSelection {
        let seed = lens_seed(set.commit, attn.commit);
        let mut adjusted = set
            .features
            .iter()
            .map(|feature| {
                let weight = Self::adjust_weight(feature.weight, attn);
                SparseFeature::new(feature.id, weight, seed)
            })
            .collect::<Vec<_>>();

        adjusted.sort_by(|left, right| {
            let left_abs = i32::from(left.weight).abs();
            let right_abs = i32::from(right.weight).abs();
            right_abs
                .cmp(&left_abs)
                .then_with(|| left.id.cmp(&right.id))
        });

        let k = self.top_k.min(adjusted.len());
        let topk = adjusted.into_iter().take(k).collect::<Vec<_>>();
        LensSelection::new(topk, seed)
    }
}

fn commit_activation_view(layer_id: u16, act_digest: Digest32, energy: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(ACT_VIEW_DOMAIN);
    hasher.update(&layer_id.to_be_bytes());
    hasher.update(act_digest.as_bytes());
    hasher.update(&energy.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_feature(seed: Digest32, id: u32, weight: i16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(FEATURE_DOMAIN);
    hasher.update(seed.as_bytes());
    hasher.update(&id.to_be_bytes());
    hasher.update(&weight.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_feature_set(seed: Digest32, features: &[SparseFeature]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(FEATURE_SET_DOMAIN);
    hasher.update(seed.as_bytes());
    hasher.update(&u64::try_from(features.len()).unwrap_or(0).to_be_bytes());
    for feature in features {
        hasher.update(feature.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_lens_selection(seed: Digest32, features: &[SparseFeature]) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(LENS_SELECTION_DOMAIN);
    hasher.update(seed.as_bytes());
    hasher.update(&u64::try_from(features.len()).unwrap_or(0).to_be_bytes());
    for feature in features {
        hasher.update(feature.commit.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn lens_seed(set_commit: Digest32, attn_commit: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(MOCK_LENS_DOMAIN);
    hasher.update(set_commit.as_bytes());
    hasher.update(attn_commit.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn clamp_weight(value: i32) -> i16 {
    value.clamp(-(MAX_WEIGHT_ABS as i32), MAX_WEIGHT_ABS as i32) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_sae_is_deterministic_and_sorted() {
        let view = ActivationView::new(3, Digest32::new([7u8; 32]), 2400);
        let port = MockSaePort::with_feature_count(4);

        let first = port.encode(&view);
        let second = port.encode(&view);

        assert_eq!(first, second);
        assert!(first
            .features
            .windows(2)
            .all(|pair| pair[0].id <= pair[1].id));
    }

    #[test]
    fn mock_lens_selects_top_k_and_scales_with_gain() {
        let seed = Digest32::new([9u8; 32]);
        let features = vec![
            SparseFeature::new(1, 100, seed),
            SparseFeature::new(2, -700, seed),
            SparseFeature::new(3, 250, seed),
            SparseFeature::new(4, -50, seed),
        ];
        let set = SparseFeatureSet::new(features, seed);
        let port = MockLensPort::with_top_k(2);

        let low_attn = AttentionWeights {
            channel: FocusChannel::Idle,
            gain: 1000,
            noise_suppress: 0,
            replay_bias: 0,
            commit: Digest32::new([3u8; 32]),
        };
        let high_attn = AttentionWeights {
            channel: FocusChannel::Idle,
            gain: 2000,
            noise_suppress: 0,
            replay_bias: 0,
            commit: Digest32::new([4u8; 32]),
        };

        let low_sel = port.select(&set, &low_attn);
        let high_sel = port.select(&set, &high_attn);

        assert_eq!(low_sel.topk.len(), 2);
        assert_eq!(low_sel.topk[0].id, 2);
        assert_eq!(low_sel.topk[1].id, 3);
        assert!(high_sel.topk[0].weight.abs() > low_sel.topk[0].weight.abs());
    }
}
