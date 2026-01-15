#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_sandbox::ControlFrameNormalized;
use ucf_types::Digest32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ActivationTap {
    pub layer: u16,
    pub token: u16,
    pub value_digest: Digest32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LensPlan {
    pub taps: Vec<(u16, u16)>,
}

pub trait LensPort {
    fn plan(&self, cf: &ControlFrameNormalized) -> LensPlan;
    fn tap(&self, plan: &LensPlan) -> Vec<ActivationTap>;
}

#[derive(Clone, Debug)]
pub struct LensMock {
    tap_count: usize,
}

impl Default for LensMock {
    fn default() -> Self {
        Self { tap_count: 3 }
    }
}

impl LensMock {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_tap_count(tap_count: usize) -> Self {
        Self { tap_count }
    }
}

impl LensPort for LensMock {
    fn plan(&self, cf: &ControlFrameNormalized) -> LensPlan {
        let digest = cf.commitment().digest;
        let bytes = digest.as_bytes();
        let tap_count = self.tap_count.min(bytes.len() / 4).max(1);
        let mut taps = Vec::with_capacity(tap_count);
        for i in 0..tap_count {
            let idx = i * 4;
            let layer = u16::from_be_bytes([bytes[idx], bytes[idx + 1]]);
            let token = u16::from_be_bytes([bytes[idx + 2], bytes[idx + 3]]);
            taps.push((layer, token));
        }
        LensPlan { taps }
    }

    fn tap(&self, plan: &LensPlan) -> Vec<ActivationTap> {
        plan.taps
            .iter()
            .map(|(layer, token)| {
                let mut hasher = Hasher::new();
                hasher.update(&layer.to_be_bytes());
                hasher.update(&token.to_be_bytes());
                let digest = Digest32::new(*hasher.finalize().as_bytes());
                ActivationTap {
                    layer: *layer,
                    token: *token,
                    value_digest: digest,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_sandbox::normalize;
    use ucf_types::v1::spec::ControlFrame;

    fn base_frame(frame_id: &str) -> ControlFrame {
        ControlFrame {
            frame_id: frame_id.to_string(),
            issued_at_ms: 1_700_000_000_000,
            decision: None,
            evidence_ids: Vec::new(),
            policy_id: "policy-1".to_string(),
        }
    }

    #[test]
    fn lens_mock_is_deterministic() {
        let mock = LensMock::new();
        let cf = normalize(base_frame("frame-1"));

        let plan_a = mock.plan(&cf);
        let plan_b = mock.plan(&cf);
        assert_eq!(plan_a, plan_b);

        let taps_a = mock.tap(&plan_a);
        let taps_b = mock.tap(&plan_b);
        assert_eq!(taps_a, taps_b);
    }
}
