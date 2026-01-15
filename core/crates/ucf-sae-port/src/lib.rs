#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_digitalbrain_port::FeatureId;
use ucf_lens_port::ActivationTap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SparseFeature {
    pub id: FeatureId,
    pub weight: i16,
}

pub trait SaePort {
    fn extract(&self, taps: &[ActivationTap]) -> Vec<SparseFeature>;
}

#[derive(Clone, Debug, Default)]
pub struct SaeMock;

impl SaeMock {
    pub fn new() -> Self {
        Self
    }

    fn feature_from_seed(seed: [u8; 4]) -> FeatureId {
        u32::from_be_bytes(seed)
    }

    fn weight_from_seed(seed: [u8; 2]) -> i16 {
        i16::from_be_bytes(seed)
    }
}

impl SaePort for SaeMock {
    fn extract(&self, taps: &[ActivationTap]) -> Vec<SparseFeature> {
        let mut features = Vec::new();
        for tap in taps {
            let mut hasher = Hasher::new();
            hasher.update(tap.value_digest.as_bytes());
            hasher.update(&tap.layer.to_be_bytes());
            hasher.update(&tap.token.to_be_bytes());
            let digest = hasher.finalize();
            let bytes = digest.as_bytes();
            let id_primary = Self::feature_from_seed([bytes[0], bytes[1], bytes[2], bytes[3]]);
            let weight_primary = Self::weight_from_seed([bytes[4], bytes[5]]);
            let id_secondary = Self::feature_from_seed([bytes[6], bytes[7], bytes[8], bytes[9]]);
            let weight_secondary = Self::weight_from_seed([bytes[10], bytes[11]]);
            features.push(SparseFeature {
                id: id_primary,
                weight: weight_primary,
            });
            features.push(SparseFeature {
                id: id_secondary,
                weight: weight_secondary,
            });
        }
        features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ucf_lens_port::ActivationTap;
    use ucf_types::Digest32;

    #[test]
    fn sae_mock_is_deterministic() {
        let tap = ActivationTap {
            layer: 1,
            token: 2,
            value_digest: Digest32::new([9u8; 32]),
        };
        let port = SaeMock::new();

        let a = port.extract(&[tap]);
        let b = port.extract(&[tap]);

        assert_eq!(a, b);
        assert_eq!(a.len(), 2);
    }
}
