use sha2::{Digest, Sha256};

use crate::ComputeError;

pub(crate) const TOY_EMBED_DIM: usize = 8;
pub(crate) const MAX_VOCAB_LEN: usize = 64;

#[derive(Debug, Clone, PartialEq)]
pub(crate) struct ToyWeights {
    pub schema_version: u16,
    pub embed: Vec<f32>,
    pub linear_w: Vec<f32>,
    pub linear_b: Vec<f32>,
    pub vocab: Vec<String>,
    pub digest: [u8; 32],
}

#[derive(Debug, serde::Deserialize)]
struct ToyWeightsFile {
    schema_version: u16,
    embed_dim: usize,
    vocab: Vec<String>,
    embed: Vec<f32>,
    linear_w: Vec<f32>,
    linear_b: Vec<f32>,
    digest_hex: String,
}

impl ToyWeights {
    pub fn load() -> Result<Self, ComputeError> {
        let json = include_str!("../../fixtures/toy_weights_v1.json");
        Self::from_json(json)
    }

    pub fn from_json(json: &str) -> Result<Self, ComputeError> {
        let file: ToyWeightsFile =
            serde_json::from_str(json).map_err(|e| ComputeError::InvalidInput {
                reason: format!("toy weights json invalid: {e}"),
            })?;

        let vocab_len = file.vocab.len();
        if vocab_len == 0
            || vocab_len > MAX_VOCAB_LEN
            || file.embed_dim != TOY_EMBED_DIM
            || file.embed.len() != vocab_len * file.embed_dim
            || file.linear_w.len() != vocab_len * file.embed_dim
            || file.linear_b.len() != vocab_len
        {
            return Err(ComputeError::InvalidInput {
                reason: "toy weights dimensions invalid".to_string(),
            });
        }

        let digest = decode_hex_32(&file.digest_hex)?;
        let expected = compute_payload_digest(
            file.schema_version,
            file.embed_dim,
            &file.vocab,
            &file.embed,
            &file.linear_w,
            &file.linear_b,
        );
        if digest != expected {
            return Err(ComputeError::InvalidInput {
                reason: "toy weights digest mismatch".to_string(),
            });
        }

        Ok(Self {
            schema_version: file.schema_version,
            embed: file.embed,
            linear_w: file.linear_w,
            linear_b: file.linear_b,
            vocab: file.vocab,
            digest,
        })
    }
}

fn compute_payload_digest(
    schema_version: u16,
    embed_dim: usize,
    vocab: &[String],
    embed: &[f32],
    linear_w: &[f32],
    linear_b: &[f32],
) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(schema_version.to_le_bytes());
    hasher.update((embed_dim as u16).to_le_bytes());
    hasher.update((vocab.len() as u16).to_le_bytes());
    for token in vocab {
        hasher.update((token.len() as u16).to_le_bytes());
        hasher.update(token.as_bytes());
    }
    for arr in [embed, linear_w, linear_b] {
        for value in arr {
            hasher.update(value.to_le_bytes());
        }
    }
    hasher.finalize().into()
}

fn decode_hex_32(value: &str) -> Result<[u8; 32], ComputeError> {
    let decoded = hex::decode(value).map_err(|_| ComputeError::InvalidInput {
        reason: "toy weights digest hex invalid".to_string(),
    })?;
    if decoded.len() != 32 {
        return Err(ComputeError::InvalidInput {
            reason: "toy weights digest length invalid".to_string(),
        });
    }
    let mut out = [0_u8; 32];
    out.copy_from_slice(&decoded);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toy_fixture_digest_is_stable() {
        let weights = ToyWeights::load().expect("fixture loads");
        assert_eq!(weights.schema_version, 1);
        let mut expected = [0_u8; 32];
        expected.copy_from_slice(
            &hex::decode("c23777c3a593fcc7277818ab684d40e107b5786ce89810403533d98dce4bdf40")
                .expect("digest"),
        );
        assert_eq!(weights.digest, expected);
    }
}
