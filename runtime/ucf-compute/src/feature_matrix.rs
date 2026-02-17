use crate::{BackendPackKind, ComputeError};

pub const FEATURE_BACKEND_STUB: u16 = 1 << 0;
pub const FEATURE_BACKEND_TOY: u16 = 1 << 1;
pub const FEATURE_LLM_CANDLE: u16 = 1 << 2;
pub const FEATURE_LFM_CANDLE: u16 = 1 << 3;
pub const FEATURE_BACKEND_BURN: u16 = 1 << 4;
pub const FEATURE_LFM_LNN: u16 = 1 << 5;
pub const FEATURE_PLASTICITY: u16 = 1 << 6;
pub const FEATURE_REPLAY: u16 = 1 << 7;
pub const FEATURE_OPS_EXPLAIN: u16 = 1 << 8;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReleaseFeatureMatrix {
    pub bits: u16,
}

impl ReleaseFeatureMatrix {
    pub const fn detect() -> Self {
        let mut bits = 0u16;
        if cfg!(feature = "backend-stub") {
            bits |= FEATURE_BACKEND_STUB;
        }
        if cfg!(feature = "backend-toy") {
            bits |= FEATURE_BACKEND_TOY;
        }
        if cfg!(feature = "llm-candle") {
            bits |= FEATURE_LLM_CANDLE;
        }
        if cfg!(feature = "lfm-candle") {
            bits |= FEATURE_LFM_CANDLE;
        }
        if cfg!(feature = "backend-burn") {
            bits |= FEATURE_BACKEND_BURN;
        }
        if cfg!(feature = "lfm-lnn") {
            bits |= FEATURE_LFM_LNN;
        }
        if cfg!(feature = "plasticity") {
            bits |= FEATURE_PLASTICITY;
        }
        if cfg!(feature = "replay") {
            bits |= FEATURE_REPLAY;
        }
        if cfg!(feature = "ops-explain") {
            bits |= FEATURE_OPS_EXPLAIN;
        }
        Self { bits }
    }

    pub fn validate_pack(self, pack: BackendPackKind) -> Result<(), ComputeError> {
        let has_toy = self.bits & FEATURE_BACKEND_TOY != 0;
        let has_lfm_candle = self.bits & FEATURE_LFM_CANDLE != 0;
        let has_burn = self.bits & FEATURE_BACKEND_BURN != 0;
        let has_lnn = self.bits & FEATURE_LFM_LNN != 0;

        match pack {
            BackendPackKind::StubV0 => Ok(()),
            BackendPackKind::ToyV1 => {
                if has_toy {
                    Ok(())
                } else {
                    Err(ComputeError::InvalidInput {
                        reason: "pack toy_v1 requires feature backend-toy".to_string(),
                    })
                }
            }
            BackendPackKind::CandleToyV1 | BackendPackKind::CandleLiquidV1 => {
                if !has_lfm_candle {
                    return Err(ComputeError::InvalidInput {
                        reason: format!("pack {} requires feature lfm-candle", pack.as_str()),
                    });
                }
                if !has_toy {
                    return Err(ComputeError::InvalidInput {
                        reason: format!("pack {} requires feature backend-toy", pack.as_str()),
                    });
                }
                Ok(())
            }
            BackendPackKind::BurnToyV1 => {
                if has_burn {
                    Ok(())
                } else {
                    Err(ComputeError::InvalidInput {
                        reason: "pack burn_toy_v1 requires feature backend-burn".to_string(),
                    })
                }
            }
            BackendPackKind::ToyLnnV1 => {
                if !has_lnn {
                    return Err(ComputeError::InvalidInput {
                        reason: "pack toy_lnn_v1 requires feature lfm-lnn".to_string(),
                    });
                }
                if !(has_toy || has_lfm_candle) {
                    return Err(ComputeError::InvalidInput {
                        reason: "pack toy_lnn_v1 requires backend-toy or lfm-candle".to_string(),
                    });
                }
                Ok(())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_missing_lnn_feature() {
        let matrix = ReleaseFeatureMatrix {
            bits: FEATURE_BACKEND_TOY,
        };
        let err = matrix
            .validate_pack(BackendPackKind::ToyLnnV1)
            .expect_err("must fail");
        assert!(format!("{err}").contains("lfm-lnn"));
    }

    #[test]
    fn accepts_toy_pack_with_toy_feature() {
        let matrix = ReleaseFeatureMatrix {
            bits: FEATURE_BACKEND_TOY,
        };
        matrix
            .validate_pack(BackendPackKind::ToyV1)
            .expect("toy supported");
    }
}
