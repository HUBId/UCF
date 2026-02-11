use candle_core::{Device, Tensor};

use crate::{
    fuse_signals, AiComputeBackend, ComputeBudget, ComputeError, ComputeInput, ComputeSignals,
    Spike, MAX_SPIKES,
};

const IN_DIM: usize = 32;
const OUT_DIM: usize = 16;
const SPIKE_TOP_K: usize = 8;
const SCALE: u64 = 8;

#[derive(Debug, Clone, Copy)]
pub struct CandleBackend {
    seed: u64,
}

impl CandleBackend {
    pub fn new(seed: u64) -> Self {
        Self { seed }
    }

    fn check_budget(
        work_units: u64,
        stage: &'static str,
        budget: ComputeBudget,
    ) -> Result<(), ComputeError> {
        let elapsed_micros = work_units / SCALE;
        if work_units > budget.max_micros.saturating_mul(SCALE) {
            return Err(ComputeError::BudgetExceeded {
                stage,
                elapsed_micros,
                limit_micros: budget.max_micros,
            });
        }
        Ok(())
    }

    fn input_vector(input: &ComputeInput) -> [f32; IN_DIM] {
        let mut x = [0.0_f32; IN_DIM];
        for (i, value) in x.iter_mut().enumerate() {
            let u = input.context_digest[i % 32] as f32;
            *value = u / 255.0;
        }
        x
    }

    fn output_to_signals(&self, y: &[f32], input: &ComputeInput) -> ComputeSignals {
        let surprise = mean(&y[0..4]).clamp(0.0, 1.0);
        let pressure = mean(&y[4..8]).clamp(0.0, 1.0);
        let energy = mean(&y[8..12]).clamp(0.0, 1.0);
        let (risk, confidence) = fuse_signals(surprise, pressure, energy);

        let mut idx: Vec<usize> = (0..OUT_DIM).collect();
        idx.sort_by(|&a, &b| y[b].total_cmp(&y[a]));
        let top_k = SPIKE_TOP_K.min(MAX_SPIKES).min(idx.len());
        let spikes = idx
            .into_iter()
            .take(top_k)
            .map(|feature_idx| Spike {
                feature_id: feature_idx as u32,
                magnitude: y[feature_idx].clamp(0.0, 1.0),
                timestamp: input.t,
            })
            .collect::<Vec<_>>();

        let mut notes = vec![
            "backend=candle".to_string(),
            format!("w_digest={}", &hex::encode(WEIGHTS_DIGEST)[..12]),
            format!("seed={}", self.seed),
            format!("frame={}", input.frame_id.0),
        ];
        notes.sort();

        ComputeSignals {
            surprise,
            pressure,
            risk,
            confidence,
            spikes,
            notes,
            sparsity: Some(1.0 - ((SPIKE_TOP_K as f32) / (OUT_DIM as f32))),
            energy: Some(energy),
            ssm_readout: None,
            ssm_digest: None,
        }
        .bounded()
    }
}

impl Default for CandleBackend {
    fn default() -> Self {
        Self::new(ComputeBudget::default().seed)
    }
}

impl AiComputeBackend for CandleBackend {
    fn name(&self) -> &'static str {
        "candle_dummy"
    }

    fn compute(
        &self,
        input: &ComputeInput,
        budget: ComputeBudget,
    ) -> Result<ComputeSignals, ComputeError> {
        // v0 strategy A: backend directly implements AiComputeBackend and returns ComputeSignals.
        if input.t == 0 {
            return Err(ComputeError::InvalidInput {
                reason: "t must be non-zero".to_string(),
            });
        }
        Self::check_budget(1, "candle/start", budget)?;

        let x = Self::input_vector(input);
        Self::check_budget((IN_DIM * OUT_DIM) as u64, "candle/forward", budget)?;

        let device = Device::Cpu;
        let w = Tensor::from_slice(&weights_flat(), (OUT_DIM, IN_DIM), &device).map_err(|e| {
            ComputeError::Internal {
                reason: e.to_string(),
            }
        })?;
        let b = Tensor::from_slice(&B, OUT_DIM, &device).map_err(|e| ComputeError::Internal {
            reason: e.to_string(),
        })?;
        let x = Tensor::from_slice(&x, IN_DIM, &device)
            .map_err(|e| ComputeError::Internal {
                reason: e.to_string(),
            })?
            .reshape((1, IN_DIM))
            .map_err(|e| ComputeError::Internal {
                reason: e.to_string(),
            })?;

        let y = w
            .broadcast_mul(&x)
            .and_then(|v| v.sum(1))
            .and_then(|v| v.broadcast_add(&b))
            .map_err(|e| ComputeError::Internal {
                reason: e.to_string(),
            })?;

        let mut yv = y.to_vec1::<f32>().map_err(|e| ComputeError::Internal {
            reason: e.to_string(),
        })?;
        for v in &mut yv {
            *v = v.clamp(0.0, 1.0);
        }

        Ok(self.output_to_signals(&yv, input))
    }
}

fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f32>() / (values.len() as f32)
}

const B: [f32; OUT_DIM] = [
    0.04, 0.01, 0.03, 0.02, 0.03, 0.02, 0.01, 0.04, 0.02, 0.01, 0.03, 0.02, 0.01, 0.02, 0.03, 0.04,
];

fn weights_flat() -> [f32; OUT_DIM * IN_DIM] {
    let mut w = [0.0_f32; OUT_DIM * IN_DIM];
    let mut i = 0;
    while i < OUT_DIM * IN_DIM {
        let phase = (i % 7) as f32;
        w[i] = 0.011 + phase * 0.001;
        i += 1;
    }
    w
}

const WEIGHTS_DIGEST: [u8; 32] = [
    0x2f, 0x3b, 0x44, 0x4d, 0x52, 0x63, 0x71, 0x80, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xee, 0xfc, 0x01,
    0x12, 0x23, 0x34, 0x45, 0x56, 0x67, 0x78, 0x89, 0x9a, 0xab, 0xbc, 0xcd, 0xde, 0xef, 0xf0, 0x0f,
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{FrameId, MAX_NOTES, MAX_NOTE_LEN};

    fn input() -> ComputeInput {
        ComputeInput {
            frame_id: FrameId(7),
            t: 9,
            context_digest: [0x2a; 32],
        }
    }

    #[test]
    fn deterministic_for_same_seed_and_input() {
        let backend = CandleBackend::new(123);
        let budget = ComputeBudget::default();
        let a = backend.compute(&input(), budget).expect("compute a");
        let b = backend.compute(&input(), budget).expect("compute b");
        assert_eq!(a, b);
    }

    #[test]
    fn bounded_outputs_respected() {
        let backend = CandleBackend::default();
        let out = backend
            .compute(&input(), ComputeBudget::default())
            .expect("compute");
        assert!(out.spikes.len() <= MAX_SPIKES);
        assert!(out.notes.len() <= MAX_NOTES);
        assert!(out.notes.iter().all(|n| n.len() <= MAX_NOTE_LEN));
        assert!((0.0..=1.0).contains(&out.surprise));
        assert!((0.0..=1.0).contains(&out.pressure));
        assert!((0.0..=1.0).contains(&out.risk));
        assert!((0.0..=1.0).contains(&out.confidence));
    }

    #[test]
    fn budget_enforced() {
        let backend = CandleBackend::default();
        let err = backend
            .compute(
                &input(),
                ComputeBudget {
                    max_micros: 1,
                    hard_timeout_micros: 1,
                    seed: 0,
                },
            )
            .expect_err("should fail budget");
        match err {
            ComputeError::BudgetExceeded { stage, .. } => assert_eq!(stage, "candle/forward"),
            other => panic!("unexpected error: {other:?}"),
        }
    }
}
