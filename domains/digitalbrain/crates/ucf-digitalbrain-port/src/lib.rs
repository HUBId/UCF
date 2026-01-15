#![forbid(unsafe_code)]

use std::collections::{BTreeMap, VecDeque};
use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};
use thiserror::Error;
use ucf_types::EvidenceId;

pub type FeatureId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BrainRegion {
    Hypothalamus,
    Insula,
    NAcc,
    Pfc,
    Thalamus,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrainCoord {
    pub region: BrainRegion,
    pub layer: u8,
    pub x: u16,
    pub y: u16,
    pub z: u16,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Spike {
    pub coord: BrainCoord,
    pub amplitude: u16,
    pub width_us: u16,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BrainStimulus {
    pub spikes: Vec<Spike>,
    pub evidence_id: EvidenceId,
    pub rationale: FeatureId,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrainStimEvent {
    pub stim: BrainStimulus,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BrainReadback {
    pub last_stimulus: Option<BrainStimulus>,
}

pub trait DigitalBrainPort {
    fn stimulate(&self, stim: &BrainStimulus) -> Result<(), BrainError>;
    fn readback(&self) -> Result<BrainReadback, BrainError>;
}

#[derive(Error, Debug)]
pub enum BrainError {
    #[error("digitalbrain feature not enabled")]
    FeatureDisabled,
    #[error("readback unavailable")]
    ReadbackUnavailable,
}

#[derive(Clone, Default)]
pub struct MockDigitalBrainPort {
    buffer: Arc<Mutex<VecDeque<BrainStimulus>>>,
    capacity: usize,
}

impl MockDigitalBrainPort {
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: Arc::new(Mutex::new(VecDeque::with_capacity(capacity))),
            capacity,
        }
    }

    pub fn recorded(&self) -> Vec<BrainStimulus> {
        self.buffer
            .lock()
            .expect("lock brain buffer")
            .iter()
            .cloned()
            .collect()
    }
}

impl DigitalBrainPort for MockDigitalBrainPort {
    fn stimulate(&self, stim: &BrainStimulus) -> Result<(), BrainError> {
        let mut buffer = self.buffer.lock().expect("lock brain buffer");
        if self.capacity > 0 && buffer.len() == self.capacity {
            buffer.pop_front();
        }
        buffer.push_back(stim.clone());
        Ok(())
    }

    fn readback(&self) -> Result<BrainReadback, BrainError> {
        let buffer = self.buffer.lock().expect("lock brain buffer");
        Ok(BrainReadback {
            last_stimulus: buffer.back().cloned(),
        })
    }
}

/// Mapping table from feature ids (u32) to brain coordinates.
///
/// Breaking change: mapping keys are numeric FeatureId values, not digest hex strings.
///
/// TOML format:
///
/// ```toml
/// [mapping]
/// "1234" = [
///   { region = "hypothalamus", layer = 4, x = 10, y = 20, z = 30 }
/// ]
/// ```
#[derive(Clone, Debug, Default)]
pub struct MappingTable {
    map: BTreeMap<FeatureId, Vec<BrainCoord>>,
}

impl MappingTable {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn from_map(map: BTreeMap<FeatureId, Vec<BrainCoord>>) -> Self {
        Self { map }
    }

    pub fn resolve(&self, feature_id: &FeatureId) -> &[BrainCoord] {
        self.map
            .get(feature_id)
            .map(|coords| coords.as_slice())
            .unwrap_or(&[])
    }

    pub fn load_toml(path: impl AsRef<Path>) -> Result<Self, MappingTableError> {
        let contents = fs::read_to_string(path)?;
        Self::load_toml_str(&contents)
    }

    pub fn load_toml_str(contents: &str) -> Result<Self, MappingTableError> {
        let file: MappingTableFile = toml::from_str(contents)?;
        let mut map = BTreeMap::new();
        for (feature, coords) in file.mapping {
            let feature_id = parse_feature_id(&feature)?;
            map.insert(feature_id, coords);
        }
        Ok(Self { map })
    }
}

#[derive(Deserialize)]
struct MappingTableFile {
    mapping: BTreeMap<String, Vec<BrainCoord>>,
}

#[derive(Error, Debug)]
pub enum MappingTableError {
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    #[error("toml parse error: {0}")]
    Toml(#[from] toml::de::Error),
    #[error("invalid feature id: {0}")]
    InvalidFeatureId(String),
}

fn parse_feature_id(value: &str) -> Result<FeatureId, MappingTableError> {
    let trimmed = value.trim();
    if let Some(hex_str) = trimmed.strip_prefix("0x") {
        u32::from_str_radix(hex_str, 16)
            .map_err(|_| MappingTableError::InvalidFeatureId(value.to_string()))
    } else {
        trimmed
            .parse::<u32>()
            .map_err(|_| MappingTableError::InvalidFeatureId(value.to_string()))
    }
}

#[cfg(feature = "digitalbrain")]
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_stimulus(seed: u8) -> BrainStimulus {
        BrainStimulus {
            spikes: vec![Spike {
                coord: BrainCoord {
                    region: BrainRegion::Pfc,
                    layer: 2,
                    x: seed as u16,
                    y: 1,
                    z: 2,
                },
                amplitude: 5,
                width_us: 20,
            }],
            evidence_id: EvidenceId::new(format!("ev-{seed}")),
            rationale: FeatureId::from(seed as u32),
        }
    }

    #[test]
    fn mock_port_records_last_n_stimuli() {
        let port = MockDigitalBrainPort::new(2);
        port.stimulate(&sample_stimulus(1)).expect("stim 1");
        port.stimulate(&sample_stimulus(2)).expect("stim 2");
        port.stimulate(&sample_stimulus(3)).expect("stim 3");

        let recorded = port.recorded();
        assert_eq!(recorded.len(), 2);
        assert_eq!(recorded[0].evidence_id, EvidenceId::new("ev-2"));
        assert_eq!(recorded[1].evidence_id, EvidenceId::new("ev-3"));
    }

    #[test]
    fn mapping_table_parses_toml() {
        let toml = r#"
            [mapping]
            "1234" = [
              { region = "thalamus", layer = 3, x = 4, y = 5, z = 6 }
            ]
        "#;

        let table = MappingTable::load_toml_str(toml).expect("parse mapping");
        let coords = table.resolve(&1234);
        assert_eq!(coords.len(), 1);
        assert_eq!(coords[0].region, BrainRegion::Thalamus);
    }
}
