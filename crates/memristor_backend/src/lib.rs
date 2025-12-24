#![forbid(unsafe_code)]

use blake3::Hasher;

pub const DEFAULT_MAX_VALUE: u16 = 100;
const DRIFT_PROFILE_VERSION: u8 = 1;

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellKey {
    AL,
    EXFIL_SENS,
    INTEGRITY_SENS,
    PROBING_SENS,
    RELIABILITY_SENS,
    RECOVERY_CONF,
}

impl CellKey {
    pub const ALL: [CellKey; 6] = [
        CellKey::AL,
        CellKey::EXFIL_SENS,
        CellKey::INTEGRITY_SENS,
        CellKey::PROBING_SENS,
        CellKey::RELIABILITY_SENS,
        CellKey::RECOVERY_CONF,
    ];

    pub fn index(self) -> usize {
        match self {
            CellKey::AL => 0,
            CellKey::EXFIL_SENS => 1,
            CellKey::INTEGRITY_SENS => 2,
            CellKey::PROBING_SENS => 3,
            CellKey::RELIABILITY_SENS => 4,
            CellKey::RECOVERY_CONF => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CalibrationReport {
    pub before_digest: [u8; 32],
    pub after_digest: [u8; 32],
    pub applied: bool,
}

pub trait MemristorBackend {
    fn read_cell(&self, key: CellKey) -> u16;

    fn write_cell(&mut self, key: CellKey, value: u16);

    fn calibrate(&mut self) -> CalibrationReport;

    fn config_digest(&self) -> [u8; 32];

    fn snapshot_digest(&self) -> [u8; 32];
}

#[derive(Debug, Clone)]
pub struct EmulatedMemristorBackend {
    cells: [u16; 6],
    drift_bias: [i16; 6],
    max_value: u16,
    seed: u64,
}

impl EmulatedMemristorBackend {
    pub fn new(seed: u64, max_value: u16) -> Self {
        let drift_bias = derive_drift_bias(seed);
        Self {
            cells: [0; 6],
            drift_bias,
            max_value,
            seed,
        }
    }

    pub fn max_value(&self) -> u16 {
        self.max_value
    }

    fn clamp(&self, value: u16) -> u16 {
        value.min(self.max_value)
    }

    fn config_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.push(DRIFT_PROFILE_VERSION);
        bytes.extend(self.seed.to_le_bytes());
        bytes.extend(self.max_value.to_le_bytes());
        for value in self.drift_bias.iter() {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }

    fn snapshot_bytes(&self) -> Vec<u8> {
        let mut bytes = self.config_bytes();
        for value in self.cells.iter() {
            bytes.extend(value.to_le_bytes());
        }
        bytes
    }
}

impl Default for EmulatedMemristorBackend {
    fn default() -> Self {
        Self::new(0, DEFAULT_MAX_VALUE)
    }
}

impl MemristorBackend for EmulatedMemristorBackend {
    fn read_cell(&self, key: CellKey) -> u16 {
        self.cells[key.index()]
    }

    fn write_cell(&mut self, key: CellKey, value: u16) {
        self.cells[key.index()] = self.clamp(value);
    }

    fn calibrate(&mut self) -> CalibrationReport {
        let before_digest = self.snapshot_digest();
        let applied = true;
        for (idx, bias) in self.drift_bias.iter().copied().enumerate() {
            let adjusted = (self.cells[idx] as i32).saturating_sub(bias as i32);
            let adjusted = adjusted.clamp(0, self.max_value as i32);
            self.cells[idx] = adjusted as u16;
        }
        let after_digest = self.snapshot_digest();
        CalibrationReport {
            before_digest,
            after_digest,
            applied,
        }
    }

    fn config_digest(&self) -> [u8; 32] {
        digest_with_domain("UCF:MEM:CFG", &self.config_bytes())
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        digest_with_domain("UCF:MEM:SNAP", &self.snapshot_bytes())
    }
}

#[cfg(feature = "microcircuit-hpa-hw")]
#[derive(Debug, Default, Clone, Copy)]
pub struct HardwareMemristorBackend;

#[cfg(feature = "microcircuit-hpa-hw")]
impl MemristorBackend for HardwareMemristorBackend {
    fn read_cell(&self, _key: CellKey) -> u16 {
        unimplemented!("Hardware memristor backend is not implemented yet")
    }

    fn write_cell(&mut self, _key: CellKey, _value: u16) {
        unimplemented!("Hardware memristor backend is not implemented yet")
    }

    fn calibrate(&mut self) -> CalibrationReport {
        unimplemented!("Hardware memristor backend is not implemented yet")
    }

    fn config_digest(&self) -> [u8; 32] {
        unimplemented!("Hardware memristor backend is not implemented yet")
    }

    fn snapshot_digest(&self) -> [u8; 32] {
        unimplemented!("Hardware memristor backend is not implemented yet")
    }
}

fn derive_drift_bias(seed: u64) -> [i16; 6] {
    let mut hasher = Hasher::new();
    hasher.update(b"UCF:MEM:BIAS");
    hasher.update(&seed.to_le_bytes());
    let digest = hasher.finalize();
    let bytes = digest.as_bytes();
    let mut bias = [0i16; 6];
    for (idx, slot) in bias.iter_mut().enumerate() {
        let mapped = bytes[idx] % 3;
        *slot = match mapped {
            0 => -1,
            1 => 0,
            _ => 1,
        };
    }
    bias
}

fn digest_with_domain(domain: &str, bytes: &[u8]) -> [u8; 32] {
    let mut hasher = Hasher::new();
    hasher.update(domain.as_bytes());
    hasher.update(bytes);
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_cell_clamps_values() {
        let mut backend = EmulatedMemristorBackend::default();
        backend.write_cell(CellKey::AL, 999);
        assert_eq!(backend.read_cell(CellKey::AL), backend.max_value());
    }

    #[test]
    fn calibrate_is_deterministic() {
        let mut backend = EmulatedMemristorBackend::new(42, DEFAULT_MAX_VALUE);
        backend.write_cell(CellKey::AL, 20);
        let report_a = backend.calibrate();
        let report_b = backend.calibrate();
        assert_ne!(report_a.before_digest, report_a.after_digest);
        assert_ne!(report_a.after_digest, report_b.after_digest);
        assert!(report_a.applied);
    }

    #[test]
    fn snapshot_digest_depends_on_state() {
        let mut backend = EmulatedMemristorBackend::new(7, DEFAULT_MAX_VALUE);
        let digest_a = backend.snapshot_digest();
        backend.write_cell(CellKey::EXFIL_SENS, 10);
        let digest_b = backend.snapshot_digest();
        assert_ne!(digest_a, digest_b);
    }
}
