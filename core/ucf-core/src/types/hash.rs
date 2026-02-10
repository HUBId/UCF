use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct Hash32([u8; 32]);

impl Hash32 {
    pub const fn zero() -> Self {
        Self([0; 32])
    }

    pub const fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub const fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

impl Display for Hash32 {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        for b in self.0 {
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}
