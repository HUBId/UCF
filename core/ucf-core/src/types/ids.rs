use std::fmt::{Display, Formatter, Result as FmtResult};

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct RegionId(u32);

impl RegionId {
    pub const fn new(v: u32) -> Self {
        Self(v)
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for RegionId {
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl From<RegionId> for u32 {
    fn from(value: RegionId) -> Self {
        value.get()
    }
}

impl Display for RegionId {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "region:{}", self.0)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct PopId(u32);

impl PopId {
    pub const fn new(v: u32) -> Self {
        Self(v)
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for PopId {
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl From<PopId> for u32 {
    fn from(value: PopId) -> Self {
        value.get()
    }
}

impl Display for PopId {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "pop:{}", self.0)
    }
}

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct EdgeId(u32);

impl EdgeId {
    pub const fn new(v: u32) -> Self {
        Self(v)
    }

    pub fn get(self) -> u32 {
        self.0
    }
}

impl From<u32> for EdgeId {
    fn from(value: u32) -> Self {
        Self::new(value)
    }
}

impl From<EdgeId> for u32 {
    fn from(value: EdgeId) -> Self {
        value.get()
    }
}

impl Display for EdgeId {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "edge:{}", self.0)
    }
}
