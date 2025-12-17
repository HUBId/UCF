#![forbid(unsafe_code)]

#[cfg(feature = "derive")]
pub use serde_derive::{Deserialize, Serialize};

pub trait Serialize {}

pub trait Deserialize<'de> {}
