#![forbid(unsafe_code)]

#[cfg(feature = "ai-burn")]
pub mod burn_backend;
#[cfg(feature = "ai-candle")]
pub mod candle_backend;

pub use ucf_ai_host_abi::MockBackend;
