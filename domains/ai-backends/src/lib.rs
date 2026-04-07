#![forbid(unsafe_code)]
//! Compatibility-only backend adapter seams for `domains/ai-host-abi`.
//!
//! Canonical runtime compute backend wiring is in `runtime/ucf-compute`.
//! The modules here are retained as non-canonical adapter surfaces.

#[cfg(feature = "ai-burn")]
pub mod burn_backend;
#[cfg(feature = "ai-candle")]
pub mod candle_backend;

pub use ucf_ai_host_abi::MockBackend;
