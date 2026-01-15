#![forbid(unsafe_code)]
//! Firewood backend für das Archiv.
//!
//! ## Aktivierung
//! - Cargo Feature: `firewood`
//! - Backend: `ArchiveBackend::Firewood`
//!
//! Beispiel:
//! ```toml
//! ucf-archive-firewood = { path = "domains/archive/crates/ucf-archive-firewood", features = ["firewood"] }
//! ```
//!
//! Danach kann `FirewoodRecordStore::open(...)` genutzt werden.

#[cfg(feature = "firewood")]
pub use ucf_archive::{
    ArchiveBackend, ArchiveStore, FirewoodRecordStore, RecordStore, SnapshotStore,
};

#[cfg(not(feature = "firewood"))]
pub struct FirewoodFeatureDisabled;
