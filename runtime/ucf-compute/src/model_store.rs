use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use sha2::{Digest, Sha256};

use crate::ComputeError;

const DEFAULT_ALLOWLIST_ROOT: &str = "models";
const DEFAULT_MANIFEST_PATH: &str = "models/manifest.toml";
const DEFAULT_MAX_BYTES: u64 = 64 * 1024 * 1024;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum ModelSlot {
    Llm,
    WorldJepa,
    WorldVljepa,
    Sae,
    Lfm,
    Ssm,
    EbmReasoner,
}

impl ModelSlot {
    pub const fn all() -> [Self; 7] {
        [
            Self::Llm,
            Self::WorldJepa,
            Self::WorldVljepa,
            Self::Sae,
            Self::Lfm,
            Self::Ssm,
            Self::EbmReasoner,
        ]
    }

    pub const fn env_key(self) -> &'static str {
        match self {
            Self::Llm => "LLM",
            Self::WorldJepa => "WORLD_JEPA",
            Self::WorldVljepa => "WORLD_VLJEPA",
            Self::Sae => "SAE",
            Self::Lfm => "LFM",
            Self::Ssm => "SSM",
            Self::EbmReasoner => "EBM",
        }
    }

    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Llm => "llm",
            Self::WorldJepa => "world_jepa",
            Self::WorldVljepa => "world_vljepa",
            Self::Sae => "sae",
            Self::Lfm => "lfm",
            Self::Ssm => "ssm",
            Self::EbmReasoner => "ebm_reasoner",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelFormat {
    CandleSafetensors,
    CandleBin,
    Burn,
    Custom,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ModelDevice {
    CpuOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelSlotSpec {
    pub slot: ModelSlot,
    pub enabled: bool,
    pub path: Option<PathBuf>,
    pub expected_sha256: [u8; 32],
    pub max_bytes: u64,
    pub format: ModelFormat,
    pub device: ModelDevice,
    pub active_hash: Option<String>,
    pub contract_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelLoadError {
    Disabled,
    MissingPath,
    MissingExpectedHash {
        slot: ModelSlot,
    },
    ManifestParse(String),
    PathOutsideAllowlist {
        path: PathBuf,
        allowlist_root: PathBuf,
    },
    PathTraversal {
        path: PathBuf,
    },
    OpenFailed {
        path: PathBuf,
        reason: String,
    },
    Oversized {
        path: PathBuf,
        max_bytes: u64,
        size_bytes: u64,
    },
    HashMismatch {
        expected: [u8; 32],
        found: [u8; 32],
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotTargetState {
    Discovered,
    Verified,
    Active,
    Candidate,
    Compare,
    Shadow,
    Disabled,
    Blocked,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelActivationError {
    ArtifactNotVerified {
        slot: ModelSlot,
        hash: String,
        reason: ModelLoadError,
    },
    IncompatiblePackContractBackend {
        slot: ModelSlot,
        expected_contract_version: Option<String>,
        requested_contract_version: Option<String>,
    },
    ActivationRejected {
        slot: ModelSlot,
        reason: String,
    },
    ActiveSlotMissing {
        slot: ModelSlot,
    },
    CompareShadowPathUnavailable {
        slot: ModelSlot,
        path_kind: SlotTargetState,
        reason: ModelLoadError,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SlotActivationPlan {
    pub slot: ModelSlot,
    pub target_hash: String,
    pub target_state: SlotTargetState,
    pub selected_via: String,
    pub contract_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SlotPathStatus {
    pub target_state: SlotTargetState,
    pub configured_hash: Option<String>,
    pub verified: bool,
    pub comparable: bool,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompareShadowContext {
    ComparableSameEffectiveConfiguration,
    ComparableWithCaveats,
    NotComparableDifferentRuntimeContext,
    BlockedMissingSignals,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ComparePathOutcome {
    ComparedSuccessfully,
    ComparisonInconclusive,
    ComparisonBlocked,
    ComparisonFailedTechnically,
    NotComparable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ShadowPathOutcome {
    ShadowedSuccessfully,
    ShadowInconclusive,
    ShadowBlocked,
    ShadowFailedTechnically,
    NotComparable,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CompareShadowEvaluation {
    pub active_reference_hash: Option<String>,
    pub candidate_hash: Option<String>,
    pub compare_hash: Option<String>,
    pub shadow_hash: Option<String>,
    pub context: CompareShadowContext,
    pub compare_outcome: ComparePathOutcome,
    pub shadow_outcome: ShadowPathOutcome,
    pub caveat: Option<String>,
    pub blocker: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SlotWarmupState {
    Cold,
    Prepared,
    Warm,
    Blocked,
    Stale,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SlotWarmupStatus {
    pub target_state: SlotTargetState,
    pub state: SlotWarmupState,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionDecisionState {
    Known,
    Candidate,
    Comparable,
    Promotable,
    BlockedForPromotion,
    Active,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionBlockerCode {
    NotComparableYet,
    InsufficientBaselineSignal,
    RuntimePathNotProductionUsable,
    GateBlocked,
    DegradedBeyondAcceptableThreshold,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct PromotionTechnicalSignals {
    pub baseline_comparison_ready: bool,
    pub runtime_path_production_usable: bool,
    pub readiness_ok: bool,
    pub degraded_beyond_acceptable_threshold: bool,
    pub compare_or_shadow_diagnostic_ready: bool,
    pub comparable_under_same_effective_configuration: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PromotionEvaluationDisposition {
    CandidateRemainsBlocked,
    CandidateMorePromotable,
    CandidateComparisonInconclusive,
    ActivePathRemainsPreferred,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SlotPromotionDecision {
    pub slot: ModelSlot,
    pub active_hash: Option<String>,
    pub candidate_hash: Option<String>,
    pub state: PromotionDecisionState,
    pub blockers: Vec<PromotionBlockerCode>,
    pub signals: PromotionTechnicalSignals,
    pub compare_shadow: CompareShadowEvaluation,
    pub disposition: PromotionEvaluationDisposition,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationOutcome {
    Pending,
    Succeeded,
    Degraded,
    Blocked,
    FailedTechnically,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActivationFallbackState {
    NotUsed,
    FallbackToPriorActive,
    FallbackUnavailable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RollbackOutcome {
    NotRequested,
    Completed,
    Unavailable,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SlotActivationAssessment {
    pub slot: ModelSlot,
    pub target_hash: String,
    pub prior_active_hash: Option<String>,
    pub resulting_active_hash: Option<String>,
    pub outcome: ActivationOutcome,
    pub fallback: ActivationFallbackState,
    pub rollback: RollbackOutcome,
    pub promotion_state: PromotionDecisionState,
    pub promotion_blockers: Vec<PromotionBlockerCode>,
    pub blocked_reason: Option<String>,
    pub degraded_reason: Option<String>,
    pub technical_failure: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct SlotRollbackAssessment {
    pub slot: ModelSlot,
    pub requested_hash: Option<String>,
    pub prior_active_hash: Option<String>,
    pub rollback_hash: Option<String>,
    pub replaced_hash: Option<String>,
    pub resulting_active_hash: Option<String>,
    pub outcome: RollbackOutcome,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedModelSlot {
    pub slot: ModelSlot,
    pub path: PathBuf,
    pub sha256: [u8; 32],
    pub size_bytes: u64,
    pub format: ModelFormat,
    pub device: ModelDevice,
    pub active_hash: Option<String>,
    pub contract_version: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelProvenance {
    pub slot: ModelSlot,
    pub enabled: bool,
    pub resolved_path: Option<String>,
    pub sha256: Option<[u8; 32]>,
    pub size_bytes: Option<u64>,
    pub format: ModelFormat,
    pub backend_pack_digest: [u8; 32],
    pub run_id: String,
    pub schema_version: u16,
    pub disable_reason: Option<String>,
    pub found_hash_prefix: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ModelStore {
    pub allowlist_root: PathBuf,
    pub specs: BTreeMap<ModelSlot, ModelSlotSpec>,
}

impl ModelStore {
    fn default_manifest_path() -> &'static str {
        DEFAULT_MANIFEST_PATH
    }

    pub fn from_manifest_and_env(manifest_path: &Path) -> Result<Self, ModelLoadError> {
        let manifest_exists = manifest_path.exists();
        let mut doc = if manifest_path.exists() {
            let text = std::fs::read_to_string(manifest_path)
                .map_err(|e| ModelLoadError::ManifestParse(e.to_string()))?;
            toml::from_str::<ModelManifest>(&text)
                .map_err(|e| ModelLoadError::ManifestParse(e.to_string()))?
        } else {
            ModelManifest::default()
        };
        doc.apply_env_overrides();
        let allowlist_root = doc
            .allowlist_root
            .clone()
            .unwrap_or_else(|| PathBuf::from(DEFAULT_ALLOWLIST_ROOT));
        let specs = doc.to_specs();
        let any_enabled = specs.values().any(|s| s.enabled);
        if any_enabled && !manifest_exists {
            return Err(ModelLoadError::ManifestParse(
                "manifest required when any model slot is enabled".to_string(),
            ));
        }
        Ok(Self {
            allowlist_root,
            specs,
        })
    }

    pub fn from_env_default() -> Result<Self, ModelLoadError> {
        let path = std::env::var("UCF_MODEL_MANIFEST")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from(Self::default_manifest_path()));
        Self::from_manifest_and_env(&path)
    }

    pub fn verify_slot(&self, slot: ModelSlot) -> Result<VerifiedModelSlot, ModelLoadError> {
        let Some(spec) = self.specs.get(&slot) else {
            return Err(ModelLoadError::Disabled);
        };
        if !spec.enabled {
            return Err(ModelLoadError::Disabled);
        }
        let pin_key = format!("UCF_MODEL_PIN_{}", slot.env_key());
        let pin_hash = std::env::var(pin_key).ok();
        let rel_path = if let Some(pin_hash) = pin_hash.as_ref() {
            PathBuf::from(format!(
                "promoted/{}/{}/model.safetensors",
                slot.as_str(),
                pin_hash
            ))
        } else if let Some(active_hash) = spec.active_hash.as_ref() {
            PathBuf::from(format!(
                "promoted/{}/{}/model.safetensors",
                slot.as_str(),
                active_hash
            ))
        } else {
            spec.path.clone().ok_or(ModelLoadError::MissingPath)?
        };
        let expected_hash = if let Some(pin_hash) = pin_hash {
            parse_hash(&pin_hash)
        } else if let Some(active_hash) = spec.active_hash.as_ref() {
            parse_hash(active_hash)
        } else {
            spec.expected_sha256
        };
        if expected_hash == [0; 32] {
            return Err(ModelLoadError::MissingExpectedHash { slot });
        }
        let joined = self.allowlist_root.join(&rel_path);
        let allowlist_root =
            self.allowlist_root
                .canonicalize()
                .map_err(|_| ModelLoadError::PathTraversal {
                    path: self.allowlist_root.clone(),
                })?;
        let canonical = joined
            .canonicalize()
            .map_err(|_| ModelLoadError::PathTraversal {
                path: rel_path.clone(),
            })?;
        if !canonical.starts_with(&allowlist_root) {
            return Err(ModelLoadError::PathOutsideAllowlist {
                path: canonical,
                allowlist_root,
            });
        }

        let mut file = File::open(&canonical).map_err(|e| ModelLoadError::OpenFailed {
            path: canonical.clone(),
            reason: e.to_string(),
        })?;
        let size = file
            .metadata()
            .map_err(|e| ModelLoadError::OpenFailed {
                path: canonical.clone(),
                reason: e.to_string(),
            })?
            .len();
        if size > spec.max_bytes {
            return Err(ModelLoadError::Oversized {
                path: canonical,
                max_bytes: spec.max_bytes,
                size_bytes: size,
            });
        }
        let mut hasher = Sha256::new();
        let mut buf = [0_u8; 16 * 1024];
        loop {
            let read = file
                .read(&mut buf)
                .map_err(|e| ModelLoadError::OpenFailed {
                    path: canonical.clone(),
                    reason: e.to_string(),
                })?;
            if read == 0 {
                break;
            }
            hasher.update(&buf[..read]);
        }
        let found: [u8; 32] = hasher.finalize().into();
        if found != expected_hash {
            return Err(ModelLoadError::HashMismatch {
                expected: expected_hash,
                found,
            });
        }

        Ok(VerifiedModelSlot {
            slot,
            path: canonical,
            sha256: found,
            size_bytes: size,
            format: spec.format,
            device: spec.device,
            active_hash: spec.active_hash.clone(),
            contract_version: spec.contract_version.clone(),
        })
    }

    pub fn read_verified_bytes(
        &self,
        verified: &VerifiedModelSlot,
    ) -> Result<Vec<u8>, ModelLoadError> {
        let mut file = File::open(&verified.path).map_err(|e| ModelLoadError::OpenFailed {
            path: verified.path.clone(),
            reason: e.to_string(),
        })?;
        let size = file
            .metadata()
            .map_err(|e| ModelLoadError::OpenFailed {
                path: verified.path.clone(),
                reason: e.to_string(),
            })?
            .len();
        let slot_spec = self
            .specs
            .get(&verified.slot)
            .ok_or(ModelLoadError::Disabled)?;
        if size > slot_spec.max_bytes {
            return Err(ModelLoadError::Oversized {
                path: verified.path.clone(),
                max_bytes: slot_spec.max_bytes,
                size_bytes: size,
            });
        }
        let mut bytes = Vec::with_capacity(size as usize);
        file.read_to_end(&mut bytes)
            .map_err(|e| ModelLoadError::OpenFailed {
                path: verified.path.clone(),
                reason: e.to_string(),
            })?;
        Ok(bytes)
    }

    pub fn verified_slots(&self) -> BTreeMap<ModelSlot, Result<VerifiedModelSlot, ModelLoadError>> {
        ModelSlot::all()
            .into_iter()
            .map(|slot| (slot, self.verify_slot(slot)))
            .collect()
    }

    pub fn model_hashes_digest(&self) -> [u8; 32] {
        let mut hasher = Sha256::new();
        let verified = self.verified_slots();
        for slot in ModelSlot::all() {
            hasher.update(slot.as_str().as_bytes());
            if let Some(Ok(v)) = verified.get(&slot) {
                hasher.update([1]);
                hasher.update(v.sha256);
            } else {
                hasher.update([0]);
            }
        }
        hasher.finalize().into()
    }

    pub fn plan_slot_activation(
        &self,
        slot: ModelSlot,
        target_hash: &str,
        requested_contract_version: Option<&str>,
    ) -> Result<SlotActivationPlan, ModelActivationError> {
        let Some(spec) = self.specs.get(&slot) else {
            return Err(ModelActivationError::ActiveSlotMissing { slot });
        };
        if !spec.enabled {
            return Err(ModelActivationError::ActivationRejected {
                slot,
                reason: "slot disabled by manifest/env".to_string(),
            });
        }
        let target_hash = target_hash.trim();
        if parse_hash(target_hash) == [0; 32] {
            return Err(ModelActivationError::ActivationRejected {
                slot,
                reason: "target hash must be 64 hex chars".to_string(),
            });
        }
        if std::env::var(format!("UCF_MODEL_PIN_{}", slot.env_key()))
            .ok()
            .as_deref()
            .is_some_and(|pin| !pin.trim().is_empty() && pin.trim() != target_hash)
        {
            return Err(ModelActivationError::ActivationRejected {
                slot,
                reason: "pin override conflicts with requested activation hash".to_string(),
            });
        }

        self.verify_promoted_hash(slot, target_hash)
            .map_err(|reason| ModelActivationError::ArtifactNotVerified {
                slot,
                hash: target_hash.to_string(),
                reason,
            })?;
        self.ensure_optional_path_verified(slot, SlotTargetState::Compare)?;
        self.ensure_optional_path_verified(slot, SlotTargetState::Shadow)?;

        let expected_contract_version = spec.contract_version.clone();
        if let Some(requested) = requested_contract_version {
            if expected_contract_version
                .as_deref()
                .is_some_and(|v| v != requested)
            {
                return Err(ModelActivationError::IncompatiblePackContractBackend {
                    slot,
                    expected_contract_version,
                    requested_contract_version: Some(requested.to_string()),
                });
            }
        }

        Ok(SlotActivationPlan {
            slot,
            target_hash: target_hash.to_string(),
            target_state: SlotTargetState::Active,
            selected_via: if std::env::var(format!("UCF_MODEL_PIN_{}", slot.env_key())).is_ok() {
                "pin_override".to_string()
            } else {
                "active_hash".to_string()
            },
            contract_version: requested_contract_version
                .map(ToOwned::to_owned)
                .or(expected_contract_version),
        })
    }

    pub fn assess_slot_activation(
        &self,
        slot: ModelSlot,
        target_hash: &str,
        requested_contract_version: Option<&str>,
    ) -> SlotActivationAssessment {
        let target_hash = target_hash.trim().to_string();
        let prior_active_hash = self.active_hash_for_slot(slot);
        let promotion = self.slot_promotion_decision(slot);
        let mut assessment = SlotActivationAssessment {
            slot,
            target_hash: target_hash.clone(),
            prior_active_hash: prior_active_hash.clone(),
            resulting_active_hash: prior_active_hash.clone(),
            outcome: ActivationOutcome::Pending,
            fallback: ActivationFallbackState::NotUsed,
            rollback: RollbackOutcome::NotRequested,
            promotion_state: promotion.state,
            promotion_blockers: promotion.blockers.clone(),
            blocked_reason: None,
            degraded_reason: None,
            technical_failure: None,
        };
        match self.plan_slot_activation(slot, &target_hash, requested_contract_version) {
            Ok(_) => {}
            Err(err) => {
                assessment.outcome = ActivationOutcome::Blocked;
                assessment.blocked_reason = Some(format!("{err:?}"));
                assessment.fallback =
                    self.fallback_state_for_prior(slot, prior_active_hash.as_deref());
                return assessment;
            }
        }
        if let Err(err) = self.prefetch_promoted_hash(slot, &target_hash) {
            assessment.outcome = ActivationOutcome::FailedTechnically;
            assessment.technical_failure = Some(format!("{err:?}"));
            assessment.fallback = self.fallback_state_for_prior(slot, prior_active_hash.as_deref());
            return assessment;
        }

        if prior_active_hash.as_deref() == Some(target_hash.as_str())
            || promotion
                .active_hash
                .as_deref()
                .is_some_and(|active| active == target_hash)
        {
            assessment.outcome = ActivationOutcome::Succeeded;
            assessment.resulting_active_hash = Some(target_hash);
            return assessment;
        }

        if promotion.candidate_hash.as_deref() == Some(target_hash.as_str())
            && !promotion.blockers.is_empty()
        {
            let hard_blocked = promotion.blockers.iter().any(|blocker| {
                matches!(
                    blocker,
                    PromotionBlockerCode::GateBlocked
                        | PromotionBlockerCode::RuntimePathNotProductionUsable
                )
            });
            if hard_blocked {
                assessment.outcome = ActivationOutcome::Blocked;
                assessment.blocked_reason = Some(format!(
                    "candidate remains blocked for promotion: {:?}",
                    promotion.blockers
                ));
                assessment.fallback =
                    self.fallback_state_for_prior(slot, prior_active_hash.as_deref());
                return assessment;
            }
        }

        if !promotion.signals.baseline_comparison_ready
            || !promotion.signals.compare_or_shadow_diagnostic_ready
            || promotion.signals.degraded_beyond_acceptable_threshold
        {
            assessment.outcome = ActivationOutcome::Degraded;
            assessment.degraded_reason = Some(format!(
                "baseline_ready={};compare_shadow_ready={};degraded_threshold={}",
                promotion.signals.baseline_comparison_ready,
                promotion.signals.compare_or_shadow_diagnostic_ready,
                promotion.signals.degraded_beyond_acceptable_threshold
            ));
            if prior_active_hash.as_deref() != Some(target_hash.as_str()) {
                assessment.fallback =
                    self.fallback_state_for_prior(slot, prior_active_hash.as_deref());
            }
            return assessment;
        }
        assessment.outcome = ActivationOutcome::Pending;
        assessment.resulting_active_hash = prior_active_hash;
        assessment
    }

    pub fn assess_slot_rollback(
        &self,
        slot: ModelSlot,
        requested_hash: Option<&str>,
        requested_contract_version: Option<&str>,
    ) -> SlotRollbackAssessment {
        let requested_hash = requested_hash
            .map(|hash| hash.trim().to_string())
            .filter(|hash| !hash.is_empty());
        let prior_active_hash = self.active_hash_for_slot(slot);
        let rollback_hash = requested_hash.clone().or_else(|| prior_active_hash.clone());
        let replaced_hash = self
            .slot_path_statuses(slot)
            .into_iter()
            .find(|status| status.target_state == SlotTargetState::Candidate)
            .and_then(|status| status.configured_hash);
        let mut assessment = SlotRollbackAssessment {
            slot,
            requested_hash,
            prior_active_hash: prior_active_hash.clone(),
            rollback_hash: rollback_hash.clone(),
            replaced_hash,
            resulting_active_hash: prior_active_hash.clone(),
            outcome: RollbackOutcome::NotRequested,
            detail: "rollback not requested".to_string(),
        };
        let Some(rollback_hash) = rollback_hash else {
            assessment.outcome = RollbackOutcome::Unavailable;
            assessment.detail = "no prior active hash available for rollback".to_string();
            return assessment;
        };

        match self.plan_slot_activation(slot, &rollback_hash, requested_contract_version) {
            Ok(_) => match self.prefetch_promoted_hash(slot, &rollback_hash) {
                Ok(()) => {
                    assessment.outcome = RollbackOutcome::Completed;
                    assessment.resulting_active_hash = Some(rollback_hash.clone());
                    assessment.detail = "rollback target is verified and warmable".to_string();
                }
                Err(err) => {
                    assessment.outcome = RollbackOutcome::Failed;
                    assessment.detail = format!("rollback prefetch failed: {err:?}");
                }
            },
            Err(err) => {
                assessment.outcome = RollbackOutcome::Failed;
                assessment.detail = format!("rollback activation blocked: {err:?}");
            }
        }
        assessment
    }

    pub fn slot_path_statuses(&self, slot: ModelSlot) -> Vec<SlotPathStatus> {
        let Some(spec) = self.specs.get(&slot) else {
            return vec![SlotPathStatus {
                target_state: SlotTargetState::Disabled,
                configured_hash: None,
                verified: false,
                comparable: false,
                detail: "slot missing from manifest spec map".to_string(),
            }];
        };
        if !spec.enabled {
            return vec![SlotPathStatus {
                target_state: SlotTargetState::Disabled,
                configured_hash: None,
                verified: false,
                comparable: false,
                detail: "slot disabled by manifest/env".to_string(),
            }];
        }
        let mut out = Vec::with_capacity(5);
        out.push(self.path_status_for(slot, SlotTargetState::Active, selected_hash(slot, spec)));
        out.push(self.path_status_for(
            slot,
            SlotTargetState::Candidate,
            std::env::var(format!("UCF_MODEL_CANDIDATE_{}", slot.env_key())).ok(),
        ));
        out.push(self.path_status_for(
            slot,
            SlotTargetState::Compare,
            std::env::var(format!("UCF_MODEL_COMPARE_{}", slot.env_key())).ok(),
        ));
        out.push(self.path_status_for(
            slot,
            SlotTargetState::Shadow,
            std::env::var(format!("UCF_MODEL_SHADOW_{}", slot.env_key())).ok(),
        ));
        let blocked = out
            .iter()
            .find(|entry| {
                matches!(
                    entry.target_state,
                    SlotTargetState::Active | SlotTargetState::Compare | SlotTargetState::Shadow
                ) && !entry.verified
                    && (entry.configured_hash.is_some()
                        || entry.target_state == SlotTargetState::Active)
            })
            .map(|entry| format!("{:?} blocked: {}", entry.target_state, entry.detail));
        out.push(SlotPathStatus {
            target_state: SlotTargetState::Blocked,
            configured_hash: None,
            verified: blocked.is_none(),
            comparable: false,
            detail: blocked.unwrap_or_else(|| "none".to_string()),
        });
        out
    }

    pub fn warmup_slot_paths(&self, slot: ModelSlot) -> Vec<SlotWarmupStatus> {
        let statuses = self.slot_path_statuses(slot);
        let mut out = Vec::with_capacity(statuses.len());
        for status in statuses {
            match status.target_state {
                SlotTargetState::Disabled => out.push(SlotWarmupStatus {
                    target_state: status.target_state,
                    state: SlotWarmupState::Blocked,
                    detail: status.detail,
                }),
                SlotTargetState::Blocked => out.push(SlotWarmupStatus {
                    target_state: status.target_state,
                    state: if status.verified {
                        SlotWarmupState::Prepared
                    } else {
                        SlotWarmupState::Blocked
                    },
                    detail: status.detail,
                }),
                SlotTargetState::Active => {
                    if status.verified {
                        match self.verify_slot(slot).and_then(|verified| {
                            self.read_verified_bytes(&verified).map(|_| verified)
                        }) {
                            Ok(_) => out.push(SlotWarmupStatus {
                                target_state: status.target_state,
                                state: SlotWarmupState::Warm,
                                detail: "artifact verified and prefetched".to_string(),
                            }),
                            Err(err) => out.push(SlotWarmupStatus {
                                target_state: status.target_state,
                                state: SlotWarmupState::Blocked,
                                detail: format!("warmup failed: {err:?}"),
                            }),
                        }
                    } else {
                        out.push(SlotWarmupStatus {
                            target_state: status.target_state,
                            state: SlotWarmupState::Cold,
                            detail: status.detail,
                        });
                    }
                }
                SlotTargetState::Candidate | SlotTargetState::Compare | SlotTargetState::Shadow => {
                    let state = if status.verified && status.configured_hash.is_some() {
                        SlotWarmupState::Prepared
                    } else if status.configured_hash.is_none() {
                        SlotWarmupState::Cold
                    } else {
                        SlotWarmupState::Blocked
                    };
                    out.push(SlotWarmupStatus {
                        target_state: status.target_state,
                        state,
                        detail: status.detail,
                    });
                }
                SlotTargetState::Discovered | SlotTargetState::Verified => {
                    out.push(SlotWarmupStatus {
                        target_state: status.target_state,
                        state: SlotWarmupState::Cold,
                        detail: status.detail,
                    })
                }
            }
        }
        out
    }

    pub fn slot_promotion_decision(&self, slot: ModelSlot) -> SlotPromotionDecision {
        let statuses = self.slot_path_statuses(slot);
        let warmup = self.warmup_slot_paths(slot);
        let active = statuses
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Active);
        let candidate = statuses
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Candidate);
        let compare = statuses
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Compare);
        let blocked = statuses
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Blocked);
        let active_warmup = warmup
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Active);
        let candidate_warmup = warmup
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Candidate);
        let compare_warmup = warmup
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Compare);
        let shadow = statuses
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Shadow);
        let shadow_warmup = warmup
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Shadow);

        let active_hash = active.and_then(|entry| entry.configured_hash.clone());
        let candidate_hash = candidate.and_then(|entry| entry.configured_hash.clone());
        let compare_hash = compare.and_then(|entry| entry.configured_hash.clone());
        let shadow_hash = shadow.and_then(|entry| entry.configured_hash.clone());

        let baseline_comparison_ready =
            compare.is_some_and(|entry| entry.verified && entry.configured_hash.is_some());
        let runtime_path_production_usable = active.is_some_and(|entry| entry.verified)
            && active_warmup.is_some_and(|entry| matches!(entry.state, SlotWarmupState::Warm));
        let readiness_ok = candidate_warmup
            .is_some_and(|entry| matches!(entry.state, SlotWarmupState::Prepared))
            || runtime_path_production_usable;
        let degraded_beyond_acceptable_threshold = compare_warmup
            .is_some_and(|entry| matches!(entry.state, SlotWarmupState::Blocked))
            && compare.is_some_and(|entry| entry.configured_hash.is_some());
        let compare_or_shadow_diagnostic_ready = compare
            .is_some_and(|entry| entry.verified && entry.configured_hash.is_some())
            || shadow.is_some_and(|entry| entry.verified && entry.configured_hash.is_some());
        let comparable_under_same_effective_configuration =
            candidate_hash.is_some() && compare_hash == candidate_hash && baseline_comparison_ready;
        let signals = PromotionTechnicalSignals {
            baseline_comparison_ready,
            runtime_path_production_usable,
            readiness_ok,
            degraded_beyond_acceptable_threshold,
            compare_or_shadow_diagnostic_ready,
            comparable_under_same_effective_configuration,
        };

        let mut blockers = Vec::new();
        let candidate_present = candidate.is_some_and(|entry| entry.configured_hash.is_some());
        if candidate_present && !candidate.is_some_and(|entry| entry.comparable) {
            blockers.push(PromotionBlockerCode::NotComparableYet);
        }
        if candidate_present && !baseline_comparison_ready {
            blockers.push(PromotionBlockerCode::InsufficientBaselineSignal);
        }
        if candidate_present && !runtime_path_production_usable {
            blockers.push(PromotionBlockerCode::RuntimePathNotProductionUsable);
        }
        if degraded_beyond_acceptable_threshold {
            blockers.push(PromotionBlockerCode::DegradedBeyondAcceptableThreshold);
        }

        let state = if runtime_path_production_usable {
            PromotionDecisionState::Active
        } else if candidate_present && blockers.is_empty() && readiness_ok {
            PromotionDecisionState::Promotable
        } else if candidate_present && !blockers.is_empty() {
            PromotionDecisionState::BlockedForPromotion
        } else if candidate.is_some_and(|entry| entry.comparable) {
            PromotionDecisionState::Comparable
        } else if candidate_present {
            PromotionDecisionState::Candidate
        } else {
            PromotionDecisionState::Known
        };

        let (context, caveat, blocker) = if !candidate_present {
            (
                CompareShadowContext::BlockedMissingSignals,
                None,
                Some("candidate path not configured".to_string()),
            )
        } else if candidate_hash == active_hash && candidate_hash.is_some() {
            (
                CompareShadowContext::ComparableWithCaveats,
                Some("candidate hash matches active reference hash".to_string()),
                None,
            )
        } else if comparable_under_same_effective_configuration {
            (
                CompareShadowContext::ComparableSameEffectiveConfiguration,
                None,
                None,
            )
        } else if candidate_hash.is_some()
            && compare_hash.is_some()
            && compare_hash != candidate_hash
            && compare.is_some_and(|entry| entry.verified)
        {
            (
                CompareShadowContext::NotComparableDifferentRuntimeContext,
                None,
                Some("compare path hash differs from candidate hash".to_string()),
            )
        } else if compare_hash.is_some() && !compare.is_some_and(|entry| entry.verified) {
            (
                CompareShadowContext::BlockedMissingSignals,
                None,
                Some("compare path configured but unavailable".to_string()),
            )
        } else if shadow_hash.is_some()
            && candidate_hash.is_some()
            && shadow_hash != candidate_hash
            && shadow.is_some_and(|entry| entry.verified)
        {
            (
                CompareShadowContext::NotComparableDifferentRuntimeContext,
                None,
                Some("shadow path hash differs from candidate hash".to_string()),
            )
        } else if shadow_hash.is_some() && !shadow.is_some_and(|entry| entry.verified) {
            (
                CompareShadowContext::BlockedMissingSignals,
                None,
                Some("shadow path configured but unavailable".to_string()),
            )
        } else if shadow_hash.is_some() && shadow_hash == candidate_hash {
            (
                CompareShadowContext::ComparableWithCaveats,
                Some("candidate compared via shadow path without compare path".to_string()),
                None,
            )
        } else {
            (
                CompareShadowContext::BlockedMissingSignals,
                None,
                Some("no compare/shadow path configured for candidate".to_string()),
            )
        };

        let compare_outcome = if baseline_comparison_ready
            && matches!(
                context,
                CompareShadowContext::ComparableSameEffectiveConfiguration
            ) {
            ComparePathOutcome::ComparedSuccessfully
        } else if compare_warmup
            .is_some_and(|entry| matches!(entry.state, SlotWarmupState::Blocked))
            && compare.is_some_and(|entry| entry.verified && entry.configured_hash.is_some())
        {
            ComparePathOutcome::ComparisonFailedTechnically
        } else if compare_hash.is_some() && !compare.is_some_and(|entry| entry.verified) {
            ComparePathOutcome::ComparisonBlocked
        } else if compare_hash.is_some() {
            ComparePathOutcome::ComparisonInconclusive
        } else {
            ComparePathOutcome::NotComparable
        };
        let shadow_outcome = if shadow_warmup
            .is_some_and(|entry| matches!(entry.state, SlotWarmupState::Prepared))
            && shadow_hash.is_some()
            && shadow_hash == candidate_hash
        {
            ShadowPathOutcome::ShadowedSuccessfully
        } else if shadow_warmup.is_some_and(|entry| matches!(entry.state, SlotWarmupState::Blocked))
            && shadow.is_some_and(|entry| entry.verified && entry.configured_hash.is_some())
        {
            ShadowPathOutcome::ShadowFailedTechnically
        } else if shadow_hash.is_some() && !shadow.is_some_and(|entry| entry.verified) {
            ShadowPathOutcome::ShadowBlocked
        } else if shadow_hash.is_some() {
            ShadowPathOutcome::ShadowInconclusive
        } else {
            ShadowPathOutcome::NotComparable
        };
        let compare_shadow = CompareShadowEvaluation {
            active_reference_hash: active_hash.clone(),
            candidate_hash: candidate_hash.clone(),
            compare_hash,
            shadow_hash,
            context,
            compare_outcome,
            shadow_outcome,
            caveat,
            blocker,
        };
        let disposition = if runtime_path_production_usable {
            PromotionEvaluationDisposition::ActivePathRemainsPreferred
        } else if candidate_present && !blockers.is_empty() {
            PromotionEvaluationDisposition::CandidateRemainsBlocked
        } else if candidate_present
            && (compare_outcome == ComparePathOutcome::ComparedSuccessfully
                || shadow_outcome == ShadowPathOutcome::ShadowedSuccessfully)
            && blockers.is_empty()
        {
            PromotionEvaluationDisposition::CandidateMorePromotable
        } else if candidate_present
            && (compare_outcome == ComparePathOutcome::ComparisonInconclusive
                || shadow_outcome == ShadowPathOutcome::ShadowInconclusive
                || compare_outcome == ComparePathOutcome::NotComparable)
        {
            PromotionEvaluationDisposition::CandidateComparisonInconclusive
        } else if candidate_present {
            PromotionEvaluationDisposition::CandidateRemainsBlocked
        } else {
            PromotionEvaluationDisposition::ActivePathRemainsPreferred
        };

        SlotPromotionDecision {
            slot,
            active_hash,
            candidate_hash,
            state,
            blockers,
            signals,
            compare_shadow,
            disposition,
            detail: blocked
                .map(|entry| entry.detail.clone())
                .unwrap_or_else(|| "none".to_string()),
        }
    }

    fn ensure_optional_path_verified(
        &self,
        slot: ModelSlot,
        path_kind: SlotTargetState,
    ) -> Result<(), ModelActivationError> {
        let env_key = match path_kind {
            SlotTargetState::Compare => format!("UCF_MODEL_COMPARE_{}", slot.env_key()),
            SlotTargetState::Shadow => format!("UCF_MODEL_SHADOW_{}", slot.env_key()),
            _ => return Ok(()),
        };
        let Some(hash) = std::env::var(env_key).ok().map(|v| v.trim().to_string()) else {
            return Ok(());
        };
        self.verify_promoted_hash(slot, &hash).map_err(|reason| {
            ModelActivationError::CompareShadowPathUnavailable {
                slot,
                path_kind,
                reason,
            }
        })?;
        Ok(())
    }

    fn active_hash_for_slot(&self, slot: ModelSlot) -> Option<String> {
        self.specs
            .get(&slot)
            .and_then(|spec| selected_hash(slot, spec))
    }

    fn verify_promoted_hash(&self, slot: ModelSlot, hash: &str) -> Result<(), ModelLoadError> {
        let Some(spec) = self.specs.get(&slot) else {
            return Err(ModelLoadError::Disabled);
        };
        let mut scoped = spec.clone();
        scoped.path = Some(PathBuf::from(format!(
            "promoted/{}/{}/model.safetensors",
            slot.as_str(),
            hash.trim()
        )));
        scoped.active_hash = None;
        scoped.expected_sha256 = parse_hash(hash);

        let mut scratch_specs = self.specs.clone();
        scratch_specs.insert(slot, scoped);
        let store = Self {
            allowlist_root: self.allowlist_root.clone(),
            specs: scratch_specs,
        };
        store.verify_slot(slot).map(|_| ())
    }

    fn prefetch_promoted_hash(&self, slot: ModelSlot, hash: &str) -> Result<(), ModelLoadError> {
        let Some(spec) = self.specs.get(&slot) else {
            return Err(ModelLoadError::Disabled);
        };
        let mut scoped = spec.clone();
        scoped.path = Some(PathBuf::from(format!(
            "promoted/{}/{}/model.safetensors",
            slot.as_str(),
            hash.trim()
        )));
        scoped.active_hash = None;
        scoped.expected_sha256 = parse_hash(hash);

        let mut scratch_specs = self.specs.clone();
        scratch_specs.insert(slot, scoped);
        let store = Self {
            allowlist_root: self.allowlist_root.clone(),
            specs: scratch_specs,
        };
        let verified = store.verify_slot(slot)?;
        store.read_verified_bytes(&verified).map(|_| ())
    }

    fn fallback_state_for_prior(
        &self,
        slot: ModelSlot,
        prior_active_hash: Option<&str>,
    ) -> ActivationFallbackState {
        let Some(prior_hash) = prior_active_hash else {
            return ActivationFallbackState::FallbackUnavailable;
        };
        if self.prefetch_promoted_hash(slot, prior_hash).is_ok() {
            ActivationFallbackState::FallbackToPriorActive
        } else {
            ActivationFallbackState::FallbackUnavailable
        }
    }

    fn path_status_for(
        &self,
        slot: ModelSlot,
        target_state: SlotTargetState,
        configured_hash: Option<String>,
    ) -> SlotPathStatus {
        let configured_hash = configured_hash
            .map(|v| v.trim().to_string())
            .filter(|v| !v.is_empty());
        let Some(hash) = configured_hash.clone() else {
            if target_state == SlotTargetState::Active {
                return match self.verify_slot(slot) {
                    Ok(_) => SlotPathStatus {
                        target_state,
                        configured_hash: None,
                        verified: true,
                        comparable: false,
                        detail: "verified".to_string(),
                    },
                    Err(err) => SlotPathStatus {
                        target_state,
                        configured_hash: None,
                        verified: false,
                        comparable: false,
                        detail: format!("{err:?}"),
                    },
                };
            }
            return SlotPathStatus {
                target_state,
                configured_hash: None,
                verified: false,
                comparable: false,
                detail: "not configured".to_string(),
            };
        };
        match self.verify_promoted_hash(slot, &hash) {
            Ok(()) => SlotPathStatus {
                target_state,
                configured_hash: Some(hash),
                verified: true,
                comparable: matches!(
                    target_state,
                    SlotTargetState::Candidate | SlotTargetState::Compare | SlotTargetState::Shadow
                ),
                detail: "verified".to_string(),
            },
            Err(err) => SlotPathStatus {
                target_state,
                configured_hash: Some(hash),
                verified: false,
                comparable: false,
                detail: format!("{err:?}"),
            },
        }
    }
}

fn selected_hash(slot: ModelSlot, spec: &ModelSlotSpec) -> Option<String> {
    std::env::var(format!("UCF_MODEL_PIN_{}", slot.env_key()))
        .ok()
        .filter(|v| !v.trim().is_empty())
        .or_else(|| spec.active_hash.clone())
}

#[derive(Debug, Deserialize, Default)]
struct ModelManifest {
    allowlist_root: Option<PathBuf>,
    slots: Option<ModelManifestSlots>,
}

#[derive(Debug, Deserialize, Default)]
struct ModelManifestSlots {
    llm: Option<ModelManifestSlotEntry>,
    world_jepa: Option<ModelManifestSlotEntry>,
    world_vljepa: Option<ModelManifestSlotEntry>,
    sae: Option<ModelManifestSlotEntry>,
    lfm: Option<ModelManifestSlotEntry>,
    ssm: Option<ModelManifestSlotEntry>,
    ebm_reasoner: Option<ModelManifestSlotEntry>,
}

#[derive(Debug, Deserialize, Clone)]
struct ModelManifestSlotEntry {
    enabled: Option<bool>,
    path: Option<PathBuf>,
    expected_sha256: Option<String>,
    max_bytes: Option<u64>,
    format: Option<ModelFormat>,
    device: Option<ModelDevice>,
    active_hash: Option<String>,
    contract_version: Option<String>,
}

impl ModelManifest {
    fn apply_env_overrides(&mut self) {
        for slot in ModelSlot::all() {
            let entry = self.entry_mut(slot);
            if let Ok(path) = std::env::var(format!("UCF_MODEL_{}_PATH", slot.env_key())) {
                entry.path = Some(PathBuf::from(path));
            }
            if let Ok(hash) = std::env::var(format!("UCF_MODEL_{}_SHA256", slot.env_key())) {
                entry.expected_sha256 = Some(hash);
            }
            if let Ok(max) = std::env::var(format!("UCF_MODEL_{}_MAX_BYTES", slot.env_key())) {
                entry.max_bytes = max.parse::<u64>().ok();
            }
            if let Ok(enabled) = std::env::var(format!("UCF_MODEL_{}_ENABLED", slot.env_key())) {
                entry.enabled = Some(matches!(enabled.as_str(), "1" | "true" | "TRUE"));
            }
        }
    }

    fn entry_mut(&mut self, slot: ModelSlot) -> &mut ModelManifestSlotEntry {
        let slots = self.slots.get_or_insert_with(ModelManifestSlots::default);
        match slot {
            ModelSlot::Llm => slots.llm.get_or_insert_with(default_entry),
            ModelSlot::WorldJepa => slots.world_jepa.get_or_insert_with(default_entry),
            ModelSlot::WorldVljepa => slots.world_vljepa.get_or_insert_with(default_entry),
            ModelSlot::Sae => slots.sae.get_or_insert_with(default_entry),
            ModelSlot::Lfm => slots.lfm.get_or_insert_with(default_entry),
            ModelSlot::Ssm => slots.ssm.get_or_insert_with(default_entry),
            ModelSlot::EbmReasoner => slots.ebm_reasoner.get_or_insert_with(default_entry),
        }
    }

    fn to_specs(&self) -> BTreeMap<ModelSlot, ModelSlotSpec> {
        let mut out = BTreeMap::new();
        for slot in ModelSlot::all() {
            let entry = self.entry(slot).cloned().unwrap_or_else(default_entry);
            let expected_sha256 = parse_hash(entry.expected_sha256.as_deref().unwrap_or(""));
            out.insert(
                slot,
                ModelSlotSpec {
                    slot,
                    enabled: entry.enabled.unwrap_or(false),
                    path: entry.path,
                    expected_sha256,
                    max_bytes: entry.max_bytes.unwrap_or(DEFAULT_MAX_BYTES),
                    format: entry.format.unwrap_or(ModelFormat::Custom),
                    device: entry.device.unwrap_or(ModelDevice::CpuOnly),
                    active_hash: entry.active_hash.clone(),
                    contract_version: entry.contract_version.clone(),
                },
            );
        }
        out
    }

    fn entry(&self, slot: ModelSlot) -> Option<&ModelManifestSlotEntry> {
        let slots = self.slots.as_ref()?;
        match slot {
            ModelSlot::Llm => slots.llm.as_ref(),
            ModelSlot::WorldJepa => slots.world_jepa.as_ref(),
            ModelSlot::WorldVljepa => slots.world_vljepa.as_ref(),
            ModelSlot::Sae => slots.sae.as_ref(),
            ModelSlot::Lfm => slots.lfm.as_ref(),
            ModelSlot::Ssm => slots.ssm.as_ref(),
            ModelSlot::EbmReasoner => slots.ebm_reasoner.as_ref(),
        }
    }
}

fn default_entry() -> ModelManifestSlotEntry {
    ModelManifestSlotEntry {
        enabled: Some(false),
        path: None,
        expected_sha256: None,
        max_bytes: Some(DEFAULT_MAX_BYTES),
        format: Some(ModelFormat::Custom),
        device: Some(ModelDevice::CpuOnly),
        active_hash: None,
        contract_version: None,
    }
}

fn parse_hash(value: &str) -> [u8; 32] {
    let trimmed = value.trim();
    let mut out = [0_u8; 32];
    if let Ok(decoded) = hex::decode(trimmed) {
        if decoded.len() == 32 {
            out.copy_from_slice(&decoded);
        }
    }
    out
}

impl From<ModelLoadError> for ComputeError {
    fn from(value: ModelLoadError) -> Self {
        ComputeError::InvalidInput {
            reason: format!("model load error: {value:?}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn hash_parse_bad_is_zero() {
        assert_eq!(parse_hash("abc"), [0; 32]);
    }

    #[test]
    fn detects_hash_mismatch() {
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let bad_hash = "0909090909090909090909090909090909090909090909090909090909090909";
        let model_path = models
            .join("promoted")
            .join("llm")
            .join(bad_hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, b"abc").expect("write");

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::Llm,
            ModelSlotSpec {
                slot: ModelSlot::Llm,
                enabled: true,
                path: None,
                expected_sha256: [9; 32],
                max_bytes: 1024,
                format: ModelFormat::Custom,
                device: ModelDevice::CpuOnly,
                active_hash: Some(bad_hash.to_string()),
                contract_version: None,
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };
        let err = store.verify_slot(ModelSlot::Llm).expect_err("must fail");
        assert!(matches!(err, ModelLoadError::HashMismatch { .. }));
    }

    #[test]
    fn enforces_size_limit() {
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let hash = "bef57ec7f53a6d40beb640a780a639c83bc29ac8a9816f1f6e8860d947f01831";
        let model_path = models
            .join("promoted")
            .join("llm")
            .join(hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, b"abcdef").expect("write");

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::Llm,
            ModelSlotSpec {
                slot: ModelSlot::Llm,
                enabled: true,
                path: None,
                expected_sha256: Sha256::digest(b"abcdef").into(),
                max_bytes: 3,
                format: ModelFormat::Custom,
                device: ModelDevice::CpuOnly,
                active_hash: Some(hash.to_string()),
                contract_version: None,
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };
        let err = store.verify_slot(ModelSlot::Llm).expect_err("must fail");
        assert!(matches!(err, ModelLoadError::Oversized { .. }));
    }

    #[test]
    fn enabled_slot_requires_nonzero_expected_hash() {
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let model_path = models
            .join("promoted")
            .join("ebm_reasoner")
            .join("zz")
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, b"abc").expect("write");

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::EbmReasoner,
            ModelSlotSpec {
                slot: ModelSlot::EbmReasoner,
                enabled: true,
                path: None,
                expected_sha256: [0; 32],
                max_bytes: 1024,
                format: ModelFormat::Custom,
                device: ModelDevice::CpuOnly,
                active_hash: Some("zz".to_string()),
                contract_version: None,
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };
        let err = store
            .verify_slot(ModelSlot::EbmReasoner)
            .expect_err("must fail");
        assert!(matches!(
            err,
            ModelLoadError::MissingExpectedHash {
                slot: ModelSlot::EbmReasoner
            }
        ));
    }

    #[test]
    fn default_manifest_path_is_lowercase_canonical() {
        assert_eq!(ModelStore::default_manifest_path(), "models/manifest.toml");
    }

    #[test]
    fn from_manifest_and_env_reads_lowercase_manifest_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path();

        let lower_models = root.join("lower").join("models");
        fs::create_dir_all(&lower_models).expect("mkdir lower models");
        let lowercase_manifest = lower_models.join("manifest.toml");
        fs::write(
            &lowercase_manifest,
            r#"
allowlist_root = "lower_models"
[slots.llm]
enabled = true
path = "llm.bin"
expected_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
max_bytes = 1024
format = "custom"
device = "cpu_only"
"#,
        )
        .expect("write lowercase manifest");

        let upper_models = root.join("upper").join("models");
        fs::create_dir_all(&upper_models).expect("mkdir upper models");
        fs::write(
            upper_models.join("MANIFEST.toml"),
            r#"
allowlist_root = "upper_models"
[slots.llm]
enabled = false
"#,
        )
        .expect("write uppercase manifest");

        let store = ModelStore::from_manifest_and_env(&lowercase_manifest).expect("load store");
        assert_eq!(
            store.allowlist_root,
            PathBuf::from("lower_models"),
            "lowercase manifest file must be honored"
        );
    }

    #[test]
    fn plan_slot_activation_accepts_verified_promoted_hash() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::remove_var("UCF_MODEL_COMPARE_WORLD_JEPA");
        std::env::remove_var("UCF_MODEL_SHADOW_WORLD_JEPA");
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let bytes = b"verified-model";
        let hash = hex::encode(Sha256::digest(bytes));
        let model_path = models
            .join("promoted")
            .join("world_jepa")
            .join(&hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, bytes).expect("write");

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::WorldJepa,
            ModelSlotSpec {
                slot: ModelSlot::WorldJepa,
                enabled: true,
                path: None,
                expected_sha256: parse_hash(&hash),
                max_bytes: 1024,
                format: ModelFormat::Burn,
                device: ModelDevice::CpuOnly,
                active_hash: Some(hash.clone()),
                contract_version: Some("v1".to_string()),
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };

        let plan = store
            .plan_slot_activation(ModelSlot::WorldJepa, &hash, Some("v1"))
            .expect("activation plan");
        assert_eq!(plan.slot, ModelSlot::WorldJepa);
        assert_eq!(plan.target_state, SlotTargetState::Active);
        assert_eq!(plan.target_hash, hash);
    }

    #[test]
    fn plan_slot_activation_rejects_incompatible_contract_version() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::remove_var("UCF_MODEL_COMPARE_SAE");
        std::env::remove_var("UCF_MODEL_SHADOW_SAE");
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let bytes = b"verified-model";
        let hash = hex::encode(Sha256::digest(bytes));
        let model_path = models
            .join("promoted")
            .join("sae")
            .join(&hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, bytes).expect("write");

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::Sae,
            ModelSlotSpec {
                slot: ModelSlot::Sae,
                enabled: true,
                path: None,
                expected_sha256: parse_hash(&hash),
                max_bytes: 1024,
                format: ModelFormat::Burn,
                device: ModelDevice::CpuOnly,
                active_hash: Some(hash.clone()),
                contract_version: Some("v2".to_string()),
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };

        let err = store
            .plan_slot_activation(ModelSlot::Sae, &hash, Some("v1"))
            .expect_err("must reject");
        assert!(matches!(
            err,
            ModelActivationError::IncompatiblePackContractBackend { .. }
        ));
    }

    #[test]
    fn slot_path_statuses_distinguish_candidate_compare_shadow_and_blocked() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::remove_var("UCF_MODEL_PIN_WORLD_JEPA");
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let bytes = b"verified-model";
        let hash = hex::encode(Sha256::digest(bytes));
        let model_path = models
            .join("promoted")
            .join("world_jepa")
            .join(&hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, bytes).expect("write");
        std::env::set_var("UCF_MODEL_CANDIDATE_WORLD_JEPA", &hash);
        std::env::set_var(
            "UCF_MODEL_COMPARE_WORLD_JEPA",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
        );

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::WorldJepa,
            ModelSlotSpec {
                slot: ModelSlot::WorldJepa,
                enabled: true,
                path: None,
                expected_sha256: parse_hash(&hash),
                max_bytes: 1024,
                format: ModelFormat::Burn,
                device: ModelDevice::CpuOnly,
                active_hash: Some(hash.clone()),
                contract_version: Some("v1".to_string()),
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };

        let statuses = store.slot_path_statuses(ModelSlot::WorldJepa);
        let candidate = statuses
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Candidate)
            .expect("candidate");
        assert!(candidate.verified);
        assert!(candidate.comparable);
        let compare = statuses
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Compare)
            .expect("compare");
        assert!(!compare.verified);
        assert!(!compare.comparable);
        let blocked = statuses
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Blocked)
            .expect("blocked");
        assert!(!blocked.verified);
        assert!(blocked.detail.contains("Compare"));

        std::env::remove_var("UCF_MODEL_CANDIDATE_WORLD_JEPA");
        std::env::remove_var("UCF_MODEL_COMPARE_WORLD_JEPA");
    }

    #[test]
    fn warmup_slot_paths_marks_active_warm_candidate_prepared_and_compare_blocked() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::remove_var("UCF_MODEL_PIN_WORLD_JEPA");
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let bytes = b"verified-model";
        let hash = hex::encode(Sha256::digest(bytes));
        let model_path = models
            .join("promoted")
            .join("world_jepa")
            .join(&hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, bytes).expect("write");
        std::env::set_var("UCF_MODEL_CANDIDATE_WORLD_JEPA", &hash);
        std::env::set_var(
            "UCF_MODEL_COMPARE_WORLD_JEPA",
            "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
        );

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::WorldJepa,
            ModelSlotSpec {
                slot: ModelSlot::WorldJepa,
                enabled: true,
                path: None,
                expected_sha256: parse_hash(&hash),
                max_bytes: 1024,
                format: ModelFormat::Burn,
                device: ModelDevice::CpuOnly,
                active_hash: Some(hash.clone()),
                contract_version: Some("v1".to_string()),
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };

        let warm = store.warmup_slot_paths(ModelSlot::WorldJepa);
        let active = warm
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Active)
            .expect("active");
        assert_eq!(active.state, SlotWarmupState::Warm);
        let candidate = warm
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Candidate)
            .expect("candidate");
        assert_eq!(candidate.state, SlotWarmupState::Prepared);
        let compare = warm
            .iter()
            .find(|entry| entry.target_state == SlotTargetState::Compare)
            .expect("compare");
        assert_eq!(compare.state, SlotWarmupState::Blocked);

        std::env::remove_var("UCF_MODEL_CANDIDATE_WORLD_JEPA");
        std::env::remove_var("UCF_MODEL_COMPARE_WORLD_JEPA");
    }

    #[test]
    fn promotion_decision_marks_active_when_runtime_path_is_warm() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::remove_var("UCF_MODEL_CANDIDATE_WORLD_JEPA");
        std::env::remove_var("UCF_MODEL_COMPARE_WORLD_JEPA");
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let bytes = b"verified-model";
        let hash = hex::encode(Sha256::digest(bytes));
        let model_path = models
            .join("promoted")
            .join("world_jepa")
            .join(&hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, bytes).expect("write");
        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::WorldJepa,
            ModelSlotSpec {
                slot: ModelSlot::WorldJepa,
                enabled: true,
                path: None,
                expected_sha256: parse_hash(&hash),
                max_bytes: 1024,
                format: ModelFormat::Burn,
                device: ModelDevice::CpuOnly,
                active_hash: Some(hash),
                contract_version: Some("v1".to_string()),
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };
        let decision = store.slot_promotion_decision(ModelSlot::WorldJepa);
        assert_eq!(decision.state, PromotionDecisionState::Active);
        assert!(decision.blockers.is_empty());
        assert!(decision.signals.runtime_path_production_usable);
    }

    #[test]
    fn promotion_decision_marks_candidate_blocked_when_baseline_or_runtime_signals_missing() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let bytes = b"verified-model";
        let hash = hex::encode(Sha256::digest(bytes));
        let model_path = models
            .join("promoted")
            .join("world_jepa")
            .join(&hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, bytes).expect("write");
        std::env::set_var("UCF_MODEL_CANDIDATE_WORLD_JEPA", &hash);
        std::env::remove_var("UCF_MODEL_COMPARE_WORLD_JEPA");
        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::WorldJepa,
            ModelSlotSpec {
                slot: ModelSlot::WorldJepa,
                enabled: true,
                path: None,
                expected_sha256: parse_hash(
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                ),
                max_bytes: 1024,
                format: ModelFormat::Burn,
                device: ModelDevice::CpuOnly,
                active_hash: Some(
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
                ),
                contract_version: Some("v1".to_string()),
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };
        let decision = store.slot_promotion_decision(ModelSlot::WorldJepa);
        assert_eq!(decision.state, PromotionDecisionState::BlockedForPromotion);
        assert!(decision
            .blockers
            .contains(&PromotionBlockerCode::InsufficientBaselineSignal));
        assert!(decision
            .blockers
            .contains(&PromotionBlockerCode::RuntimePathNotProductionUsable));
        assert_eq!(
            decision.compare_shadow.context,
            CompareShadowContext::BlockedMissingSignals
        );
        assert_eq!(
            decision.disposition,
            PromotionEvaluationDisposition::CandidateRemainsBlocked
        );
        std::env::remove_var("UCF_MODEL_CANDIDATE_WORLD_JEPA");
    }

    #[test]
    fn promotion_decision_marks_compare_success_with_same_candidate_context() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::remove_var("UCF_MODEL_PIN_WORLD_JEPA");
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let bytes = b"verified-model";
        let hash = hex::encode(Sha256::digest(bytes));
        let model_path = models
            .join("promoted")
            .join("world_jepa")
            .join(&hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, bytes).expect("write");
        std::env::set_var("UCF_MODEL_CANDIDATE_WORLD_JEPA", &hash);
        std::env::set_var("UCF_MODEL_COMPARE_WORLD_JEPA", &hash);

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::WorldJepa,
            ModelSlotSpec {
                slot: ModelSlot::WorldJepa,
                enabled: true,
                path: None,
                expected_sha256: parse_hash(
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
                ),
                max_bytes: 1024,
                format: ModelFormat::Burn,
                device: ModelDevice::CpuOnly,
                active_hash: Some(
                    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".to_string(),
                ),
                contract_version: Some("v1".to_string()),
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };
        let decision = store.slot_promotion_decision(ModelSlot::WorldJepa);
        assert_eq!(
            decision.compare_shadow.context,
            CompareShadowContext::ComparableSameEffectiveConfiguration
        );
        assert_eq!(
            decision.compare_shadow.compare_outcome,
            ComparePathOutcome::ComparedSuccessfully
        );
        assert!(decision.signals.compare_or_shadow_diagnostic_ready);
        assert!(
            decision
                .signals
                .comparable_under_same_effective_configuration
        );
        assert_eq!(
            decision.disposition,
            PromotionEvaluationDisposition::CandidateRemainsBlocked
        );
        std::env::remove_var("UCF_MODEL_CANDIDATE_WORLD_JEPA");
        std::env::remove_var("UCF_MODEL_COMPARE_WORLD_JEPA");
    }

    #[test]
    fn promotion_decision_marks_runtime_context_mismatch_as_not_comparable() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::remove_var("UCF_MODEL_PIN_WORLD_JEPA");
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let candidate_bytes = b"verified-model-candidate";
        let compare_bytes = b"verified-model-compare";
        let active_bytes = b"verified-model-active";
        let candidate_hash = hex::encode(Sha256::digest(candidate_bytes));
        let compare_hash = hex::encode(Sha256::digest(compare_bytes));
        let active_hash = hex::encode(Sha256::digest(active_bytes));
        for (hash, bytes) in [
            (candidate_hash.as_str(), candidate_bytes.as_slice()),
            (compare_hash.as_str(), compare_bytes.as_slice()),
            (active_hash.as_str(), active_bytes.as_slice()),
        ] {
            let model_path = models
                .join("promoted")
                .join("world_jepa")
                .join(hash)
                .join("model.safetensors");
            fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
            fs::write(&model_path, bytes).expect("write");
        }
        std::env::set_var("UCF_MODEL_CANDIDATE_WORLD_JEPA", &candidate_hash);
        std::env::set_var("UCF_MODEL_COMPARE_WORLD_JEPA", &compare_hash);

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::WorldJepa,
            ModelSlotSpec {
                slot: ModelSlot::WorldJepa,
                enabled: true,
                path: None,
                expected_sha256: parse_hash(&active_hash),
                max_bytes: 1024,
                format: ModelFormat::Burn,
                device: ModelDevice::CpuOnly,
                active_hash: Some(active_hash),
                contract_version: Some("v1".to_string()),
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };
        let decision = store.slot_promotion_decision(ModelSlot::WorldJepa);
        assert_eq!(
            decision.compare_shadow.context,
            CompareShadowContext::NotComparableDifferentRuntimeContext
        );
        assert_eq!(
            decision.compare_shadow.compare_outcome,
            ComparePathOutcome::ComparisonInconclusive
        );
        assert_eq!(
            decision.disposition,
            PromotionEvaluationDisposition::ActivePathRemainsPreferred
        );
        std::env::remove_var("UCF_MODEL_CANDIDATE_WORLD_JEPA");
        std::env::remove_var("UCF_MODEL_COMPARE_WORLD_JEPA");
    }

    #[test]
    fn activation_assessment_reports_succeeded_for_current_verified_active_hash() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::remove_var("UCF_MODEL_CANDIDATE_WORLD_JEPA");
        std::env::remove_var("UCF_MODEL_COMPARE_WORLD_JEPA");
        std::env::remove_var("UCF_MODEL_SHADOW_WORLD_JEPA");
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let bytes = b"verified-model-active";
        let hash = hex::encode(Sha256::digest(bytes));
        let model_path = models
            .join("promoted")
            .join("world_jepa")
            .join(&hash)
            .join("model.safetensors");
        fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&model_path, bytes).expect("write");

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::WorldJepa,
            ModelSlotSpec {
                slot: ModelSlot::WorldJepa,
                enabled: true,
                path: None,
                expected_sha256: parse_hash(&hash),
                max_bytes: 1024,
                format: ModelFormat::Burn,
                device: ModelDevice::CpuOnly,
                active_hash: Some(hash.clone()),
                contract_version: Some("v1".to_string()),
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };

        let assessment = store.assess_slot_activation(ModelSlot::WorldJepa, &hash, Some("v1"));
        assert_eq!(assessment.outcome, ActivationOutcome::Succeeded);
        assert_eq!(assessment.fallback, ActivationFallbackState::NotUsed);
        assert_eq!(assessment.prior_active_hash.as_deref(), Some(hash.as_str()));
        assert_eq!(
            assessment.resulting_active_hash.as_deref(),
            Some(hash.as_str())
        );
    }

    #[test]
    fn activation_assessment_distinguishes_blocked_and_fallback_to_prior_active() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        std::env::remove_var("UCF_MODEL_COMPARE_WORLD_JEPA");
        std::env::remove_var("UCF_MODEL_SHADOW_WORLD_JEPA");
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let active_bytes = b"verified-model-active";
        let active_hash = hex::encode(Sha256::digest(active_bytes));
        let active_path = models
            .join("promoted")
            .join("world_jepa")
            .join(&active_hash)
            .join("model.safetensors");
        fs::create_dir_all(active_path.parent().expect("parent")).expect("mkdirs");
        fs::write(&active_path, active_bytes).expect("write");

        let blocked_hash = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::WorldJepa,
            ModelSlotSpec {
                slot: ModelSlot::WorldJepa,
                enabled: true,
                path: None,
                expected_sha256: parse_hash(&active_hash),
                max_bytes: 1024,
                format: ModelFormat::Burn,
                device: ModelDevice::CpuOnly,
                active_hash: Some(active_hash.clone()),
                contract_version: Some("v1".to_string()),
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };

        let assessment =
            store.assess_slot_activation(ModelSlot::WorldJepa, blocked_hash, Some("v1"));
        assert_eq!(assessment.outcome, ActivationOutcome::Blocked);
        assert_eq!(
            assessment.fallback,
            ActivationFallbackState::FallbackToPriorActive
        );
        assert_eq!(
            assessment.resulting_active_hash.as_deref(),
            Some(active_hash.as_str())
        );
    }

    #[test]
    fn activation_assessment_distinguishes_degraded_from_rollback_semantics() {
        let _lock = crate::test_env::env_lock()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let temp = tempfile::tempdir().expect("tempdir");
        let models = temp.path().join("models");
        fs::create_dir_all(&models).expect("models");
        let active_bytes = b"verified-model-active";
        let active_hash = hex::encode(Sha256::digest(active_bytes));
        let candidate_bytes = b"verified-model-candidate";
        let candidate_hash = hex::encode(Sha256::digest(candidate_bytes));
        for (hash, bytes) in [
            (active_hash.as_str(), active_bytes.as_slice()),
            (candidate_hash.as_str(), candidate_bytes.as_slice()),
        ] {
            let model_path = models
                .join("promoted")
                .join("world_jepa")
                .join(hash)
                .join("model.safetensors");
            fs::create_dir_all(model_path.parent().expect("parent")).expect("mkdirs");
            fs::write(&model_path, bytes).expect("write");
        }
        std::env::set_var("UCF_MODEL_CANDIDATE_WORLD_JEPA", &candidate_hash);
        std::env::remove_var("UCF_MODEL_COMPARE_WORLD_JEPA");
        std::env::remove_var("UCF_MODEL_SHADOW_WORLD_JEPA");

        let mut specs = BTreeMap::new();
        specs.insert(
            ModelSlot::WorldJepa,
            ModelSlotSpec {
                slot: ModelSlot::WorldJepa,
                enabled: true,
                path: None,
                expected_sha256: parse_hash(&active_hash),
                max_bytes: 1024,
                format: ModelFormat::Burn,
                device: ModelDevice::CpuOnly,
                active_hash: Some(active_hash.clone()),
                contract_version: Some("v1".to_string()),
            },
        );
        let store = ModelStore {
            allowlist_root: models,
            specs,
        };

        let activation =
            store.assess_slot_activation(ModelSlot::WorldJepa, &candidate_hash, Some("v1"));
        assert_eq!(activation.outcome, ActivationOutcome::Degraded);
        assert_eq!(
            activation.fallback,
            ActivationFallbackState::FallbackToPriorActive
        );
        assert_eq!(activation.rollback, RollbackOutcome::NotRequested);

        let rollback = store.assess_slot_rollback(ModelSlot::WorldJepa, None, Some("v1"));
        assert_eq!(rollback.outcome, RollbackOutcome::Completed);
        assert_eq!(
            rollback.rollback_hash.as_deref(),
            Some(active_hash.as_str())
        );

        std::env::remove_var("UCF_MODEL_CANDIDATE_WORLD_JEPA");
    }
}
