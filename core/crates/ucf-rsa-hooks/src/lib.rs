#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_tcf_port::PulseKind;
use ucf_types::Digest32;

const DOMAIN_PROPOSAL_ID: &[u8] = b"ucf.rsa_hooks.proposal_id.v1";
const DOMAIN_PROPOSAL_COMMIT: &[u8] = b"ucf.rsa_hooks.proposal_commit.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaContext {
    pub cycle_id: u64,
    pub pulse_kind: PulseKind,
    pub phi: u16,
    pub surprise_score: u16,
    pub workspace_commit: Digest32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RsaProposal {
    pub id: Digest32,
    pub target: String,
    pub expected_gain: u16,
    pub risks: u16,
    pub commit: Digest32,
}

impl RsaProposal {
    pub fn new(id: Digest32, target: String, expected_gain: u16, risks: u16) -> Self {
        let commit = proposal_commit(&id, &target, expected_gain, risks);
        Self {
            id,
            target,
            expected_gain,
            risks,
            commit,
        }
    }
}

pub trait RsaHook: Send + Sync {
    fn propose(&self, context: &RsaContext) -> Vec<RsaProposal>;
}

#[derive(Clone, Debug)]
pub struct MockRsaHook {
    pub phi_threshold: u16,
    pub surprise_threshold: u16,
}

impl Default for MockRsaHook {
    fn default() -> Self {
        Self {
            phi_threshold: 3200,
            surprise_threshold: 7000,
        }
    }
}

impl MockRsaHook {
    pub fn new() -> Self {
        Self::default()
    }

    fn build_proposal(&self, context: &RsaContext, index: u16, target: &str) -> RsaProposal {
        let mut hasher = Hasher::new();
        hasher.update(DOMAIN_PROPOSAL_ID);
        hasher.update(&context.cycle_id.to_be_bytes());
        hasher.update(context.workspace_commit.as_bytes());
        hasher.update(&context.phi.to_be_bytes());
        hasher.update(&context.surprise_score.to_be_bytes());
        hasher.update(&index.to_be_bytes());
        hasher.update(target.as_bytes());
        let id = Digest32::new(*hasher.finalize().as_bytes());
        let expected_gain = (context.surprise_score / 3).saturating_add(200);
        let risks = context.phi.saturating_add(index * 10);
        RsaProposal::new(id, target.to_string(), expected_gain, risks)
    }

    fn should_emit(&self, context: &RsaContext) -> bool {
        matches!(context.pulse_kind, PulseKind::Sleep)
            && context.phi < self.phi_threshold
            && context.surprise_score >= self.surprise_threshold
    }
}

impl RsaHook for MockRsaHook {
    fn propose(&self, context: &RsaContext) -> Vec<RsaProposal> {
        if !self.should_emit(context) {
            return Vec::new();
        }

        vec![
            self.build_proposal(context, 0, "config.attention.replay_bias"),
            self.build_proposal(context, 1, "config.workspace.broadcast_cap"),
        ]
    }
}

fn proposal_commit(id: &Digest32, target: &str, expected_gain: u16, risks: u16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_PROPOSAL_COMMIT);
    hasher.update(id.as_bytes());
    hasher.update(&expected_gain.to_be_bytes());
    hasher.update(&risks.to_be_bytes());
    hasher.update(&u16::try_from(target.len()).unwrap_or(0).to_be_bytes());
    hasher.update(target.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mock_hook_emits_only_on_sleep_with_low_phi_and_high_surprise() {
        let hook = MockRsaHook::default();
        let base_context = RsaContext {
            cycle_id: 12,
            pulse_kind: PulseKind::Sleep,
            phi: 2000,
            surprise_score: 9000,
            workspace_commit: Digest32::new([3u8; 32]),
        };

        let proposals = hook.propose(&base_context);
        assert!(!proposals.is_empty());

        let mut other = base_context.clone();
        other.pulse_kind = PulseKind::Verify;
        assert!(hook.propose(&other).is_empty());

        let mut other = base_context.clone();
        other.phi = 9000;
        assert!(hook.propose(&other).is_empty());

        let mut other = base_context.clone();
        other.surprise_score = 1000;
        assert!(hook.propose(&other).is_empty());
    }
}
