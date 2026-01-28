#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

const SNAPSHOT_DOMAIN: &[u8] = b"ucf.params.snapshot.v1";
const SNAPSHOT_DELTA_DOMAIN: &[u8] = b"ucf.params.snapshot.delta.v1";
const SNAPSHOT_CHAIN_DOMAIN: &[u8] = b"ucf.params.snapshot.chain.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ParamTargetId(pub u16);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamSnapshotDelta {
    pub target: ParamTargetId,
    pub delta: i16,
    pub commit: Digest32,
}

impl ParamSnapshotDelta {
    pub fn new(target: ParamTargetId, delta: i16) -> Self {
        let commit = commit_snapshot_delta(target, delta);
        Self {
            target,
            delta,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ParamSnapshot {
    pub cycle_id: u64,
    pub onn_params_commit: Digest32,
    pub tcf_params_commit: Digest32,
    pub ncde_params_commit: Digest32,
    pub cde_params_commit: Digest32,
    pub feature_params_commit: Digest32,
    pub deltas: Vec<ParamSnapshotDelta>,
    pub applied_root: Digest32,
    pub commit: Digest32,
}

impl ParamSnapshot {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        cycle_id: u64,
        onn_params_commit: Digest32,
        tcf_params_commit: Digest32,
        ncde_params_commit: Digest32,
        cde_params_commit: Digest32,
        feature_params_commit: Digest32,
        mut deltas: Vec<ParamSnapshotDelta>,
        applied_root: Digest32,
    ) -> Self {
        deltas.sort_by(|a, b| a.target.cmp(&b.target));
        let commit = commit_snapshot(
            cycle_id,
            onn_params_commit,
            tcf_params_commit,
            ncde_params_commit,
            cde_params_commit,
            feature_params_commit,
            &deltas,
            applied_root,
        );
        Self {
            cycle_id,
            onn_params_commit,
            tcf_params_commit,
            ncde_params_commit,
            cde_params_commit,
            feature_params_commit,
            deltas,
            applied_root,
            commit,
        }
    }
}

pub fn commit_snapshot_chain(prev_snapshot: Digest32, current_snapshot: Digest32) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SNAPSHOT_CHAIN_DOMAIN);
    hasher.update(prev_snapshot.as_bytes());
    hasher.update(current_snapshot.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn commit_snapshot_delta(target: ParamTargetId, delta: i16) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SNAPSHOT_DELTA_DOMAIN);
    hasher.update(&target.0.to_be_bytes());
    hasher.update(&delta.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[allow(clippy::too_many_arguments)]
fn commit_snapshot(
    cycle_id: u64,
    onn_params_commit: Digest32,
    tcf_params_commit: Digest32,
    ncde_params_commit: Digest32,
    cde_params_commit: Digest32,
    feature_params_commit: Digest32,
    deltas: &[ParamSnapshotDelta],
    applied_root: Digest32,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(SNAPSHOT_DOMAIN);
    hasher.update(&cycle_id.to_be_bytes());
    hasher.update(onn_params_commit.as_bytes());
    hasher.update(tcf_params_commit.as_bytes());
    hasher.update(ncde_params_commit.as_bytes());
    hasher.update(cde_params_commit.as_bytes());
    hasher.update(feature_params_commit.as_bytes());
    hasher.update(&(deltas.len() as u16).to_be_bytes());
    for delta in deltas {
        hasher.update(delta.commit.as_bytes());
    }
    hasher.update(applied_root.as_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn snapshot_commit_is_deterministic() {
        let delta = ParamSnapshotDelta::new(ParamTargetId(1), 50);
        let snapshot_a = ParamSnapshot::new(
            7,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            vec![delta.clone()],
            Digest32::new([9u8; 32]),
        );
        let snapshot_b = ParamSnapshot::new(
            7,
            Digest32::new([1u8; 32]),
            Digest32::new([2u8; 32]),
            Digest32::new([3u8; 32]),
            Digest32::new([4u8; 32]),
            Digest32::new([5u8; 32]),
            vec![delta],
            Digest32::new([9u8; 32]),
        );
        assert_eq!(snapshot_a.commit, snapshot_b.commit);
    }
}
