#![forbid(unsafe_code)]

use blake3::Hasher;
use ucf_types::Digest32;

const ANCHOR_ID_DOMAIN: &[u8] = b"ucf.ism.anchor.id.v1";
const ANCHOR_COMMIT_DOMAIN: &[u8] = b"ucf.ism.anchor.commit.v1";
const ROOT_COMMIT_DOMAIN: &[u8] = b"ucf.ism.root.commit.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IsmAnchor {
    pub anchor_id: Digest32,
    pub source_macro: Digest32,
    pub traits_commit: Digest32,
    pub policy_class: u16,
    pub created_cycle: u64,
    pub commit: Digest32,
}

impl IsmAnchor {
    pub fn new(
        source_macro: Digest32,
        traits_commit: Digest32,
        policy_class: u16,
        created_cycle: u64,
    ) -> Self {
        let anchor_id = hash_anchor_id(source_macro, traits_commit, policy_class, created_cycle);
        let commit = hash_anchor_commit(
            anchor_id,
            source_macro,
            traits_commit,
            policy_class,
            created_cycle,
        );
        Self {
            anchor_id,
            source_macro,
            traits_commit,
            policy_class,
            created_cycle,
            commit,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IsmStore {
    anchors: Vec<IsmAnchor>,
    cap: usize,
}

impl IsmStore {
    pub fn new(cap: usize) -> Self {
        Self {
            anchors: Vec::new(),
            cap,
        }
    }

    pub fn append(&mut self, anchor: IsmAnchor) {
        if self.cap == 0 {
            return;
        }
        self.anchors.push(anchor);
        while self.anchors.len() > self.cap {
            self.anchors.remove(0);
        }
    }

    pub fn anchors(&self) -> &[IsmAnchor] {
        &self.anchors
    }

    pub fn root_commit(&self) -> Digest32 {
        let mut hasher = Hasher::new();
        hasher.update(ROOT_COMMIT_DOMAIN);
        for anchor in &self.anchors {
            hasher.update(anchor.commit.as_bytes());
        }
        Digest32::new(*hasher.finalize().as_bytes())
    }
}

fn hash_anchor_id(
    source_macro: Digest32,
    traits_commit: Digest32,
    policy_class: u16,
    created_cycle: u64,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(ANCHOR_ID_DOMAIN);
    hasher.update(source_macro.as_bytes());
    hasher.update(traits_commit.as_bytes());
    hasher.update(&policy_class.to_be_bytes());
    hasher.update(&created_cycle.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

fn hash_anchor_commit(
    anchor_id: Digest32,
    source_macro: Digest32,
    traits_commit: Digest32,
    policy_class: u16,
    created_cycle: u64,
) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(ANCHOR_COMMIT_DOMAIN);
    hasher.update(anchor_id.as_bytes());
    hasher.update(source_macro.as_bytes());
    hasher.update(traits_commit.as_bytes());
    hasher.update(&policy_class.to_be_bytes());
    hasher.update(&created_cycle.to_be_bytes());
    Digest32::new(*hasher.finalize().as_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_commit_deterministic_for_same_sequence() {
        let mut store_a = IsmStore::new(8);
        let mut store_b = IsmStore::new(8);
        let anchor_a = IsmAnchor::new(Digest32::new([1u8; 32]), Digest32::new([2u8; 32]), 3, 10);
        let anchor_b = IsmAnchor::new(Digest32::new([4u8; 32]), Digest32::new([5u8; 32]), 6, 11);
        store_a.append(anchor_a);
        store_a.append(anchor_b);
        store_b.append(anchor_a);
        store_b.append(anchor_b);

        assert_eq!(store_a.root_commit(), store_b.root_commit());
    }
}
