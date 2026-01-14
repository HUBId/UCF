#![forbid(unsafe_code)]

use std::collections::VecDeque;
use std::sync::Mutex;

use blake3::Hasher;
use ucf_cde_port::CdeHypothesis;
use ucf_nsr_port::NsrReport;
use ucf_types::{AiOutput, Digest32, OutputChannel};

const DOMAIN_INPUT: &[u8] = b"ucf.sle.input.v1";
const DOMAIN_OUTPUT: &[u8] = b"ucf.sle.output.v1";
const DOMAIN_REPORT: &[u8] = b"ucf.sle.report.v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoopFrame {
    pub input_commit: Digest32,
    pub output_commit: Digest32,
    pub report_commit: Digest32,
}

impl LoopFrame {
    pub fn seed(digest: Digest32) -> Self {
        Self {
            input_commit: digest,
            output_commit: digest,
            report_commit: digest,
        }
    }
}

#[derive(Debug)]
pub struct StrangeLoopEngine {
    pub depth: u8,
    frames: Mutex<VecDeque<LoopFrame>>,
}

impl StrangeLoopEngine {
    pub fn new(depth: u8) -> Self {
        Self {
            depth,
            frames: Mutex::new(VecDeque::new()),
        }
    }

    pub fn latest(&self) -> Option<LoopFrame> {
        self.frames.lock().ok()?.back().cloned()
    }

    pub fn reflect(
        &self,
        prev: &LoopFrame,
        current_output: &AiOutput,
        nsr_report: Option<&NsrReport>,
        cde_hyp: Option<&CdeHypothesis>,
    ) -> LoopFrame {
        let input_commit = hash_with_domain(DOMAIN_INPUT, |hasher| {
            hasher.update(prev.input_commit.as_bytes());
            hasher.update(prev.output_commit.as_bytes());
            hasher.update(prev.report_commit.as_bytes());
            encode_output(hasher, current_output);
        });

        let output_commit = hash_with_domain(DOMAIN_OUTPUT, |hasher| {
            hasher.update(input_commit.as_bytes());
            encode_output(hasher, current_output);
        });

        let report_commit = hash_with_domain(DOMAIN_REPORT, |hasher| {
            hasher.update(output_commit.as_bytes());
            encode_nsr_report(hasher, nsr_report);
            encode_cde_hypothesis(hasher, cde_hyp);
        });

        let frame = LoopFrame {
            input_commit,
            output_commit,
            report_commit,
        };

        if self.depth > 0 {
            if let Ok(mut frames) = self.frames.lock() {
                frames.push_back(frame.clone());
                let max_len = usize::from(self.depth);
                while frames.len() > max_len {
                    frames.pop_front();
                }
            }
        }

        frame
    }
}

fn hash_with_domain(domain: &[u8], f: impl FnOnce(&mut Hasher)) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(domain);
    f(&mut hasher);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn encode_output(hasher: &mut Hasher, output: &AiOutput) {
    let channel_tag: u8 = match output.channel {
        OutputChannel::Thought => 0,
        OutputChannel::Speech => 1,
    };
    hasher.update(&[channel_tag]);
    hasher.update(&output.confidence.to_be_bytes());
    hasher.update(
        &u64::try_from(output.content.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    hasher.update(output.content.as_bytes());
    match output.rationale_commit {
        Some(commit) => {
            hasher.update(&[1]);
            hasher.update(commit.as_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
    match output.integration_score {
        Some(score) => {
            hasher.update(&[1]);
            hasher.update(&score.to_be_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn encode_nsr_report(hasher: &mut Hasher, report: Option<&NsrReport>) {
    match report {
        Some(report) => {
            hasher.update(&[1]);
            hasher.update(&[report.ok as u8]);
            hasher.update(
                &u64::try_from(report.violations.len())
                    .unwrap_or(0)
                    .to_be_bytes(),
            );
            for violation in &report.violations {
                hasher.update(violation.as_bytes());
            }
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

fn encode_cde_hypothesis(hasher: &mut Hasher, hyp: Option<&CdeHypothesis>) {
    match hyp {
        Some(hyp) => {
            hasher.update(&[1]);
            hasher.update(hyp.digest.as_bytes());
            hasher.update(&u64::try_from(hyp.nodes).unwrap_or(0).to_be_bytes());
            hasher.update(&u64::try_from(hyp.edges).unwrap_or(0).to_be_bytes());
        }
        None => {
            hasher.update(&[0]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reflect_is_deterministic() {
        let engine = StrangeLoopEngine::new(4);
        let prev = LoopFrame::seed(Digest32::new([1u8; 32]));
        let output = AiOutput {
            channel: OutputChannel::Thought,
            content: "ok".to_string(),
            confidence: 900,
            rationale_commit: Some(Digest32::new([2u8; 32])),
            integration_score: Some(1234),
        };
        let report = NsrReport {
            ok: true,
            violations: vec!["rule-a".to_string()],
        };
        let hyp = CdeHypothesis {
            digest: Digest32::new([3u8; 32]),
            nodes: 2,
            edges: 1,
        };

        let first = engine.reflect(&prev, &output, Some(&report), Some(&hyp));
        let second = engine.reflect(&prev, &output, Some(&report), Some(&hyp));

        assert_eq!(first, second);
        let latest = engine.latest().expect("frame stored");
        assert_eq!(latest, second);
    }
}
