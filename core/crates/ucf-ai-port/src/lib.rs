#![forbid(unsafe_code)]

use std::sync::{Arc, Mutex};

use blake3::Hasher;
#[cfg(feature = "ai-runtime")]
use ucf_ai_runtime::backend::{AiRuntimeBackend, NoopRuntimeBackend};
use ucf_attn_controller::AttentionWeights;
#[cfg(feature = "digitalbrain")]
use ucf_bus::{BusPublisher, MessageEnvelope};
use ucf_cde_port::{CdeHypothesis, CdePort};
use ucf_digitalbrain_port::BrainError;
#[cfg(feature = "digitalbrain")]
use ucf_digitalbrain_port::{BrainStimEvent, BrainStimulus, MappingTable, Spike};
use ucf_feature_translator::ActivationView;
use ucf_iit_monitor::IitMonitor;
use ucf_lens_port::{LensMock, LensPort};
use ucf_ncde_port::{NcdeContext, NcdePort};
use ucf_nsr_port::{NsrPort, NsrReport};
use ucf_policy_ecology::{PolicyEcology, PolicyRule};
use ucf_sae_port::{SaeMock, SaePort, SparseFeature};
use ucf_sandbox::ControlFrameNormalized;
use ucf_scm_port::{CounterfactualQuery, Intervention, ScmDag, ScmPort};
use ucf_sle::{LoopFrame, StrangeLoopEngine};
use ucf_ssm_port::{SsmInput, SsmPort, SsmState};
use ucf_tcf_port::{idle_attention, TcfPort};
use ucf_types::v1::spec::DecisionKind;
use ucf_types::{
    CausalCounterfactual, CausalGraphStub, CausalIntervention, CausalReport, Claim, Digest32,
    EvidenceId, SymbolicClaims, ThoughtVec, WorldStateVec,
};

pub use ucf_types::{AiOutput, OutputChannel};

pub trait AiPort {
    fn infer(&self, input: &ControlFrameNormalized) -> Vec<AiOutput>;

    fn infer_with_context(&self, input: &ControlFrameNormalized) -> AiInference {
        AiInference::new(self.infer(input))
    }

    fn update_attention(&self, _weights: &AttentionWeights) {}
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct AiInference {
    pub outputs: Vec<AiOutput>,
    pub nsr_report: Option<NsrReport>,
    pub scm_dag: Option<ScmDag>,
    pub cde_confidence: Option<u16>,
    pub ssm_state: Option<SsmState>,
    pub activation_view: Option<ActivationView>,
}

impl AiInference {
    pub fn new(outputs: Vec<AiOutput>) -> Self {
        Self {
            outputs,
            nsr_report: None,
            scm_dag: None,
            cde_confidence: None,
            ssm_state: None,
            activation_view: None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputSuppressed {
    pub channel: OutputChannel,
    pub reason_digest: Digest32,
    pub risk: u16,
}

pub trait OutputSuppressionSink {
    fn publish(&self, event: OutputSuppressed);
}

#[derive(Clone)]
pub struct AiPillars {
    pub tcf: Option<Arc<Mutex<dyn TcfPort + Send + Sync>>>,
    pub ssm: Option<Arc<Mutex<dyn SsmPort + Send + Sync>>>,
    pub cde: Option<Arc<dyn CdePort + Send + Sync>>,
    pub scm: Option<Arc<Mutex<dyn ScmPort + Send + Sync>>>,
    pub nsr: Option<Arc<NsrPort>>,
    pub ncde: Option<Arc<dyn NcdePort + Send + Sync>>,
    pub sle: Option<Arc<StrangeLoopEngine>>,
    pub iit_monitor: Option<Arc<Mutex<IitMonitor>>>,
    pub lens: Option<Arc<dyn LensPort + Send + Sync>>,
    pub sae: Option<Arc<dyn SaePort + Send + Sync>>,
    pub digital_brain: Option<Arc<DigitalBrainBridge>>,
}

impl AiPillars {
    fn enabled(&self) -> bool {
        self.tcf.is_some()
            || self.ssm.is_some()
            || self.cde.is_some()
            || self.scm.is_some()
            || self.nsr.is_some()
            || self.ncde.is_some()
            || self.sle.is_some()
            || self.iit_monitor.is_some()
    }
}

#[cfg(feature = "digitalbrain")]
const DEFAULT_SPIKE_WIDTH_US: u16 = 100;

#[cfg(feature = "digitalbrain")]
fn weight_to_amplitude(weight: i16) -> u16 {
    let magnitude = weight.checked_abs().unwrap_or(i16::MAX);
    magnitude as u16
}

impl Default for AiPillars {
    fn default() -> Self {
        Self {
            tcf: None,
            ssm: None,
            cde: None,
            scm: None,
            nsr: None,
            ncde: None,
            sle: None,
            iit_monitor: None,
            lens: Some(Arc::new(LensMock::new())),
            sae: Some(Arc::new(SaeMock::new())),
            digital_brain: None,
        }
    }
}

#[cfg(feature = "digitalbrain")]
#[derive(Clone)]
pub struct DigitalBrainBridge {
    mapping: MappingTable,
    publisher: Arc<dyn BusPublisher<MessageEnvelope<BrainStimEvent>> + Send + Sync>,
}

#[cfg(feature = "digitalbrain")]
impl DigitalBrainBridge {
    pub fn new(
        mapping: MappingTable,
        publisher: Arc<dyn BusPublisher<MessageEnvelope<BrainStimEvent>> + Send + Sync>,
    ) -> Self {
        Self { mapping, publisher }
    }

    pub fn mirror_features(
        &self,
        evidence_id: EvidenceId,
        features: &[SparseFeature],
    ) -> Result<(), BrainError> {
        let mut spikes: Vec<Spike> = Vec::new();
        for feature in features {
            let coords = self.mapping.resolve(&feature.id);
            if coords.is_empty() {
                continue;
            }
            let amplitude = weight_to_amplitude(feature.weight);
            if amplitude == 0 {
                continue;
            }
            for coord in coords {
                if let Some(existing) = spikes.iter_mut().find(|spike| spike.coord == *coord) {
                    existing.amplitude = existing.amplitude.saturating_add(amplitude);
                } else {
                    spikes.push(Spike {
                        coord: *coord,
                        amplitude,
                        width_us: DEFAULT_SPIKE_WIDTH_US,
                    });
                }
            }
        }
        if spikes.is_empty() {
            return Ok(());
        }
        let rationale = features
            .first()
            .map(|feature| feature.id)
            .unwrap_or_default();
        let stim = BrainStimulus {
            spikes,
            evidence_id,
            rationale,
        };
        let event = BrainStimEvent { stim };
        self.publisher.publish(MessageEnvelope {
            node_id: ucf_types::NodeId::new("ai-port"),
            stream_id: ucf_types::StreamId::new("digitalbrain"),
            logical_time: ucf_types::LogicalTime::new(0),
            wall_time: ucf_types::WallTime::new(0),
            payload: event,
        });
        Ok(())
    }
}

#[cfg(not(feature = "digitalbrain"))]
#[derive(Clone, Default)]
pub struct DigitalBrainBridge;

#[cfg(not(feature = "digitalbrain"))]
impl DigitalBrainBridge {
    pub fn mirror_features(
        &self,
        _evidence_id: EvidenceId,
        _features: &[SparseFeature],
    ) -> Result<(), BrainError> {
        Err(BrainError::FeatureDisabled)
    }
}

#[derive(Clone)]
pub struct MockAiPort {
    pillars: AiPillars,
    attention: Arc<Mutex<Option<AttentionWeights>>>,
}

impl MockAiPort {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_pillars(pillars: AiPillars) -> Self {
        Self {
            pillars,
            attention: Arc::new(Mutex::new(None)),
        }
    }

    fn set_attention(&self, weights: AttentionWeights) {
        if let Ok(mut guard) = self.attention.lock() {
            *guard = Some(weights);
        }
    }

    fn attention_snapshot(&self) -> Option<AttentionWeights> {
        self.attention.lock().ok().and_then(|guard| guard.clone())
    }
}

#[cfg(feature = "ai-runtime")]
pub fn noop_runtime_backend(mock: MockAiPort) -> NoopRuntimeBackend {
    let delegate = Arc::new(move |cf: &ControlFrameNormalized| mock.infer(cf));
    NoopRuntimeBackend::new(delegate)
}

#[derive(Clone)]
pub struct AiOrchestrator {
    mock: MockAiPort,
    #[cfg(feature = "ai-runtime")]
    runtime_backend: Option<Arc<dyn AiRuntimeBackend + Send + Sync>>,
}

impl AiOrchestrator {
    pub fn new(mock: MockAiPort) -> Self {
        Self {
            mock,
            #[cfg(feature = "ai-runtime")]
            runtime_backend: None,
        }
    }

    pub fn with_pillars(pillars: AiPillars) -> Self {
        Self::new(MockAiPort::with_pillars(pillars))
    }

    #[cfg(feature = "ai-runtime")]
    pub fn with_runtime_backend(
        mut self,
        backend: Arc<dyn AiRuntimeBackend + Send + Sync>,
    ) -> Self {
        self.runtime_backend = Some(backend);
        self
    }

    #[cfg(feature = "ai-runtime")]
    pub fn with_noop_runtime(mut self) -> Self {
        let backend = noop_runtime_backend(self.mock.clone());
        self.runtime_backend = Some(Arc::new(backend));
        self
    }
}

impl Default for AiOrchestrator {
    fn default() -> Self {
        Self::new(MockAiPort::new())
    }
}

impl AiPort for AiOrchestrator {
    fn infer(&self, input: &ControlFrameNormalized) -> Vec<AiOutput> {
        #[cfg(feature = "ai-runtime")]
        if let Some(runtime) = &self.runtime_backend {
            return runtime.infer_runtime(input);
        }
        self.mock.infer(input)
    }

    fn update_attention(&self, weights: &AttentionWeights) {
        self.mock.set_attention(weights.clone());
    }
}

impl AiPort for MockAiPort {
    fn infer(&self, input: &ControlFrameNormalized) -> Vec<AiOutput> {
        let (outputs, _) = self.infer_with_artifacts(input);
        outputs
    }

    fn infer_with_context(&self, input: &ControlFrameNormalized) -> AiInference {
        let (outputs, artifacts) = self.infer_with_artifacts(input);
        let activation_view = activation_view_from_outputs(input, &outputs);
        AiInference {
            outputs,
            nsr_report: artifacts.nsr_report,
            scm_dag: artifacts.scm_dag,
            cde_confidence: artifacts
                .cde_hyp
                .as_ref()
                .map(|hypothesis| hypothesis.confidence),
            ssm_state: artifacts.ssm_state,
            activation_view: Some(activation_view),
        }
    }

    fn update_attention(&self, weights: &AttentionWeights) {
        self.set_attention(weights.clone());
    }
}

impl Default for MockAiPort {
    fn default() -> Self {
        Self {
            pillars: AiPillars::default(),
            attention: Arc::new(Mutex::new(None)),
        }
    }
}

impl MockAiPort {
    fn infer_with_artifacts(
        &self,
        input: &ControlFrameNormalized,
    ) -> (Vec<AiOutput>, PillarArtifacts) {
        let attention = self.attention_snapshot();
        let artifacts = if self.pillars.enabled() {
            run_pillars(&self.pillars, input, attention.as_ref())
        } else {
            PillarArtifacts::baseline(input.commitment().digest)
        };
        let mut thought = AiOutput {
            channel: OutputChannel::Thought,
            content: "ok".to_string(),
            confidence: 1000,
            rationale_commit: Some(artifacts.rationale_commit),
            integration_score: None,
        };

        let loop_frame = reflect_sle(
            &self.pillars,
            input.commitment().digest,
            &artifacts,
            &thought,
        );
        if let Some(frame) = loop_frame {
            thought.rationale_commit = Some(frame.report_commit);
        }
        let integration_score = sample_iit_monitor(&self.pillars, &artifacts, &thought);
        if let Some(score) = integration_score {
            thought.integration_score = Some(score);
        }
        let sparse_features = extract_sparse_features(&self.pillars, input);
        if let Some(bridge) = &self.pillars.digital_brain {
            let evidence_id = evidence_id_from_frame(input);
            let _ = bridge.mirror_features(evidence_id, &sparse_features);
        }

        let mut outputs = vec![thought];
        if input.as_ref().frame_id == "ping" {
            let rationale_commit = outputs[0].rationale_commit;
            let integration_score = outputs[0].integration_score;
            outputs.push(AiOutput {
                channel: OutputChannel::Speech,
                content: "ok".to_string(),
                confidence: 1000,
                rationale_commit,
                integration_score,
            });
        }

        (outputs, artifacts)
    }
}

struct PillarArtifacts {
    rationale_commit: Digest32,
    nsr_report: Option<NsrReport>,
    ssm_state: Option<SsmState>,
    cde_hyp: Option<CdeHypothesis>,
    scm_dag: Option<ScmDag>,
    causal_report: Option<CausalReport>,
    nsr_digest: Option<Digest32>,
}

impl PillarArtifacts {
    fn baseline(digest: Digest32) -> Self {
        Self {
            rationale_commit: digest,
            nsr_report: None,
            ssm_state: None,
            cde_hyp: None,
            scm_dag: None,
            causal_report: None,
            nsr_digest: None,
        }
    }
}

fn evidence_id_from_frame(input: &ControlFrameNormalized) -> EvidenceId {
    input
        .as_ref()
        .evidence_ids
        .first()
        .map(|id| EvidenceId::new(id.clone()))
        .unwrap_or_else(|| EvidenceId::new("unknown"))
}

fn run_pillars(
    pillars: &AiPillars,
    input: &ControlFrameNormalized,
    attention: Option<&AttentionWeights>,
) -> PillarArtifacts {
    let base_digest = input.commitment().digest;
    let mut graph = CausalGraphStub::new(Vec::new(), Vec::new());
    let ctx = NcdeContext::new(base_digest);
    let world_state = WorldStateVec::new(base_digest.as_bytes().to_vec(), vec![Digest32::LEN]);
    let mut thought = ThoughtVec::new(base_digest.as_bytes().to_vec());
    let claims = SymbolicClaims::new(vec![Claim::new_from_strs(
        "frame",
        vec![input.as_ref().frame_id.as_str()],
    )]);

    let mut artifacts = Vec::new();
    let mut nsr_report = None;
    let mut ssm_state = None;
    let mut cde_hyp = None;
    let mut scm_dag = None;
    let mut causal_report = None;
    let mut nsr_digest = None;

    if let Some(tcf) = &pillars.tcf {
        let fallback = idle_attention();
        let attn = attention.unwrap_or(&fallback);
        if let Ok(mut guard) = tcf.lock() {
            let plan = guard.step(attn, None);
            artifacts.push(plan.commit);
        }
    }
    if let Some(ssm) = &pillars.ssm {
        if let Ok(mut guard) = ssm.lock() {
            let input_dim = guard.state().s.len().max(1);
            if let Ok(ssm_input) = SsmInput::from_commitment(base_digest, input_dim) {
                let output = guard.update(&ssm_input, attention);
                artifacts.push(output.commit);
            } else {
                guard.reset();
            }
            let state = guard.state().clone();
            ssm_state = Some(state);
        }
    }
    let context_state = ssm_state
        .as_ref()
        .map(world_state_from_ssm_state)
        .unwrap_or_else(|| apply_attention_to_world_state(&world_state, attention));
    if let Some(cde) = &pillars.cde {
        let hypothesis = cde.infer(&mut graph, &context_state);
        artifacts.push(hypothesis.digest);
        cde_hyp = Some(hypothesis);
    }
    if let (Some(hypothesis), Some(scm)) = (cde_hyp.as_ref(), &pillars.scm) {
        if let Ok(mut guard) = scm.lock() {
            let dag = guard.update(&context_state, Some(hypothesis));
            let dag_commit = ucf_scm_port::digest_dag(&dag);
            scm_dag = Some(dag.clone());
            let counterfactual = default_counterfactual(&dag).map(|query| {
                let result = guard.counterfactual(&query);
                CausalCounterfactual::new(
                    query
                        .interventions
                        .iter()
                        .map(|intervention| {
                            CausalIntervention::new(intervention.node, intervention.value)
                        })
                        .collect(),
                    query.target,
                    result.predicted,
                    result.confidence,
                )
            });
            let report = CausalReport::new(dag_commit, counterfactual);
            artifacts.push(digest_causal_report(&report));
            causal_report = Some(report);
        }
    }
    if let Some(nsr) = &pillars.nsr {
        let report = nsr.check(&claims);
        let report_digest = digest_nsr_report(&report);
        artifacts.push(report_digest);
        nsr_report = Some(report);
        nsr_digest = Some(report_digest);
    }
    if let Some(ncde) = &pillars.ncde {
        thought = ncde.integrate(&ctx, &thought);
        artifacts.push(digest_thought(&thought));
    }

    let rationale_commit = if artifacts.is_empty() {
        base_digest
    } else {
        hash_digests(&artifacts)
    };

    PillarArtifacts {
        rationale_commit,
        nsr_report,
        ssm_state,
        cde_hyp,
        scm_dag,
        causal_report,
        nsr_digest,
    }
}

fn reflect_sle(
    pillars: &AiPillars,
    base_digest: Digest32,
    artifacts: &PillarArtifacts,
    output: &AiOutput,
) -> Option<LoopFrame> {
    pillars.sle.as_ref().map(|sle| {
        let prev = sle.latest().unwrap_or_else(|| LoopFrame::seed(base_digest));
        sle.reflect(
            &prev,
            output,
            artifacts.nsr_report.as_ref(),
            artifacts.cde_hyp.as_ref(),
        )
    })
}

fn sample_iit_monitor(
    pillars: &AiPillars,
    artifacts: &PillarArtifacts,
    output: &AiOutput,
) -> Option<u16> {
    let monitor = pillars.iit_monitor.as_ref()?;
    let mut guard = monitor.lock().ok()?;
    if let (Some(ssm), Some(cde)) = (artifacts.ssm_state.as_ref(), artifacts.cde_hyp.as_ref()) {
        guard.sample(ssm.commit, cde.digest);
    }
    if let (Some(cde), Some(nsr)) = (artifacts.cde_hyp.as_ref(), artifacts.nsr_digest) {
        guard.sample(cde.digest, nsr);
    }
    if let (Some(report), Some(cde)) =
        (artifacts.causal_report.as_ref(), artifacts.cde_hyp.as_ref())
    {
        guard.sample(cde.digest, report.dag_commit);
    }
    if let Some(nsr) = artifacts.nsr_digest {
        let output_digest = digest_ai_output(output);
        guard.sample(nsr, output_digest);
    }
    Some(guard.aggregate())
}

fn extract_sparse_features(
    pillars: &AiPillars,
    input: &ControlFrameNormalized,
) -> Vec<SparseFeature> {
    let lens = match &pillars.lens {
        Some(port) => port,
        None => return Vec::new(),
    };
    let sae = match &pillars.sae {
        Some(port) => port,
        None => return Vec::new(),
    };
    let plan = lens.plan(input);
    let taps = lens.tap(&plan);
    sae.extract(&taps)
}

fn default_counterfactual(dag: &ScmDag) -> Option<CounterfactualQuery> {
    let first = dag.nodes.first()?;
    let target = dag.nodes.get(1)?;
    Some(CounterfactualQuery::new(
        vec![Intervention::new(first.id, 1)],
        target.id,
    ))
}

fn digest_causal_report(report: &CausalReport) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.causal.report.v1");
    hasher.update(report.dag_commit.as_bytes());
    match &report.counterfactual {
        Some(counterfactual) => {
            hasher.update(&[1]);
            hasher.update(&counterfactual.target.to_be_bytes());
            hasher.update(&counterfactual.predicted.to_be_bytes());
            hasher.update(&counterfactual.confidence.to_be_bytes());
            hasher.update(
                &u64::try_from(counterfactual.interventions.len())
                    .unwrap_or(0)
                    .to_be_bytes(),
            );
            for intervention in &counterfactual.interventions {
                hasher.update(&intervention.node.to_be_bytes());
                hasher.update(&intervention.value.to_be_bytes());
            }
        }
        None => {
            hasher.update(&[0]);
        }
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_ai_output(output: &AiOutput) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ai.output.v1");
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
    Digest32::new(*hasher.finalize().as_bytes())
}

fn activation_view_from_outputs(
    input: &ControlFrameNormalized,
    outputs: &[AiOutput],
) -> ActivationView {
    let mut hasher = Hasher::new();
    hasher.update(b"ucf.ai.activation.view.v1");
    hasher.update(input.commitment().digest.as_bytes());
    for output in outputs {
        let digest = digest_ai_output(output);
        hasher.update(digest.as_bytes());
    }
    let act_digest = Digest32::new(*hasher.finalize().as_bytes());
    let bytes = act_digest.as_bytes();
    let layer_id = u16::from_be_bytes([bytes[0], bytes[1]]);
    let energy = u16::from_be_bytes([bytes[2], bytes[3]]);
    ActivationView::new(layer_id, act_digest, energy)
}

fn apply_attention_to_world_state(
    world_state: &WorldStateVec,
    attention: Option<&AttentionWeights>,
) -> WorldStateVec {
    let Some(attn) = attention else {
        return world_state.clone();
    };
    if attn.gain == 1000 {
        return world_state.clone();
    }
    let scale = attn.gain as u32;
    let bytes = world_state
        .bytes
        .iter()
        .map(|byte| {
            let scaled = (*byte as u32).saturating_mul(scale) / 1000;
            scaled.min(u8::MAX as u32) as u8
        })
        .collect();
    WorldStateVec::new(bytes, world_state.dims.clone())
}

fn world_state_from_ssm_state(state: &SsmState) -> WorldStateVec {
    let mut bytes = Vec::with_capacity(state.s.len().saturating_mul(4));
    for value in &state.s {
        bytes.extend_from_slice(&value.to_be_bytes());
    }
    WorldStateVec::new(bytes, vec![state.s.len()])
}

fn hash_digests(digests: &[Digest32]) -> Digest32 {
    let mut hasher = Hasher::new();
    for digest in digests {
        hasher.update(digest.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_thought(thought: &ThoughtVec) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(&thought.bytes);
    Digest32::new(*hasher.finalize().as_bytes())
}

fn digest_nsr_report(report: &NsrReport) -> Digest32 {
    let mut hasher = Hasher::new();
    hasher.update(&[report.ok as u8]);
    hasher.update(
        &u64::try_from(report.violations.len())
            .unwrap_or(0)
            .to_be_bytes(),
    );
    for violation in &report.violations {
        hasher.update(violation.as_bytes());
    }
    Digest32::new(*hasher.finalize().as_bytes())
}

pub trait SpeechGate {
    fn allow_speech(&self, cf: &ControlFrameNormalized, out: &AiOutput) -> bool;
}

#[derive(Clone, Debug)]
pub struct PolicySpeechGate {
    policy: PolicyEcology,
}

impl PolicySpeechGate {
    pub fn new(policy: PolicyEcology) -> Self {
        Self { policy }
    }

    fn decision_class(cf: &ControlFrameNormalized) -> Option<u16> {
        cf.as_ref()
            .decision
            .as_ref()
            .and_then(|decision| DecisionKind::try_from(decision.kind).ok())
            .map(|kind| kind as u16)
    }

    fn allow_for_class(&self, class: u16) -> bool {
        self.policy.rules().iter().any(|rule| {
            matches!(
                rule,
                PolicyRule::AllowExternalSpeechIfDecisionClass { class: rule_class }
                    if *rule_class == class
            )
        })
    }
}

impl SpeechGate for PolicySpeechGate {
    fn allow_speech(&self, cf: &ControlFrameNormalized, out: &AiOutput) -> bool {
        if out.channel != OutputChannel::Speech {
            return true;
        }

        let Some(class) = Self::decision_class(cf) else {
            return false;
        };

        self.allow_for_class(class)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(feature = "ai-runtime")]
    use ucf_ai_runtime::backend::{AiRuntimeBackend, NoopRuntimeBackend};
    use ucf_sandbox::normalize;
    use ucf_types::v1::spec::ControlFrame;
    use ucf_types::SymbolicClaims;
    use ucf_types::ThoughtVec;
    use ucf_types::WorldStateVec;

    use ucf_cde_port::{CdeHypothesis, CdePort, MockCdePort};
    use ucf_iit_monitor::IitMonitor;
    use ucf_ncde_port::{NcdeContext, NcdePort};
    use ucf_nsr_port::{NsrBackend, NsrPort, NsrReport};
    use ucf_scm_port::{CounterfactualQuery, CounterfactualResult, MockScmPort, ScmDag, ScmPort};
    use ucf_sle::StrangeLoopEngine;
    use ucf_ssm_port::{DeterministicSsmPort, SsmConfig, SsmInput, SsmOutput, SsmPort, SsmState};
    use ucf_tcf_port::{CyclePlan, Pulse, PulseKind, TcfPort, TcfState};

    fn base_frame(frame_id: &str) -> ControlFrame {
        ControlFrame {
            frame_id: frame_id.to_string(),
            issued_at_ms: 1_700_000_000_000,
            decision: None,
            evidence_ids: Vec::new(),
            policy_id: "policy-1".to_string(),
        }
    }

    #[test]
    fn mock_ai_port_emits_thought_and_speech_for_ping() {
        let pillars = AiPillars {
            nsr: Some(Arc::new(NsrPort::default())),
            ..AiPillars::default()
        };
        let port = MockAiPort::with_pillars(pillars);
        let normalized = normalize(base_frame("ping"));

        let outputs = port.infer(&normalized);

        assert_eq!(outputs.len(), 2);
        assert!(outputs
            .iter()
            .any(|out| out.channel == OutputChannel::Thought));
        assert!(outputs
            .iter()
            .any(|out| out.channel == OutputChannel::Speech));
        assert!(outputs.iter().all(|out| out.content == "ok"));
    }

    #[test]
    fn mock_ai_port_emits_only_thought_for_other_frames() {
        let port = MockAiPort::new();
        let normalized = normalize(base_frame("frame-1"));

        let outputs = port.infer(&normalized);

        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].channel, OutputChannel::Thought);
        assert_eq!(outputs[0].content, "ok");
    }

    #[test]
    fn mock_ai_port_emits_activation_view_without_raw_content() {
        let port = MockAiPort::new();
        let normalized = normalize(base_frame("frame-1"));

        let inference = port.infer_with_context(&normalized);
        let view = inference.activation_view.expect("activation view");
        let expected = activation_view_from_outputs(&normalized, &inference.outputs);

        assert_eq!(view, expected);
        let debug = format!("{:?}", view);
        assert!(!debug.contains("ok"));
    }

    #[test]
    fn pipeline_order_is_deterministic() {
        let order: Arc<std::sync::Mutex<Vec<&'static str>>> =
            Arc::new(std::sync::Mutex::new(Vec::new()));

        struct OrderTcf {
            order: Arc<std::sync::Mutex<Vec<&'static str>>>,
            state: TcfState,
        }
        impl TcfPort for OrderTcf {
            fn step(
                &mut self,
                _attn: &AttentionWeights,
                _surprise: Option<&ucf_predictive_coding::SurpriseSignal>,
            ) -> CyclePlan {
                self.order.lock().unwrap().push("tcf");
                self.state.commit = Digest32::new([1u8; 32]);
                CyclePlan {
                    cycle_id: 0,
                    pulses: vec![Pulse {
                        kind: PulseKind::Sense,
                        weight: 1,
                        slot: 0,
                    }],
                    commit: Digest32::new([1u8; 32]),
                }
            }

            fn state(&self) -> &TcfState {
                &self.state
            }
        }

        struct OrderSsm {
            order: Arc<std::sync::Mutex<Vec<&'static str>>>,
            state: SsmState,
        }
        impl SsmPort for OrderSsm {
            fn reset(&mut self) {
                self.state.s.fill(0);
                self.state.commit = Digest32::new([0u8; 32]);
            }

            fn update(&mut self, _input: &SsmInput, _attn: Option<&AttentionWeights>) -> SsmOutput {
                self.order.lock().unwrap().push("ssm");
                self.state.commit = Digest32::new([3u8; 32]);
                SsmOutput {
                    y: vec![1, 2],
                    commit: Digest32::new([2u8; 32]),
                }
            }

            fn state(&self) -> &SsmState {
                &self.state
            }
        }

        struct OrderCde {
            order: Arc<std::sync::Mutex<Vec<&'static str>>>,
        }
        impl CdePort for OrderCde {
            fn infer(&self, _graph: &mut CausalGraphStub, _obs: &WorldStateVec) -> CdeHypothesis {
                self.order.lock().unwrap().push("cde");
                CdeHypothesis {
                    digest: Digest32::new([4u8; 32]),
                    nodes: 0,
                    edges: 0,
                    confidence: 9000,
                    interventions: Vec::new(),
                }
            }
        }

        struct OrderNsr {
            order: Arc<std::sync::Mutex<Vec<&'static str>>>,
        }
        impl NsrBackend for OrderNsr {
            fn check(&self, _claims: &SymbolicClaims) -> NsrReport {
                self.order.lock().unwrap().push("nsr");
                NsrReport {
                    ok: true,
                    violations: Vec::new(),
                }
            }
        }

        struct OrderScm {
            order: Arc<std::sync::Mutex<Vec<&'static str>>>,
        }
        impl ScmPort for OrderScm {
            fn update(&mut self, _obs: &WorldStateVec, _hint: Option<&CdeHypothesis>) -> ScmDag {
                self.order.lock().unwrap().push("scm");
                ScmDag::new(Vec::new(), Vec::new())
            }

            fn counterfactual(&self, _q: &CounterfactualQuery) -> CounterfactualResult {
                CounterfactualResult {
                    predicted: 0,
                    confidence: 0,
                }
            }
        }

        struct OrderNcde {
            order: Arc<std::sync::Mutex<Vec<&'static str>>>,
        }
        impl NcdePort for OrderNcde {
            fn integrate(&self, _ctx: &NcdeContext, _control: &ThoughtVec) -> ThoughtVec {
                self.order.lock().unwrap().push("ncde");
                ThoughtVec::new(vec![1, 2, 3])
            }
        }

        let pillars = AiPillars {
            tcf: Some(Arc::new(Mutex::new(OrderTcf {
                order: order.clone(),
                state: TcfState {
                    phase: ucf_tcf_port::Phase { q: 0 },
                    energy: 0,
                    commit: Digest32::new([0u8; 32]),
                },
            }))),
            ssm: Some(Arc::new(Mutex::new(OrderSsm {
                order: order.clone(),
                state: SsmState::new(vec![0, 0]),
            }))),
            cde: Some(Arc::new(OrderCde {
                order: order.clone(),
            })),
            scm: Some(Arc::new(Mutex::new(OrderScm {
                order: order.clone(),
            }))),
            nsr: Some(Arc::new(NsrPort::new(Arc::new(OrderNsr {
                order: order.clone(),
            })))),
            ncde: Some(Arc::new(OrderNcde {
                order: order.clone(),
            })),
            ..AiPillars::default()
        };

        let port = MockAiPort::with_pillars(pillars);
        let normalized = normalize(base_frame("frame-1"));

        let _ = port.infer(&normalized);

        let collected = order.lock().unwrap().clone();
        assert_eq!(collected, vec!["tcf", "ssm", "cde", "scm", "nsr", "ncde"]);
    }

    #[test]
    fn pipeline_emits_stable_meta_observer_outputs() {
        let build_port = || {
            let pillars = AiPillars {
                ssm: Some(Arc::new(Mutex::new(DeterministicSsmPort::new(
                    SsmConfig::new(4, 4, 2),
                )))),
                cde: Some(Arc::new(MockCdePort::new())),
                nsr: Some(Arc::new(NsrPort::default())),
                sle: Some(Arc::new(StrangeLoopEngine::new(4))),
                iit_monitor: Some(Arc::new(Mutex::new(IitMonitor::new(4)))),
                ..AiPillars::default()
            };
            MockAiPort::with_pillars(pillars)
        };

        let normalized = normalize(base_frame("frame-1"));
        let port_a = build_port();
        let port_b = build_port();

        let out_a = port_a.infer(&normalized);
        let out_b = port_b.infer(&normalized);
        let thought_a = out_a
            .iter()
            .find(|out| out.channel == OutputChannel::Thought)
            .expect("thought output");
        let thought_b = out_b
            .iter()
            .find(|out| out.channel == OutputChannel::Thought)
            .expect("thought output");

        assert!(thought_a.rationale_commit.is_some());
        assert!(thought_a.integration_score.is_some());
        assert_eq!(thought_a.rationale_commit, thought_b.rationale_commit);
        assert_eq!(thought_a.integration_score, thought_b.integration_score);
    }

    #[test]
    fn causal_report_influences_rationale_commit() {
        let normalized_a = normalize(base_frame("frame-1"));
        let normalized_b = normalize(base_frame("frame-2"));

        let no_scm = MockAiPort::with_pillars(AiPillars {
            ssm: Some(Arc::new(Mutex::new(DeterministicSsmPort::new(
                SsmConfig::new(4, 4, 2),
            )))),
            cde: Some(Arc::new(MockCdePort::new())),
            ..AiPillars::default()
        });
        let with_scm = MockAiPort::with_pillars(AiPillars {
            ssm: Some(Arc::new(Mutex::new(DeterministicSsmPort::new(
                SsmConfig::new(4, 4, 2),
            )))),
            cde: Some(Arc::new(MockCdePort::new())),
            scm: Some(Arc::new(Mutex::new(MockScmPort::new()))),
            ..AiPillars::default()
        });

        let thought_no_scm = no_scm
            .infer(&normalized_a)
            .into_iter()
            .find(|out| out.channel == OutputChannel::Thought)
            .expect("thought output");
        let thought_with_scm_a = with_scm
            .infer(&normalized_a)
            .into_iter()
            .find(|out| out.channel == OutputChannel::Thought)
            .expect("thought output");
        let thought_with_scm_b = with_scm
            .infer(&normalized_b)
            .into_iter()
            .find(|out| out.channel == OutputChannel::Thought)
            .expect("thought output");

        assert_ne!(
            thought_no_scm.rationale_commit,
            thought_with_scm_a.rationale_commit
        );
        assert_ne!(
            thought_with_scm_a.rationale_commit,
            thought_with_scm_b.rationale_commit
        );
    }

    #[cfg(feature = "ai-runtime")]
    #[test]
    fn noop_runtime_backend_compiles_and_delegates() {
        let mock = MockAiPort::new();
        let backend = NoopRuntimeBackend::new(Arc::new(move |cf| mock.infer(cf)));
        let normalized = normalize(base_frame("ping"));

        let outputs = backend.infer_runtime(&normalized);

        assert!(!outputs.is_empty());
    }

    #[cfg(feature = "nsr-smt")]
    #[test]
    fn nsr_smt_backend_compiles() {
        let backend = ucf_nsr_smt::NsrSmtBackend::new();
        let _port = NsrPort::new(Arc::new(backend));
    }

    #[cfg(feature = "nsr-datalog")]
    #[test]
    fn nsr_datalog_backend_compiles() {
        let backend = ucf_nsr_datalog::NsrDatalogBackend::new();
        let _port = NsrPort::new(Arc::new(backend));
    }

    #[cfg(feature = "digitalbrain")]
    mod digitalbrain_tests {
        use super::*;
        use std::collections::BTreeMap;
        use std::sync::mpsc;
        use std::sync::Arc;

        use ucf_bus::{BusSubscriber, InMemoryBus, MessageEnvelope};
        use ucf_digitalbrain_port::{
            BrainCoord, BrainRegion, BrainStimEvent, FeatureId, MappingTable,
        };
        use ucf_lens_port::{LensMock, LensPort};
        use ucf_sae_port::{SaeMock, SaePort, SparseFeature};

        #[test]
        fn mapping_emits_brain_stim_event() {
            let feature_id: FeatureId = 42;
            let mut map = BTreeMap::new();
            map.insert(
                feature_id,
                vec![BrainCoord {
                    region: BrainRegion::Insula,
                    layer: 4,
                    x: 1,
                    y: 2,
                    z: 3,
                }],
            );
            let table = MappingTable::from_map(map);
            let bus: InMemoryBus<MessageEnvelope<BrainStimEvent>> = InMemoryBus::new();
            let receiver = bus.subscribe();
            let bridge = DigitalBrainBridge::new(table, Arc::new(bus.clone()));

            let features = vec![SparseFeature {
                id: feature_id,
                weight: 3,
            }];
            bridge
                .mirror_features(EvidenceId::new("ev-1"), &features)
                .expect("mirror thought");

            let envelope = receiver.try_recv().expect("stim event");
            assert_eq!(envelope.payload.stim.spikes.len(), 1);
            assert_eq!(envelope.payload.stim.rationale, feature_id);
        }

        #[test]
        fn missing_mapping_emits_nothing() {
            let feature_id: FeatureId = 7;
            let table = MappingTable::empty();
            let bus: InMemoryBus<MessageEnvelope<BrainStimEvent>> = InMemoryBus::new();
            let receiver = bus.subscribe();
            let bridge = DigitalBrainBridge::new(table, Arc::new(bus.clone()));

            let features = vec![SparseFeature {
                id: feature_id,
                weight: 2,
            }];
            bridge
                .mirror_features(EvidenceId::new("ev-2"), &features)
                .expect("mirror thought");

            assert!(matches!(
                receiver.try_recv(),
                Err(mpsc::TryRecvError::Empty)
            ));
        }

        #[test]
        fn pipeline_emits_brain_stim_event_for_mapped_feature() {
            let normalized = normalize(base_frame("frame-1"));
            let lens = Arc::new(LensMock::new());
            let sae = Arc::new(SaeMock::new());
            let plan = lens.plan(&normalized);
            let taps = lens.tap(&plan);
            let features = sae.extract(&taps);
            let mapped_feature = features.first().expect("feature");

            let mut map = BTreeMap::new();
            let coord = BrainCoord {
                region: BrainRegion::Thalamus,
                layer: 1,
                x: 9,
                y: 8,
                z: 7,
            };
            map.insert(mapped_feature.id, vec![coord]);
            let table = MappingTable::from_map(map);
            let bus: InMemoryBus<MessageEnvelope<BrainStimEvent>> = InMemoryBus::new();
            let receiver = bus.subscribe();
            let bridge = DigitalBrainBridge::new(table, Arc::new(bus.clone()));
            let pillars = AiPillars {
                lens: Some(lens),
                sae: Some(sae),
                digital_brain: Some(Arc::new(bridge)),
                ..AiPillars::default()
            };
            let port = MockAiPort::with_pillars(pillars);

            let _ = port.infer(&normalized);

            let envelope = receiver.try_recv().expect("stim event");
            assert_eq!(envelope.payload.stim.spikes.len(), 1);
            assert_eq!(envelope.payload.stim.spikes[0].coord, coord);
            assert_eq!(envelope.payload.stim.rationale, mapped_feature.id);
        }
    }
}
