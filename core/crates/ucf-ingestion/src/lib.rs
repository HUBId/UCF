#![forbid(unsafe_code)]

use std::sync::{mpsc, Arc, Mutex};

use ucf_ai_port::AiOutput;
use ucf_bus::{BusPublisher, BusSubscriber, MessageEnvelope};
use ucf_events::{OutcomeEvent, SandboxRejectEvent, SpeechEvent, WorkspaceBroadcast};
use ucf_router::{Router, RouterOutcome};
use ucf_sandbox::{ControlFrameValidator, SandboxError};
use ucf_sleep_coordinator::{
    SleepPhaseRunner, SleepStateHandle, SleepStateUpdater, SleepTriggered,
};
use ucf_types::v1::spec::ControlFrame;
use ucf_types::EvidenceId;
use ucf_workspace::{Workspace, WorkspaceSignal};

pub struct IngestionService<S, P, R, E, W, T> {
    router: Arc<Router>,
    subscriber: S,
    publisher: Option<P>,
    reject_publisher: Option<R>,
    speech_publisher: Option<E>,
    workspace_publisher: Option<W>,
    sleep: Option<SleepLoop<T>>,
    validator: ControlFrameValidator,
    receiver: Option<mpsc::Receiver<MessageEnvelope<ControlFrame>>>,
}

pub struct SleepLoop<T> {
    pub state: SleepStateHandle,
    pub runner: Arc<dyn SleepPhaseRunner + Send + Sync>,
    pub trigger_bus: T,
}

impl<S, P, R, E, W, T> IngestionService<S, P, R, E, W, T>
where
    S: BusSubscriber<MessageEnvelope<ControlFrame>>,
    P: BusPublisher<MessageEnvelope<OutcomeEvent>>,
    R: BusPublisher<MessageEnvelope<SandboxRejectEvent>>,
    E: BusPublisher<MessageEnvelope<SpeechEvent>>,
    W: BusPublisher<MessageEnvelope<WorkspaceBroadcast>>,
    T: BusPublisher<MessageEnvelope<SleepTriggered>>,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        router: Arc<Router>,
        subscriber: S,
        publisher: Option<P>,
        reject_publisher: Option<R>,
        speech_publisher: Option<E>,
        workspace_publisher: Option<W>,
        sleep: Option<SleepLoop<T>>,
        validator: ControlFrameValidator,
    ) -> Self {
        Self {
            router,
            subscriber,
            publisher,
            reject_publisher,
            speech_publisher,
            workspace_publisher,
            sleep,
            validator,
            receiver: None,
        }
    }

    pub fn start(&mut self) {
        self.receiver = Some(self.subscriber.subscribe());
    }

    pub fn drain(&mut self) -> usize {
        let receiver = self
            .receiver
            .as_ref()
            .expect("ingestion service must be started");
        let mut processed = 0;

        loop {
            match receiver.try_recv() {
                Ok(message) => {
                    processed += 1;
                    let MessageEnvelope {
                        node_id,
                        stream_id,
                        logical_time,
                        wall_time,
                        payload,
                    } = message;
                    match self.validator.validate_and_normalize(payload) {
                        Ok(normalized) => {
                            if let Ok(outcome) = self.router.handle_control_frame(normalized) {
                                self.publish_outcome(
                                    &outcome,
                                    &node_id,
                                    &stream_id,
                                    logical_time,
                                    wall_time,
                                );
                                self.update_sleep(
                                    &outcome,
                                    &node_id,
                                    &stream_id,
                                    logical_time,
                                    wall_time,
                                );
                                self.publish_speech(
                                    &outcome.speech_outputs,
                                    &outcome.evidence_id,
                                    &node_id,
                                    &stream_id,
                                    logical_time,
                                    wall_time,
                                );
                            }
                        }
                        Err(err) => {
                            self.publish_reject(err, node_id, stream_id, logical_time, wall_time);
                        }
                    }
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }

        processed
    }

    fn publish_outcome(
        &self,
        outcome: &RouterOutcome,
        node_id: &ucf_types::NodeId,
        stream_id: &ucf_types::StreamId,
        logical_time: ucf_types::LogicalTime,
        wall_time: ucf_types::WallTime,
    ) {
        if let Some(publisher) = &self.publisher {
            publisher.publish(MessageEnvelope {
                node_id: node_id.clone(),
                stream_id: stream_id.clone(),
                logical_time,
                wall_time,
                payload: OutcomeEvent {
                    evidence_id: outcome.evidence_id.clone(),
                    decision_kind: outcome.decision_kind,
                },
            });
        }
        if let (Some(commit), Some(publisher)) =
            (outcome.workspace_snapshot_commit, &self.workspace_publisher)
        {
            publisher.publish(MessageEnvelope {
                node_id: node_id.clone(),
                stream_id: stream_id.clone(),
                logical_time,
                wall_time,
                payload: WorkspaceBroadcast {
                    snapshot_commit: commit,
                },
            });
        }
    }

    fn publish_speech(
        &self,
        speech_outputs: &[AiOutput],
        evidence_id: &EvidenceId,
        node_id: &ucf_types::NodeId,
        stream_id: &ucf_types::StreamId,
        logical_time: ucf_types::LogicalTime,
        wall_time: ucf_types::WallTime,
    ) {
        if let Some(publisher) = &self.speech_publisher {
            for output in speech_outputs {
                publisher.publish(MessageEnvelope {
                    node_id: node_id.clone(),
                    stream_id: stream_id.clone(),
                    logical_time,
                    wall_time,
                    payload: SpeechEvent {
                        evidence_id: evidence_id.clone(),
                        content: output.content.clone(),
                        confidence: output.confidence,
                        rationale_commit: output.rationale_commit,
                    },
                });
            }
        }
    }

    fn publish_reject(
        &self,
        err: SandboxError,
        node_id: ucf_types::NodeId,
        stream_id: ucf_types::StreamId,
        logical_time: ucf_types::LogicalTime,
        wall_time: ucf_types::WallTime,
    ) {
        if let Some(publisher) = &self.reject_publisher {
            publisher.publish(MessageEnvelope {
                node_id,
                stream_id,
                logical_time,
                wall_time,
                payload: SandboxRejectEvent {
                    code: err.code,
                    message: err.message,
                    field: err.field,
                },
            });
        }
    }

    fn update_sleep(
        &self,
        outcome: &RouterOutcome,
        node_id: &ucf_types::NodeId,
        stream_id: &ucf_types::StreamId,
        logical_time: ucf_types::LogicalTime,
        wall_time: ucf_types::WallTime,
    ) {
        let Some(sleep) = &self.sleep else {
            return;
        };
        let Ok(mut guard) = sleep.state.lock() else {
            return;
        };
        guard.record_derived_record(outcome.evidence_id.clone());
        if let Some(score) = outcome.integration_score {
            guard.record_integration_score(score);
        }
        if let Some(signal) = outcome.surprise_signal.as_ref() {
            guard.record_surprise_band(signal.band);
        }
        if let Some(stats) = outcome.structural_stats.clone() {
            guard.record_structural_stats(stats);
        }
        if let Some(proposal) = outcome.structural_proposal.clone() {
            guard.record_structural_proposal(proposal);
        }
        let publisher = SleepEnvelopePublisher {
            publisher: &sleep.trigger_bus,
            workspace: Some(self.router.workspace_handle()),
            node_id: node_id.clone(),
            stream_id: stream_id.clone(),
            logical_time,
            wall_time,
        };
        let _ = guard.maybe_trigger(sleep.runner.as_ref(), &publisher);
    }
}

struct SleepEnvelopePublisher<'a, T> {
    publisher: &'a T,
    workspace: Option<Arc<Mutex<Workspace>>>,
    node_id: ucf_types::NodeId,
    stream_id: ucf_types::StreamId,
    logical_time: ucf_types::LogicalTime,
    wall_time: ucf_types::WallTime,
}

impl<T> BusPublisher<SleepTriggered> for SleepEnvelopePublisher<'_, T>
where
    T: BusPublisher<MessageEnvelope<SleepTriggered>>,
{
    fn publish(&self, message: SleepTriggered) {
        if let Some(workspace) = &self.workspace {
            if let Ok(mut guard) = workspace.lock() {
                guard.publish(WorkspaceSignal::from_sleep_triggered(&message, None, None));
            }
        }
        self.publisher.publish(MessageEnvelope {
            node_id: self.node_id.clone(),
            stream_id: self.stream_id.clone(),
            logical_time: self.logical_time,
            wall_time: self.wall_time,
            payload: message,
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use prost::Message;
    use ucf_ai_port::{AiPillars, MockAiPort, PolicySpeechGate};
    use ucf_archive::InMemoryArchive;
    use ucf_archive_store::{ArchiveStore, InMemoryArchiveStore, RecordKind};
    use ucf_bus::InMemoryBus;
    use ucf_digital_brain::InMemoryDigitalBrain;
    use ucf_nsr_port::{NsrPort, NsrStubBackend};
    use ucf_policy_ecology::{PolicyEcology, PolicyRule, PolicyWeights};
    use ucf_policy_gateway::NoOpPolicyEvaluator;
    use ucf_risk_gate::PolicyRiskGate;
    use ucf_sandbox::{
        ControlFrameNormalized, ControlFrameValidator, SandboxErrorCode, ValidatorLimits,
    };
    use ucf_sleep_coordinator::SleepTriggered;
    use ucf_tom_port::{
        ActorProfile, IntentHypothesis, IntentType, KnowledgeGap, SocialRiskSignals, TomPort,
        TomReport,
    };
    use ucf_types::v1::spec::{
        ActionCode, ControlFrame, DecisionKind, ExperienceRecord, PolicyDecision,
    };
    use ucf_types::{LogicalTime, NodeId, StreamId, WallTime};

    #[derive(Clone)]
    struct LowRiskTomPort;

    impl TomPort for LowRiskTomPort {
        fn analyze(
            &self,
            cf: &ControlFrameNormalized,
            _outputs: &[ucf_types::AiOutput],
        ) -> TomReport {
            TomReport {
                actors: vec![ActorProfile {
                    id: 1,
                    label: "actor-1".to_string(),
                }],
                intent: IntentHypothesis {
                    intent: IntentType::Unknown,
                    confidence: 0,
                },
                gaps: vec![KnowledgeGap {
                    topic: "context".to_string(),
                    uncertainty: 0,
                }],
                risk: SocialRiskSignals {
                    deception_likelihood: 0,
                    consent_uncertainty: 0,
                    manipulation_risk: 0,
                    overall: 0,
                },
                commit: cf.commitment().digest,
            }
        }
    }

    #[test]
    fn ingestion_service_routes_control_frame_and_publishes_outcome() {
        let bus = InMemoryBus::new();
        let outcome_bus = InMemoryBus::new();
        let outcome_receiver = outcome_bus.subscribe();
        let reject_bus = InMemoryBus::new();
        let reject_receiver = reject_bus.subscribe();
        let speech_bus = InMemoryBus::new();
        let speech_receiver = speech_bus.subscribe();

        let policy = Arc::new(NoOpPolicyEvaluator::new());
        let archive = Arc::new(InMemoryArchive::new());
        let archive_store = Arc::new(InMemoryArchiveStore::new());
        let brain = Arc::new(InMemoryDigitalBrain::new());
        let ai_port = Arc::new(MockAiPort::with_pillars(AiPillars {
            nsr: Some(Arc::new(NsrPort::default())),
            ..AiPillars::default()
        }));
        let speech_gate = Arc::new(PolicySpeechGate::new(PolicyEcology::allow_all()));
        let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
        let tom_port = Arc::new(LowRiskTomPort);
        let nsr_backend = Arc::new(NsrStubBackend::new());
        let router = Arc::new(
            Router::new(
                policy,
                archive.clone(),
                archive_store.clone(),
                Some(brain),
                ai_port,
                speech_gate,
                risk_gate,
                tom_port,
                None,
            )
            .with_nsr_port(Arc::new(NsrPort::new(nsr_backend))),
        );

        let validator = ControlFrameValidator::new(ValidatorLimits::default());
        let mut service = IngestionService::new(
            router,
            bus.clone(),
            Some(outcome_bus.clone()),
            Some(reject_bus.clone()),
            Some(speech_bus.clone()),
            None::<InMemoryBus<MessageEnvelope<WorkspaceBroadcast>>>,
            None::<SleepLoop<InMemoryBus<MessageEnvelope<SleepTriggered>>>>,
            validator,
        );
        service.start();

        let frame = ControlFrame {
            frame_id: "frame-1".to_string(),
            issued_at_ms: 1_700_000_000_000,
            decision: Some(PolicyDecision {
                kind: DecisionKind::DecisionKindAllow as i32,
                action: ActionCode::ActionCodeContinue as i32,
                rationale: "ok".to_string(),
                confidence_bp: 1000,
                constraint_ids: Vec::new(),
            }),
            evidence_ids: Vec::new(),
            policy_id: "policy-1".to_string(),
        };

        bus.publish(MessageEnvelope {
            node_id: NodeId::new("node-a"),
            stream_id: StreamId::new("stream-1"),
            logical_time: LogicalTime::new(11),
            wall_time: WallTime::new(1_700_000_000_000),
            payload: frame,
        });

        let processed = service.drain();

        assert_eq!(processed, 1);
        assert_eq!(archive.list().len(), 9);

        let outcome = outcome_receiver.try_recv().expect("outcome event");
        assert_eq!(outcome.payload.evidence_id, EvidenceId::new("exp-frame-1"));
        assert_eq!(
            outcome.payload.decision_kind,
            DecisionKind::DecisionKindUnspecified
        );
        assert_eq!(outcome.logical_time.tick, 11);

        assert!(reject_receiver.try_recv().is_err());
        assert!(speech_receiver.try_recv().is_err());

        let payload = archive
            .list()
            .iter()
            .find_map(|envelope| {
                let proof = envelope.proof.as_ref()?;
                let record = ExperienceRecord::decode(proof.payload.as_slice()).ok()?;
                let payload = String::from_utf8(record.payload).ok()?;
                payload.contains("ai_thoughts=ok").then_some(payload)
            })
            .expect("experience payload");
        assert!(payload.contains("ai_thoughts=ok"));
    }

    #[test]
    fn ingestion_service_rejects_invalid_control_frame() {
        let bus = InMemoryBus::new();
        let outcome_bus = InMemoryBus::new();
        let outcome_receiver = outcome_bus.subscribe();
        let reject_bus = InMemoryBus::new();
        let reject_receiver = reject_bus.subscribe();

        let policy = Arc::new(NoOpPolicyEvaluator::new());
        let archive = Arc::new(InMemoryArchive::new());
        let archive_store = Arc::new(InMemoryArchiveStore::new());
        let brain = Arc::new(InMemoryDigitalBrain::new());
        let ai_port = Arc::new(MockAiPort::with_pillars(AiPillars {
            nsr: Some(Arc::new(NsrPort::default())),
            ..AiPillars::default()
        }));
        let speech_gate = Arc::new(PolicySpeechGate::new(PolicyEcology::allow_all()));
        let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
        let tom_port = Arc::new(LowRiskTomPort);
        let router = Arc::new(Router::new(
            policy,
            archive.clone(),
            archive_store.clone(),
            Some(brain),
            ai_port,
            speech_gate,
            risk_gate,
            tom_port,
            None,
        ));

        let limits = ValidatorLimits {
            max_context_items: 1,
            ..ValidatorLimits::default()
        };
        let validator = ControlFrameValidator::new(limits);
        let mut service = IngestionService::new(
            router,
            bus.clone(),
            Some(outcome_bus.clone()),
            Some(reject_bus.clone()),
            None::<InMemoryBus<MessageEnvelope<SpeechEvent>>>,
            None::<InMemoryBus<MessageEnvelope<WorkspaceBroadcast>>>,
            None::<SleepLoop<InMemoryBus<MessageEnvelope<SleepTriggered>>>>,
            validator,
        );
        service.start();

        let frame = ControlFrame {
            frame_id: "frame-2".to_string(),
            issued_at_ms: 1_700_000_000_001,
            decision: Some(PolicyDecision {
                kind: DecisionKind::DecisionKindAllow as i32,
                action: ActionCode::ActionCodeContinue as i32,
                rationale: "ok".to_string(),
                confidence_bp: 1000,
                constraint_ids: Vec::new(),
            }),
            evidence_ids: vec!["e1".to_string(), "e2".to_string()],
            policy_id: "policy-2".to_string(),
        };

        bus.publish(MessageEnvelope {
            node_id: NodeId::new("node-b"),
            stream_id: StreamId::new("stream-2"),
            logical_time: LogicalTime::new(12),
            wall_time: WallTime::new(1_700_000_000_001),
            payload: frame,
        });

        let processed = service.drain();

        assert_eq!(processed, 1);
        assert!(outcome_receiver.try_recv().is_err());
        assert_eq!(archive.list().len(), 0);

        let reject = reject_receiver.try_recv().expect("reject event");
        assert_eq!(
            reject.payload.code,
            SandboxErrorCode::INVALID_CONTEXT_BINDING
        );
        assert_eq!(reject.logical_time.tick, 12);
    }

    #[test]
    fn ingestion_service_publishes_speech_event_when_gate_allows() {
        let bus = InMemoryBus::new();
        let outcome_bus = InMemoryBus::new();
        let speech_bus = InMemoryBus::new();
        let speech_receiver = speech_bus.subscribe();

        let policy = Arc::new(NoOpPolicyEvaluator::new());
        let archive = Arc::new(InMemoryArchive::new());
        let archive_store = Arc::new(InMemoryArchiveStore::new());
        let brain = Arc::new(InMemoryDigitalBrain::new());
        let ai_port = Arc::new(MockAiPort::with_pillars(AiPillars {
            nsr: Some(Arc::new(NsrPort::default())),
            ..AiPillars::default()
        }));
        let speech_policy = PolicyEcology::new(
            1,
            vec![PolicyRule::AllowExternalSpeechIfDecisionClass {
                class: DecisionKind::DecisionKindAllow as u16,
            }],
            PolicyWeights,
        );
        let speech_gate = Arc::new(PolicySpeechGate::new(speech_policy.clone()));
        let risk_gate = Arc::new(PolicyRiskGate::new(speech_policy));
        let tom_port = Arc::new(LowRiskTomPort);
        let nsr_backend = Arc::new(NsrStubBackend::new());
        let router = Arc::new(
            Router::new(
                policy,
                archive.clone(),
                archive_store.clone(),
                Some(brain),
                ai_port,
                speech_gate,
                risk_gate,
                tom_port,
                None,
            )
            .with_nsr_port(Arc::new(NsrPort::new(nsr_backend))),
        );

        let validator = ControlFrameValidator::new(ValidatorLimits::default());
        let mut service = IngestionService::new(
            router,
            bus.clone(),
            Some(outcome_bus.clone()),
            None::<InMemoryBus<MessageEnvelope<SandboxRejectEvent>>>,
            Some(speech_bus.clone()),
            None::<InMemoryBus<MessageEnvelope<WorkspaceBroadcast>>>,
            None::<SleepLoop<InMemoryBus<MessageEnvelope<SleepTriggered>>>>,
            validator,
        );
        service.start();

        let frame = ControlFrame {
            frame_id: "ping".to_string(),
            issued_at_ms: 1_700_000_000_000,
            decision: Some(PolicyDecision {
                kind: DecisionKind::DecisionKindAllow as i32,
                action: ActionCode::ActionCodeContinue as i32,
                rationale: "ok".to_string(),
                confidence_bp: 1000,
                constraint_ids: Vec::new(),
            }),
            evidence_ids: Vec::new(),
            policy_id: "policy-1".to_string(),
        };

        bus.publish(MessageEnvelope {
            node_id: NodeId::new("node-a"),
            stream_id: StreamId::new("stream-1"),
            logical_time: LogicalTime::new(21),
            wall_time: WallTime::new(1_700_000_000_000),
            payload: frame,
        });

        let processed = service.drain();

        assert_eq!(processed, 1);
        assert_eq!(archive.list().len(), 10);

        let speech = speech_receiver.try_recv().expect("speech event");
        assert_eq!(speech.payload.content, "ok");
        assert_eq!(speech.payload.evidence_id, EvidenceId::new("exp-ping"));
        assert_eq!(speech.logical_time.tick, 21);
    }

    #[test]
    fn ingestion_service_emits_workspace_snapshot_event_and_archives_record() {
        let bus = InMemoryBus::new();
        let workspace_bus = InMemoryBus::new();
        let workspace_receiver = workspace_bus.subscribe();

        let policy = Arc::new(NoOpPolicyEvaluator::new());
        let archive = Arc::new(InMemoryArchive::new());
        let archive_store = Arc::new(InMemoryArchiveStore::new());
        let brain = Arc::new(InMemoryDigitalBrain::new());
        let ai_port = Arc::new(MockAiPort::with_pillars(AiPillars {
            nsr: Some(Arc::new(NsrPort::default())),
            ..AiPillars::default()
        }));
        let speech_gate = Arc::new(PolicySpeechGate::new(PolicyEcology::allow_all()));
        let risk_gate = Arc::new(PolicyRiskGate::new(PolicyEcology::allow_all()));
        let tom_port = Arc::new(LowRiskTomPort);
        let router = Arc::new(Router::new(
            policy,
            archive.clone(),
            archive_store.clone(),
            Some(brain),
            ai_port,
            speech_gate,
            risk_gate,
            tom_port,
            None,
        ));

        let validator = ControlFrameValidator::new(ValidatorLimits::default());
        let mut service = IngestionService::new(
            router,
            bus.clone(),
            None::<InMemoryBus<MessageEnvelope<OutcomeEvent>>>,
            None::<InMemoryBus<MessageEnvelope<SandboxRejectEvent>>>,
            None::<InMemoryBus<MessageEnvelope<SpeechEvent>>>,
            Some(workspace_bus.clone()),
            None::<SleepLoop<InMemoryBus<MessageEnvelope<SleepTriggered>>>>,
            validator,
        );
        service.start();

        let frame = ControlFrame {
            frame_id: "frame-workspace".to_string(),
            issued_at_ms: 1_700_000_000_002,
            decision: Some(PolicyDecision {
                kind: DecisionKind::DecisionKindAllow as i32,
                action: ActionCode::ActionCodeContinue as i32,
                rationale: "ok".to_string(),
                confidence_bp: 1000,
                constraint_ids: Vec::new(),
            }),
            evidence_ids: Vec::new(),
            policy_id: "policy-1".to_string(),
        };

        bus.publish(MessageEnvelope {
            node_id: NodeId::new("node-c"),
            stream_id: StreamId::new("stream-3"),
            logical_time: LogicalTime::new(31),
            wall_time: WallTime::new(1_700_000_000_002),
            payload: frame,
        });

        let processed = service.drain();

        assert_eq!(processed, 1);
        assert_eq!(archive.list().len(), 10);

        let event = workspace_receiver.try_recv().expect("workspace broadcast");

        let records: Vec<_> = archive_store
            .iter_kind(RecordKind::WorkspaceSnapshot, None)
            .collect();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].payload_commit, event.payload.snapshot_commit);
    }
}
