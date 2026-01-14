#![forbid(unsafe_code)]

use std::sync::{mpsc, Arc};

use ucf_bus::{BusPublisher, BusSubscriber, MessageEnvelope};
use ucf_router::{Router, RouterOutcome};
use ucf_sandbox::{ControlFrameValidator, SandboxError, SandboxErrorCode};
use ucf_types::v1::spec::{ControlFrame, DecisionKind};
use ucf_types::EvidenceId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutcomeEvent {
    pub evidence_id: EvidenceId,
    pub decision_kind: DecisionKind,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SandboxRejectEvent {
    pub code: SandboxErrorCode,
    pub message: String,
    pub field: Option<&'static str>,
}

pub struct IngestionService<S, P, R> {
    router: Arc<Router>,
    subscriber: S,
    publisher: Option<P>,
    reject_publisher: Option<R>,
    validator: ControlFrameValidator,
    receiver: Option<mpsc::Receiver<MessageEnvelope<ControlFrame>>>,
}

impl<S, P, R> IngestionService<S, P, R>
where
    S: BusSubscriber<MessageEnvelope<ControlFrame>>,
    P: BusPublisher<MessageEnvelope<OutcomeEvent>>,
    R: BusPublisher<MessageEnvelope<SandboxRejectEvent>>,
{
    pub fn new(
        router: Arc<Router>,
        subscriber: S,
        publisher: Option<P>,
        reject_publisher: Option<R>,
        validator: ControlFrameValidator,
    ) -> Self {
        Self {
            router,
            subscriber,
            publisher,
            reject_publisher,
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
                            if let Ok(outcome) =
                                self.router.handle_control_frame(normalized.into_inner())
                            {
                                self.publish_outcome(
                                    outcome,
                                    node_id,
                                    stream_id,
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
        outcome: RouterOutcome,
        node_id: ucf_types::NodeId,
        stream_id: ucf_types::StreamId,
        logical_time: ucf_types::LogicalTime,
        wall_time: ucf_types::WallTime,
    ) {
        if let Some(publisher) = &self.publisher {
            publisher.publish(MessageEnvelope {
                node_id,
                stream_id,
                logical_time,
                wall_time,
                payload: OutcomeEvent {
                    evidence_id: outcome.evidence_id,
                    decision_kind: outcome.decision_kind,
                },
            });
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
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use ucf_archive::InMemoryArchive;
    use ucf_bus::InMemoryBus;
    use ucf_digital_brain::InMemoryDigitalBrain;
    use ucf_policy_gateway::NoOpPolicyEvaluator;
    use ucf_sandbox::{ControlFrameValidator, ValidatorLimits};
    use ucf_types::v1::spec::{ActionCode, ControlFrame, DecisionKind, PolicyDecision};
    use ucf_types::{LogicalTime, NodeId, StreamId, WallTime};

    #[test]
    fn ingestion_service_routes_control_frame_and_publishes_outcome() {
        let bus = InMemoryBus::new();
        let outcome_bus = InMemoryBus::new();
        let outcome_receiver = outcome_bus.subscribe();
        let reject_bus = InMemoryBus::new();
        let reject_receiver = reject_bus.subscribe();

        let policy = Arc::new(NoOpPolicyEvaluator::new());
        let archive = Arc::new(InMemoryArchive::new());
        let brain = Arc::new(InMemoryDigitalBrain::new());
        let router = Arc::new(Router::new(policy, archive.clone(), Some(brain)));

        let validator = ControlFrameValidator::new(ValidatorLimits::default());
        let mut service = IngestionService::new(
            router,
            bus.clone(),
            Some(outcome_bus.clone()),
            Some(reject_bus.clone()),
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
        assert_eq!(archive.list().len(), 1);

        let outcome = outcome_receiver.try_recv().expect("outcome event");
        assert_eq!(outcome.payload.evidence_id, EvidenceId::new("exp-frame-1"));
        assert_eq!(
            outcome.payload.decision_kind,
            DecisionKind::DecisionKindUnspecified
        );
        assert_eq!(outcome.logical_time.tick, 11);

        assert!(reject_receiver.try_recv().is_err());
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
        let brain = Arc::new(InMemoryDigitalBrain::new());
        let router = Arc::new(Router::new(policy, archive.clone(), Some(brain)));

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
}
