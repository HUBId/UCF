#![forbid(unsafe_code)]

use std::sync::{mpsc, Arc};

use ucf_bus::{BusPublisher, BusSubscriber, MessageEnvelope};
use ucf_router::{Router, RouterOutcome};
use ucf_types::v1::spec::{ControlFrame, DecisionKind};
use ucf_types::EvidenceId;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutcomeEvent {
    pub evidence_id: EvidenceId,
    pub decision_kind: DecisionKind,
}

pub struct IngestionService<S, P> {
    router: Arc<Router>,
    subscriber: S,
    publisher: Option<P>,
    receiver: Option<mpsc::Receiver<MessageEnvelope<ControlFrame>>>,
}

impl<S, P> IngestionService<S, P>
where
    S: BusSubscriber<MessageEnvelope<ControlFrame>>,
    P: BusPublisher<MessageEnvelope<OutcomeEvent>>,
{
    pub fn new(router: Arc<Router>, subscriber: S, publisher: Option<P>) -> Self {
        Self {
            router,
            subscriber,
            publisher,
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
                    if let Ok(outcome) = self.router.handle_control_frame(payload) {
                        self.publish_outcome(outcome, node_id, stream_id, logical_time, wall_time);
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
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use ucf_archive::InMemoryArchive;
    use ucf_bus::InMemoryBus;
    use ucf_digital_brain::InMemoryDigitalBrain;
    use ucf_policy_gateway::NoOpPolicyEvaluator;
    use ucf_types::v1::spec::ControlFrame;
    use ucf_types::{LogicalTime, NodeId, StreamId, WallTime};

    #[test]
    fn ingestion_service_routes_control_frame_and_publishes_outcome() {
        let bus = InMemoryBus::new();
        let outcome_bus = InMemoryBus::new();
        let outcome_receiver = outcome_bus.subscribe();

        let policy = Arc::new(NoOpPolicyEvaluator::new());
        let archive = Arc::new(InMemoryArchive::new());
        let brain = Arc::new(InMemoryDigitalBrain::new());
        let router = Arc::new(Router::new(policy, archive.clone(), Some(brain)));

        let mut service = IngestionService::new(router, bus.clone(), Some(outcome_bus.clone()));
        service.start();

        let frame = ControlFrame {
            frame_id: "frame-1".to_string(),
            issued_at_ms: 1_700_000_000_000,
            decision: None,
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
    }
}
