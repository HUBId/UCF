#![forbid(unsafe_code)]

use std::sync::{mpsc, Arc, Mutex};

use ucf::boundary::Envelope as BoundaryEnvelope;
use ucf_bus::{BusSubscriber, MessageEnvelope};
use ucf_events::SpeechEvent;
use ucf_types::EvidenceId;

pub mod external {
    //! External-facing boundary types.
    //!
    //! This module is intentionally serialization-free for now. Protobuf support can be
    //! added here later without affecting the internal event types.

    use ucf::boundary::{v1::ExternalSpeechV1, Envelope, MessageKind, ProtocolVersion};
    use ucf_events::SpeechEvent;
    use ucf_types::EvidenceId;

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct ExternalSpeechMessage {
        pub evidence_id: EvidenceId,
        pub text: String,
    }

    impl From<SpeechEvent> for ExternalSpeechMessage {
        fn from(event: SpeechEvent) -> Self {
            Self {
                evidence_id: event.evidence_id,
                text: event.content,
            }
        }
    }

    impl From<&SpeechEvent> for ExternalSpeechMessage {
        fn from(event: &SpeechEvent) -> Self {
            Self {
                evidence_id: event.evidence_id.clone(),
                text: event.content.clone(),
            }
        }
    }

    pub fn envelope_from_event(event: &SpeechEvent) -> Envelope {
        let message = ExternalSpeechV1 {
            evidence_id: event.evidence_id.as_str().to_string(),
            text: event.content.clone(),
            risk_bucket: 0,
            nsr_ok: true,
            tom_intent: String::new(),
        };
        Envelope::new(
            ProtocolVersion::V1,
            MessageKind::ExternalSpeech,
            message.digest(),
        )
    }
}

pub use external::ExternalSpeechMessage;

pub trait SpeechSink {
    fn emit(&self, line: &str);
}

#[derive(Clone, Default)]
pub struct BufferSpeechSink {
    lines: Arc<Mutex<Vec<String>>>,
}

impl BufferSpeechSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn lines(&self) -> Vec<String> {
        self.lines.lock().expect("lock speech buffer").clone()
    }
}

impl SpeechSink for BufferSpeechSink {
    fn emit(&self, line: &str) {
        let mut lines = self.lines.lock().expect("lock speech buffer");
        lines.push(line.to_string());
    }
}

#[derive(Clone, Default)]
pub struct StdoutSpeechSink;

impl SpeechSink for StdoutSpeechSink {
    fn emit(&self, line: &str) {
        use std::io::{self, Write};

        let mut stdout = io::stdout();
        let _ = writeln!(stdout, "{line}");
    }
}

pub struct TerminalService<S, K> {
    subscriber: S,
    sink: K,
    receiver: Option<mpsc::Receiver<MessageEnvelope<SpeechEvent>>>,
}

impl<S, K> TerminalService<S, K>
where
    S: BusSubscriber<MessageEnvelope<SpeechEvent>>,
    K: SpeechSink,
{
    pub fn new(subscriber: S, sink: K) -> Self {
        Self {
            subscriber,
            sink,
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
            .expect("terminal service must be started");
        let mut processed = 0;

        loop {
            match receiver.try_recv() {
                Ok(envelope) => {
                    processed += 1;
                    let _boundary: BoundaryEnvelope =
                        external::envelope_from_event(&envelope.payload);
                    let line = format_speech_line(&envelope.payload);
                    self.sink.emit(&line);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }

        processed
    }
}

fn format_speech_line(event: &SpeechEvent) -> String {
    format!(
        "[SPEECH] {} {}",
        evidence_id_prefix(&event.evidence_id),
        event.content
    )
}

fn evidence_id_prefix(evidence_id: &EvidenceId) -> String {
    evidence_id.as_str().chars().take(8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    use ucf_bus::{BusPublisher, InMemoryBus};
    use ucf_types::{LogicalTime, NodeId, StreamId, WallTime};

    #[test]
    fn terminal_service_formats_speech_lines() {
        let bus = InMemoryBus::new();
        let sink = BufferSpeechSink::new();
        let mut service = TerminalService::new(bus.clone(), sink.clone());
        service.start();

        bus.publish(MessageEnvelope {
            node_id: NodeId::new("node-a"),
            stream_id: StreamId::new("stream-1"),
            logical_time: LogicalTime::new(7),
            wall_time: WallTime::new(1_700_000_000_000),
            payload: SpeechEvent {
                evidence_id: EvidenceId::new("abcdef1234567890"),
                content: "hello".to_string(),
                confidence: 900,
                rationale_commit: None,
            },
        });

        assert_eq!(service.drain(), 1);
        assert_eq!(sink.lines(), vec!["[SPEECH] abcdef12 hello"]);
    }

    #[test]
    fn external_speech_message_converts_from_event() {
        let event = SpeechEvent {
            evidence_id: EvidenceId::new("rec-1"),
            content: "hi".to_string(),
            confidence: 1000,
            rationale_commit: None,
        };

        let message = ExternalSpeechMessage::from(&event);

        assert_eq!(message.evidence_id, EvidenceId::new("rec-1"));
        assert_eq!(message.text, "hi");
    }
}
