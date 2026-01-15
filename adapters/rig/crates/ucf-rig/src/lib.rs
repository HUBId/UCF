#![forbid(unsafe_code)]

use std::sync::{mpsc, Arc, Mutex};

use ucf_bus::{BusSubscriber, MessageEnvelope};
use ucf_digitalbrain_port::BrainStimEvent;

pub mod external {
    //! External-facing boundary types for future FFI integration.
    //!
    //! Integration points for cxx/PyO3 should convert from these types without
    //! impacting internal event flow.

    use ucf_digitalbrain_port::BrainStimEvent;

    #[derive(Clone, Debug, PartialEq, Eq)]
    pub struct ExternalBrainStimMessage {
        pub stim: BrainStimEvent,
    }

    impl From<BrainStimEvent> for ExternalBrainStimMessage {
        fn from(event: BrainStimEvent) -> Self {
            Self { stim: event }
        }
    }

    impl From<&BrainStimEvent> for ExternalBrainStimMessage {
        fn from(event: &BrainStimEvent) -> Self {
            Self {
                stim: event.clone(),
            }
        }
    }
}

pub use external::ExternalBrainStimMessage;

pub trait RigSink {
    fn emit(&self, event: &BrainStimEvent);
}

#[derive(Clone, Default)]
pub struct BufferRigSink {
    events: Arc<Mutex<Vec<BrainStimEvent>>>,
}

impl BufferRigSink {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> Vec<BrainStimEvent> {
        self.events.lock().expect("lock rig buffer").clone()
    }
}

impl RigSink for BufferRigSink {
    fn emit(&self, event: &BrainStimEvent) {
        let mut events = self.events.lock().expect("lock rig buffer");
        events.push(event.clone());
    }
}

#[derive(Clone, Default)]
pub struct StdoutRigSink;

impl RigSink for StdoutRigSink {
    fn emit(&self, event: &BrainStimEvent) {
        use std::io::{self, Write};

        let mut stdout = io::stdout();
        let _ = writeln!(stdout, "[RIG] stimuli={}", event.stim.spikes.len());
    }
}

pub struct RigService<S, K> {
    subscriber: S,
    sink: K,
    receiver: Option<mpsc::Receiver<MessageEnvelope<BrainStimEvent>>>,
}

impl<S, K> RigService<S, K>
where
    S: BusSubscriber<MessageEnvelope<BrainStimEvent>>,
    K: RigSink,
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
        let receiver = self.receiver.as_ref().expect("rig service must be started");
        let mut processed = 0;

        loop {
            match receiver.try_recv() {
                Ok(envelope) => {
                    processed += 1;
                    self.sink.emit(&envelope.payload);
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }

        processed
    }
}
