#![forbid(unsafe_code)]

use std::sync::{mpsc, Arc, Mutex};

use ucf_types::{LogicalTime, NodeId, StreamId, WallTime};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MessageEnvelope<T> {
    pub node_id: NodeId,
    pub stream_id: StreamId,
    pub logical_time: LogicalTime,
    pub wall_time: WallTime,
    pub payload: T,
}

pub trait BusPublisher<T> {
    fn publish(&self, message: T);
}

pub trait BusSubscriber<T> {
    fn subscribe(&self) -> mpsc::Receiver<T>;
}

#[derive(Clone, Default)]
pub struct InMemoryBus<T> {
    subscribers: Arc<Mutex<Vec<mpsc::Sender<T>>>>,
}

impl<T> InMemoryBus<T> {
    pub fn new() -> Self {
        Self {
            subscribers: Arc::new(Mutex::new(Vec::new())),
        }
    }
}

impl<T> BusPublisher<T> for InMemoryBus<T>
where
    T: Clone + Send + 'static,
{
    fn publish(&self, message: T) {
        let subscribers = self.subscribers.lock().expect("lock subscribers");
        for sender in subscribers.iter() {
            let _ = sender.send(message.clone());
        }
    }
}

impl<T> BusSubscriber<T> for InMemoryBus<T>
where
    T: Send + 'static,
{
    fn subscribe(&self) -> mpsc::Receiver<T> {
        let (sender, receiver) = mpsc::channel();
        let mut subscribers = self.subscribers.lock().expect("lock subscribers");
        subscribers.push(sender);
        receiver
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn in_memory_bus_publishes_messages() {
        let bus = InMemoryBus::new();
        let receiver = bus.subscribe();

        bus.publish(MessageEnvelope {
            node_id: NodeId::new("node-a"),
            stream_id: StreamId::new("stream-1"),
            logical_time: LogicalTime::new(7),
            wall_time: WallTime::new(1_700_000_000_000),
            payload: "hello".to_string(),
        });

        let received = receiver.recv().expect("message");
        assert_eq!(received.payload, "hello");
        assert_eq!(received.logical_time.tick, 7);
    }
}
