use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use crossbeam_channel::{Receiver, Sender, bounded};
use tch::{Device, Tensor};

use crate::evaluation::PolicyValueModel;

/// A batched wrapper around a policy/value model.
///
/// Calls to `forward` enqueue the tensor and block until the background
/// worker flushes a batch and returns the corresponding slice of the output.
pub struct BatchingModel<M: PolicyValueModel + Send + Sync + 'static> {
    inner: Arc<BatchingInner<M>>,
}

struct BatchWork {
    input: Tensor,
    resp_tx: Sender<Result<(Tensor, Tensor), tch::TchError>>,
}

struct BatchingInner<M: PolicyValueModel + Send + Sync + 'static> {
    sender: Sender<BatchWork>,
    _handle: thread::JoinHandle<()>,
    device: Device,
    _marker: std::marker::PhantomData<M>,
}

impl<M: PolicyValueModel + Send + Sync + 'static> BatchingModel<M> {
    /// Create a new batching wrapper.
    ///
    /// - `base_model`: the underlying model that will be owned by the worker thread
    /// - `batch_size`: maximum items per batch (>=1)
    /// - `timeout`: maximum time to wait before flushing a partial batch
    pub fn new(base_model: M, batch_size: usize, timeout: Duration) -> Self {
        let batch_size = batch_size.max(1);
        let (tx, rx) = bounded::<BatchWork>(batch_size * 4);

        let device = base_model.device();
        let handle = thread::spawn(move || worker_loop(base_model, batch_size, timeout, rx));

        Self {
            inner: Arc::new(BatchingInner {
                sender: tx,
                _handle: handle,
                device,
                _marker: std::marker::PhantomData,
            }),
        }
    }
}

impl<M: PolicyValueModel + Send + Sync + 'static> Clone for BatchingModel<M> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<M: PolicyValueModel + Send + Sync + 'static> PolicyValueModel for BatchingModel<M> {
    fn forward(&self, x: &Tensor) -> tch::Result<(Tensor, Tensor)> {
        let (resp_tx, resp_rx) = bounded::<Result<(Tensor, Tensor), tch::TchError>>(1);

        // The caller typically passes [1, C, H, W]; we want to batch on the first
        // dimension, so strip the leading singleton to avoid producing a 5D tensor
        // after stacking.
        let input = if x.dim() == 4 && x.size()[0] == 1 {
            x.squeeze_dim(0)
        } else {
            x.shallow_clone()
        };

        // Enqueue the work. If the worker has stopped, surface an error.
        self.inner
            .sender
            .send(BatchWork { input, resp_tx })
            .map_err(|_| tch::TchError::Kind("Batching worker stopped".into()))?;

        // Block until the worker returns the matching slice.
        resp_rx
            .recv()
            .map_err(|_| tch::TchError::Kind("Batching worker stopped".into()))?
    }

    fn device(&self) -> Device {
        self.inner.device
    }
}

fn worker_loop<M: PolicyValueModel + Send + Sync + 'static>(
    model: M,
    batch_size: usize,
    timeout: Duration,
    rx: Receiver<BatchWork>,
) {
    // Loop until all senders are dropped.
    while let Ok(first) = rx.recv() {
        let mut batch_inputs = Vec::with_capacity(batch_size);
        let mut responders = Vec::with_capacity(batch_size);

        batch_inputs.push(first.input);
        responders.push(first.resp_tx);

        // Fill up to batch_size or until timeout expires.
        let deadline = Instant::now() + timeout;
        while responders.len() < batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let recv_result = rx.recv_timeout(remaining);
            match recv_result {
                Ok(work) => {
                    batch_inputs.push(work.input);
                    responders.push(work.resp_tx);
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => break,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        }

        // Run a single forward pass with stacked inputs.
        let stacked = Tensor::stack(&batch_inputs, 0);
        let forward_result = model.forward(&stacked);

        match forward_result {
            Ok((policy_batch, value_batch)) => {
                // Slice per item and send.
                for (i, tx) in responders.into_iter().enumerate() {
                    let policy = policy_batch.get(i as i64);
                    let value = value_batch.get(i as i64);
                    let _ = tx.send(Ok((policy, value)));
                }
            }
            Err(err) => {
                // Convert to a simple Kind error to clone for each responder.
                let msg = format!("Batch forward failed: {err}");
                for tx in responders {
                    let _ = tx.send(Err(tch::TchError::Kind(msg.clone())));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tch::{Kind, Tensor};

    struct DummyModel {
        calls: AtomicUsize,
    }

    impl DummyModel {
        fn new() -> Self {
            Self {
                calls: AtomicUsize::new(0),
            }
        }
    }

    impl PolicyValueModel for DummyModel {
        fn forward(&self, x: &Tensor) -> tch::Result<(Tensor, Tensor)> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            // policy: zeros, value: zeros
            let batch = x.size()[0];
            Ok((
                Tensor::zeros([batch, 64], (Kind::Float, x.device())),
                Tensor::zeros([batch, 1], (Kind::Float, x.device())),
            ))
        }

        fn device(&self) -> Device {
            Device::Cpu
        }
    }

    #[test]
    fn batches_multiple_requests() {
        let model = DummyModel::new();
        let batching = BatchingModel::new(model, 8, Duration::from_millis(10));

        let inputs: Vec<_> = (0..4)
            .map(|_| Tensor::zeros([1, 3, 8, 8], (Kind::Float, Device::Cpu)))
            .collect();

        let handles: Vec<_> = inputs
            .iter()
            .map(|t| {
                let cloned = t.shallow_clone();
                std::thread::spawn({
                    let batching = batching.clone();
                    move || batching.forward(&cloned)
                })
            })
            .collect();

        for h in handles {
            let (policy, value) = h.join().unwrap().unwrap();
            assert_eq!(policy.size(), [64]);
            assert_eq!(value.numel(), 1, "expected scalar value tensor");
        }
    }
}
