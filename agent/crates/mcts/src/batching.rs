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
    // Always normalized to [B, C, H, W]. A work item may contain a complete
    // MCTS expansion batch rather than one leaf.
    input: Tensor,
    unbatched: bool,
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
        let (input, unbatched) = match x.dim() {
            3 => (x.unsqueeze(0), true),
            4 => (x.shallow_clone(), false),
            dimensions => {
                return Err(tch::TchError::Kind(format!(
                    "Expected a 3D or 4D input tensor, got {dimensions} dimensions"
                )));
            }
        };
        let (resp_tx, resp_rx) = bounded::<Result<(Tensor, Tensor), tch::TchError>>(1);

        self.inner
            .sender
            .send(BatchWork {
                input,
                unbatched,
                resp_tx,
            })
            .map_err(|_| tch::TchError::Kind("Batching worker stopped".into()))?;

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
    let mut pending = None;

    // Loop until all senders are dropped.
    loop {
        let first = match pending.take() {
            Some(work) => work,
            None => match rx.recv() {
                Ok(work) => work,
                Err(_) => break,
            },
        };
        let mut sample_count = first.input.size()[0] as usize;
        let mut works = Vec::with_capacity(batch_size.min(64));
        works.push(first);

        // Fill up to batch_size or until timeout expires.
        let deadline = Instant::now() + timeout;
        while sample_count < batch_size {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                break;
            }
            let recv_result = rx.recv_timeout(remaining);
            match recv_result {
                Ok(work) => {
                    let work_samples = work.input.size()[0] as usize;
                    if sample_count + work_samples > batch_size {
                        pending = Some(work);
                        break;
                    }
                    sample_count += work_samples;
                    works.push(work);
                }
                Err(crossbeam_channel::RecvTimeoutError::Timeout) => break,
                Err(crossbeam_channel::RecvTimeoutError::Disconnected) => break,
            }
        }

        // Concatenate complete request batches on CPU, then let the underlying
        // model perform one device transfer for the full inference batch.
        let batch_inputs: Vec<_> = works
            .iter()
            .map(|work| work.input.shallow_clone())
            .collect();
        let combined = Tensor::cat(&batch_inputs, 0);
        let forward_result = model.forward(&combined);

        match forward_result {
            Ok((policy_batch, value_batch)) => {
                // Outputs have already crossed to CPU as complete batches.
                // Split them per request without further device synchronization.
                let mut offset = 0i64;
                for work in works {
                    let request_size = work.input.size()[0];
                    let mut policy = policy_batch.narrow(0, offset, request_size);
                    let mut value = value_batch.narrow(0, offset, request_size);
                    if work.unbatched {
                        policy = policy.squeeze_dim(0);
                        value = value.squeeze_dim(0);
                    }
                    let _ = work.resp_tx.send(Ok((policy, value)));
                    offset += request_size;
                }
            }
            Err(err) => {
                // Convert to a simple Kind error to clone for each responder.
                let msg = format!("Batch forward failed: {err}");
                for work in works {
                    let _ = work.resp_tx.send(Err(tch::TchError::Kind(msg.clone())));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Barrier;
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
            assert_eq!(policy.size(), [1, 64]);
            assert_eq!(value.size(), [1, 1]);
        }
    }

    struct RecordingModel {
        calls: Arc<AtomicUsize>,
        largest_batch: Arc<AtomicUsize>,
    }

    impl PolicyValueModel for RecordingModel {
        fn forward(&self, x: &Tensor) -> tch::Result<(Tensor, Tensor)> {
            let batch = x.size()[0] as usize;
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.largest_batch.fetch_max(batch, Ordering::SeqCst);
            Ok((
                Tensor::zeros([batch as i64, 64], (Kind::Float, Device::Cpu)),
                Tensor::zeros([batch as i64, 1], (Kind::Float, Device::Cpu)),
            ))
        }

        fn device(&self) -> Device {
            Device::Cpu
        }
    }

    #[test]
    fn combines_complete_request_batches_and_preserves_shapes() {
        let calls = Arc::new(AtomicUsize::new(0));
        let largest_batch = Arc::new(AtomicUsize::new(0));
        let model = RecordingModel {
            calls: Arc::clone(&calls),
            largest_batch: Arc::clone(&largest_batch),
        };
        let batching = BatchingModel::new(model, 8, Duration::from_millis(100));
        let barrier = Arc::new(Barrier::new(3));

        let handles: Vec<_> = (0..2)
            .map(|_| {
                let batching = batching.clone();
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let input = Tensor::zeros([4, 3, 8, 8], (Kind::Float, Device::Cpu));
                    barrier.wait();
                    batching.forward(&input)
                })
            })
            .collect();
        barrier.wait();

        for handle in handles {
            let (policy, value) = handle.join().unwrap().unwrap();
            assert_eq!(policy.size(), [4, 64]);
            assert_eq!(value.size(), [4, 1]);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 1);
        assert_eq!(largest_batch.load(Ordering::SeqCst), 8);
    }

    #[test]
    fn preserves_unbatched_output_contract() {
        let model = DummyModel::new();
        let batching = BatchingModel::new(model, 8, Duration::from_millis(1));
        let input = Tensor::zeros([3, 8, 8], (Kind::Float, Device::Cpu));

        let (policy, value) = batching.forward(&input).unwrap();

        assert_eq!(policy.size(), [64]);
        assert_eq!(value.size(), [1]);
    }
}
