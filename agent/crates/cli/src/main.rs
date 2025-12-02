use reversi_nn::NnModel;
use tch::{Cuda, Device, IndexOp, Kind, Tensor};

fn main() -> anyhow::Result<()> {
    println!("tch cuda available: {}", Cuda::is_available());
    println!("tch cuda device count: {}", Cuda::device_count());
    let model_path = "../models/ts/latest.pt";

    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    println!("Loading TorchScript model from {:?}...", model_path);
    let model = NnModel::load(model_path, device)?;

    let x = Tensor::zeros(&[1, 3, 8, 8], (Kind::Float, Device::cuda_if_available()));

    let (policy, value) = model.forward(&x)?;

    println!("policy shape: {:?}", policy.size());
    println!("value shape:  {:?}", value.size());

    println!("policy[0, 0..8]: {:?}", policy.i((0, 0..8)));
    println!("value[0]: {:?}", value.i(0));

    Ok(())
}
