use reversi_core::Board;
use reversi_nn::NnModel;
use tch::{Device, IndexOp};

fn main() -> anyhow::Result<()> {
    let model_path = "../models/ts/latest.pt";

    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    println!("Loading TorchScript model from {:?}...", model_path);
    let model = NnModel::load(model_path, device)?;

    let board = Board::new();
    let board_tensor = board.to_tensor();
    let x = board_tensor.unsqueeze(0).to_device(device);

    let (policy, value) = model.forward(&x)?;

    println!("policy shape: {:?}", policy.size());
    println!("value shape:  {:?}", value.size());

    println!("policy[0, 0..8]: {:?}", policy.i((0, 0..8)));
    println!("value[0]: {:?}", value.i(0));

    Ok(())
}
