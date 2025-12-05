use reversi_core::Board;
use reversi_mcts::{Mcts, MctsConfig, PolicyValueModel};
use tch::{Device, Tensor};

struct DummyModel;

impl DummyModel {
    fn forward_impl(&self, _x: &Tensor) -> tch::Result<(Tensor, Tensor)> {
        Ok((
            Tensor::zeros([1, 64], (tch::Kind::Float, tch::Device::Cpu)),
            Tensor::zeros([1], (tch::Kind::Float, tch::Device::Cpu)),
        ))
    }

    fn device_impl(&self) -> Device {
        Device::Cpu
    }
}

impl PolicyValueModel for DummyModel {
    fn forward(&self, x: &Tensor) -> tch::Result<(Tensor, Tensor)> {
        self.forward_impl(x)
    }

    fn device(&self) -> Device {
        self.device_impl()
    }
}

struct FixedModel {
    logits: Vec<f32>,
    value: f32,
}

impl PolicyValueModel for FixedModel {
    fn forward(&self, _x: &Tensor) -> tch::Result<(Tensor, Tensor)> {
        let policy = Tensor::from_slice(&self.logits).view([1, 64]);
        let value = Tensor::from_slice(&[self.value]).view([1]);
        Ok((policy, value))
    }

    fn device(&self) -> Device {
        Device::Cpu
    }
}

#[test]
fn search_with_dummy_model_uniform_policy() {
    let mut mcts = Mcts::new();
    let board = Board::new();
    let config = MctsConfig::default()
        .with_simulations(10)
        .with_temperature(0.0);

    let result = mcts.search(&board, &DummyModel, &config).unwrap();

    // With uniform policy/logits and low sims, best_move should be a legal move index (0-63)
    assert!(result.best_move.is_some());
    assert!(result.best_move.unwrap() < 64);

    // Policy distribution sums to ~1
    let sum: f32 = result.policy_distribution.iter().sum();
    assert!((sum - 1.0).abs() < 1e-3);

    // Visit counts are recorded for legal moves only
    assert!(!result.root_visit_counts.is_empty());
}

#[test]
fn search_picks_high_prior_move() {
    // Prefer move 19 (row 2, col 3) which is legal in the initial position
    let mut logits = vec![0.0f32; 64];
    logits[19] = 10.0;

    let model = FixedModel { logits, value: 0.0 };
    let mut mcts = Mcts::new();
    let board = Board::new();
    let config = MctsConfig::default()
        .with_simulations(8)
        .with_temperature(0.0);

    let result = mcts.search(&board, &model, &config).unwrap();

    assert_eq!(result.best_move, Some(19));
    assert!(result.policy_distribution.iter().sum::<f32>() > 0.99);
    assert!(result.root_visit_counts.iter().any(|(m, _)| *m == 19));
}
