// Test if BatchingModel handles duplicate inputs correctly
use reversi_mcts::{BatchingModel, PolicyValueModel};
use reversi_nn::NnModel;
use tch::{Device, Tensor};
use std::time::Duration;

fn main() {
    // Load model
    let model_path = "../trainer/models/ts/model_iter_0.pt";
    let device = Device::Cpu;
    let base_model = NnModel::load(model_path, device).expect("Failed to load model");

    let batching_model = BatchingModel::new(base_model, 512, Duration::from_millis(2));

    // Create identical inputs (same board state)
    let same_input = Tensor::randn([1, 3, 8, 8], (tch::Kind::Float, device));

    println!("Testing BatchingModel with duplicate inputs...");
    println!("Sending 10 identical inputs concurrently\n");

    // Spawn 10 threads with the same input
    let handles: Vec<_> = (0..10).map(|i| {
        let input = same_input.shallow_clone();
        let model = batching_model.clone();

        std::thread::spawn(move || {
            let result = model.forward(&input).expect("Forward failed");
            (i, result)
        })
    }).collect();

    // Collect results
    let mut results = Vec::new();
    for h in handles {
        let (i, (policy, value)) = h.join().unwrap();
        results.push((i, policy, value));
    }

    // Check if all results are identical
    println!("Checking if all outputs are identical...");
    let (_, first_policy, first_value) = &results[0];

    let mut all_identical = true;
    for (i, policy, value) in &results[1..] {
        let policy_equal = policy.allclose(first_policy, 1e-6, 1e-6, false);
        let value_equal = value.allclose(first_value, 1e-6, 1e-6, false);

        if !policy_equal || !value_equal {
            println!("Thread {} has different output!", i);
            all_identical = false;
        }
    }

    if all_identical {
        println!("✓ All 10 outputs are identical (EXPECTED for same input)");
    } else {
        println!("✗ Outputs differ (UNEXPECTED - possible bug!)");
    }

    // Now test with 32 games running MCTS simultaneously
    println!("\n\nTesting 32 concurrent MCTS searches from initial board...");

    use reversi_core::Board;
    use reversi_mcts::{Mcts, MctsConfig};

    let config = MctsConfig::default()
        .with_simulations(500)
        .with_batch_size(512)
        .with_batch_timeout_ms(2)
        .with_c_puct(1.5)
        .with_temperature(1.0)
        .with_dirichlet_noise(0.3, 0.25);

    let handles: Vec<_> = (0..32).map(|i| {
        let model = batching_model.clone();
        let cfg = config.clone();

        std::thread::spawn(move || {
            let board = Board::new();
            let mut mcts = Mcts::new();
            let result = mcts.search(&board, &model, &cfg).expect("MCTS failed");
            (i, result.best_move, result.policy_distribution)
        })
    }).collect();

    let mut mcts_results = Vec::new();
    for h in handles {
        let (i, best_move, policy) = h.join().unwrap();
        mcts_results.push((i, best_move, policy));
    }

    // Check diversity
    let unique_moves: std::collections::HashSet<_> = mcts_results.iter()
        .filter_map(|(_, m, _)| *m)
        .collect();

    println!("Number of unique first moves: {} / 32", unique_moves.len());

    if unique_moves.len() == 1 {
        println!("✗ BUG CONFIRMED: All 32 games chose the same move!");
        println!("   Move chosen: {:?}", mcts_results[0].1);
    } else if unique_moves.len() < 5 {
        println!("⚠ Low diversity: Only {} unique moves", unique_moves.len());
    } else {
        println!("✓ Good diversity: {} different moves chosen", unique_moves.len());
    }
}
