use reversi_core::Board;
use reversi_mcts::{Mcts, MctsConfig};
use reversi_nn::NnModel;
use tch::Device;

fn main() -> anyhow::Result<()> {
    let model_path = "../models/ts/latest.pt";

    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    println!("Loading TorchScript model from {:?}...", model_path);
    let model = NnModel::load(model_path, device)?;
    println!("Model loaded successfully!\n");

    // Create MCTS
    let mut mcts = Mcts::new();
    println!("Created MCTS instance\n");

    // Configure MCTS for quick demo
    let config = MctsConfig::default()
        .with_simulations(100) // Quick demo with 100 simulations
        .with_c_puct(1.5)
        .with_temperature(0.0); // Deterministic best move

    println!("MCTS Configuration:");
    println!("  Simulations: {}", config.num_simulations);
    println!("  C_PUCT: {}", config.c_puct);
    println!("  Temperature: {}", config.temperature);
    println!("  Dirichlet noise: {}\n", config.add_dirichlet_noise);

    // Initial board
    let board = Board::new();
    println!("Starting position (Black to move)");
    println!("Running MCTS search...\n");

    // Run MCTS search
    let result = mcts.search(&board, &model, &config)?;

    println!("Search completed!");
    println!("  Tree size: {} nodes", mcts.tree_size());
    match result.best_move {
        Some(best_move) => {
            println!("  Best move: {}", best_move);
            println!("  Root value: {:.4}", result.root_value);
            println!("\nTop moves by visit count:");

            // Sort and display top 5 moves
            let mut visits = result.root_visit_counts.clone();
            visits.sort_by_key(|(_, v)| std::cmp::Reverse(*v));

            for (i, (move_idx, count)) in visits.iter().take(5).enumerate() {
                let row = move_idx / 8;
                let col = move_idx % 8;
                let prob = result.policy_distribution[*move_idx];
                println!(
                    "  {}. Move {} (row {}, col {}): {} visits ({:.1}%)",
                    i + 1,
                    move_idx,
                    row,
                    col,
                    count,
                    prob * 100.0
                );
            }
        }
        None => {
            println!("  Best move: pass (no legal moves)");
            println!("  Root value: {:.4}", result.root_value);
        }
    }

    println!("\nDemo completed successfully!");
    Ok(())
}
