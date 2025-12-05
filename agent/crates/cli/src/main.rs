use anyhow::Result;
use reversi_mcts::MctsConfig;
use reversi_nn::NnModel;
use reversi_selfplay::{game_to_training_examples, play_game, save_training_data};
use tch::Device;

fn main() -> Result<()> {
    // Configuration
    let model_path =
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "../models/ts/latest.pt".to_string());
    let num_games = std::env::var("NUM_GAMES")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let num_simulations = std::env::var("NUM_SIMULATIONS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(800);
    let output_path = std::env::var("OUTPUT_PATH").unwrap_or_else(|_| "selfplay_data".to_string());

    // Device selection
    let device = Device::cuda_if_available();
    println!("Using device: {:?}", device);

    // Load model
    println!("Loading model from {}...", model_path);
    let model = NnModel::load(&model_path, device)?;
    println!("Model loaded successfully!");

    // Configure MCTS for self-play
    let config = MctsConfig::default()
        .with_simulations(num_simulations)
        .with_c_puct(1.5)
        .with_temperature(1.0) // Stochastic for exploration
        .with_dirichlet_noise(0.3, 0.25); // Add exploration noise

    println!("\nSelf-play configuration:");
    println!("  Number of games: {}", num_games);
    println!("  MCTS simulations: {}", config.num_simulations);
    println!("  C_PUCT: {}", config.c_puct);
    println!("  Temperature: {}", config.temperature);
    println!(
        "  Dirichlet noise: {} (alpha={}, epsilon={})",
        config.add_dirichlet_noise, config.dirichlet_alpha, config.dirichlet_epsilon
    );
    println!("  Output path: {}", output_path);
    println!();

    // Play multiple games
    let mut all_examples = Vec::new();

    for game_num in 0..num_games {
        println!("Playing game {}/{}...", game_num + 1, num_games);

        // Play one game
        let record = play_game(&model, &config)?;
        println!(
            "  Game finished: {} moves, result: {:?}",
            record.len(),
            record.winner
        );

        // Convert to training examples
        let examples = game_to_training_examples(&record);
        println!("  Generated {} training examples", examples.len());

        all_examples.extend(examples);
    }

    // Save all training data
    println!(
        "\nSaving {} total training examples to {}...",
        all_examples.len(),
        output_path
    );
    save_training_data(&all_examples, &output_path)?;

    println!("\nSelf-play completed successfully!");
    println!("Created files:");
    println!("  - {}_states.npy", output_path);
    println!("  - {}_policies.npy", output_path);
    println!("  - {}_values.npy", output_path);

    Ok(())
}
