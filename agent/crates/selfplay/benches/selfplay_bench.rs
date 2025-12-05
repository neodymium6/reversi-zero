use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use reversi_mcts::MctsConfig;
use reversi_nn::NnModel;
use reversi_selfplay::play_game;
use tch::Device;

/// Benchmark for a single self-play game with different MCTS simulation counts
fn bench_single_game(c: &mut Criterion) {
    let mut group = c.benchmark_group("single_game");

    // Load model once for all benchmarks
    let model_path =
        std::env::var("MODEL_PATH").unwrap_or_else(|_| "../../../models/ts/latest.pt".to_string());

    let device = Device::Cpu; // Use CPU for consistent benchmarking
    let model = NnModel::load(&model_path, device).expect("Failed to load model");

    // Benchmark with different simulation counts
    for num_sims in [1, 10, 30].iter() {
        let config = MctsConfig::default()
            .with_simulations(*num_sims)
            .with_c_puct(1.5)
            .with_temperature(1.0)
            .with_dirichlet_noise(0.3, 0.25);

        group.bench_with_input(BenchmarkId::from_parameter(num_sims), num_sims, |b, _| {
            b.iter(|| {
                let record = play_game(black_box(&model), black_box(&config)).expect("Game failed");
                black_box(record)
            });
        });
    }

    group.finish();
}

criterion_group!(benches, bench_single_game);
criterion_main!(benches);
