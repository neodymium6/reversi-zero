# Reversi-Zero

AlphaZero-style reinforcement learning system for Reversi (Othello), combining high-performance Rust components with flexible Python training.

## Overview

This project implements a complete AlphaZero training pipeline:
- **Self-play game generation** (Rust) - Fast parallel MCTS with batched neural network inference
- **Neural network training** (Python/PyTorch) - Policy and value network optimization
- **Arena evaluation** - Automated testing against baseline opponents

## Project Structure

```
reversi-zero/
├── agent/               # Rust implementation (performance-critical)
│   ├── crates/
│   │   ├── core/       # Board representation & tensor conversion
│   │   ├── mcts/       # Monte Carlo Tree Search
│   │   ├── nn/         # Neural network inference (TorchScript)
│   │   ├── selfplay/   # Self-play game generation
│   │   └── reversi_zero_rs/  # PyO3 Python bindings
│   └── target/         # Build artifacts
├── trainer/            # Python implementation (training & evaluation)
│   ├── src/reversi_zero_trainer/
│   │   ├── models/     # Neural network architectures
│   │   ├── logging/    # Logging system
│   │   ├── training.py # AlphaZero trainer
│   │   └── train_main.py  # Main training loop
│   ├── players/        # Arena evaluation players
│   ├── data/           # Training data (self-play games)
│   └── checkpoints/    # Model checkpoints
└── models/             # Exported TorchScript models
```

## Installation

### Prerequisites

- **Rust** (stable) - [Install](https://rustup.rs/)
- **Python** 3.10+
- **uv** - [Install](https://docs.astral.sh/uv/)
- **PyTorch** 2.9.0 with libtorch
- **CUDA** (optional, for GPU acceleration)

### Setup

```bash
# Clone repository
git clone https://github.com/neodymium6/reversi-zero.git
cd reversi-zero

# Initialize development environment
make init

# Build Rust components
cd agent && cargo build --release

# Install Python dependencies
cd ../trainer && uv sync
```

## Quick Start

### Train a Model

```bash
cd trainer
uv run train
```

This runs the full AlphaZero training loop (10 iterations by default):
1. **Self-play generation** - Configurable games per iteration (default: 128)
2. **Neural network training** - 10 epochs per iteration
3. **Arena evaluation** - Test vs Alpha-Beta and Random players

Training configuration is in `trainer/src/reversi_zero_trainer/train_main.py`.

**Note**: Adjust `factor` variable in `train_main.py` to scale game generation (e.g., `factor=32` → 4,096 games/iteration).

### Training Output

```
data/
├── selfplay_iter_0/    # Self-play data for iteration 0
│   ├── states.npy      # Board states (N, 3, 8, 8)
│   ├── policies.npy    # MCTS visit distributions (N, 64)
│   └── values.npy      # Game outcomes (N,)
├── selfplay_iter_1/
└── ...

checkpoints/
├── checkpoint_iter_0.pt   # Training checkpoint
├── checkpoint_iter_1.pt
└── ...

models/
├── model_iter_0.pt     # TorchScript model (for self-play)
├── model_iter_1.pt
└── ...
```

## Key Features

### MCTS Implementation

- **PUCT selection** with proper minimax Q-value handling
- **Dirichlet noise** for exploration during self-play
- **Batched inference** - Groups neural network calls for GPU efficiency
- **Parallel self-play** - 32 concurrent games using rayon
- **Temperature sampling** - Configurable exploration vs exploitation

### Training System

- **Policy + Value loss** - Cross-entropy for policy, MSE for value
- **Arena evaluation** - Automated testing during training
- **Rich logging** - Real-time metrics with console output
- **Checkpointing** - Save/load training state

### Arena Evaluation

Tests trained models against baseline opponents:
- **Alpha-Beta** - Traditional minimax search
- **Random** - Random move selection
- **Temperature control** - Adds diversity to deterministic opponents

## Configuration

### Self-Play Parameters

Key parameters in `train_main.py`:

```python
# Game generation
selfplay_games_per_iter = 128 * factor    # Games per iteration
selfplay_game_concurrency = 32             # Parallel games
selfplay_num_simulations = 100             # MCTS simulations per move

# MCTS configuration
selfplay_c_puct = 3.0                      # Exploration constant
selfplay_expansion_batch_size = 2          # Expansion batch size
```

### Training Parameters

```python
# Training configuration
batch_size = 256                           # Training batch size
num_epochs = 10                            # Epochs per iteration
learning_rate = 0.001                      # Adam learning rate
weight_decay = 1e-4                        # L2 regularization
```

### Arena Evaluation

```python
# Arena configuration
arena_enabled = True                       # Enable arena evaluation
arena_games = 10                           # Games per opponent
arena_mcts_sims = 400                      # MCTS simulations
arena_alphabeta_temperature = 0.5          # Temperature vs Alpha-Beta
arena_random_temperature = 0.0             # Temperature vs Random
```

## Development

### Run Tests

```bash
# All tests
make test

# Rust tests only
cd agent && cargo test

# Python tests only
cd trainer && uv run pytest
```

### Code Formatting

```bash
# Format all code
make format

# Lint all code
make lint
```

### Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

## Architecture Details

### Rust → Python Integration

1. **Model Export**: PyTorch models traced to TorchScript format (`.pt`)
2. **Self-Play**: Rust loads TorchScript, runs MCTS, saves NumPy arrays
3. **Training**: Python loads NumPy arrays, trains model, exports TorchScript
4. **Evaluation**: Python spawns Rust player processes via Arena

### Data Flow

```
Python Trainer
    ↓ (TorchScript export)
Rust Self-Play
    ↓ (NumPy arrays)
Python Training
    ↓ (Updated model)
Arena Evaluation
    ↓ (Metrics)
Next Iteration
```

## License

MIT

## Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Run tests and formatting
4. Submit a pull request
