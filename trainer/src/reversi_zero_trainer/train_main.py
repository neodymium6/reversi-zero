"""
Main AlphaZero training loop integrating self-play and neural network training.

Each iteration:
1. Generate self-play games with current model
2. Train neural network on the generated data
3. Export updated model to TorchScript
4. Repeat
"""

from pathlib import Path
from typing import Literal

import torch
from reversi_zero_rs import (
    BatchConfigArgs,
    MctsConfigArgs,
    SelfPlayStream,
)

from reversi_zero_trainer.logging import (
    ConsoleConfig,
    LoggerKind,
    LoggingConfig,
    create_logger,
    log_hyperparameters,
    log_selfplay_stats,
    log_training_metrics,
)
from reversi_zero_trainer.models.dummy import ResNetReversiNet
from reversi_zero_trainer.training import AlphaZeroTrainer, TrainingConfig


def export_model_to_torchscript(
    model: torch.nn.Module,
    output_path: Path | str,
    device: str = "cuda",
) -> None:
    """
    Export PyTorch model to TorchScript format for Rust inference.

    Args:
        model: PyTorch model to export
        output_path: Path to save TorchScript model
        device: Device to use for export
    """
    model.eval()
    model.to(device)

    # Create dummy input
    dummy_input = torch.randn(1, 3, 8, 8, device=device)

    # Trace the model
    traced_model = torch.jit.trace(model, dummy_input)
    if not isinstance(traced_model, torch.jit.ScriptModule):
        raise ValueError(
            "Model must be a torch.jit.ScriptModule for TorchScript export."
        )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    traced_model.save(str(output_path))


def main() -> None:
    """Run AlphaZero training loop."""

    # Global configuration
    num_iterations = 10
    device: Literal["cuda", "cpu"] = "cuda" if torch.cuda.is_available() else "cpu"

    # Self-play configuration
    factor = 1
    selfplay_games_per_iter = 128 * factor
    selfplay_report_interval = 128 * factor // 8
    selfplay_batch_size = 128
    selfplay_game_concurrency = 32
    selfplay_batch_timeout_ms = 1
    selfplay_num_simulations = 100
    selfplay_expansion_batch_size = 2
    selfplay_c_puct = 3.0

    # Training configuration
    train_config = TrainingConfig(
        batch_size=256,
        num_workers=4,
        num_epochs=10,
        learning_rate=0.001,
        weight_decay=1e-4,
        policy_loss_weight=1.0,
        value_loss_weight=1.0,
        device=device,
        checkpoint_dir=Path("checkpoints"),
        save_every_n_epochs=5,
        # Arena evaluation
        arena_enabled=True,
        arena_vs_alphabeta=True,
        arena_vs_random=True,
        arena_games=10,
        arena_mcts_sims=400,
        arena_alphabeta_temperature=0.5,  # Add diversity vs deterministic opponent
        arena_random_temperature=0.0,  # Deterministic vs random opponent
        arena_device=None,  # Use training device
    )

    # Model configuration
    model_channels = 64
    model_num_blocks = 6

    # Paths
    data_base_dir = Path("data")
    models_dir = Path("models/ts")
    models_dir.mkdir(parents=True, exist_ok=True)

    # Logging configuration
    logging_cfg = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(
                verbose=True,
                show_params_table=True,
                show_timestamp=True,
            ),
        }
    )

    # Initialize model
    model = ResNetReversiNet(
        in_channels=3,
        channels=model_channels,
        num_blocks=model_num_blocks,
    )

    # Export initial model (random weights)
    initial_model_path = models_dir / "model_iter_0.pt"
    export_model_to_torchscript(model, initial_model_path, device=device)

    # Initialize trainer (will be reused across iterations)
    trainer = AlphaZeroTrainer(model=model, config=train_config)

    with create_logger(logging_cfg) as logger:
        # Log all hyperparameters upfront (before any metrics)
        log_hyperparameters(
            logger=logger,
            num_iterations=num_iterations,
            selfplay_config={
                "games_per_iter": selfplay_games_per_iter,
                "report_interval": selfplay_report_interval,
                "batch_size": selfplay_batch_size,
                "game_concurrency": selfplay_game_concurrency,
                "num_simulations": selfplay_num_simulations,
            },
            train_config=train_config,
            model_config={
                "channels": model_channels,
                "num_blocks": model_num_blocks,
            },
            arena_config={
                "enabled": train_config.arena_enabled,
                "vs_alphabeta": train_config.arena_vs_alphabeta,
                "vs_random": train_config.arena_vs_random,
                "games": train_config.arena_games,
                "mcts_sims": train_config.arena_mcts_sims,
                "alphabeta_temperature": train_config.arena_alphabeta_temperature,
                "random_temperature": train_config.arena_random_temperature,
            },
            paths={
                "data_base_dir": data_base_dir,
                "models_dir": models_dir,
                "checkpoint_dir": Path(train_config.checkpoint_dir),
            },
            device=device,
        )

        # Global step counter (continuous across all iterations)
        global_step = 0

        # Main AlphaZero loop
        for iteration in range(num_iterations):
            # Step 1: Self-play with current model
            current_model_path = models_dir / f"model_iter_{iteration}.pt"
            selfplay_data_dir = data_base_dir / f"selfplay_iter_{iteration}"

            # Run self-play
            stream = SelfPlayStream(
                total_games=selfplay_games_per_iter,
                report_interval=selfplay_report_interval,
                batch=BatchConfigArgs(
                    batch_size=selfplay_batch_size,
                    game_concurrency=selfplay_game_concurrency,
                    batch_timeout_ms=selfplay_batch_timeout_ms,
                ),
                mcts=MctsConfigArgs(
                    num_simulations=selfplay_num_simulations,
                    c_puct=selfplay_c_puct,
                    expansion_batch_size=selfplay_expansion_batch_size,
                ),
                model_path=str(current_model_path),
                device=device,
                save_dir=str(selfplay_data_dir),
            )

            for step, stats in enumerate(stream):
                log_selfplay_stats(logger, stats, iteration, global_step)
                global_step += 1

            # Step 2: Train on self-play data
            for epoch_metrics in trainer.train(
                data_path=selfplay_data_dir, num_epochs=train_config.num_epochs
            ):
                log_training_metrics(logger, epoch_metrics, iteration, global_step)
                global_step += 1

            # Step 3: Save checkpoint and export to TorchScript for next iteration
            checkpoint_path = trainer.save_checkpoint(
                iteration, filename=f"checkpoint_iter_{iteration}.pt"
            )
            logger.log_artifact("checkpoint", str(checkpoint_path))

            # Export updated model for next iteration's self-play
            next_model_path = models_dir / f"model_iter_{iteration + 1}.pt"
            export_model_to_torchscript(model, next_model_path, device=device)
            logger.log_artifact("model", str(next_model_path))

        # Save final model
        final_model_path = models_dir / "model_final.pt"
        export_model_to_torchscript(model, final_model_path, device=device)
        logger.log_artifact("final model", str(final_model_path))


if __name__ == "__main__":
    main()
