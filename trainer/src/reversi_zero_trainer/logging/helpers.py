"""Helper functions for logging metrics in training."""

from pathlib import Path
from typing import Any

from reversi_zero_rs import SelfPlayStats

from .base import BaseLogger
from ..training import TrainingConfig


def log_selfplay_stats(
    logger: BaseLogger,
    stats: SelfPlayStats,
    iteration: int,
    step: int,
) -> None:
    """Log self-play statistics.

    Args:
        logger: Logger instance
        stats: Self-play statistics
        iteration: Current iteration number
        step: Global step counter
    """
    prefix = f"iter_{iteration}/selfplay"

    # Log all public attributes from SelfPlayStats
    for attr_name in dir(stats):
        # Skip private/magic attributes and methods
        if attr_name.startswith("_") or callable(getattr(stats, attr_name)):
            continue

        value = getattr(stats, attr_name)
        logger.log_metric(
            f"{prefix}/{attr_name}", float(value), step=step, color="magenta"
        )


def log_training_metrics(
    logger: BaseLogger,
    metrics: dict[str, Any],
    iteration: int,
    step: int,
) -> None:
    """Log training metrics.

    Args:
        logger: Logger instance
        metrics: Training metrics dictionary (contains all metrics from trainer)
        iteration: Current iteration number
        step: Global step counter
    """
    prefix = f"iter_{iteration}"

    # Log all metrics from the dictionary
    for metric_name, metric_value in metrics.items():
        # Metric names already contain the path (e.g., "loss/total", "eval/loss_total", "epoch", etc.)
        logger.log_metric(
            f"{prefix}/{metric_name}", float(metric_value), step=step, color="cyan"
        )


def log_hyperparameters(
    logger: BaseLogger,
    num_iterations: int,
    selfplay_config: dict[str, Any],
    train_config: TrainingConfig,
    model_config: dict[str, Any],
    arena_config: dict[str, Any],
    paths: dict[str, Path],
    device: str,
) -> None:
    """Log all hyperparameters at the beginning.

    Args:
        logger: Logger instance
        num_iterations: Number of training iterations
        selfplay_config: Self-play configuration
        train_config: Training configuration
        model_config: Model configuration
        arena_config: Arena evaluation configuration
        paths: Dictionary of paths (data_base_dir, models_dir, checkpoint_dir)
        device: Device being used
    """
    # Global settings
    logger.log_param("num_iterations", num_iterations)
    logger.log_param("device", device)

    # Self-play parameters
    logger.log_param("selfplay_games_per_iter", selfplay_config["games_per_iter"])
    logger.log_param("selfplay_report_interval", selfplay_config["report_interval"])
    logger.log_param("selfplay_batch_size", selfplay_config["batch_size"])
    logger.log_param("selfplay_game_concurrency", selfplay_config["game_concurrency"])
    logger.log_param("selfplay_num_simulations", selfplay_config["num_simulations"])

    # Training parameters
    logger.log_param("train_batch_size", train_config.batch_size)
    logger.log_param("train_num_epochs", train_config.num_epochs)
    logger.log_param("train_learning_rate", train_config.learning_rate)
    logger.log_param("train_weight_decay", train_config.weight_decay)
    logger.log_param("train_policy_loss_weight", train_config.policy_loss_weight)
    logger.log_param("train_value_loss_weight", train_config.value_loss_weight)

    # Model parameters
    logger.log_param("model_channels", model_config["channels"])
    logger.log_param("model_num_blocks", model_config["num_blocks"])

    # Arena evaluation parameters
    logger.log_param("arena_enabled", arena_config["enabled"])
    logger.log_param("arena_vs_alphabeta", arena_config["vs_alphabeta"])
    logger.log_param("arena_vs_random", arena_config["vs_random"])
    logger.log_param("arena_games", arena_config["games"])
    logger.log_param("arena_mcts_sims", arena_config["mcts_sims"])
    logger.log_param(
        "arena_alphabeta_temperature", arena_config["alphabeta_temperature"]
    )
    logger.log_param("arena_random_temperature", arena_config["random_temperature"])

    # Paths
    logger.log_param("data_base_dir", str(paths["data_base_dir"]))
    logger.log_param("models_dir", str(paths["models_dir"]))
    logger.log_param("checkpoint_dir", str(paths["checkpoint_dir"]))
