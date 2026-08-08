"""Helper functions for logging metrics in training."""

from pathlib import Path
from typing import Any

from reversi_zero_rs import SelfPlayStats

from .base import BaseLogger
from ..training import TrainingConfig


def log_selfplay_stats(
    logger: BaseLogger,
    stats: SelfPlayStats,
    step: int,
) -> None:
    """Log self-play statistics.

    Args:
        logger: Logger instance
        stats: Self-play statistics
        step: Position in the self-play report series
    """
    prefix = "selfplay"

    metrics: dict[str, float] = {}
    for attr_name in dir(stats):
        # Skip private/magic attributes and methods
        if attr_name.startswith("_") or callable(getattr(stats, attr_name)):
            continue

        metrics[f"{prefix}/{attr_name}"] = float(getattr(stats, attr_name))
    logger.log_metrics(metrics, step=step, color="magenta")


def log_training_metrics(
    logger: BaseLogger,
    metrics: dict[str, Any],
    step: int,
) -> None:
    """Log training metrics.

    Args:
        logger: Logger instance
        metrics: Training metrics dictionary (contains all metrics from trainer)
        step: Position in the training epoch series
    """
    prefix = "train"

    logger.log_metrics(
        {
            f"{prefix}/{metric_name}": float(metric_value)
            for metric_name, metric_value in metrics.items()
        },
        step=step,
        color="cyan",
    )


def log_promotion_metrics(
    logger: BaseLogger,
    report: dict[str, Any],
    accepted: bool,
    step: int,
) -> None:
    """Log candidate-vs-incumbent promotion results."""
    summary = report["summary"]
    interval = summary["score_interval_95"]
    metrics = {
        "score": summary["score"],
        "score_interval_low": interval[0],
        "score_interval_high": interval[1],
        "wins": summary["wins"],
        "draws": summary["draws"],
        "losses": summary["losses"],
        "accepted": float(accepted),
    }
    logger.log_metrics(
        {f"promotion/{name}": float(value) for name, value in metrics.items()},
        step=step,
        color="green",
    )


def log_reference_metrics(
    logger: BaseLogger,
    opponent: str,
    report: dict[str, Any],
    step: int,
) -> None:
    """Log paired incumbent-vs-reference evaluation results."""
    summary = report["summary"]
    interval = summary["score_interval_95"]
    metrics = {
        "score": summary["score"],
        "score_interval_low": interval[0],
        "score_interval_high": interval[1],
        "wins": summary["wins"],
        "draws": summary["draws"],
        "losses": summary["losses"],
    }
    logger.log_metrics(
        {
            f"reference/{opponent}/{name}": float(value)
            for name, value in metrics.items()
        },
        step=step,
        color="yellow",
    )


def log_hyperparameters(
    logger: BaseLogger,
    num_iterations: int,
    seed: int,
    replay_window: int,
    selfplay_config: dict[str, Any],
    train_config: TrainingConfig,
    model_config: dict[str, Any],
    reference_config: dict[str, Any],
    promotion_config: dict[str, Any],
    paths: dict[str, Path],
    device: str,
) -> None:
    """Log all hyperparameters at the beginning.

    Args:
        logger: Logger instance
        num_iterations: Number of training iterations
        seed: PyTorch seed for model initialization and training shuffles
        replay_window: Number of recent self-play iterations used for training
        selfplay_config: Self-play configuration
        train_config: Training configuration
        model_config: Model configuration
        reference_config: Fixed reference evaluation configuration
        promotion_config: Candidate promotion configuration
        paths: Dictionary of paths (data_base_dir, models_dir, checkpoint_dir)
        device: Device being used
    """
    logger.log_params(
        {
            "num_iterations": num_iterations,
            "device": device,
            "seed": seed,
            "train_replay_window": replay_window,
            "selfplay_games_per_iter": selfplay_config["games_per_iter"],
            "selfplay_report_interval": selfplay_config["report_interval"],
            "selfplay_batch_size": selfplay_config["batch_size"],
            "selfplay_game_concurrency": selfplay_config["game_concurrency"],
            "selfplay_batch_timeout_ms": selfplay_config["batch_timeout_ms"],
            "selfplay_expansion_batch_size": selfplay_config["expansion_batch_size"],
            "torch_threads": selfplay_config["torch_threads"],
            "selfplay_num_simulations": selfplay_config["num_simulations"],
            "train_batch_size": train_config.batch_size,
            "train_num_workers": train_config.num_workers,
            "train_num_epochs": train_config.num_epochs,
            "train_symmetry_augmentation": train_config.symmetry_augmentation,
            "train_learning_rate": train_config.learning_rate,
            "train_lr_schedule": train_config.lr_schedule,
            "train_weight_decay": train_config.weight_decay,
            "train_policy_loss_weight": train_config.policy_loss_weight,
            "train_value_loss_weight": train_config.value_loss_weight,
            "train_dtype": train_config.dtype,
            "model_type": model_config["type"],
            "model_channels": model_config["channels"],
            "model_num_blocks": model_config["num_blocks"],
            "reference_eval_enabled": reference_config["enabled"],
            "reference_games": reference_config["games"],
            "reference_opponents": "bitmatrix",
            "promotion_enabled": promotion_config["enabled"],
            "promotion_num_openings": promotion_config["num_openings"],
            "promotion_opening_plies": promotion_config["opening_plies"],
            "promotion_mcts_sims": promotion_config["mcts_sims"],
            "promotion_c_puct": promotion_config["c_puct"],
            "promotion_expansion_batch_size": promotion_config["expansion_batch_size"],
            "promotion_threshold": promotion_config["threshold"],
            "promotion_require_confidence": promotion_config["require_confidence"],
            "run_dir": str(paths["run_dir"]),
            "data_base_dir": str(paths["data_base_dir"]),
            "models_dir": str(paths["models_dir"]),
            "checkpoint_dir": str(paths["checkpoint_dir"]),
        }
    )
