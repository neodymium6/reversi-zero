"""Safe, configurable AlphaZero training entry point."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal, Sequence

import torch
from reversi_zero_rs import BatchConfigArgs, MctsConfigArgs, SelfPlayStream

from reversi_zero_trainer.logging import (
    ConsoleConfig,
    LoggerKind,
    LoggingConfig,
    create_logger,
    log_hyperparameters,
    log_selfplay_stats,
    log_training_metrics,
)
from reversi_zero_trainer.models.dummy import DummyReversiNet, ResNetReversiNet
from reversi_zero_trainer.runtime import configure_training_threads
from reversi_zero_trainer.training import AlphaZeroTrainer, TrainingConfig


@dataclass(frozen=True)
class RunConfig:
    """Configuration for one isolated AlphaZero run."""

    run_dir: Path | None = None
    resume: bool = False
    num_iterations: int = 10
    device: Literal["auto", "cuda", "cpu"] = "auto"
    torch_threads: int | None = None

    selfplay_games_per_iter: int = 128
    selfplay_report_interval: int | None = None
    selfplay_batch_size: int | None = None
    selfplay_game_concurrency: int | None = None
    selfplay_batch_timeout_ms: int = 1
    selfplay_num_simulations: int = 100
    selfplay_expansion_batch_size: int = 2
    selfplay_c_puct: float = 3.0

    train_batch_size: int = 256
    train_num_workers: int = 4
    train_num_epochs: int = 10
    train_learning_rate: float = 0.001
    train_weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    arena_enabled: bool = True
    arena_games: int = 10
    arena_mcts_sims: int = 400
    arena_alphabeta_temperature: float = 0.5
    arena_random_temperature: float = 0.0

    model_type: Literal["dummy", "resnet"] = "dummy"
    model_channels: int = 64
    model_num_blocks: int = 6

    def report_interval(self) -> int:
        if self.selfplay_report_interval is not None:
            return self.selfplay_report_interval
        concurrency_floor = min(self.selfplay_games_per_iter, 16)
        return max(1, self.selfplay_games_per_iter // 8, concurrency_floor)

    def resolved_device(self) -> Literal["cuda", "cpu"]:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but it is not available")
        return self.device

    def resolved_torch_threads(self, device: str) -> int | None:
        if device != "cpu":
            return None
        return self.torch_threads if self.torch_threads is not None else 4

    def resolved_selfplay_batch_size(self, device: str) -> int:
        if self.selfplay_batch_size is not None:
            return self.selfplay_batch_size
        return 32 if device == "cpu" else 128

    def resolved_selfplay_game_concurrency(self, device: str) -> int:
        if self.selfplay_game_concurrency is not None:
            configured = self.selfplay_game_concurrency
        elif device == "cpu":
            affinity = getattr(os, "sched_getaffinity", None)
            available_cpus = (
                len(affinity(0)) if affinity is not None else os.cpu_count()
            )
            configured = min(16, available_cpus or 1)
        else:
            configured = 32
        return min(configured, self.selfplay_games_per_iter)

    def validate(self) -> None:
        positive_ints = {
            "num_iterations": self.num_iterations,
            "selfplay_games_per_iter": self.selfplay_games_per_iter,
            "selfplay_report_interval": self.report_interval(),
            "selfplay_batch_timeout_ms": self.selfplay_batch_timeout_ms,
            "selfplay_num_simulations": self.selfplay_num_simulations,
            "selfplay_expansion_batch_size": self.selfplay_expansion_batch_size,
            "train_batch_size": self.train_batch_size,
            "train_num_epochs": self.train_num_epochs,
            "arena_games": self.arena_games,
            "arena_mcts_sims": self.arena_mcts_sims,
            "model_channels": self.model_channels,
            "model_num_blocks": self.model_num_blocks,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        optional_positive_ints = {
            "torch_threads": self.torch_threads,
            "selfplay_batch_size": self.selfplay_batch_size,
            "selfplay_game_concurrency": self.selfplay_game_concurrency,
        }
        invalid.extend(
            name
            for name, value in optional_positive_ints.items()
            if value is not None and value <= 0
        )
        if invalid:
            raise ValueError(f"These settings must be > 0: {', '.join(invalid)}")
        if self.train_num_workers < 0:
            raise ValueError("train_num_workers must be >= 0")


@dataclass(frozen=True)
class RunState:
    run_dir: Path
    start_iteration: int
    checkpoint_path: Path | None


def default_run_dir(now: datetime | None = None) -> Path:
    """Return a unique-by-default run path relative to the current directory."""
    timestamp = (now or datetime.now()).strftime("%Y%m%d_%H%M%S_%f")
    return Path("runs") / timestamp


def _config_payload(config: RunConfig, run_dir: Path) -> dict[str, object]:
    payload = asdict(config)
    device = config.resolved_device()
    payload["run_dir"] = str(run_dir)
    payload["report_interval"] = config.report_interval()
    payload["torch_threads"] = config.resolved_torch_threads(device)
    payload["selfplay_batch_size"] = config.resolved_selfplay_batch_size(device)
    payload["selfplay_game_concurrency"] = config.resolved_selfplay_game_concurrency(
        device
    )
    return payload


def _write_run_config(config: RunConfig, run_dir: Path) -> None:
    config_path = run_dir / "run_config.json"
    next_path = run_dir / ".run_config.json.next"
    next_path.write_text(
        json.dumps(_config_payload(config, run_dir), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    next_path.replace(config_path)


def _validate_resume_model_config(config: RunConfig, run_dir: Path) -> None:
    config_path = run_dir / "run_config.json"
    if not config_path.is_file():
        raise RuntimeError(f"Cannot resume without {config_path}")

    stored = json.loads(config_path.read_text(encoding="utf-8"))
    for key in ("model_type", "model_channels", "model_num_blocks"):
        if stored.get(key) != getattr(config, key):
            raise RuntimeError(
                f"Resume configuration mismatch for {key}: "
                f"stored={stored.get(key)!r}, requested={getattr(config, key)!r}"
            )


def _find_resume_iteration(run_dir: Path) -> tuple[int, Path]:
    checkpoints_dir = run_dir / "checkpoints"
    models_dir = run_dir / "models" / "ts"
    pattern = re.compile(r"checkpoint_iter_(\d+)\.pt$")
    completed: list[tuple[int, Path]] = []

    for checkpoint in checkpoints_dir.glob("checkpoint_iter_*.pt"):
        match = pattern.fullmatch(checkpoint.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        model_path = models_dir / f"model_iter_{iteration + 1}.pt"
        if not model_path.is_file():
            raise RuntimeError(
                f"Incomplete run: {checkpoint} exists but {model_path} does not"
            )
        completed.append((iteration, checkpoint))

    if not completed:
        raise RuntimeError(f"No completed iteration found in {run_dir}")

    completed.sort(key=lambda item: item[0])
    iteration_numbers = [item[0] for item in completed]
    expected = list(range(iteration_numbers[-1] + 1))
    if iteration_numbers != expected:
        raise RuntimeError(
            f"Checkpoint sequence is not contiguous: found {iteration_numbers}"
        )

    latest_iteration, checkpoint_path = completed[-1]
    return latest_iteration + 1, checkpoint_path


def prepare_run(config: RunConfig) -> RunState:
    """Create a new run or validate a resumable existing run."""
    config.validate()
    run_dir = (config.run_dir or default_run_dir()).resolve()

    if not config.resume:
        if run_dir.exists():
            raise FileExistsError(
                f"Run directory already exists: {run_dir}. "
                "Choose another --run-dir or use --resume."
            )
        run_dir.mkdir(parents=True)
        _write_run_config(config, run_dir)
        return RunState(run_dir, 0, None)

    if config.run_dir is None:
        raise ValueError("--resume requires an explicit --run-dir")
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    _validate_resume_model_config(config, run_dir)
    start_iteration, checkpoint_path = _find_resume_iteration(run_dir)
    partial_data_dir = run_dir / "data" / f"selfplay_iter_{start_iteration}"
    if partial_data_dir.exists():
        raise RuntimeError(
            f"Refusing to resume with ambiguous partial data: {partial_data_dir}. "
            "Move it aside before resuming."
        )
    if start_iteration >= config.num_iterations:
        raise RuntimeError(
            f"Run already completed {start_iteration} iterations; "
            f"requested total is {config.num_iterations}"
        )

    return RunState(run_dir, start_iteration, checkpoint_path)


def export_model_to_torchscript(
    model: torch.nn.Module,
    output_path: Path | str,
    device: str = "cuda",
) -> None:
    """Atomically export a PyTorch model to TorchScript."""
    model.eval()
    model.to(device)
    dummy_input = torch.randn(1, 3, 8, 8, device=device)
    traced_model = torch.jit.trace(model, dummy_input)
    if not isinstance(traced_model, torch.jit.ScriptModule):
        raise ValueError(
            "Model must be a torch.jit.ScriptModule for TorchScript export."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    next_path = output_path.with_name(f".{output_path.name}.next")
    try:
        traced_model.save(str(next_path))
        next_path.replace(output_path)
    finally:
        next_path.unlink(missing_ok=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Reversi Zero safely")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num-iterations", type=int, default=10)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument(
        "--torch-threads",
        type=int,
        help="Torch CPU threads for self-play and training (default: 4 on CPU)",
    )

    parser.add_argument("--games-per-iteration", type=int, default=128)
    parser.add_argument("--report-interval", type=int)
    parser.add_argument(
        "--selfplay-batch-size",
        type=int,
        help="NN inference batch size (default: 32 on CPU, 128 on CUDA)",
    )
    parser.add_argument(
        "--game-concurrency",
        type=int,
        help="Parallel games (default: up to 16 on CPU, 32 on CUDA)",
    )
    parser.add_argument("--batch-timeout-ms", type=int, default=1)
    parser.add_argument("--simulations", type=int, default=100)
    parser.add_argument("--expansion-batch-size", type=int, default=2)
    parser.add_argument("--c-puct", type=float, default=3.0)

    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-loss-weight", type=float, default=1.0)
    parser.add_argument("--value-loss-weight", type=float, default=1.0)

    parser.add_argument("--arena", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--arena-games", type=int, default=10)
    parser.add_argument("--arena-mcts-sims", type=int, default=400)
    parser.add_argument("--arena-alphabeta-temperature", type=float, default=0.5)
    parser.add_argument("--arena-random-temperature", type=float, default=0.0)

    parser.add_argument("--model", choices=["dummy", "resnet"], default="dummy")
    parser.add_argument("--model-channels", type=int, default=64)
    parser.add_argument("--model-blocks", type=int, default=6)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> RunConfig:
    args = _build_parser().parse_args(argv)
    return RunConfig(
        run_dir=args.run_dir,
        resume=args.resume,
        num_iterations=args.num_iterations,
        device=args.device,
        torch_threads=args.torch_threads,
        selfplay_games_per_iter=args.games_per_iteration,
        selfplay_report_interval=args.report_interval,
        selfplay_batch_size=args.selfplay_batch_size,
        selfplay_game_concurrency=args.game_concurrency,
        selfplay_batch_timeout_ms=args.batch_timeout_ms,
        selfplay_num_simulations=args.simulations,
        selfplay_expansion_batch_size=args.expansion_batch_size,
        selfplay_c_puct=args.c_puct,
        train_batch_size=args.train_batch_size,
        train_num_workers=args.num_workers,
        train_num_epochs=args.epochs,
        train_learning_rate=args.learning_rate,
        train_weight_decay=args.weight_decay,
        policy_loss_weight=args.policy_loss_weight,
        value_loss_weight=args.value_loss_weight,
        arena_enabled=args.arena,
        arena_games=args.arena_games,
        arena_mcts_sims=args.arena_mcts_sims,
        arena_alphabeta_temperature=args.arena_alphabeta_temperature,
        arena_random_temperature=args.arena_random_temperature,
        model_type=args.model,
        model_channels=args.model_channels,
        model_num_blocks=args.model_blocks,
    )


def main(argv: Sequence[str] | None = None) -> None:
    """Run the configured AlphaZero training loop."""
    config = parse_args(argv)
    device = config.resolved_device()
    torch_threads = config.resolved_torch_threads(device)
    configure_training_threads(device, torch_threads)
    selfplay_batch_size = config.resolved_selfplay_batch_size(device)
    selfplay_game_concurrency = config.resolved_selfplay_game_concurrency(device)
    state = prepare_run(config)
    run_dir = state.run_dir
    data_base_dir = run_dir / "data"
    models_dir = run_dir / "models" / "ts"
    checkpoints_dir = run_dir / "checkpoints"
    models_dir.mkdir(parents=True, exist_ok=True)

    train_config = TrainingConfig(
        batch_size=config.train_batch_size,
        num_workers=config.train_num_workers,
        num_epochs=config.train_num_epochs,
        learning_rate=config.train_learning_rate,
        weight_decay=config.train_weight_decay,
        policy_loss_weight=config.policy_loss_weight,
        value_loss_weight=config.value_loss_weight,
        device=device,
        checkpoint_dir=checkpoints_dir,
        arena_enabled=config.arena_enabled,
        arena_vs_alphabeta=True,
        arena_vs_random=True,
        arena_games=config.arena_games,
        arena_mcts_sims=config.arena_mcts_sims,
        arena_alphabeta_temperature=config.arena_alphabeta_temperature,
        arena_random_temperature=config.arena_random_temperature,
        arena_device=None,
    )

    if config.model_type == "resnet":
        model: torch.nn.Module = ResNetReversiNet(
            in_channels=3,
            channels=config.model_channels,
            num_blocks=config.model_num_blocks,
        )
    else:
        model = DummyReversiNet(in_channels=3)

    trainer = AlphaZeroTrainer(model=model, config=train_config)
    if state.checkpoint_path is None:
        export_model_to_torchscript(model, models_dir / "model_iter_0.pt", device)
    else:
        trainer.load_checkpoint(state.checkpoint_path)
        trainer.config = train_config

    logging_cfg = LoggingConfig(
        backends={
            LoggerKind.CONSOLE: ConsoleConfig(
                verbose=True,
                show_params_table=True,
                show_timestamp=True,
            ),
        }
    )

    with create_logger(logging_cfg) as logger:
        log_hyperparameters(
            logger=logger,
            num_iterations=config.num_iterations,
            selfplay_config={
                "games_per_iter": config.selfplay_games_per_iter,
                "report_interval": config.report_interval(),
                "batch_size": selfplay_batch_size,
                "game_concurrency": selfplay_game_concurrency,
                "batch_timeout_ms": config.selfplay_batch_timeout_ms,
                "expansion_batch_size": config.selfplay_expansion_batch_size,
                "torch_threads": torch_threads,
                "num_simulations": config.selfplay_num_simulations,
            },
            train_config=train_config,
            model_config={
                "type": config.model_type,
                "channels": config.model_channels,
                "num_blocks": config.model_num_blocks,
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
                "run_dir": run_dir,
                "data_base_dir": data_base_dir,
                "models_dir": models_dir,
                "checkpoint_dir": checkpoints_dir,
            },
            device=device,
        )

        steps_per_iteration = (
            math.ceil(config.selfplay_games_per_iter / config.report_interval())
            + config.train_num_epochs
        )
        global_step = state.start_iteration * steps_per_iteration

        for iteration in range(state.start_iteration, config.num_iterations):
            current_model_path = models_dir / f"model_iter_{iteration}.pt"
            selfplay_data_dir = data_base_dir / f"selfplay_iter_{iteration}"
            if selfplay_data_dir.exists():
                raise FileExistsError(
                    f"Refusing to append to existing iteration data: {selfplay_data_dir}"
                )

            stream = SelfPlayStream(
                total_games=config.selfplay_games_per_iter,
                report_interval=config.report_interval(),
                batch=BatchConfigArgs(
                    batch_size=selfplay_batch_size,
                    game_concurrency=selfplay_game_concurrency,
                    batch_timeout_ms=config.selfplay_batch_timeout_ms,
                ),
                mcts=MctsConfigArgs(
                    num_simulations=config.selfplay_num_simulations,
                    c_puct=config.selfplay_c_puct,
                    expansion_batch_size=config.selfplay_expansion_batch_size,
                ),
                model_path=str(current_model_path),
                device=device,
                save_dir=str(selfplay_data_dir),
            )

            for stats in stream:
                log_selfplay_stats(logger, stats, iteration, global_step)
                global_step += 1

            for epoch_metrics in trainer.train(
                data_path=selfplay_data_dir,
                num_epochs=train_config.num_epochs,
            ):
                log_training_metrics(logger, epoch_metrics, iteration, global_step)
                global_step += 1

            checkpoint_path = trainer.save_checkpoint(
                iteration,
                filename=f"checkpoint_iter_{iteration}.pt",
            )
            logger.log_artifact("checkpoint", str(checkpoint_path))

            next_model_path = models_dir / f"model_iter_{iteration + 1}.pt"
            export_model_to_torchscript(model, next_model_path, device=device)
            logger.log_artifact("model", str(next_model_path))

        final_model_path = models_dir / "model_final.pt"
        export_model_to_torchscript(model, final_model_path, device=device)
        logger.log_artifact("final model", str(final_model_path))


if __name__ == "__main__":
    main()
