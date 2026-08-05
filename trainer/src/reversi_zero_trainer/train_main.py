"""Safe, configurable AlphaZero training entry point."""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
from argparse import Namespace
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Sequence

import hydra
import torch
from omegaconf import DictConfig
from reversi_zero_rs import BatchConfigArgs, MctsConfigArgs, SelfPlayStream

from reversi_zero_trainer.logging import (
    ConsoleConfig,
    LoggerKind,
    LoggingConfig,
    create_logger,
    log_hyperparameters,
    log_promotion_metrics,
    log_reference_metrics,
    log_selfplay_stats,
    log_training_metrics,
)
from reversi_zero_trainer.evaluate_main import (
    run as run_model_evaluation,
    write_report as write_evaluation_report,
)
from reversi_zero_trainer.hydra_config import (
    materialize_train_config,
    register_train_config,
)
from reversi_zero_trainer.models.dummy import DummyReversiNet, ResNetReversiNet
from reversi_zero_trainer.runtime import configure_training_threads
from reversi_zero_trainer.training import AlphaZeroTrainer, TrainingConfig


register_train_config()


@dataclass(frozen=True)
class RunConfig:
    """Configuration for one isolated AlphaZero run."""

    run_dir: Path | None = None
    resume: bool = False
    num_iterations: int = 10
    device: Literal["auto", "cuda", "cpu"] = "auto"
    seed: int = 0
    torch_threads: int | None = None

    selfplay_games_per_iter: int = 128
    selfplay_report_interval: int | None = None
    selfplay_batch_size: int | None = None
    selfplay_game_concurrency: int | None = None
    selfplay_batch_timeout_ms: int = 1
    selfplay_num_simulations: int = 100
    selfplay_expansion_batch_size: int = 4
    selfplay_c_puct: float = 3.0
    inference_dtype: Literal["auto", "float32", "float16"] = "auto"

    train_batch_size: int = 256
    train_num_workers: int | None = None
    train_num_epochs: int = 10
    train_replay_window: int = 5
    train_symmetry_augmentation: Literal[1, 2, 4, 8] = 8
    train_learning_rate: float = 0.001
    train_weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0

    reference_eval_enabled: bool = True
    reference_games: int = 40

    promotion_enabled: bool = True
    promotion_num_openings: int = 80
    promotion_opening_plies: int = 8
    promotion_seed: int = 0
    promotion_mcts_sims: int | None = None
    promotion_c_puct: float = 1.5
    promotion_expansion_batch_size: int | None = None
    promotion_threshold: float = 0.55
    promotion_require_confidence: bool = False

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

    def resolved_train_num_workers(self, device: str) -> int:
        if self.train_num_workers is not None:
            return self.train_num_workers
        return 0 if device == "cpu" else 4

    def resolved_inference_dtype(self, device: str) -> Literal["float32", "float16"]:
        if self.inference_dtype == "auto":
            return "float16" if device == "cuda" else "float32"
        return self.inference_dtype

    def resolved_promotion_mcts_sims(self) -> int:
        return self.promotion_mcts_sims or self.selfplay_num_simulations

    def resolved_promotion_expansion_batch_size(self) -> int:
        return self.promotion_expansion_batch_size or self.selfplay_expansion_batch_size

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
            "train_replay_window": self.train_replay_window,
            "reference_games": self.reference_games,
            "promotion_num_openings": self.promotion_num_openings,
            "promotion_mcts_sims": self.resolved_promotion_mcts_sims(),
            "promotion_expansion_batch_size": (
                self.resolved_promotion_expansion_batch_size()
            ),
            "model_channels": self.model_channels,
            "model_num_blocks": self.model_num_blocks,
        }
        invalid = [name for name, value in positive_ints.items() if value <= 0]
        optional_positive_ints = {
            "torch_threads": self.torch_threads,
            "selfplay_batch_size": self.selfplay_batch_size,
            "selfplay_game_concurrency": self.selfplay_game_concurrency,
            "promotion_mcts_sims": self.promotion_mcts_sims,
            "promotion_expansion_batch_size": self.promotion_expansion_batch_size,
        }
        invalid.extend(
            name
            for name, value in optional_positive_ints.items()
            if value is not None and value <= 0
        )
        if invalid:
            raise ValueError(f"These settings must be > 0: {', '.join(invalid)}")
        if self.train_num_workers is not None and self.train_num_workers < 0:
            raise ValueError("train_num_workers must be >= 0")
        if self.train_symmetry_augmentation not in (1, 2, 4, 8):
            raise ValueError("train_symmetry_augmentation must be one of 1, 2, 4, or 8")
        if self.reference_games % 2 != 0:
            raise ValueError("reference_games must be even for paired color swaps")
        if self.promotion_opening_plies < 0:
            raise ValueError("promotion_opening_plies must be >= 0")
        if self.promotion_c_puct <= 0:
            raise ValueError("promotion_c_puct must be > 0")
        if not 0.0 <= self.promotion_threshold <= 1.0:
            raise ValueError("promotion_threshold must be between 0 and 1")
        device = self.resolved_device()
        if self.resolved_inference_dtype(device) == "float16" and device != "cuda":
            raise ValueError("float16 inference requires CUDA")


@dataclass(frozen=True)
class RunState:
    run_dir: Path
    start_iteration: int
    checkpoint_path: Path | None


@dataclass(frozen=True)
class TrainerSnapshot:
    """Restorable incumbent model and optimizer state."""

    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]


def capture_trainer_snapshot(trainer: AlphaZeroTrainer) -> TrainerSnapshot:
    """Capture incumbent state before candidate training."""
    return TrainerSnapshot(
        model_state_dict=copy.deepcopy(trainer.model.state_dict()),
        optimizer_state_dict=copy.deepcopy(trainer.optimizer.state_dict()),
    )


def restore_trainer_snapshot(
    trainer: AlphaZeroTrainer, snapshot: TrainerSnapshot
) -> None:
    """Restore a rejected candidate to the incumbent training state."""
    trainer.model.load_state_dict(snapshot.model_state_dict)
    trainer.optimizer.load_state_dict(snapshot.optimizer_state_dict)


def promotion_is_accepted(
    report: dict[str, Any], threshold: float, require_confidence: bool
) -> bool:
    """Apply the configured promotion rule to an evaluation report."""
    summary = report["summary"]
    score = float(summary["score"])
    interval_low = float(summary["score_interval_95"][0])
    return score >= threshold and (not require_confidence or interval_low > 0.5)


def evaluate_promotion_candidate(
    candidate_path: Path,
    incumbent_path: Path,
    report_path: Path,
    config: RunConfig,
    iteration: int,
    device: str,
    torch_threads: int | None,
) -> dict[str, Any]:
    """Evaluate a candidate against the incumbent and persist the report."""
    args = Namespace(
        challenger=candidate_path,
        reference_model=incumbent_path,
        reference_alphabeta=False,
        reference_bitmatrix=False,
        reference_random=False,
        output=report_path,
        overwrite=False,
        openings_from=None,
        num_openings=config.promotion_num_openings,
        opening_plies=config.promotion_opening_plies,
        seed=config.promotion_seed + iteration,
        device=device,
        torch_threads=torch_threads or 1,
        simulations=config.resolved_promotion_mcts_sims(),
        c_puct=config.promotion_c_puct,
        challenger_expansion_batch_size=(
            config.resolved_promotion_expansion_batch_size()
        ),
        reference_expansion_batch_size=(
            config.resolved_promotion_expansion_batch_size()
        ),
        alphabeta_depth=3,
        bitmatrix_depth=3,
        promotion_threshold=config.promotion_threshold,
        show_progress=False,
    )
    report = run_model_evaluation(args)
    report["config"]["promotion_require_confidence"] = (
        config.promotion_require_confidence
    )
    report["summary"]["promotion_accepted"] = promotion_is_accepted(
        report,
        threshold=config.promotion_threshold,
        require_confidence=config.promotion_require_confidence,
    )
    write_evaluation_report(report, report_path)
    return report


def evaluate_bitmatrix_reference(
    model_path: Path,
    evaluations_dir: Path,
    config: RunConfig,
    iteration: int,
    device: str,
    torch_threads: int | None,
) -> tuple[dict[str, Any], Path]:
    """Evaluate the selected incumbent against the fixed BitMatrix reference."""
    report_path = evaluations_dir / f"reference_bitmatrix_iter_{iteration}.json"
    args = Namespace(
        challenger=model_path,
        reference_model=None,
        reference_alphabeta=False,
        reference_bitmatrix=True,
        reference_random=False,
        output=report_path,
        overwrite=False,
        openings_from=None,
        num_openings=config.reference_games // 2,
        opening_plies=config.promotion_opening_plies,
        seed=config.promotion_seed,
        device=device,
        torch_threads=torch_threads or 1,
        simulations=config.selfplay_num_simulations,
        c_puct=config.promotion_c_puct,
        challenger_expansion_batch_size=config.selfplay_expansion_batch_size,
        reference_expansion_batch_size=1,
        alphabeta_depth=3,
        bitmatrix_depth=3,
        promotion_threshold=0.5,
        show_progress=False,
    )
    report = run_model_evaluation(args)
    write_evaluation_report(report, report_path)
    return report, report_path


def _copy_file_atomically(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    next_path = destination.with_name(f".{destination.name}.next")
    if next_path.exists():
        raise FileExistsError(f"Refusing ambiguous partial file: {next_path}")
    try:
        shutil.copyfile(source, next_path)
        next_path.replace(destination)
    finally:
        next_path.unlink(missing_ok=True)


def replay_data_paths(
    data_base_dir: Path, iteration: int, replay_window: int
) -> list[Path]:
    """Return the available rolling window ending at the current iteration."""
    first_iteration = max(0, iteration - replay_window + 1)
    paths = [
        data_base_dir / f"selfplay_iter_{index}"
        for index in range(first_iteration, iteration + 1)
    ]
    missing = [path for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(f"Replay data directory not found: {missing[0]}")
    return paths


def finalize_candidate(
    trainer: AlphaZeroTrainer,
    snapshot: TrainerSnapshot,
    iteration: int,
    incumbent_model_path: Path,
    candidate_model_path: Path,
    candidate_checkpoint_path: Path,
    next_model_path: Path,
    accepted: bool,
) -> Path:
    """Promote a candidate or restore the incumbent, then mark iteration complete."""
    checkpoint_path = Path(trainer.config.checkpoint_dir) / (
        f"checkpoint_iter_{iteration}.pt"
    )
    if checkpoint_path.exists() or next_model_path.exists():
        raise FileExistsError("Refusing to overwrite completed iteration artifacts")

    if accepted:
        candidate_model_path.replace(next_model_path)
        candidate_checkpoint_path.replace(checkpoint_path)
        return checkpoint_path

    restore_trainer_snapshot(trainer, snapshot)
    _copy_file_atomically(incumbent_model_path, next_model_path)
    return trainer.save_checkpoint(
        iteration,
        filename=checkpoint_path.name,
    )


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
    payload["train_num_workers"] = config.resolved_train_num_workers(device)
    payload["inference_dtype"] = config.resolved_inference_dtype(device)
    payload["promotion_mcts_sims"] = config.resolved_promotion_mcts_sims()
    payload["promotion_expansion_batch_size"] = (
        config.resolved_promotion_expansion_batch_size()
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
    for key in (
        "model_type",
        "model_channels",
        "model_num_blocks",
        "inference_dtype",
    ):
        stored_value = stored.get(key, "float32" if key == "inference_dtype" else None)
        requested_value = (
            config.resolved_inference_dtype(config.resolved_device())
            if key == "inference_dtype"
            else getattr(config, key)
        )
        if stored_value != requested_value:
            raise RuntimeError(
                f"Resume configuration mismatch for {key}: "
                f"stored={stored_value!r}, requested={requested_value!r}"
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
                "Choose another run.dir or use run.resume=true."
            )
        run_dir.mkdir(parents=True)
        _write_run_config(config, run_dir)
        return RunState(run_dir, 0, None)

    if config.run_dir is None:
        raise ValueError("run.resume=true requires an explicit run.dir")
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


class _Float16InferenceWrapper(torch.nn.Module):
    """Keep the Rust boundary FP32 while running FP16 channels-last inference."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        inference_inputs = inputs.to(
            dtype=torch.float16,
            memory_format=torch.channels_last,
        )
        policy, value = self.model(inference_inputs)
        return policy.float(), value.float()


def export_model_to_torchscript(
    model: torch.nn.Module,
    output_path: Path | str,
    device: str = "cuda",
    inference_dtype: Literal["float32", "float16"] = "float32",
) -> None:
    """Atomically export a PyTorch model to TorchScript."""
    model.eval()
    model.to(device)
    export_model = model
    if inference_dtype == "float16":
        if device != "cuda":
            raise ValueError("float16 inference requires CUDA")
        export_model = _Float16InferenceWrapper(
            copy.deepcopy(model).to(
                device=device,
                dtype=torch.float16,
                memory_format=torch.channels_last,
            )
        ).eval()
    dummy_input = torch.randn(1, 3, 8, 8, device=device)
    traced_model = torch.jit.trace(export_model, dummy_input)
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


def run_config_from_hydra(config: DictConfig) -> RunConfig:
    """Convert Hydra's nested application config into the training domain config."""
    typed = materialize_train_config(config)
    return RunConfig(
        run_dir=Path(typed.run.dir) if typed.run.dir is not None else None,
        resume=typed.run.resume,
        num_iterations=typed.run.num_iterations,
        device=typed.hardware.device.value,
        seed=typed.run.seed,
        torch_threads=typed.hardware.torch_threads,
        selfplay_games_per_iter=typed.selfplay.games_per_iteration,
        selfplay_report_interval=typed.selfplay.report_interval,
        selfplay_batch_size=typed.selfplay.batch_size,
        selfplay_game_concurrency=typed.selfplay.game_concurrency,
        selfplay_batch_timeout_ms=typed.selfplay.batch_timeout_ms,
        selfplay_num_simulations=typed.selfplay.simulations,
        selfplay_expansion_batch_size=typed.selfplay.expansion_batch_size,
        selfplay_c_puct=typed.selfplay.c_puct,
        inference_dtype=typed.hardware.inference_dtype.value,
        train_batch_size=typed.training.batch_size,
        train_num_workers=typed.training.num_workers,
        train_num_epochs=typed.training.epochs,
        train_replay_window=typed.training.replay_window,
        train_symmetry_augmentation=typed.training.symmetry_augmentation.value,
        train_learning_rate=typed.training.learning_rate,
        train_weight_decay=typed.training.weight_decay,
        policy_loss_weight=typed.training.policy_loss_weight,
        value_loss_weight=typed.training.value_loss_weight,
        reference_eval_enabled=typed.reference.enabled,
        reference_games=typed.reference.games,
        promotion_enabled=typed.promotion.enabled,
        promotion_num_openings=typed.promotion.openings,
        promotion_opening_plies=typed.promotion.opening_plies,
        promotion_seed=typed.promotion.seed,
        promotion_mcts_sims=typed.promotion.simulations,
        promotion_c_puct=typed.promotion.c_puct,
        promotion_expansion_batch_size=typed.promotion.expansion_batch_size,
        promotion_threshold=typed.promotion.threshold,
        promotion_require_confidence=typed.promotion.require_confidence,
        model_type=typed.model.type.value,
        model_channels=typed.model.channels,
        model_num_blocks=typed.model.blocks,
    )


def compose_run_config(overrides: Sequence[str] = ()) -> RunConfig:
    """Compose a training config for tests and programmatic callers."""
    config_dir = Path(__file__).parent / "conf"
    with hydra.initialize_config_dir(
        version_base="1.3", config_dir=str(config_dir.resolve())
    ):
        config = hydra.compose(config_name="train", overrides=list(overrides))
    return run_config_from_hydra(config)


def run_training(config: RunConfig) -> None:
    """Run the configured AlphaZero training loop."""
    device = config.resolved_device()
    inference_dtype = config.resolved_inference_dtype(device)
    torch_threads = config.resolved_torch_threads(device)
    configure_training_threads(device, torch_threads)
    selfplay_batch_size = config.resolved_selfplay_batch_size(device)
    selfplay_game_concurrency = config.resolved_selfplay_game_concurrency(device)
    train_num_workers = config.resolved_train_num_workers(device)
    state = prepare_run(config)
    run_dir = state.run_dir
    data_base_dir = run_dir / "data"
    models_dir = run_dir / "models" / "ts"
    candidate_models_dir = models_dir / "candidates"
    checkpoints_dir = run_dir / "checkpoints"
    evaluations_dir = run_dir / "evaluations"
    models_dir.mkdir(parents=True, exist_ok=True)
    if config.promotion_enabled:
        candidate_models_dir.mkdir(parents=True, exist_ok=True)
    if config.promotion_enabled or config.reference_eval_enabled:
        evaluations_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(config.seed)

    train_config = TrainingConfig(
        batch_size=config.train_batch_size,
        num_workers=train_num_workers,
        num_epochs=config.train_num_epochs,
        symmetry_augmentation=config.train_symmetry_augmentation,
        learning_rate=config.train_learning_rate,
        weight_decay=config.train_weight_decay,
        policy_loss_weight=config.policy_loss_weight,
        value_loss_weight=config.value_loss_weight,
        device=device,
        checkpoint_dir=checkpoints_dir,
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
        export_model_to_torchscript(
            model,
            models_dir / "model_iter_0.pt",
            device,
            inference_dtype,
        )
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
            seed=config.seed,
            replay_window=config.train_replay_window,
            selfplay_config={
                "games_per_iter": config.selfplay_games_per_iter,
                "report_interval": config.report_interval(),
                "batch_size": selfplay_batch_size,
                "game_concurrency": selfplay_game_concurrency,
                "batch_timeout_ms": config.selfplay_batch_timeout_ms,
                "expansion_batch_size": config.selfplay_expansion_batch_size,
                "torch_threads": torch_threads,
                "num_simulations": config.selfplay_num_simulations,
                "inference_dtype": inference_dtype,
            },
            train_config=train_config,
            model_config={
                "type": config.model_type,
                "channels": config.model_channels,
                "num_blocks": config.model_num_blocks,
            },
            reference_config={
                "enabled": config.reference_eval_enabled,
                "games": config.reference_games,
            },
            promotion_config={
                "enabled": config.promotion_enabled,
                "num_openings": config.promotion_num_openings,
                "opening_plies": config.promotion_opening_plies,
                "mcts_sims": config.resolved_promotion_mcts_sims(),
                "c_puct": config.promotion_c_puct,
                "expansion_batch_size": (
                    config.resolved_promotion_expansion_batch_size()
                ),
                "threshold": config.promotion_threshold,
                "require_confidence": config.promotion_require_confidence,
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
            + int(config.promotion_enabled)
            + int(config.reference_eval_enabled)
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

            incumbent_snapshot = (
                capture_trainer_snapshot(trainer) if config.promotion_enabled else None
            )
            for epoch_metrics in trainer.train(
                data_path=replay_data_paths(
                    data_base_dir,
                    iteration,
                    config.train_replay_window,
                ),
                num_epochs=train_config.num_epochs,
            ):
                log_training_metrics(logger, epoch_metrics, iteration, global_step)
                global_step += 1

            next_model_path = models_dir / f"model_iter_{iteration + 1}.pt"
            if config.promotion_enabled:
                if incumbent_snapshot is None:
                    raise RuntimeError("Missing incumbent snapshot")
                candidate_model_path = (
                    candidate_models_dir / f"candidate_iter_{iteration}.pt"
                )
                candidate_checkpoint_path = trainer.save_checkpoint(
                    iteration,
                    filename=f"candidate_iter_{iteration}.pt",
                )
                export_model_to_torchscript(
                    model,
                    candidate_model_path,
                    device=device,
                    inference_dtype=inference_dtype,
                )
                report_path = evaluations_dir / f"promotion_iter_{iteration}.json"
                report = evaluate_promotion_candidate(
                    candidate_path=candidate_model_path,
                    incumbent_path=current_model_path,
                    report_path=report_path,
                    config=config,
                    iteration=iteration,
                    device=device,
                    torch_threads=torch_threads,
                )
                accepted = bool(report["summary"]["promotion_accepted"])
                log_promotion_metrics(
                    logger,
                    report,
                    accepted=accepted,
                    iteration=iteration,
                    step=global_step,
                )
                global_step += 1
                logger.log_artifact("promotion evaluation", str(report_path))
                checkpoint_path = finalize_candidate(
                    trainer=trainer,
                    snapshot=incumbent_snapshot,
                    iteration=iteration,
                    incumbent_model_path=current_model_path,
                    candidate_model_path=candidate_model_path,
                    candidate_checkpoint_path=candidate_checkpoint_path,
                    next_model_path=next_model_path,
                    accepted=accepted,
                )
                if not accepted:
                    logger.log_artifact(
                        "rejected candidate checkpoint",
                        str(candidate_checkpoint_path),
                    )
                    logger.log_artifact(
                        "rejected candidate model",
                        str(candidate_model_path),
                    )
            else:
                checkpoint_path = trainer.save_checkpoint(
                    iteration,
                    filename=f"checkpoint_iter_{iteration}.pt",
                )
                export_model_to_torchscript(
                    model,
                    next_model_path,
                    device=device,
                    inference_dtype=inference_dtype,
                )

            if config.reference_eval_enabled:
                reference_report, reference_path = evaluate_bitmatrix_reference(
                    model_path=next_model_path,
                    evaluations_dir=evaluations_dir,
                    config=config,
                    iteration=iteration,
                    device=device,
                    torch_threads=torch_threads,
                )
                log_reference_metrics(
                    logger,
                    opponent="bitmatrix",
                    report=reference_report,
                    iteration=iteration,
                    step=global_step,
                )
                global_step += 1
                logger.log_artifact(
                    "bitmatrix reference evaluation",
                    str(reference_path),
                )

            logger.log_artifact("checkpoint", str(checkpoint_path))
            logger.log_artifact("model", str(next_model_path))

        final_model_path = models_dir / "model_final.pt"
        export_model_to_torchscript(
            model,
            final_model_path,
            device=device,
            inference_dtype=inference_dtype,
        )
        logger.log_artifact("final model", str(final_model_path))


@hydra.main(version_base="1.3", config_path="conf", config_name="train")
def main(config: DictConfig) -> None:
    """Compose the Hydra configuration and start training."""
    run_training(run_config_from_hydra(config))


if __name__ == "__main__":
    main()
