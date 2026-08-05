"""Tests for safe training run setup and CLI configuration."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from reversi_zero_trainer.train_main import (
    RunConfig,
    default_run_dir,
    parse_args,
    prepare_run,
)


def test_default_run_dir_is_timestamped():
    now = datetime(2026, 8, 5, 10, 40, 30, 123456)
    assert default_run_dir(now) == Path("runs/20260805_104030_123456")


def test_prepare_run_creates_new_isolated_directory(tmp_path):
    run_dir = tmp_path / "run"
    state = prepare_run(RunConfig(run_dir=run_dir, device="cpu"))

    assert state.run_dir == run_dir.resolve()
    assert state.start_iteration == 0
    assert state.checkpoint_path is None
    assert (run_dir / "run_config.json").is_file()
    stored = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    assert stored["torch_threads"] == 4
    assert stored["selfplay_batch_size"] == 32
    assert 1 <= stored["selfplay_game_concurrency"] <= 16
    assert stored["train_num_workers"] == 0


def test_prepare_run_refuses_existing_directory_without_resume(tmp_path):
    run_dir = tmp_path / "existing"
    run_dir.mkdir()

    with pytest.raises(FileExistsError, match="already exists"):
        prepare_run(RunConfig(run_dir=run_dir, device="cpu"))


def _create_completed_iteration(run_dir: Path, iteration: int) -> None:
    checkpoints_dir = run_dir / "checkpoints"
    models_dir = run_dir / "models" / "ts"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    (checkpoints_dir / f"checkpoint_iter_{iteration}.pt").touch()
    (models_dir / f"model_iter_{iteration + 1}.pt").touch()


def test_prepare_run_resumes_after_latest_complete_iteration(tmp_path):
    run_dir = tmp_path / "resume"
    prepare_run(RunConfig(run_dir=run_dir, device="cpu", num_iterations=3))
    _create_completed_iteration(run_dir, 0)
    _create_completed_iteration(run_dir, 1)

    state = prepare_run(
        RunConfig(run_dir=run_dir, resume=True, device="cpu", num_iterations=3)
    )

    assert state.start_iteration == 2
    assert state.checkpoint_path == run_dir / "checkpoints/checkpoint_iter_1.pt"


def test_prepare_run_refuses_ambiguous_partial_iteration(tmp_path):
    run_dir = tmp_path / "partial"
    prepare_run(RunConfig(run_dir=run_dir, device="cpu", num_iterations=2))
    _create_completed_iteration(run_dir, 0)
    (run_dir / "data/selfplay_iter_1").mkdir(parents=True)

    with pytest.raises(RuntimeError, match="ambiguous partial data"):
        prepare_run(
            RunConfig(run_dir=run_dir, resume=True, device="cpu", num_iterations=2)
        )


def test_parse_args_maps_training_options():
    config = parse_args(
        [
            "--run-dir",
            "example-run",
            "--num-iterations",
            "2",
            "--games-per-iteration",
            "16",
            "--epochs",
            "3",
            "--model",
            "resnet",
            "--no-arena",
        ]
    )

    assert config.run_dir == Path("example-run")
    assert config.num_iterations == 2
    assert config.selfplay_games_per_iter == 16
    assert config.train_num_epochs == 3
    assert config.model_type == "resnet"
    assert not config.arena_enabled


def test_cpu_runtime_defaults_are_tuned_for_selfplay():
    config = RunConfig(device="cpu", selfplay_games_per_iter=64)

    assert config.resolved_torch_threads("cpu") == 4
    assert config.resolved_selfplay_batch_size("cpu") == 32
    assert config.resolved_selfplay_game_concurrency("cpu") == 16
    assert config.resolved_train_num_workers("cpu") == 0
    assert config.report_interval() == 16


def test_cuda_runtime_defaults_remain_unchanged():
    config = RunConfig(device="cuda")

    assert config.resolved_torch_threads("cuda") is None
    assert config.resolved_selfplay_batch_size("cuda") == 128
    assert config.resolved_selfplay_game_concurrency("cuda") == 32
    assert config.resolved_train_num_workers("cuda") == 4


def test_explicit_selfplay_runtime_settings_are_respected():
    config = RunConfig(
        device="cpu",
        selfplay_games_per_iter=64,
        selfplay_report_interval=8,
        torch_threads=2,
        selfplay_batch_size=16,
        selfplay_game_concurrency=4,
        train_num_workers=2,
    )

    assert config.resolved_torch_threads("cpu") == 2
    assert config.resolved_selfplay_batch_size("cpu") == 16
    assert config.resolved_selfplay_game_concurrency("cpu") == 4
    assert config.resolved_train_num_workers("cpu") == 2
