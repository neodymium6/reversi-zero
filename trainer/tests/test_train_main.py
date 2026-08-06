"""Tests for safe training run setup and CLI configuration."""

import json
from datetime import datetime
from pathlib import Path

import pytest
import torch
from hydra.errors import ConfigCompositionException

import reversi_zero_trainer.train_main as train_main
from reversi_zero_trainer.train_main import (
    RunConfig,
    _Float16InferenceWrapper,
    capture_trainer_snapshot,
    compose_run_config,
    default_run_dir,
    evaluate_promotion_candidate,
    evaluate_bitmatrix_reference,
    finalize_candidate,
    prepare_run,
    promotion_is_accepted,
    replay_data_paths,
)
from reversi_zero_trainer.models.dummy import DummyReversiNet
from reversi_zero_trainer.training import AlphaZeroTrainer, TrainingConfig


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
    assert stored["seed"] == 0
    assert stored["selfplay_batch_size"] == 32
    assert 1 <= stored["selfplay_game_concurrency"] <= 16
    assert stored["train_num_workers"] == 0
    assert stored["train_num_epochs"] == 1
    assert stored["train_symmetry_augmentation"] == 8
    assert stored["train_replay_window"] == 5
    assert stored["reference_eval_enabled"] is True
    assert stored["reference_games"] == 40
    assert stored["promotion_enabled"] is True
    assert stored["promotion_num_openings"] == 80
    assert stored["promotion_mcts_sims"] == 100
    assert stored["promotion_expansion_batch_size"] == 4
    assert stored["promotion_require_confidence"] is False
    assert stored["inference_dtype"] == "float32"
    assert stored["train_dtype"] == "float32"


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


def test_hydra_config_maps_training_options():
    config = compose_run_config(
        [
            "run.dir=example-run",
            "run.num_iterations=2",
            "run.seed=123",
            "selfplay.games_per_iteration=16",
            "training.epochs=3",
            "training.lr_schedule=wsd",
            "training.symmetry_augmentation=4",
            "training.replay_window=7",
            "promotion.openings=12",
            "promotion.opening_plies=6",
            "promotion.simulations=50",
            "promotion.expansion_batch_size=2",
            "promotion.threshold=0.6",
            "promotion.require_confidence=true",
            "model=resnet",
            "hardware.inference_dtype=float16",
            "training.dtype=bfloat16",
            "reference.enabled=false",
            "reference.games=20",
        ]
    )

    assert config.run_dir == Path("example-run")
    assert config.num_iterations == 2
    assert config.seed == 123
    assert config.selfplay_games_per_iter == 16
    assert config.train_num_epochs == 3
    assert config.train_lr_schedule == "wsd"
    assert config.train_symmetry_augmentation == 4
    assert config.train_replay_window == 7
    assert config.promotion_num_openings == 12
    assert config.promotion_opening_plies == 6
    assert config.resolved_promotion_mcts_sims() == 50
    assert config.resolved_promotion_expansion_batch_size() == 2
    assert config.promotion_threshold == pytest.approx(0.6)
    assert config.promotion_require_confidence
    assert config.model_type == "resnet"
    assert config.inference_dtype == "float16"
    assert config.train_dtype == "bfloat16"
    assert not config.reference_eval_enabled
    assert config.reference_games == 20


def test_hydra_training_epochs_default_to_one():
    assert compose_run_config([]).train_num_epochs == 1


def test_training_dtype_auto_uses_float32_on_cpu():
    assert RunConfig().resolved_train_dtype("cpu") == "float32"


def test_training_dtype_auto_uses_bfloat16_on_supported_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)

    assert RunConfig().resolved_train_dtype("cuda") == "bfloat16"


def test_bfloat16_training_rejects_cpu():
    config = RunConfig(device="cpu", train_dtype="bfloat16")

    with pytest.raises(ValueError, match="bfloat16 training requires CUDA"):
        config.validate()


@pytest.mark.parametrize(
    ("profile", "device"),
    [("auto", "auto"), ("cpu", "cpu"), ("gpu", "cuda")],
)
def test_hydra_profile_selects_device(profile, device):
    config = compose_run_config([f"profile={profile}"])

    assert config.device == device


def test_hydra_defaults_use_160_game_score_only_promotion_gate():
    config = compose_run_config()

    assert config.promotion_num_openings == 80
    assert config.promotion_threshold == pytest.approx(0.55)
    assert not config.promotion_require_confidence


@pytest.mark.parametrize(
    "override",
    [
        "hardware.device=banana",
        "hardware.inference_dtype=bfloat16",
        "training.dtype=float16",
        "model.type=other",
        "training.symmetry_augmentation=3",
        "training.epochs=abc",
        "training.lr_schedule=cyclic",
        "reference.enabled=maybe",
        "selfplay.simmulations=10",
    ],
)
def test_hydra_schema_rejects_invalid_overrides(override):
    with pytest.raises(ConfigCompositionException):
        compose_run_config([override])


def test_hydra_conversion_rejects_explicit_unknown_keys():
    with pytest.raises(ValueError, match="selfplay.simmulations"):
        compose_run_config(["+selfplay.simmulations=10"])


def test_run_config_rejects_invalid_symmetry_augmentation():
    config = RunConfig(train_symmetry_augmentation=3)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="one of 1, 2, 4, or 8"):
        config.validate()


def test_replay_data_paths_selects_recent_complete_window(tmp_path):
    for iteration in range(6):
        (tmp_path / f"selfplay_iter_{iteration}").mkdir()

    assert replay_data_paths(tmp_path, iteration=5, replay_window=3) == [
        tmp_path / "selfplay_iter_3",
        tmp_path / "selfplay_iter_4",
        tmp_path / "selfplay_iter_5",
    ]
    assert replay_data_paths(tmp_path, iteration=1, replay_window=5) == [
        tmp_path / "selfplay_iter_0",
        tmp_path / "selfplay_iter_1",
    ]


def test_replay_data_paths_rejects_missing_iteration(tmp_path):
    (tmp_path / "selfplay_iter_0").mkdir()

    with pytest.raises(FileNotFoundError, match="selfplay_iter_1"):
        replay_data_paths(tmp_path, iteration=1, replay_window=2)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"promotion_num_openings": 0}, "must be > 0"),
        ({"promotion_opening_plies": -1}, "must be >= 0"),
        ({"promotion_mcts_sims": 0}, "must be > 0"),
        ({"promotion_expansion_batch_size": 0}, "must be > 0"),
        ({"promotion_c_puct": 0.0}, "must be > 0"),
        ({"promotion_threshold": 1.1}, "between 0 and 1"),
        ({"train_replay_window": 0}, "must be > 0"),
        ({"reference_games": 3}, "must be even"),
    ],
)
def test_run_config_rejects_invalid_promotion_settings(kwargs, message):
    with pytest.raises(ValueError, match=message):
        RunConfig(**kwargs).validate()


def test_promotion_rule_can_optionally_require_confidence():
    report = {"summary": {"score": 0.6, "score_interval_95": [0.49, 0.7]}}

    assert promotion_is_accepted(report, threshold=0.55, require_confidence=False)
    assert not promotion_is_accepted(report, threshold=0.55, require_confidence=True)


def test_evaluate_promotion_candidate_persists_decision(tmp_path, monkeypatch):
    candidate = tmp_path / "candidate.pt"
    incumbent = tmp_path / "incumbent.pt"
    candidate.touch()
    incumbent.touch()
    output = tmp_path / "promotion.json"

    def fake_evaluation(args):
        assert args.seed == 12
        assert args.simulations == 25
        assert args.challenger == candidate
        assert args.reference_model == incumbent
        return {
            "config": {},
            "summary": {
                "score": 0.6,
                "score_interval_95": [0.49, 0.7],
            },
        }

    monkeypatch.setattr(train_main, "run_model_evaluation", fake_evaluation)
    report = evaluate_promotion_candidate(
        candidate_path=candidate,
        incumbent_path=incumbent,
        report_path=output,
        config=RunConfig(
            promotion_seed=10,
            promotion_mcts_sims=25,
            promotion_require_confidence=False,
        ),
        iteration=2,
        device="cpu",
        torch_threads=1,
    )

    assert report["summary"]["promotion_accepted"] is True
    assert output.is_file()
    stored = json.loads(output.read_text(encoding="utf-8"))
    assert stored["summary"]["promotion_accepted"] is True


def test_bitmatrix_reference_uses_training_search_budget(tmp_path, monkeypatch):
    model = tmp_path / "model.pt"
    model.touch()
    evaluations_dir = tmp_path / "evaluations"
    evaluations_dir.mkdir()
    calls = []

    def fake_evaluation(args):
        calls.append(args)
        return {
            "reference": {"type": "bitmatrix"},
            "summary": {
                "score": 0.5,
                "score_interval_95": [0.4, 0.6],
                "wins": 5,
                "draws": 2,
                "losses": 5,
            },
        }

    monkeypatch.setattr(train_main, "run_model_evaluation", fake_evaluation)
    report, report_path = evaluate_bitmatrix_reference(
        model_path=model,
        evaluations_dir=evaluations_dir,
        config=RunConfig(
            reference_games=12,
            selfplay_num_simulations=50,
            selfplay_expansion_batch_size=2,
            promotion_opening_plies=6,
            promotion_seed=10,
        ),
        iteration=2,
        device="cpu",
        torch_threads=3,
    )

    assert report["reference"]["type"] == "bitmatrix"
    assert report_path == evaluations_dir / "reference_bitmatrix_iter_2.json"
    assert report_path.is_file()
    assert len(calls) == 1
    call = calls[0]
    assert call.challenger == model
    assert call.reference_bitmatrix
    assert not call.reference_alphabeta
    assert not call.reference_random
    assert call.openings_from is None
    assert call.num_openings == 6
    assert call.opening_plies == 6
    assert call.seed == 10
    assert call.simulations == 50
    assert call.challenger_expansion_batch_size == 2
    assert call.torch_threads == 3


def test_finalize_candidate_promotes_candidate_files(tmp_path):
    trainer = AlphaZeroTrainer(
        model=DummyReversiNet(),
        config=TrainingConfig(device="cpu", checkpoint_dir=tmp_path / "checkpoints"),
    )
    snapshot = capture_trainer_snapshot(trainer)
    incumbent_model = tmp_path / "incumbent.pt"
    candidate_model = tmp_path / "candidate.pt"
    candidate_checkpoint = tmp_path / "checkpoints" / "candidate_iter_0.pt"
    next_model = tmp_path / "next.pt"
    incumbent_model.write_bytes(b"incumbent")
    candidate_model.write_bytes(b"candidate")
    candidate_checkpoint.write_bytes(b"candidate checkpoint")

    checkpoint = finalize_candidate(
        trainer,
        snapshot,
        iteration=0,
        incumbent_model_path=incumbent_model,
        candidate_model_path=candidate_model,
        candidate_checkpoint_path=candidate_checkpoint,
        next_model_path=next_model,
        accepted=True,
    )

    assert next_model.read_bytes() == b"candidate"
    assert checkpoint.read_bytes() == b"candidate checkpoint"
    assert not candidate_model.exists()
    assert not candidate_checkpoint.exists()


def test_finalize_candidate_restores_incumbent_optimizer_and_scheduler(tmp_path):
    trainer = AlphaZeroTrainer(
        model=DummyReversiNet(),
        config=TrainingConfig(
            device="cpu",
            checkpoint_dir=tmp_path / "checkpoints",
            lr_schedule="wsd",
            lr_schedule_iterations=10,
        ),
    )
    snapshot = capture_trainer_snapshot(trainer)
    expected_scheduler_state = snapshot.lr_scheduler_state_dict
    expected_parameters = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    inputs = torch.randn(2, 3, 8, 8)
    policy, value = trainer.model(inputs)
    (policy.sum() + value.sum()).backward()
    trainer.optimizer.step()
    assert trainer.lr_scheduler is not None
    trainer.lr_scheduler.begin_iteration(5, 2)
    trainer.lr_scheduler.step()
    assert trainer.optimizer.state_dict()["state"]

    incumbent_model = tmp_path / "incumbent.pt"
    candidate_model = tmp_path / "candidate.pt"
    incumbent_model.write_bytes(b"incumbent")
    candidate_model.write_bytes(b"candidate")
    candidate_checkpoint = trainer.save_checkpoint(0, filename="candidate_iter_0.pt")
    next_model = tmp_path / "next.pt"

    checkpoint = finalize_candidate(
        trainer,
        snapshot,
        iteration=0,
        incumbent_model_path=incumbent_model,
        candidate_model_path=candidate_model,
        candidate_checkpoint_path=candidate_checkpoint,
        next_model_path=next_model,
        accepted=False,
    )

    assert next_model.read_bytes() == b"incumbent"
    assert candidate_model.read_bytes() == b"candidate"
    assert candidate_checkpoint.exists()
    assert checkpoint.exists()
    assert trainer.optimizer.state_dict()["state"] == {}
    assert trainer.lr_scheduler.state_dict() == expected_scheduler_state
    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(parameter, expected_parameters[name])


def test_float16_inference_requires_cuda():
    config = RunConfig(device="cpu", inference_dtype="float16")

    with pytest.raises(ValueError, match="requires CUDA"):
        config.validate()


def test_inference_dtype_defaults_to_float16_on_cuda_and_float32_on_cpu():
    config = RunConfig()

    assert config.resolved_inference_dtype("cuda") == "float16"
    assert config.resolved_inference_dtype("cpu") == "float32"


def test_float16_wrapper_uses_channels_last_and_returns_float32():
    class LayoutProbe(torch.nn.Module):
        def forward(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            assert inputs.dtype == torch.float16
            assert inputs.is_contiguous(memory_format=torch.channels_last)
            batch_size = inputs.size(0)
            return (
                torch.zeros(batch_size, 64, dtype=inputs.dtype),
                torch.zeros(batch_size, 1, dtype=inputs.dtype),
            )

    policy, value = _Float16InferenceWrapper(LayoutProbe())(torch.zeros(2, 3, 8, 8))

    assert policy.dtype == torch.float32
    assert value.dtype == torch.float32


def test_cpu_runtime_defaults_are_tuned_for_selfplay():
    config = RunConfig(device="cpu", selfplay_games_per_iter=64)

    assert config.resolved_torch_threads("cpu") == 4
    assert config.resolved_selfplay_batch_size("cpu") == 32
    assert config.resolved_selfplay_game_concurrency("cpu") == 16
    assert config.resolved_train_num_workers("cpu") == 0
    assert config.selfplay_expansion_batch_size == 4
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
