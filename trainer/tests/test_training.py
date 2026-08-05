"""
Tests for the training system.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from reversi_zero_trainer.data import (
    ReplayBufferDataset,
    SelfPlayDataset,
    SymmetryAugmentedDataset,
)
from reversi_zero_trainer.models.dummy import DummyReversiNet, ResNetReversiNet
from reversi_zero_trainer.training import AlphaZeroTrainer, TrainingConfig


@pytest.fixture
def dummy_training_data():
    """Create dummy training data for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create small dummy dataset
        num_samples = 100
        states = np.random.randn(num_samples, 3, 8, 8).astype(np.float32)

        # Random policy distributions (must sum to 1)
        policies = np.random.rand(num_samples, 64).astype(np.float32)
        policies = policies / policies.sum(axis=1, keepdims=True)

        # Random values in [-1, 1]
        values = np.random.uniform(-1, 1, num_samples).astype(np.float32)

        # Save to disk
        np.save(tmpdir / "states.npy", states)
        np.save(tmpdir / "policies.npy", policies)
        np.save(tmpdir / "values.npy", values)

        yield tmpdir


def test_selfplay_dataset_loading(dummy_training_data):
    """Test that SelfPlayDataset can load data correctly."""
    dataset = SelfPlayDataset(dummy_training_data)

    assert len(dataset) == 100

    # Check a single sample
    state, policy, value = dataset[0]
    assert state.shape == (3, 8, 8)
    assert policy.shape == (64,)
    assert value.shape == ()

    # Check stats
    stats = dataset.get_stats()
    assert stats["num_samples"] == 100
    assert "mean_value" in stats
    assert "std_value" in stats


def test_replay_buffer_lazily_concatenates_iterations(tmp_path):
    data_dirs = []
    for iteration, (num_samples, marker, value) in enumerate(
        [(2, 1.0, -1.0), (3, 2.0, 1.0)]
    ):
        data_dir = tmp_path / f"selfplay_iter_{iteration}"
        data_dir.mkdir()
        states = np.full((num_samples, 3, 8, 8), marker, dtype=np.float32)
        policies = np.full((num_samples, 64), marker, dtype=np.float32)
        values = np.full(num_samples, value, dtype=np.float32)
        np.save(data_dir / "states.npy", states)
        np.save(data_dir / "policies.npy", policies)
        np.save(data_dir / "values.npy", values)
        data_dirs.append(data_dir)

    dataset = ReplayBufferDataset(data_dirs)

    assert len(dataset) == 5
    assert dataset[1][0][0, 0, 0].item() == pytest.approx(1.0)
    assert dataset[2][0][0, 0, 0].item() == pytest.approx(2.0)
    assert dataset[-1][2].item() == pytest.approx(1.0)
    stats = dataset.get_stats()
    assert stats["mean_value"] == pytest.approx(0.2)
    assert stats["positive_values"] == 3
    assert stats["negative_values"] == 2


def test_replay_buffer_requires_data():
    with pytest.raises(ValueError, match="at least one"):
        ReplayBufferDataset([])


def test_symmetry_augmentation_transforms_state_and_policy_together(tmp_path):
    grid = np.arange(64, dtype=np.float32).reshape(8, 8)
    states = np.zeros((1, 3, 8, 8), dtype=np.float32)
    states[0, 0] = grid
    policies = grid.reshape(1, 64)
    values = np.array([0.5], dtype=np.float32)
    np.save(tmp_path / "states.npy", states)
    np.save(tmp_path / "policies.npy", policies)
    np.save(tmp_path / "values.npy", values)

    base_dataset = SelfPlayDataset(tmp_path)
    dataset = SymmetryAugmentedDataset(base_dataset, symmetry_count=8)

    assert len(dataset) == 8
    transformed_policies = set()
    for idx in range(8):
        state, policy, value = dataset[idx]
        assert torch.equal(state[0].reshape(64), policy)
        assert value.item() == pytest.approx(0.5)
        transformed_policies.add(tuple(policy.tolist()))
    assert len(transformed_policies) == 8
    assert torch.equal(dataset[0][0], base_dataset[0][0])
    assert torch.equal(dataset[0][1], base_dataset[0][1])


@pytest.mark.parametrize("symmetry_count", [1, 2, 4, 8])
def test_symmetry_augmentation_supported_lengths(dummy_training_data, symmetry_count):
    base_dataset = SelfPlayDataset(dummy_training_data)
    dataset = SymmetryAugmentedDataset(
        base_dataset,
        symmetry_count=symmetry_count,
    )

    assert len(dataset) == len(base_dataset) * symmetry_count


def test_symmetry_augmentation_rejects_invalid_count(dummy_training_data):
    with pytest.raises(ValueError, match="one of 1, 2, 4, or 8"):
        SymmetryAugmentedDataset(  # type: ignore[arg-type]
            SelfPlayDataset(dummy_training_data),
            symmetry_count=3,
        )


def test_trainer_initialization():
    """Test that trainer can be initialized."""
    model = DummyReversiNet()
    config = TrainingConfig(
        batch_size=32,
        num_workers=0,
        num_epochs=2,
        symmetry_augmentation=1,
        device="cpu",
    )

    trainer = AlphaZeroTrainer(model=model, config=config)

    assert trainer.batch_step == 0
    assert trainer.total_epochs_trained == 0


def test_bfloat16_training_config_rejects_cpu():
    with pytest.raises(ValueError, match="bfloat16 training requires CUDA"):
        TrainingConfig(device="cpu", dtype="bfloat16")


def test_training_single_epoch(dummy_training_data):
    """Test that we can train for a single epoch."""
    model = DummyReversiNet()
    config = TrainingConfig(
        batch_size=32,
        num_workers=0,
        num_epochs=2,
        symmetry_augmentation=1,
        device="cpu",
    )

    trainer = AlphaZeroTrainer(model=model, config=config)

    # Train for one iteration
    metrics_list = list(trainer.train(data_path=dummy_training_data, num_epochs=1))

    assert len(metrics_list) == 1
    metrics = metrics_list[0]

    # Check that all expected metrics are present
    assert "loss/total" in metrics
    assert "loss/policy" in metrics
    assert "loss/value" in metrics
    assert "eval/loss_total" in metrics
    assert "eval/value_mae" in metrics
    assert "eval/value_correlation" in metrics
    assert metrics["epoch"] == 0
    assert metrics["batch_step"] > 0


def test_training_uses_augmented_data_but_evaluation_does_not(
    dummy_training_data,
):
    model = DummyReversiNet()
    config = TrainingConfig(
        batch_size=32,
        num_workers=0,
        symmetry_augmentation=8,
        device="cpu",
    )
    trainer = AlphaZeroTrainer(model=model, config=config)

    training_dataloader, evaluation_dataloader = trainer._create_dataloaders(
        dummy_training_data
    )

    assert len(training_dataloader.dataset) == 800
    assert len(evaluation_dataloader.dataset) == 100


def test_training_applies_symmetry_after_replay_concatenation(dummy_training_data):
    model = DummyReversiNet()
    config = TrainingConfig(
        batch_size=32,
        num_workers=0,
        symmetry_augmentation=8,
        device="cpu",
    )
    trainer = AlphaZeroTrainer(model=model, config=config)

    training_dataloader, evaluation_dataloader = trainer._create_dataloaders(
        [dummy_training_data, dummy_training_data]
    )

    assert len(training_dataloader.dataset) == 1600
    assert len(evaluation_dataloader.dataset) == 200


def test_training_multiple_epochs(dummy_training_data):
    """Test that we can train for multiple epochs."""
    model = DummyReversiNet()
    config = TrainingConfig(
        batch_size=32,
        num_workers=0,
        num_epochs=3,
        symmetry_augmentation=1,
        device="cpu",
    )

    trainer = AlphaZeroTrainer(model=model, config=config)

    metrics_list = list(trainer.train(data_path=dummy_training_data, num_epochs=3))

    assert len(metrics_list) == 3
    assert metrics_list[0]["epoch"] == 0
    assert metrics_list[1]["epoch"] == 1
    assert metrics_list[2]["epoch"] == 2


def test_trainer_reuse_with_different_data(dummy_training_data):
    """Test that trainer can be reused with different datasets."""
    model = DummyReversiNet()
    config = TrainingConfig(
        batch_size=32,
        num_workers=0,
        num_epochs=2,
        symmetry_augmentation=1,
        device="cpu",
    )

    trainer = AlphaZeroTrainer(model=model, config=config)

    # First training
    metrics1 = list(trainer.train(data_path=dummy_training_data, num_epochs=2))
    assert len(metrics1) == 2
    assert trainer.total_epochs_trained == 2

    # Second training (simulating new self-play data)
    metrics2 = list(trainer.train(data_path=dummy_training_data, num_epochs=2))
    assert len(metrics2) == 2
    assert trainer.total_epochs_trained == 4  # Cumulative


def test_checkpoint_save_and_load(dummy_training_data):
    """Test checkpoint saving and loading."""
    model = DummyReversiNet()
    config = TrainingConfig(
        batch_size=32,
        num_workers=0,
        num_epochs=2,
        symmetry_augmentation=1,
        device="cpu",
    )

    trainer = AlphaZeroTrainer(model=model, config=config)

    # Train for a bit
    list(trainer.train(data_path=dummy_training_data, num_epochs=2))

    # Save checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        config.checkpoint_dir = Path(tmpdir)
        checkpoint_path = trainer.save_checkpoint(
            epoch=2, filename="test_checkpoint.pt"
        )

        assert checkpoint_path.exists()
        assert not (Path(tmpdir) / ".test_checkpoint.pt.next").exists()

        # Create new trainer and load checkpoint
        new_model = DummyReversiNet()
        new_trainer = AlphaZeroTrainer(model=new_model, config=config)

        new_trainer.load_checkpoint(checkpoint_path)

        assert new_trainer.total_epochs_trained == 2
        assert new_trainer.batch_step == trainer.batch_step

        checkpoint = torch.load(checkpoint_path, weights_only=False)
        del checkpoint["config"].symmetry_augmentation
        del checkpoint["config"].dtype
        legacy_checkpoint_path = Path(tmpdir) / "legacy_checkpoint.pt"
        torch.save(checkpoint, legacy_checkpoint_path)

        legacy_trainer = AlphaZeroTrainer(model=DummyReversiNet(), config=config)
        legacy_trainer.load_checkpoint(legacy_checkpoint_path)

        assert legacy_trainer.config.symmetry_augmentation == 1
        assert legacy_trainer.config.dtype == "float32"


def test_resnet_model():
    """Test that ResNet model works with trainer."""
    model = ResNetReversiNet(in_channels=3, channels=32, num_blocks=2)
    config = TrainingConfig(
        batch_size=16,
        num_workers=0,
        num_epochs=1,
        device="cpu",
    )

    trainer = AlphaZeroTrainer(model=model, config=config)

    # Just check it doesn't crash
    assert trainer is not None


def test_loss_computation():
    """Test loss computation."""
    model = DummyReversiNet()
    config = TrainingConfig(device="cpu")
    trainer = AlphaZeroTrainer(model=model, config=config)

    # Create dummy tensors
    batch_size = 4
    policy_logits = torch.randn(batch_size, 64)
    value_pred = torch.randn(batch_size, 1)

    target_policy = torch.softmax(torch.randn(batch_size, 64), dim=1)
    target_value = torch.randn(batch_size)

    total_loss, policy_loss, value_loss = trainer.compute_loss(
        policy_logits, value_pred, target_policy, target_value
    )

    assert total_loss.numel() == 1
    assert policy_loss.numel() == 1
    assert value_loss.numel() == 1
    assert total_loss.item() > 0
