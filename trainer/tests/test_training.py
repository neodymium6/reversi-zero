"""
Tests for the training system.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from reversi_zero_trainer.data import SelfPlayDataset
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


def test_trainer_initialization():
    """Test that trainer can be initialized."""
    model = DummyReversiNet()
    config = TrainingConfig(
        batch_size=32,
        num_workers=0,
        num_epochs=2,
        device="cpu",
    )

    trainer = AlphaZeroTrainer(model=model, config=config)

    assert trainer.batch_step == 0
    assert trainer.total_epochs_trained == 0


def test_training_single_epoch(dummy_training_data):
    """Test that we can train for a single epoch."""
    model = DummyReversiNet()
    config = TrainingConfig(
        batch_size=32,
        num_workers=0,
        num_epochs=2,
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


def test_training_multiple_epochs(dummy_training_data):
    """Test that we can train for multiple epochs."""
    model = DummyReversiNet()
    config = TrainingConfig(
        batch_size=32,
        num_workers=0,
        num_epochs=3,
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

        # Create new trainer and load checkpoint
        new_model = DummyReversiNet()
        new_trainer = AlphaZeroTrainer(model=new_model, config=config)

        new_trainer.load_checkpoint(checkpoint_path)

        assert new_trainer.total_epochs_trained == 2
        assert new_trainer.batch_step == trainer.batch_step


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
