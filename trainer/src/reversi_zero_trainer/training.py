"""
AlphaZero training loop implementation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Literal, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

from reversi_zero_trainer.data import (
    ReplayBufferDataset,
    SelfPlayDataset,
    SymmetryAugmentedDataset,
)

TrainingDataSource = Path | str | Sequence[Path | str]


@dataclass
class TrainingConfig:
    """Configuration for AlphaZero training."""

    # Training
    batch_size: int = 256
    num_workers: int = 4
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    symmetry_augmentation: Literal[1, 2, 4, 8] = 8

    # Device
    device: Literal["cuda", "cpu"] = "cuda"

    # Checkpointing
    checkpoint_dir: Path | str = "checkpoints"
    save_every_n_epochs: int = 1

    def __post_init__(self) -> None:
        if self.symmetry_augmentation not in (1, 2, 4, 8):
            raise ValueError("symmetry_augmentation must be one of 1, 2, 4, or 8")
        self.checkpoint_dir = Path(self.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)


class AlphaZeroTrainer:
    """
    Trainer for AlphaZero-style networks.

    Combines policy loss (cross-entropy with MCTS distribution) and
    value loss (MSE with game outcome).

    This trainer is designed to be reusable across multiple self-play iterations.
    You can call train() multiple times with different data paths.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        optimizer: Optimizer | None = None,
    ):
        self.model = model
        self.config = config
        self.device = torch.device(config.device)

        # Move model to device
        self.model.to(self.device)

        # Setup optimizer
        if optimizer is None:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )
        else:
            self.optimizer = optimizer

        # Training state (persists across multiple train() calls)
        self.batch_step = 0  # Total number of batches processed
        self.total_epochs_trained = 0

    def _create_dataloaders(
        self, data_path: TrainingDataSource
    ) -> tuple[DataLoader, DataLoader]:
        """Create augmented training and unmodified evaluation dataloaders."""
        evaluation_dataset: SelfPlayDataset | ReplayBufferDataset
        if isinstance(data_path, (Path, str)):
            evaluation_dataset = SelfPlayDataset(data_path)
        else:
            evaluation_dataset = ReplayBufferDataset(list(data_path))
        training_dataset = (
            evaluation_dataset
            if self.config.symmetry_augmentation == 1
            else SymmetryAugmentedDataset(
                evaluation_dataset,
                symmetry_count=self.config.symmetry_augmentation,
            )
        )
        common_options = {
            "batch_size": self.config.batch_size,
            "num_workers": self.config.num_workers,
            "pin_memory": self.config.device == "cuda",
        }
        training_dataloader = DataLoader(
            training_dataset,
            shuffle=True,
            **common_options,
        )
        evaluation_dataloader = DataLoader(
            evaluation_dataset,
            shuffle=False,
            **common_options,
        )
        return training_dataloader, evaluation_dataloader

    def compute_loss(
        self,
        policy_logits: torch.Tensor,
        value_pred: torch.Tensor,
        target_policy: torch.Tensor,
        target_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute combined AlphaZero loss.

        Args:
            policy_logits: (B, 64) raw policy logits
            value_pred: (B, 1) predicted value
            target_policy: (B, 64) target MCTS visit distribution
            target_value: (B,) target game outcome

        Returns:
            total_loss: combined loss
            policy_loss: cross-entropy loss
            value_loss: MSE loss
        """
        # Policy loss: cross-entropy between MCTS distribution and network output
        # Note: target_policy is already a probability distribution from MCTS
        log_probs = F.log_softmax(policy_logits, dim=1)
        policy_loss = -(target_policy * log_probs).sum(dim=1).mean()

        # Value loss: MSE between predicted value and actual outcome
        value_pred = value_pred.squeeze(1)  # (B, 1) -> (B,)
        value_loss = F.mse_loss(value_pred, target_value)

        # Combined loss
        total_loss = (
            self.config.policy_loss_weight * policy_loss
            + self.config.value_loss_weight * value_loss
        )

        return total_loss, policy_loss, value_loss

    def train_epoch(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: DataLoader for training data

        Returns:
            Dictionary of average metrics for the epoch.
        """
        self.model.train()

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        for batch_idx, (states, target_policies, target_values) in enumerate(
            dataloader
        ):
            # Move to device
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)

            # Forward pass
            policy_logits, value_pred = self.model(states)

            # Compute loss
            total_loss, policy_loss, value_loss = self.compute_loss(
                policy_logits, value_pred, target_policies, target_values
            )

            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss_sum += total_loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            num_batches += 1
            self.batch_step += 1

        # Return average metrics
        return {
            "loss/total": total_loss_sum / num_batches,
            "loss/policy": policy_loss_sum / num_batches,
            "loss/value": value_loss_sum / num_batches,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """
        Evaluate model on full dataset without training.

        Args:
            dataloader: DataLoader for evaluation data

        Returns:
            Dictionary of evaluation metrics.
        """
        self.model.eval()

        total_loss_sum = 0.0
        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        num_batches = 0

        # Track value predictions for analysis
        value_preds = []
        value_targets = []

        for states, target_policies, target_values in dataloader:
            states = states.to(self.device)
            target_policies = target_policies.to(self.device)
            target_values = target_values.to(self.device)

            # Forward pass
            policy_logits, value_pred = self.model(states)

            # Compute loss
            total_loss, policy_loss, value_loss = self.compute_loss(
                policy_logits, value_pred, target_policies, target_values
            )

            # Accumulate metrics
            total_loss_sum += total_loss.item()
            policy_loss_sum += policy_loss.item()
            value_loss_sum += value_loss.item()
            num_batches += 1

            # Track predictions
            value_preds.append(value_pred.squeeze(1).cpu())
            value_targets.append(target_values.cpu())

        # Concatenate all predictions
        all_preds = torch.cat(value_preds)
        all_targets = torch.cat(value_targets)

        # Compute additional metrics
        value_mae = (all_preds - all_targets).abs().mean().item()
        value_correlation = torch.corrcoef(torch.stack([all_preds, all_targets]))[
            0, 1
        ].item()

        return {
            "eval/loss_total": total_loss_sum / num_batches,
            "eval/loss_policy": policy_loss_sum / num_batches,
            "eval/loss_value": value_loss_sum / num_batches,
            "eval/value_mae": value_mae,
            "eval/value_correlation": value_correlation,
        }

    def save_checkpoint(self, epoch: int, filename: str | None = None) -> Path:
        """
        Save model checkpoint.

        Args:
            epoch: Current epoch number
            filename: Optional custom filename (default: checkpoint_epoch_{epoch}.pt)

        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{epoch}.pt"

        if not isinstance(self.config.checkpoint_dir, Path):
            self.config.checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_path = self.config.checkpoint_dir / filename

        checkpoint = {
            "epoch": epoch,
            "total_epochs_trained": self.total_epochs_trained,
            "batch_step": self.batch_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
        }

        next_path = checkpoint_path.with_name(f".{checkpoint_path.name}.next")
        try:
            torch.save(checkpoint, next_path)  # nosec B614
            next_path.replace(checkpoint_path)
        finally:
            next_path.unlink(missing_ok=True)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: Path | str) -> None:
        """
        Load model checkpoint and restore training state.

        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )  # nosec B614

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.total_epochs_trained = checkpoint["total_epochs_trained"]
        self.batch_step = checkpoint["batch_step"]

        # Restore config if available
        if "config" in checkpoint:
            self.config = checkpoint["config"]
            if "symmetry_augmentation" not in vars(self.config):
                self.config.symmetry_augmentation = 1
            # Ensure checkpoint_dir is a Path object
            self.config.checkpoint_dir = Path(self.config.checkpoint_dir)

    def train(
        self,
        data_path: TrainingDataSource,
        num_epochs: int | None = None,
    ) -> Generator[dict[str, float], None, None]:
        """
        Run training loop, yielding metrics for each epoch.

        This method can be called multiple times with different data paths
        to continue training on new self-play data.

        Args:
            data_path: One or more directories containing self-play NPY files
            num_epochs: Number of epochs to train. If None, uses config.num_epochs

        Yields:
            Dictionary containing training and evaluation metrics for each epoch

        Example:
            # Train on initial data
            for metrics in trainer.train("data/selfplay_iter1"):
                logger.log_metric("loss", metrics["loss/total"])

            # Later, train on new data
            for metrics in trainer.train("data/selfplay_iter2"):
                logger.log_metric("loss", metrics["loss/total"])
        """
        training_dataloader, evaluation_dataloader = self._create_dataloaders(data_path)

        if num_epochs is None:
            num_epochs = self.config.num_epochs

        for epoch in range(num_epochs):
            # Train one epoch
            train_metrics = self.train_epoch(training_dataloader)

            # Evaluate
            eval_metrics = self.evaluate(evaluation_dataloader)

            # Combine metrics
            metrics = {
                **train_metrics,
                **eval_metrics,
                "epoch": self.total_epochs_trained,
                "batch_step": self.batch_step,
            }

            self.total_epochs_trained += 1

            yield metrics
